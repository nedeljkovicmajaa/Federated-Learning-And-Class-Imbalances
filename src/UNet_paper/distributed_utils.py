import datetime
import time
from collections import defaultdict, deque

import torch
import torch.nn.functional as F

from src.UNet_paper.dice_coefficient_loss import (build_target,
                                                  multiclass_dice_coeff)


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(
            maxlen=window_size
        )  # A deque to store the values within the sliding window
        self.total = 0.0  # To keep track of the cumulative sum of the values
        self.count = 0  # Number of updates (n)
        self.fmt = fmt  # Format string for the output

    def update(self, value, n=1):
        self.deque.append(value)  # Append new value to the deque
        self.count += n  # Increment the count by n
        self.total += value * n  # Add value*n to the total for averaging

    @property
    def median(self):
        # Median of the values in the deque
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        # Average of the values in the deque
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # Global average across all updates
        return self.total / self.count

    @property
    def max(self):
        # Maximum value in the deque
        return max(self.deque)

    @property
    def value(self):
        # Most recent value in the deque
        return self.deque[-1]

    def __str__(self):
        # For printing the statistics
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # Initialize all zeros
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # Create a mask to find the pixels within valid class range (a >= 0 and a < n)
            k = (a >= 0) & (a < n)
            # Calculate the index of the confusion matrix (a[k] for the true class and b[k] for the predicted class)
            inds = n * a[k].to(torch.int64) + b[k]
            # Increment the corresponding confusion matrix values
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # Global accuracy: total correct predictions / total predictions
        acc_global = torch.diag(h).sum() / h.sum()
        # Accuracy per class: correct predictions per class / total samples per class
        acc = torch.diag(h) / h.sum(1)
        # Intersection over Union (IoU) for each class
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        # String representation of confusion matrix metrics
        acc_global, acc, iu = self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )


class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None  # Store cumulative dice score
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # Class index to ignore during computation
        self.count = None  # Number of updates

    def update(self, pred, target):
        # Update the cumulative dice score based on the current prediction and target
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        # One-hot encode the predicted output
        pred = (
            F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        )
        # Build the target for dice calculation
        dice_target = build_target(target, self.num_classes, self.ignore_index)
        # Update cumulative dice coefficient
        self.cumulative_dice += multiclass_dice_coeff(
            pred[:, 1:], dice_target[:, 1:], ignore_index=self.ignore_index
        )
        self.count += 1

    @property
    def value(self):
        # Return the average dice coefficient
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        # Reset the cumulative dice and count
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)  # Dictionary to track metrics
        self.delimiter = delimiter  # Delimiter for separating printed metrics

    def update(self, **kwargs):
        # Update the values for each metric
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()  # Convert tensor to number if necessary
            assert isinstance(
                v, (float, int)
            )  # Ensure that the value is a float or integer
            self.meters[k].update(v)  # Update the corresponding meter for each metric

    def __getattr__(self, attr):
        # Access meters dynamically using attribute access
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        # String representation for printing the metrics
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        # Add a custom meter for tracking
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        # Log and print metrics at regular intervals during iteration
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {}".format(header, total_time_str))
