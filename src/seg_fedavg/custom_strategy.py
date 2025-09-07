from typing import Optional

from flwr.common import Parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from configs.fedavg_config import *
from src.seg_fedavg.model import UNetModel, dice_coef, dice_coef_loss, iou


class SaveModelStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters: Optional[Parameters] = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd, results, failures
        )
        if aggregated_parameters is not None:

            # Save the aggregated parameters (weights) after each round
            weights = parameters_to_ndarrays(aggregated_parameters)
            model = UNetModel().model
            model.compile(
                optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef, iou]
            )
            model.set_weights(weights)
            # Save with round number in filename
            filename = INITIAL_FED_PATH + PROBLEM_TYPE + f"global_model_round_{rnd}.h5"
            model.save(filename)

            self.final_parameters = aggregated_parameters  # Save final weights

        return aggregated_parameters, aggregated_metrics
