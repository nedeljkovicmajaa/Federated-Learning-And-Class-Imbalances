import pytest
import torch
from src.MNIST_classification.client import *
from src.MNIST_classification.mnist_centralised import *
from src.MNIST_classification.model import *
from src.MNIST_classification.server import *
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def dummy_data():
    # Random dummy MNIST-like data (100 samples)
    x = torch.rand((100, 1, 28, 28))
    y = torch.randint(0, 10, (100,))
    return TensorDataset(x, y)


@pytest.fixture
def dummy_loader(dummy_data):
    return DataLoader(dummy_data, batch_size=10)


@pytest.fixture
def dummy_model():
    return SimpleNN()


@pytest.fixture
def dummy_optim(dummy_model):
    return torch.optim.SGD(dummy_model.parameters(), lr=0.01)


@pytest.fixture
def dummy_loss():
    return torch.nn.CrossEntropyLoss()


def test_model_forward_shape(dummy_model):
    dummy_input = torch.randn(4, 1, 28, 28)
    output = dummy_model(dummy_input)
    assert output.shape == (4, 10), "Model output should be (batch_size, 10)"


def test_train_loop_runs(dummy_model, dummy_loader, dummy_loss, dummy_optim):
    try:
        train(
            dummy_model, dummy_loader, dummy_loader, dummy_loss, dummy_optim, epochs=1
        )
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")


def test_evaluate_loop_runs(dummy_model, dummy_loader, dummy_loss):
    try:
        evaluate(dummy_model, dummy_loader, dummy_loss)
    except Exception as e:
        pytest.fail(f"Testing failed with error: {e}")


def test_partitioning_returns_subset(dummy_data):
    subset = partition_dataset(dummy_data, client_id=0, num_clients=5)
    assert isinstance(
        subset, torch.utils.data.Subset
    ), "Partitioning should return a Subset"
    assert (
        len(subset) == len(dummy_data) // 5
    ), "Subset length should be 1/num_clients of the original dataset"
