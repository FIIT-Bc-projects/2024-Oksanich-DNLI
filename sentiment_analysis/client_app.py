from typing import Dict

from flwr.common import NDArrays, Scalar, Context
from flwr.client import NumPyClient, ClientApp

from .task import Transformer, get_weights, set_weights, train, test, load_data

from torch.utils.data import DataLoader

from transformers import AutoModel

import torch

#torch.cuda.set_per_process_memory_fraction(0.25, device=0)

class FlowerClient(NumPyClient):
    def __init__(self, model: Transformer, training_loader: DataLoader, validation_loader: DataLoader):
        super().__init__()

        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters: NDArrays, config):
        set_weights(self.model, parameters)

        loss, accuracy, precision = train(self.model, self.training_loader, self.device)

        torch.cuda.empty_cache()

        return get_weights(self.model), len(self.training_loader), {"loss": loss, "accuracy": accuracy, "precision": precision}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.model, parameters)

        loss, accuracy, precision = test(self.model, self.validation_loader, self.device)

        torch.cuda.empty_cache()

        return float(loss), len(self.validation_loader), {"accuracy": accuracy, "precision": precision}


def client_fn(context: Context) -> FlowerClient:
    tf = AutoModel.from_pretrained("distilbert-base-uncased")
    model = Transformer(tf, num_classes=3, freeze=False)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    training_loader, validation_loader = load_data(partition_id, num_partitions)

    return FlowerClient(model, training_loader, validation_loader).to_client()


app = ClientApp(
    client_fn,
)