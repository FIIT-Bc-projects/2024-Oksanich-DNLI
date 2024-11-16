from typing import Dict

from flwr.common import NDArrays, Scalar, Context
from flwr.client import NumPyClient, ClientApp

from sentiment_analysis.task import Transformer, get_weights, set_weights, train, test, load_data

from transformers import AutoModel

import torch


class FlowerClient(NumPyClient):
    def __init__(self, model, training_loader, validation_loader):
        super().__init__()

        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.model: Transformer = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        loss, accuracy = train(self.model, self.training_loader, self.device)

        torch.cuda.empty_cache()

        return get_weights(self.model), len(self.training_loader), {"loss": loss, "accuracy": accuracy}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.model, parameters)

        loss, accuracy = test(self.model, self.validation_loader, self.device)

        torch.cuda.empty_cache()

        return float(loss), len(self.validation_loader), {"accuracy": accuracy}


def client_fn(context: Context):
    tf = AutoModel.from_pretrained("bert-base-uncased")
    model = Transformer(tf, num_classes=3, freeze=False)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    training_loader, validation_loader = load_data(partition_id, num_partitions)

    return FlowerClient(model, training_loader, validation_loader).to_client()


app = ClientApp(
    client_fn,
)