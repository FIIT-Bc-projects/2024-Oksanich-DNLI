from datasets import Dataset

from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents, ServerApp

from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer

from typing import List, Tuple

from .strategy import CustomFedAvg
from .task import Transformer, get_collate_fn, get_weights, set_weights, test, tokenize_data, label_mapping

import pandas as pd
import torch
import torch.nn as nn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def gen_evaluate_fn(
    validation_loader: DataLoader,
    device: torch.device,
):
    def evaluate(server_round, parameters_ndarrays, config):
        tf = AutoModel.from_pretrained("distilbert-base-uncased")
        model = Transformer(tf, num_classes=3, freeze=False)

        set_weights(model, parameters_ndarrays)

        model.to(device)

        loss, accuracy = test(model, validation_loader, device, -1)

        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]

    #tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tf = AutoModel.from_pretrained("distilbert-base-uncased")
    model = Transformer(tf, num_classes=3, freeze=False)

    global_model_init = ndarrays_to_parameters(get_weights(model))

    # validation_data = pd.read_csv("data/twitter_validation.csv")
    # validation_data = validation_data[validation_data.label != "Irrelevant"].drop(columns=["tweet_id", "entity"]).dropna()
    # validation_data["label"] = validation_data["label"].map(label_mapping)
    # validation_ds = Dataset.from_pandas(validation_data, preserve_index=False)
    # validation_ds = tokenize_data(validation_ds, tokenizer)

    # validation_loader = DataLoader(
    #     dataset=validation_ds,
    #     batch_size=24,
    #     collate_fn=get_collate_fn(tokenizer.pad_token_id),
    #     pin_memory=True,
    # )

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        evaluate_fn=None,#gen_evaluate_fn(validation_loader, server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_model_init,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)