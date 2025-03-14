from datasets import Dataset

from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents, ServerApp

from torch.utils.data import DataLoader

from transformers import AutoModel

from typing import List, Tuple

from .strategy import CustomFedAvg
from .task import *

import pandas as pd
import torch


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def gen_evaluate_fn(validation_loader: DataLoader, device):
    def evaluate(server_round, parameters_ndarrays, config):
        distilbert_tf = AutoModel.from_pretrained("distilbert-base-uncased", attn_implementation="eager")
        model = Transformer(distilbert_tf, num_classes=3, freeze=False)

        set_weights(model, parameters_ndarrays)

        model.to(device)

        loss, accuracy, precision, recall, f1_score = test(model, validation_loader, device, -1)

        return loss, {
            "centralized_accuracy": accuracy,
            "centralized_precision": precision,
            "centralized_recall": recall,
            "centralized_f1_score": f1_score,
        }

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]

    distilbert_tf = AutoModel.from_pretrained("distilbert-base-uncased", attn_implementation="eager")
    model = Transformer(distilbert_tf, num_classes=3, freeze=False)

    #checkpoint = torch.load("model/model_state_acc_0.5252777777777777_round_3.pth", weights_only=True)
    #model.load_state_dict(checkpoint)

    global_model_init = ndarrays_to_parameters(get_weights(model))

    validation_data = pd.concat([
        preprocess_data(pd.read_json("data/dynasent-v1.1-round01-yelp-test.jsonl", lines=True)),
        preprocess_data(pd.read_json("data/dynasent-v1.1-round01-yelp-dev.jsonl", lines=True)),
    ], ignore_index=True).drop_duplicates()
    validation_ds = Dataset.from_pandas(validation_data, preserve_index=False)
    validation_ds = tokenize_data(validation_ds, distilbert_tokenizer)

    validation_loader = DataLoader(
        dataset=validation_ds,
        batch_size=24,
        collate_fn=get_collate_fn(distilbert_tokenizer.pad_token_id),
        pin_memory=True,
    )

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_fn=gen_evaluate_fn(validation_loader, server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_model_init,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)