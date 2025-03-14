from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from .task import Transformer, set_weights

from transformers import AutoModel

import json

import torch


class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cur_best_accuracy = 0.0
        self.results = {}
        self.base_dir = "./model"


    def _store_results(self, tag: str, results_dict):
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        with open(f"{self.base_dir}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)


    def update_best_accuracy(self, round: int, accuracy, parameters):
        if accuracy > self.cur_best_accuracy:
            self.cur_best_accuracy = accuracy

            ndarrays = parameters_to_ndarrays(parameters)

            distilbert_tf = AutoModel.from_pretrained("distilbert-base-uncased", attn_implementation="eager")
            model = Transformer(distilbert_tf, num_classes=3, freeze=False)

            set_weights(model, ndarrays)

            torch.save(model.state_dict(), f"{self.base_dir}/model_state_acc_{accuracy}_round_{round}.pth")


    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )


    def evaluate(self, server_round: int, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        loss = float(loss)
        metrics = {k: float(v) for k, v in metrics.items()}

        self.update_best_accuracy(server_round, metrics["centralized_accuracy"], parameters)

        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )

        return loss, metrics


    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        loss = float(loss)
        metrics = {k: float(v) for k, v in metrics.items()}

        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )

        return loss, metrics
