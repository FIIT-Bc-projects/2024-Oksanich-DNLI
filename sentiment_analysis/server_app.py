from typing import List, Tuple

from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents, ServerApp
from flwr.server.strategy import FedAvg

from transformers import AutoModel

from .task import Transformer, get_weights


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]

    tf = AutoModel.from_pretrained("distilbert-base-uncased")
    model = Transformer(tf, num_classes=3, freeze=False)

    global_model_init = ndarrays_to_parameters(get_weights(model))

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_model_init,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)