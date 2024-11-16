from collections import OrderedDict
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm


class Transformer(nn.Module):
    def __init__(self, transformer, num_classes, freeze):
        super().__init__()

        self.transformer = transformer
        self.fc = nn.Linear(transformer.config.hidden_size, num_classes)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        output = self.transformer(ids, output_attentions=True)

        cls_hidden = output.last_hidden_state[:, 0, :]

        prediction = self.fc(torch.tanh(cls_hidden))

        return prediction


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    model.load_state_dict(state_dict, strict=True)


dataset = None
partitioner = None

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def load_data(partition_id: int, num_partitions: int, batch_size=24):
    global dataset, partitioner

    def collate_fn(batch):
        batch_ids = nn.utils.rnn.pad_sequence(
            sequences=[i["ids"] for i in batch],
            padding_value=tokenizer.pad_token_id,
            batch_first=True,
        )

        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)

        return {
            "ids": batch_ids,
            "label": batch_label,
        }

    if dataset is None:
        label_mapping = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
        }

        training_data = pd.read_csv("data/twitter_training.csv", names=["tweet_id", "entity", "label", "text"])
        training_data = training_data[training_data.label != "Irrelevant"].drop(columns=["tweet_id", "entity"]).dropna()
        training_data["label"] = training_data["label"].map(label_mapping)

        validation_data = pd.read_csv("data/twitter_validation.csv", names=["tweet_id", "entity", "label", "text"])
        validation_data = validation_data[validation_data.label != "Irrelevant"].drop(columns=["tweet_id", "entity"]).dropna()
        validation_data["label"] = validation_data["label"].map(label_mapping)

        data = pd.concat([training_data, validation_data])
        data.drop_duplicates(inplace=True)

        dataset = Dataset.from_pandas(data, preserve_index=False)
        dataset = dataset.map(lambda s, t: {"ids": t(s["text"], truncation=True)["input_ids"]}, fn_kwargs={"t": tokenizer})
        dataset = dataset.with_format(type="torch", columns=["ids", "label"])

        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset

    partition = partitioner.load_partition(partition_id)

    train_test = partition.train_test_split(test_size=0.2, seed=42)
    training_partition = train_test["train"]
    validation_partition = train_test["test"]

    training_loader = DataLoader(
        dataset=training_partition,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    validation_loader = DataLoader(
        dataset=validation_partition,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    return training_loader, validation_loader


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size

    return accuracy


def train(model, data_loader, device):
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()

    epoch_losses = []
    epoch_accuracies = []

    for batch in tqdm.tqdm(data_loader, desc="Training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)

        prediction = model(ids)

        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy.item())

    return np.mean(epoch_losses), np.mean(epoch_accuracies)


def test(model, data_loader, device):
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model.eval()

    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)

            prediction = model(ids)

            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy.item())

    return np.mean(epoch_losses), np.mean(epoch_accuracies)