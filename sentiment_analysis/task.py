from collections import OrderedDict

from datasets import Dataset

from flwr.common import NDArrays
from flwr_datasets.partitioner import IidPartitioner

from sklearn.metrics import accuracy_score, precision_score

from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from typing import Tuple

import numpy as np

import pandas as pd

import torch
import torch.nn as nn

import tqdm


class Transformer(nn.Module):
    def __init__(self, transformer, num_classes: int, freeze: bool):
        super().__init__()

        self.transformer = transformer
        self.fc = nn.Linear(transformer.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.transformer(ids, attention_mask=attention_mask, output_attentions=True)

        cls_hidden = output.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(torch.tanh(cls_hidden))

        prediction = self.fc(cls_hidden)

        return prediction, output.attentions


def get_weights(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, parameters: NDArrays):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    model.load_state_dict(state_dict, strict=True)


dataset = None
partitioner = None

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def load_data(partition_id: int, num_partitions: int, batch_size=24) -> Tuple[DataLoader, DataLoader]:
    global dataset, partitioner

    pad_index = tokenizer.pad_token_id

    def collate_fn(batch):
        batch_ids = nn.utils.rnn.pad_sequence([i["ids"] for i in batch], padding_value=pad_index, batch_first=True)
        batch_label = torch.stack([i["label"] for i in batch])
        batch_mask = nn.utils.rnn.pad_sequence([i["attention_mask"] for i in batch], padding_value=pad_index, batch_first=True)

        return {
            "ids": batch_ids,
            "label": batch_label,
            "attention_mask": batch_mask,
        }

    if dataset is None:
        label_mapping = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
        }

        training_data = pd.read_csv("data/twitter_training.csv")
        training_data = training_data[training_data.label != "Irrelevant"].drop(columns=["tweet_id", "entity"]).dropna()
        training_data["label"] = training_data["label"].map(label_mapping)

        validation_data = pd.read_csv("data/twitter_validation.csv")
        validation_data = validation_data[validation_data.label != "Irrelevant"].drop(columns=["tweet_id", "entity"]).dropna()
        validation_data["label"] = validation_data["label"].map(label_mapping)

        data = pd.concat([training_data, validation_data])
        data.drop_duplicates(inplace=True)

        dataset = Dataset.from_pandas(data, preserve_index=False)
        dataset = dataset.map(
            lambda s, t: {
                "ids": (encoded := t(s["text"], truncation=True))["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            fn_kwargs={"t": tokenizer},
        )
        dataset = dataset.with_format(type="torch", columns=["ids", "label", "attention_mask"])

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
        pin_memory=True,
    )

    validation_loader = DataLoader(
        dataset=validation_partition,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return training_loader, validation_loader


def get_accuracy(prediction, label):
    predicted_classes = prediction.argmax(dim=-1).cpu().numpy()
    actual_labels = label.cpu().numpy()

    return accuracy_score(actual_labels, predicted_classes)

def get_precision(prediction, label):
    predicted_classes = prediction.argmax(dim=-1).cpu().numpy()
    actual_labels = label.cpu().numpy()
    
    return precision_score(actual_labels, predicted_classes, average="macro", zero_division=0)


def train(model, data_loader, device) -> Tuple[np.float64, np.float64, np.float64]:
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()

    batch_losses = []
    batch_accuracies = []
    batch_precisions = []

    for batch in tqdm.tqdm(data_loader, desc="Training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        prediction, _ = model(ids, attention_mask)

        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        precision = get_precision(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        batch_accuracies.append(accuracy)
        batch_precisions.append(precision.item())

    return np.mean(batch_losses), np.mean(batch_accuracies), np.mean(batch_precisions)


def test(model, data_loader, device) -> Tuple[np.float64, np.float64, np.float64]:
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model.eval()

    batch_losses = []
    batch_accuracies = []
    batch_precisions = []

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            prediction = model(ids, attention_mask)

            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            precision = get_precision(prediction, label)

            batch_losses.append(loss.item())
            batch_accuracies.append(accuracy)
            batch_precisions.append(precision.item())

    return np.mean(batch_losses), np.mean(batch_accuracies), np.mean(batch_precisions)
