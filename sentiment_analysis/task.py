from collections import OrderedDict

from datasets import Dataset

from flwr.common import NDArrays
from flwr_datasets.partitioner import IidPartitioner

from sklearn.metrics import accuracy_score, precision_score, recall_score

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, PreTrainedTokenizer

import numpy as np

import os

import pandas as pd

import torch
import torch.nn as nn

import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Transformer(nn.Module):
    def __init__(self, transformer, num_classes: int, freeze: bool):
        super().__init__()

        self.transformer = transformer
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(transformer.config.hidden_size, num_classes)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False


    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.transformer(ids, attention_mask=attention_mask, output_attentions=True)
        pooled_mean = torch.mean(output.last_hidden_state, dim=1)
        cls_hidden = self.dropout(pooled_mean)
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

global distilbert_tokenizer
distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = nn.utils.rnn.pad_sequence([i["ids"] for i in batch], padding_value=pad_index, batch_first=True)
        batch_label = torch.stack([i["label"] for i in batch])
        batch_mask = nn.utils.rnn.pad_sequence([i["attention_mask"] for i in batch], padding_value=pad_index, batch_first=True)

        return {
            "ids": batch_ids,
            "label": batch_label,
            "attention_mask": batch_mask,
        }
    
    return collate_fn


def tokenize_data(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    copied = dataset.map(
        lambda s, tok: {
            "ids": (encoded := tok(s["text"], truncation=True, padding=True))["input_ids"],
            "attention_mask": encoded["attention_mask"],
        },
        fn_kwargs={"tok": tokenizer},
    )
    copied = copied.with_format(type="torch", columns=["ids", "label", "attention_mask"])

    return copied


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    copied = df[["sentence", "gold_label"]].rename(columns={"sentence": "text", "gold_label": "label"})
    copied = copied[copied.label != "mixed"].dropna()
    copied["label"] = copied["label"].map(label_mapping)

    return copied


def load_data(partition_id: int, num_partitions: int, batch_size: int=24) -> tuple[DataLoader, DataLoader]:
    global dataset, partitioner

    if dataset is None:
        data = preprocess_data(pd.read_json("data/dynasent-v1.1-round01-yelp-train.jsonl", lines=True))
        #validation_data = preprocess_data(pd.read_json("data/dynasent-v1.1-round01-yelp-test.jsonl", lines=True))

        #data = pd.concat([training_data, validation_data], ignore_index=True)
        #data.drop_duplicates(inplace=True)

        dataset = Dataset.from_pandas(data, preserve_index=False)
        dataset = tokenize_data(dataset, distilbert_tokenizer)

        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset

    partition = partitioner.load_partition(partition_id)

    train_test = partition.train_test_split(test_size=0.15, seed=42)
    training_partition = train_test["train"]
    validation_partition = train_test["test"]

    training_loader = DataLoader(
        dataset=training_partition,
        batch_size=batch_size,
        collate_fn=get_collate_fn(distilbert_tokenizer.pad_token_id),
        shuffle=True,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        dataset=validation_partition,
        batch_size=batch_size,
        collate_fn=get_collate_fn(distilbert_tokenizer.pad_token_id),
        pin_memory=True,
    )

    return training_loader, validation_loader


def get_accuracy(prediction, label) -> np.float64:
    predicted_classes = prediction.argmax(dim=-1).cpu().numpy()
    actual_labels = label.cpu().numpy()
    
    return accuracy_score(actual_labels, predicted_classes)

def get_precision(prediction, label) -> np.float64:
    predicted_classes = prediction.argmax(dim=-1).cpu().numpy()
    actual_labels = label.cpu().numpy()
    
    return precision_score(actual_labels, predicted_classes, average="macro", zero_division=0)

def get_recall(prediction, label) -> np.float64:
    predicted_classes = prediction.argmax(dim=-1).cpu().numpy()
    actual_labels = label.cpu().numpy()
    
    return recall_score(actual_labels, predicted_classes, average="macro", zero_division=0)

def get_f1_score(precision: np.float64, recall: np.float64) -> np.float64:
    return np.float64(2.0) * (precision * recall) / (precision + recall)


def train(
        model: Transformer,
        data_loader: DataLoader,
        device: torch.device,
        id: int,
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64]:
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()

    batch_losses = []
    batch_accuracies = []
    batch_precisions = []
    batch_recalls = []

    for batch in tqdm.tqdm(data_loader, desc=f"Training (partition {id})..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        prediction, _ = model(ids, attention_mask)

        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        precision = get_precision(prediction, label)
        recall = get_recall(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss)
        batch_accuracies.append(accuracy)
        batch_precisions.append(precision)
        batch_recalls.append(recall)

    avg_loss = np.mean(batch_losses)
    avg_accuracy = np.mean(batch_accuracies)
    avg_precision = np.mean(batch_precisions)
    avg_recall = np.mean(batch_recalls)
    f1_score = get_f1_score(avg_precision, avg_recall)

    return avg_loss, avg_accuracy, avg_precision, avg_recall, f1_score


def test(
        model: Transformer,
        data_loader: DataLoader,
        device: torch.device,
        id: int,
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64]:
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model.eval()

    batch_losses = []
    batch_accuracies = []
    batch_precisions = []
    batch_recalls = []

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc=f"Evaluating (partition {id})..." if id >= 0 else "Evaluating (centralized)..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            prediction, _ = model(ids, attention_mask)

            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            precision = get_precision(prediction, label)
            recall = get_recall(prediction, label)

            batch_losses.append(loss)
            batch_accuracies.append(accuracy)
            batch_precisions.append(precision)
            batch_recalls.append(recall)

    avg_loss = np.mean(batch_losses)
    avg_accuracy = np.mean(batch_accuracies)
    avg_precision = np.mean(batch_precisions)
    avg_recall = np.mean(batch_recalls)
    f1_score = get_f1_score(avg_precision, avg_recall)

    return avg_loss, avg_accuracy, avg_precision, avg_recall, f1_score
