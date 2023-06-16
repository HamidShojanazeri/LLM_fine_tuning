import datasets
import torch

from functools import partial
from itertools import chain


from ft_datasets import grammar_dataset
from ft_datasets.alpaca_dataset import InstructionDataset
from utils.generation_utils import generate_and_tokenize_prompt
from typing import Optional


VALID_DATASET = ["alpaca_dataset", "cnn_dailymail_dataset", "grammar_dataset"]


def get_sharded_datasets(
    data_path: str, val_set_size: int = 0, num_shards: int = 1
) -> tuple[datasets.Dataset, Optional[datasets.Dataset]]:
    """
    Loads a dataset from a given path and returns sharded train and validation datasets.

    Args:
    - data_path (str): Path to the dataset file. Supports .json and .jsonl formats.
    - val_set_size (int): Size of the validation set. If 0, validation set is not created.
    - num_shards (int): Number of shards to split the dataset into.

    Returns:
    A tuple of train and validation datasets (or None if no validation set is created), each sharded into
    the specified number of shards.
    """

    # Load dataset from path
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = datasets.load_dataset("json", data_files=data_path)
    else:
        data = datasets.load_dataset(data_path)

    # Split dataset into train and validation sets if requested
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )

        # Preprocess and tokenize train and validation sets
        shard_dataset_train = train_val["train"].shard(num_shards, index=0)
        train_data = shard_dataset_train.shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"]
        shard_dataset_val = (
            None if val_data is None else val_data.shard(num_shards, index=0)
        )
        val_data = shard_dataset_val.shuffle().map(generate_and_tokenize_prompt)

    else:
        # Preprocess and tokenize train set
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Create sharded datasets
    # shard_dataset_train = train_data.shard(num_shards, index=0)
    # shard_dataset_val = None if val_data is None else val_data.shard(num_shards, index=0)

    return train_data, val_data


residual = {"input_ids": [], "attention_mask": []}


def _get_preprocessed_cnn_dailymail(tokenizer, split):
    dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split=split)

    prompt = (
        f"Summarize this article:\n{{article}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                article=sample["article"],
                summary=sample["highlights"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def concatenate_batches(batch, chunk_size=512):
        global residual
        concatenated_samples = residual
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= chunk_size:
            chunk_num = total_length // chunk_size
            result = {
                k: [
                    v[i : i + chunk_size]
                    for i in range(0, chunk_num * chunk_size, chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            residual = {
                k: v[(chunk_num * chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(concatenate_batches, batched=True)
    return dataset


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in VALID_DATASET:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    if dataset_config.dataset == "cnn_dailymail_dataset":
        return _get_preprocessed_cnn_dailymail(tokenizer, dataset_config.train_split)

    elif dataset_config.dataset == "grammar_dataset":
        return grammar_dataset.get_dataset(
            tokenizer,
            get_split(),
            512,
            512,
            True,
        )

    elif dataset_config.dataset == "alpaca_dataset":
        return InstructionDataset(
            data_path=dataset_config.data_path,
            tokenizer=tokenizer,
            max_words=224,
            partition=get_split(),
        )
