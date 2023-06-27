# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import datasets
import torch

from functools import partial

from ft_datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_cnn_dailymail_dataset,
    get_samsum_dataset,
)
from utils.generation_utils import generate_and_tokenize_prompt
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "cnn_dailymail_dataset": get_cnn_dailymail_dataset,
    "grammar_dataset": partial(get_grammar_dataset, num_samples=512, input_length=512),
    "samsum_dataset": get_samsum_dataset,
}


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


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
