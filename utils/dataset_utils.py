import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.generation_utils import generate_and_tokenize_prompt
from typing import Optional
import datasets

def get_sharded_datasets(
    data_path: str, 
    val_set_size: int = 0, 
    num_shards: int = 1
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
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        
    else:
        # Preprocess and tokenize train set
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
        
    # Create sharded datasets
    shard_dataset_train = train_data.shard(num_shards, index=0)
    shard_dataset_val = None if val_data is None else val_data.shard(num_shards, index=0)

    return shard_dataset_train, shard_dataset_val
