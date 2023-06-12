import os
import sys
from functools import partial
from itertools import chain
from typing import Any, List

import fire
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
import json
import copy
from sentencepiece import SentencePieceProcessor

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

VALID_DATASET = ["cnn_dailymail"]


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


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        # logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        # logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


class InstructionDataset(Dataset):
    def __init__(self, data_path, model_path, max_words=30, partition="train"):
        self.ann = json.load(open(data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer1.encode(prompt, bos=True, eos=False), dtype=torch.int64
        )
        example = torch.tensor(
            self.tokenizer1.encode(example, bos=True, eos=True), dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return example, labels, example_mask


residual = {"input_ids": [], "attention_mask": []}
def _get_preprocessed_cnn_dailymail(tokenizer, split):
    dataset = datasets.load_dataset("cnn_dailymail", "3.0.0" ,split=split)

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

    def concatenate_batches(batch, chunk_size=2048):
        global residual
        concatenated_samples = residual
        concatenated_samples = {k: v + list(chain(*batch[k])) for k, v in residual.items()}

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= chunk_size:
            chunk_num = total_length // chunk_size
            result = {
                k: [v[i : i + chunk_size] for i in range(0, chunk_num * chunk_size, chunk_size)]
                for k, v in concatenated_samples.items()
            }
            residual = {
                k: v[(chunk_num * chunk_size) :] for k, v in concatenated_samples.items()
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


def get_preprocessed_dataset(tokenizer, dataset_ident: str, split: str = "train") -> torch.utils.data.Dataset:
    if not dataset_ident in VALID_DATASET:
        raise NotImplemented

    return _get_preprocessed_cnn_dailymail(tokenizer, split)
