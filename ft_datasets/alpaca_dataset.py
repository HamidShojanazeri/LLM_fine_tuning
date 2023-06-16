import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

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


# class Tokenizer:
#     def __init__(self, model_path: str):
#         # reload tokenizer
#         assert os.path.isfile(model_path), model_path
#         self.sp_model = SentencePieceProcessor(model_file=model_path)
#         # logger.info(f"Reloaded SentencePiece model from {model_path}")

#         # BOS / EOS token IDs
#         self.n_words: int = self.sp_model.vocab_size()
#         self.bos_id: int = self.sp_model.bos_id()
#         self.eos_id: int = self.sp_model.eos_id()
#         self.pad_id: int = self.sp_model.pad_id()
#         # logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
#         assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

#     def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
#         assert type(s) is str
#         t = self.sp_model.encode(s)
#         if bos:
#             t = [self.bos_id] + t
#         if eos:
#             t = t + [self.eos_id]
#         return t

#     def decode(self, t: List[int]) -> str:
#         return self.sp_model.decode(t)


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_words=30, partition="train"):
        self.ann = json.load(open(data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

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
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
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

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
