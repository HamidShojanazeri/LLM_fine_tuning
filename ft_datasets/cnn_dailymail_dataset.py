# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import datasets
from .utils import Concatenator

def get_preprocessed_cnn_dailymail(dataset_config, tokenizer, split):
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
    
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(chunk_size=1024), batched=True)
    return dataset
