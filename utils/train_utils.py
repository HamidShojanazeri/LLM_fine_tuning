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


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


def train(model, train_dataloader, num_epochs):
    for epoch in range(num_epochs):
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
       print("GPU Memory before entering the train : {}".format(b2mb(memtrace.begin)))
       print("GPU Memory consumed at the end of the train (end-begin): {}".format(memtrace.used))
       print("GPU Peak Memory consumed during the train (max-begin): {}".format(memtrace.peaked))
       print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                memtrace.peaked + b2mb(memtrace.begin)
            )
        )

       print("CPU Memory before entering the train : {}".format(b2mb(memtrace.cpu_begin)))
       print("CPU Memory consumed at the end of the train (end-begin): {}".format(memtrace.cpu_used))
       print("CPU Peak Memory consumed during the train (max-begin): {}".format(memtrace.cpu_peaked))
       print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                memtrace.cpu_peaked + b2mb(memtrace.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
       print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
       
def evaluation(model, eval_dataloader)    
    model.eval()
    eval_preds = []
    with MemoryTrace() as memtrace:
        for _, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather_for_metrics(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(memtrace.begin)))
    accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(memtrace.used))
    accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(memtrace.peaked))
    accelerator.print(
        "GPU Total Peak Memory consumed during the eval (max): {}".format(
            memtrace.peaked + b2mb(memtrace.begin)
        )
    )

    accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(memtrace.cpu_begin)))
    accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(memtrace.cpu_used))
    accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(memtrace.cpu_peaked))
    accelerator.print(
        "CPU Total Peak Memory consumed during the eval (max): {}".format(
            memtrace.cpu_peaked + b2mb(memtrace.cpu_begin)
        )
    )

    correct = 0
    total = 0
    assert len(eval_preds) == len(
        dataset["train"][label_column]
    ), f"{len(eval_preds)} != {len(dataset['train'][label_column])}"
    for pred, true in zip(eval_preds, dataset["train"][label_column]):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total * 100
    accelerator.print(f"{accuracy=}")
    accelerator.print(f"{eval_preds[:10]=}")
    accelerator.print(f"{dataset['train'][label_column][:10]=}")
