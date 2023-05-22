import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import tqdm
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from torch.nn import functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch.distributed as dist
from memory_utils import MemoryTrace

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, num_epochs):
    for epoch in range(num_epochs):
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss = loss / gradient_accumulation_steps
                loss.backward(loss)
                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        print("GPU Memory before entering the train : {}".format(byte2mb(memtrace.begin)))
        print("GPU Memory consumed at the end of the train (end-begin): {}".format(memtrace.used))
        print("GPU Peak Memory consumed during the train (max-begin): {}".format(memtrace.peaked))
        print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                memtrace.peaked + byte2mb(memtrace.begin)
            )
        )

        print("CPU Memory before entering the train : {}".format(byte2mb(memtrace.cpu_begin)))
        print("CPU Memory consumed at the end of the train (end-begin): {}".format(memtrace.cpu_used))
        print("CPU Peak Memory consumed during the train (max-begin): {}".format(memtrace.cpu_peaked))
        print(
                "CPU Total Peak Memory consumed during the train (max): {}".format(
                    memtrace.cpu_peaked + byte2mb(memtrace.cpu_begin)
                )
            )
    if torch.device_count()>1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_perplexity = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_perplexity=} {train_epoch_loss=}")
       
def evaluation(model, eval_dataloader):
    model.eval()
    eval_preds = []
    with MemoryTrace() as memtrace:
        for _, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                pred = model.generate(
                    **batch, max_new_tokens=10
                )  
            
            loss = F.cross_entropy(pred.flatten(0, 1), batch["labels"].flatten(0, 1), reduction="sum")
            metric += loss.item()
            n_toks += batch["labels"].nelement()


    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    print("GPU Memory before entering the eval : {}".format(b2mb(memtrace.begin)))
    print("GPU Memory consumed at the end of the eval (end-begin): {}".format(memtrace.used))
    print("GPU Peak Memory consumed during the eval (max-begin): {}".format(memtrace.peaked))
    print(
        "GPU Total Peak Memory consumed during the eval (max): {}".format(
            memtrace.peaked + byte2mb(memtrace.begin)
        )
    )

    print("CPU Memory before entering the eval : {}".format(b2mb(memtrace.cpu_begin)))
    print("CPU Memory consumed at the end of the eval (end-begin): {}".format(memtrace.cpu_used))
    print("CPU Peak Memory consumed during the eval (max-begin): {}".format(memtrace.cpu_peaked))
    print(
        "CPU Total Peak Memory consumed during the eval (max): {}".format(
            memtrace.cpu_peaked + byte2mb(memtrace.cpu_begin)
        )
    )

    return torch.exp(metric / n_toks)
    
   
