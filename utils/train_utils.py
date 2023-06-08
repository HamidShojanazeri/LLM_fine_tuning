import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
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
from .memory_utils import MemoryTrace
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
scaler = ShardedGradScaler()

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, num_epochs, local_rank, train_config):
    for epoch in range(num_epochs):
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0
            # for step, (examples, labels, example_mask) in enumerate(tqdm(train_dataloader)):
            #     print(f"================== type(batch) : {type(examples)}, len(batch): {len(examples)}, actual example {examples.size()}===================")
            #     inputs = {'input_ids': examples, 'attention_mask': example_mask, 'labels': labels}
            for step, batch in enumerate(tqdm(train_dataloader)):
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                outputs = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"], labels=batch["target_ids"])
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss = loss / gradient_accumulation_steps
                if train_config.fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                loss.backward()
                if (step+1)% gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
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
    print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}")
       
def evaluation(model, eval_dataloader,local_rank ):
    model.eval()
    eval_preds = []
    metric = 0.0
    n_toks = 0
    with MemoryTrace() as memtrace:
        # for _, (examples, labels, example_mask) in enumerate(tqdm(eval_dataloader)):
        for step, batch in enumerate(tqdm(eval_dataloader)):
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            # batch = {k: v for k, v in examples.items() if k != "labels"}
            with torch.no_grad():
                pred = model.generate(
                    **batch, max_new_tokens=10
                )  
            
            loss = F.cross_entropy(pred.flatten(0, 1), labels=batch["target_ids"].flatten(0, 1), reduction="sum")
            metric += loss.item()
            n_toks += batch["target_ids"].nelement()


    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    print("GPU Memory before entering the eval : {}".format(byte2mb(memtrace.begin)))
    print("GPU Memory consumed at the end of the eval (end-begin): {}".format(memtrace.used))
    print("GPU Peak Memory consumed during the eval (max-begin): {}".format(memtrace.peaked))
    print(
        "GPU Total Peak Memory consumed during the eval (max): {}".format(
            memtrace.peaked + byte2mb(memtrace.begin)
        )
    )

    print("CPU Memory before entering the eval : {}".format(byte2mb(memtrace.cpu_begin)))
    print("CPU Memory consumed at the end of the eval (end-begin): {}".format(memtrace.cpu_used))
    print("CPU Peak Memory consumed during the eval (max-begin): {}".format(memtrace.cpu_peaked))
    print(
        "CPU Total Peak Memory consumed during the eval (max): {}".format(
            memtrace.cpu_peaked + byte2mb(memtrace.cpu_begin)
        )
    )
    if torch.device_count()>1:
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    eval_perplexity = torch.exp(torch.tensor(metric / n_toks))
    print(f"Evaluation perplexity: {eval_perplexity:.4f}")
    return eval_perplexity
    
   
