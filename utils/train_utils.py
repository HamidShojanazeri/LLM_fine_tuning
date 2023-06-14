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
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
    
    Returns: train_perplexity, train_epoch_loss
    """
    # Create a gradient scaler for fp16
    scaler = torch.cuda.amp.GradScaler() if train_config.fp16 else None

    for epoch in range(num_epochs):
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(tqdm(train_dataloader)):
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss = loss / gradient_accumulation_steps
                
                if train_config.fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # perform optimization step if enough gradients have been accumulated
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

        #Printing the GPU and CPU memory usage details
        print("GPU Memory before entering the train : {}".format(byte2mb(memtrace.begin)))
        print("GPU Memory consumed at the end of the train (end-begin): {}".format(memtrace.used))
        print("GPU Peak Memory consumed during the train (max-begin): {}".format(memtrace.peaked))
        print("GPU Total Peak Memory consumed during the train (max): {}".format(memtrace.peaked + byte2mb(memtrace.begin)))
        print("CPU Memory before entering the train : {}".format(byte2mb(memtrace.cpu_begin)))
        print("CPU Memory consumed at the end of the train (end-begin): {}".format(memtrace.cpu_used))
        print("CPU Peak Memory consumed during the train (max-begin): {}".format(memtrace.cpu_peaked))
        print("CPU Total Peak Memory consumed during the train (max): {}".format(memtrace.cpu_peaked + byte2mb(memtrace.cpu_begin)))

        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        
        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}")
        return train_perplexity, train_epoch_loss

   
def evaluation(model, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader)):
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
                    
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )
    
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    
    # Print evaluation metrics
    print(f" {eval_ppl=} {eval_epoch_loss=}")
    return eval_ppl, eval_epoch_loss

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")