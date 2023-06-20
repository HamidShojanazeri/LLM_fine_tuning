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
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from .memory_utils import MemoryTrace
import model_checkpointing
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
scaler = ShardedGradScaler()

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None):
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
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    scaler = torch.cuda.amp.GradScaler() if train_config.use_fp16 else None

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]
    results = {}
    best_val_loss = float("inf")

    for epoch in range(train_config.num_epochs):
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(tqdm(train_dataloader)):
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda')       
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss = loss / gradient_accumulation_steps
                
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer)
            if local_rank == 0 and eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                val_loss.append(best_val_loss)
                val_prep.append(eval_ppl)
                
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                
                if not train_config.enable_fsdp:
                    model.save_pretrained(train_config.output_dir)   
                else:
                    if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        model_checkpointing.save_model_checkpoint(
                            model, optimizer, rank, train_config, epoch=1
                        )
                    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config)
                        if fsdp_config.save_optimizer:
                            model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)

                    if fsdp_config.save_optimizer:
                        model_checkpointing.save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=1
                        )           

        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}")

    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    avg_eval_prep = sum(val_prep)/len(val_prep) if val_prep else float('inf')  # to handle case when validation is not run
    avg_eval_loss = sum(val_loss)/len(val_loss) if val_loss else float('inf')  # to handle case when validation is not run

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results['avg_eval_prep'] = avg_eval_prep
    results['avg_eval_loss'] = avg_eval_loss

    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer):
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
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda')
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