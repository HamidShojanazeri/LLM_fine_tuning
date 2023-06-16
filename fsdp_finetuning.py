import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset
import os.path as osp
from tqdm import tqdm

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
)
import torch.distributed as dist
from utils.generation_utils import Prompter, generate_and_tokenize_prompt, tokenize

# Unused imports removed
from utils.train_utils import (
    set_tokenizer_params,
    train,
    evaluation,
    freeze_transformer_layers,
    check_frozen_layers_peft_model,
)
from utils.dataset_utils import (
    get_sharded_datasets,
    InstructionDataset,
    get_preprocessed_dataset,
)
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import get_peft_model, TaskType, prepare_model_for_int8_training
import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)
from anyprecision_optimizer import AnyPrecisionAdamW
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened
import policies
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(cfg, rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if rank == 0:
            print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")

    return mixed_precision_policy


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def main(**kwargs):
    update_config((train_config, fsdp_config), **kwargs)

    torch.cuda.manual_seed(fsdp_config.seed)
    torch.manual_seed(fsdp_config.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size

    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        load_in_8bit=True if train_config.quantization else None,
    )
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    if fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if rank == 0:
        parameter_dtypes = get_parameter_dtypes(model)
        for name, dtype in parameter_dtypes.items():
            print(f"Parameter '{name}' dtype: {dtype}")

    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "bos_token": "</s>",
            "unk_token": "</s>",
            "pad_token": '[PAD]',
        }
    )
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    setup()

    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)

    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            
            freeze_transformer_layers(train_config.num_freeze_layers)

        mp_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        mp_policy = None
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy,
            mixed_precision=mp_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=False,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)

    dataset_config = generate_dataset_config(train_config, kwargs)

    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    if rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )

    if fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    train(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config.num_epochs,
        local_rank,
        train_config,
    )
    evaluation(model, eval_dataloader, local_rank, tokenizer)
    model.save_pretrained(train_config.output_dir)


if __name__ == "__main__":
    fire.Fire(main)