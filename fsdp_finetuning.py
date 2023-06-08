import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset
import os.path as osp
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
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch.distributed as dist
from utils.generation_utils import Prompter, generate_and_tokenize_prompt, tokenize

from utils.train_utils import set_tokenizer_params, train, evaluation

from utils.dataset_utils import get_sharded_datasets
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType, AdaptionPromptConfig
import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)
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

verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)

def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")

def setup_environ_flags(cfg, rank):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if rank == 0:
            print(f"--> running with torch dist debug set to detail")


def cleanup():
    dist.destroy_process_group()

def clear_gpu_cache(rank=None):
    if rank == 0:
        print(f"clearing gpu cache for all ranks")
    torch.cuda.empty_cache() 

def get_policies(cfg, rank):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    # wrapping policy -------
    # print(f"**overriding mp to fp16 - remove")
    # mixed_precision_policy = policies.fpSixteen

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy

def get_parameter_dtypes(model):
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes
      
def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    num_shards: int=20,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # PEFT configs
    peft_method: str="lora",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
       
        assert (
            train_config.model_name
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
        
    #setup process group
    setup()
    torch.cuda.manual_seed(fsdp_config.seed)
    torch.manual_seed(fsdp_config.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
     
    
    if peft_method=="lora":
        from configs import lora_config
        print(lora_config)
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )
    elif peft_method=="llama_adapter":
        from configs import llama_adapter_config
        peft_config = AdaptionPromptConfig(
            adapter_len=llama_adapter_config.adapter_len,
            adapter_layers=llama_adapter_config.adapter_layers, 
            task_type=llama_adapter_config.task_type)
        
        
    elif peft_method=="prefix":
        from configs import prefix_config
        peft_config = PrefixTuningConfig(
            num_virtual_tokens=prefix_config.num_virtual_tokens,
            task_type=prefix_config.task_type
            )
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size



    model = AutoModelForSeq2SeqLM.from_pretrained(
        train_config.model_name,
        # load_in_8bit=True,
        # torch_dtype=torch.float16,
        # device_map=device_map,
    )
    if rank==0:
        parameter_dtypes = get_parameter_dtypes(model)
        for name, dtype in parameter_dtypes.items():
            print(f"Parameter '{name}' dtype: {dtype}")
        
    # model.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

    set_tokenizer_params(tokenizer)
    

    # model = prepare_model_for_int8_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.train()
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    #Getting fsdp configs
    mp_policy, wrapping_policy = get_policies(fsdp_config, rank)
    
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=False,
        # param_init_fn=my_init_fn
    )

    shard_dataset_train, shard_dataset_val = get_sharded_datasets(data_path, val_set_size, num_shards)
    
    train_sampler = DistributedSampler(
        shard_dataset_train, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True
    )

    if train_config.run_validation:
        val_sampler = DistributedSampler(
            shard_dataset_val, rank=dist.get_rank(), num_replicas=dist.get_world_size()
        )
    train_dataloader = torch.utils.data.DataLoader(
        shard_dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=False,
        sampler=train_sampler,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            shard_dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=False,
            sampler=val_sampler,
        )

    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model.bin"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")



    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    train(model, train_dataloader, optimizer, scheduler, gradient_accumulation_steps, num_epochs)
    evaluation(model, eval_dataloader)
    # model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)