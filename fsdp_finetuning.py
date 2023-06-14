import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset
import os.path as osp
from tqdm import tqdm
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from utils import fsdp_auto_wrap_policy
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch.distributed as dist
from utils.generation_utils import Prompter, generate_and_tokenize_prompt, tokenize

from utils.train_utils import set_tokenizer_params, train, evaluation

from utils.dataset_utils import get_sharded_datasets, InstructionDataset

# from datasets import get_dataset
import grammer_dataset as dg
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

    # wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy #, wrapping_policy

def get_parameter_dtypes(model):
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes
      
def main(
    quantization: bool= False,
    one_gpu: bool= False,
):
    
    torch.cuda.manual_seed(fsdp_config.seed)
    torch.manual_seed(fsdp_config.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    #setup process group 
  
     
    peft_method = train_config.peft_method
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
    
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size

    
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        load_in_8bit=True if quantization else False,
        torch_dtype=torch.float16 if one_gpu else torch.float32,
        device_map="auto" if quantization else False,
    )
    if quantization:
        model = prepare_model_for_int8_training(model)
    
    if fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
        
    if rank==0:
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

    # set_tokenizer_params(tokenizer)
    

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    setup()
    
    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
    
    #Getting fsdp configs
    if train_config.train_strategy == "fsdp":
        mp_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        mp_policy = None
        model = FSDP(
            model,
            # use_orig_params=True,
            auto_wrap_policy=my_auto_wrapping_policy,
            mixed_precision=mp_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=False,
            # param_init_fn=my_init_fn
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
            
    # shard_dataset_train, shard_dataset_val = get_sharded_datasets(data_path, val_set_size, num_shards)
    if train_config.dataset == "grammer_dataset":
        dataset_train = dg.get_dataset(tokenizer, train_config.dataset_train, 512, 512, True)
        if 0 == os.getenv("RANK"):
            print(f"--> Training Set Len = {len(dataset_train)}")
            print(f"using dataset {train_config.dataset_train}")
        # print("bailing")

        dataset_val = dg.get_dataset(tokenizer,train_config.dataset_test, 512, 512, True)
        
    elif train_config.dataset == "alpaca":
        
        dataset_train = InstructionDataset(
            data_path=train_config.data_path, model_path=train_config.model_path, max_words=224, partition="train"
        )
        dataset_val = InstructionDataset(
            data_path=train_config.data_path, model_path=train_config.model_path, max_words=224, partition="val"
        )
    
    if train_config.train_strategy == "fsdp":
        train_sampler = DistributedSampler(
            dataset_train, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True
        )

    if train_config.run_validation:
        val_sampler = DistributedSampler(
            dataset_val, rank=dist.get_rank(), num_replicas=dist.get_world_size()
        )
 
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=False,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        # collate_fn = data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=False,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            # collate_fn = data_collator,
        )
  
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    train(model, train_dataloader, optimizer, scheduler, gradient_accumulation_steps, train_config.num_epochs, local_rank, train_config)
    evaluation(model, eval_dataloader, local_rank,tokenizer)
    # model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
