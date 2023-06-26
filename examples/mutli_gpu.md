# Fine-tuning with Single GPU

To run fine-tuning on multi-GPUs, we will  make use of two packages

1- [PEFT](https://huggingface.co/blog/peft) methods and in specific using HuggingFace [PEFT](https://github.com/huggingface/peft)library. 

2- [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html) which help us to parallelize the training over mutiple GPUs, [more details](../README.md#2-full-partial-parameter-finetuning).

Given combination of PEFT and FSDP, we would be able to fine_tune a LLaMA model on multiple GPUs in one node of multi-node.


## How to run it?

Get access to a machine with mutiple GPUs ( in this case we tested with 4 A100 and A10s). It runs by default with `grammar_dataset` for grammar correction application.

**Multiple GPUs one node**:

```bash

pip install -r requirements.txt
torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

The args used in the command above are:

* `--enable_fsdp` boolean flag to enable FSDP  in the script

* `--use_peft` boolean flag to enable PEFT methods in the script

* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`. 

We use `torchrun` here to spawn multiple processes for FSDP.

**Multi GPU multi node**:

Here we use a slurm script to schedule a job with slurm over multiple nodes.

```bash

pip install -r requirements.txt
sbatch multi_node.slurm 
# Change the num nodes and GPU per nodes in the script before running.

```

## How to run with different datasets?

Currenty 4 datasets are supported that can be found in [Datasets config file](../configs/datasets.py).

* `grammar_dataset`
* `alpaca_dataset`
* `cnn_dailymail_dataset`
* `samsum_dataset`

to run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset
torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --dataset grammar_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

# alpaca_dataset

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --dataset alpaca_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

# cnn_dailymail_dataset

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --dataset cnn_dailymail_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

# samsum_dataset

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --dataset samsum_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

## Where to configure settings?

* [Training config file](../configs/training.py) is the main config file that help to specify the settings for our run can be found in

It let us specify the training settings, everything from `model_name` to `dataset_name`, `batch_size` etc. can be set here. Below is the list of supported settings:

```python

model_name: str="decapoda-research/llama-7b-hf"
enable_fsdp: bool= False 
run_validation: bool=True
batch_size_training: int=4
num_epochs: int=3
num_workers_dataloader: int=2
lr: float=2e-4
weight_decay: float=0.0
gamma: float= 0.85
use_fp16: bool=False
mixed_precision: bool=True
val_batch_size: int=4
dataset = "grammar_dataset" # alpaca_dataset, cnn_dailymail_dataset,samsum_dataset
micro_batch_size: int=1
peft_method: str = "lora" # None , llama_adapter, prefix
use_peft: bool=False
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False

```

* [Datasets config file](../configs/datasets.py) provides the avaiable options for datasets.

* [peft config file](../configs/peft.py) provides the suported PEFT methods and respective settings that can be modified.  

* [FSDP config file](../configs/fsdp.py) provides FSDP settings such as:

    * `mixed_precision` boolean flag to specify using mixed precision, defatult to true.

    * `use_fp16` boolean flag to specify using FP16 for mixed precision, defatult to False. We recommond not setting this flag, and only set `mixed_precision` that will use `BF16`, this would help with speed and memory savings while avoid challenges of scaler accuracies with `FP16`.

    *  `sharding_strategy` this specifies the sharding strategy for FSDP, it can be 
        * `FULL_SHARD` that shards model parameters, gradients and optimizer states, results in the most memory savings.

        * `SHARD_GRAD_OP` that shards gradinets and optimizer states and keep the parameters after the first `all_gather`. This reduces communication overhead specially if you are using slower networks more spcifically beneficial on multi-node cases. This comes with the trade off higher memory consumption.

        * `NO_SHARD` this is equivalant to DDP, does not shard model parameters, gradinets or optimizer states. It keep the full parameter after the first `all_gather`.

        * `HYBRID_SHARD` availbel on PyTorch Nightlies, it does FSDP within a node and DDP between nodes. It's agian for multi-node cases and helpful for slower networks, given your model will fit into one node. 

* `checkpoint_type` specifies the state dict checkpoint type for saving the model. `FULL_STATE_DICT` streams state_dict of each model shard from a rank to CPU and assembels the full state_dict on CPU. `SHARDED_STATE_DICT` saves one checkpoint per rank, and enables the re-loading the model in a different world size. 

* `fsdp_activation_checkpointing` enables activation checkpoining for FSDP, this saves siginificant amount of memory with the trade off of recomputing itermediate activations during the backward pass. The saved memory can be re-invest in higher batch sizes to increase the throughput. We recommond to use this option. 

* `pure_bf16` it moves the  model to `BFloat16` and if set `optimizer` to `anyprecision` then optimizer states will be kept in `BFloat16` as well. You can use this option if neccessary. 