#!/bin/bash

#SBATCH --job-name=Nano-2d-trainer-20b-8nodes

# Here, $1 and $2 are the first and second arguments from the command line respectively

#SBATCH --nodes=$1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=$2
#SBATCH --partition=train 
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# export PYTHONFAULTHANDLER=1

srun  torchrun --nproc_per_node $2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 fsdp_finetuning.py  --enable_fsdp --use_peft --peft_method lora
