# Fine-tuning with Single GPU

To run fine-tuning on a single GPU, we will  make use of two packages

1- [PEFT](https://huggingface.co/blog/peft) methods and in specific using HuggingFace [PEFT](https://github.com/huggingface/peft)library. 

2- [BitandBytes](https://github.com/TimDettmers/bitsandbytes) int8 quantization.

Given combination of PEFT and Int8 quantization, we would be able to fine_tune a LLaMA 7B model on one consumer grade GPU such as A10.


## How to run it?

Get access to a machine with one GPU or if using a multi-GPU macine please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id` and run the following. It runs by default with `grammar_dataset` for grammar correction application.


```bash
pip install -r requirements.txt
python ../llama_finetuning.py  --use_peft --peft_method lora --quantization 

```
The args used in the command above are:

* `--use_peft` boolean flag to enable PEFT methods in the script

* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`. 

* `--quantization` boolean flag to enable int8 quantization


## How to run with different datasets?

Currenty 4 datasets are supported that can be found in [Datasets config file](../configs/datasets.py).

* `grammar_dataset`
* `alpaca_dataset`
* `cnn_dailymail_dataset`
* `samsum_dataset`

to run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization  --dataset grammar_dataset

# alpaca_dataset

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization  --dataset alpaca_dataset

# cnn_dailymail_dataset

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization  --dataset cnn_dailymail_dataset

# samsum_dataset

python ../llama_finetuning.py  --use_peft --peft_method lora --quantization  --dataset samsum_dataset

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
model_path: str=""
data_path: str = "path to alpaca data json file"
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
one_gpu: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False

```

* [Datasets config file](../configs/datasets.py) provides the avaiable options for datasets.

* [peft config file](../configs/peft.py) provides the suported PEFT methods and respective settings that can be modified.  