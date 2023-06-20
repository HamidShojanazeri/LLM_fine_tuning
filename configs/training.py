from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
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
    dataset = "grammar_dataset"
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
    save_model: bool = True
    dist_checkpoint_root_folder: str="model_checkpoints"
    dist_checkpoint_folder: str="fine-tuned"

    
    
    