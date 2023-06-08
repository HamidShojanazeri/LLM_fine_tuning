from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str="decapoda-research/llama-7b-hf"
    run_validation: bool=True
    batch_size_training: int=4
    num_workers_dataloader: int=2
    lr: float=2e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=4
    dataset: str="grammer_dataset"
    dataset_train: str = "grammer_dataset/gtrain_1k.csv"  # grammar_13k.csv
    dataset_test: str = "grammer_dataset/grammar_validation.csv"

    
    
    