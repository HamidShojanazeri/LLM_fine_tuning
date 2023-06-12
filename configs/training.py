from dataclasses import dataclass
from typing import ClassVar

@dataclass
class cnn_dailymail_config:
    dataset: str =  "cnn_dailymail"
    train_split: str = "train[0:100]"
    test_split: str = "validation[0:100]"
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "grammer_dataset/gtrain_1k.csv"  # grammar_13k.csv
    test_split: str = "grammer_dataset/grammar_validation.csv"
    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca",
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/data/home/hamidnazeri/stanford_alpaca/alpaca_data.json",
    model_path = "/data/home/hamidnazeri/LLM_fine_tuning/model/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348/",

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
    # dataset_config = cnn_dailymail_config
    dataset_config = grammar_dataset
    fp16: bool=False

    
    
    