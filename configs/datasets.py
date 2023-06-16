from dataclasses import dataclass

@dataclass
class cnn_dailymail_dataset:
    dataset: str =  "cnn_dailymail_dataset"
    train_split: str = "train[0:100]"
    test_split: str = "validation[0:100]"
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_1k.csv"  # grammar_13k.csv
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/data/home/hamidnazeri/stanford_alpaca/alpaca_data.json"
    model_path = "/data/home/hamidnazeri/LLM_fine_tuning/model/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348/"