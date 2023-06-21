from dataclasses import dataclass

@dataclass
class cnn_dailymail_dataset:
    dataset: str =  "cnn_dailymail_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
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
    data_path: str = "ft_datasets/alpaca_data.json"