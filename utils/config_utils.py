from dataclasses import fields
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)

from configs import lora_config, llama_adapter_config, prefix_config, train_config
from configs.datasets import grammar_dataset, alpaca_dataset, cnn_dailymail_dataset


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")
                        
                        
def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)
    
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    
    config = configs[names.index(train_config.peft_method)]
    update_config(config, **kwargs)
    params = {k.name: getattr(config, k.name) for k in fields(config)}
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    
    return peft_config


def generate_dataset_config(train_config, kwargs):
    datasets = (grammar_dataset, alpaca_dataset, cnn_dailymail_dataset)
    names = tuple(ds.__name__ for ds in datasets)
    
    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"
    
    dataset_config = datasets[names.index(train_config.dataset)]
    update_config(dataset_config, **kwargs)
    
    return  dataset_config