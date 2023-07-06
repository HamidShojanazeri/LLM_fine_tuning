import torch

def clear_gpu_cache():
    """Clear the GPU cache """
  
    torch.cuda.empty_cache()

def print_model_size(model) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
    """
   
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> {model.config._name_or_path} has {total_params / 1e6} Million params\n")
