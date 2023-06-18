from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import fire
import torch
import os
import sys
from peft import PeftModel, PeftConfig
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)

from configs import train_config
from utils.config_utils import update_config

peft_model_id = "ft-output"
    

def main(**kwargs):
    update_config(train_config, **kwargs)
    
    max_new_tokens = kwargs.get("max_new_tokens", 100)
    
    if "prompt_file" in kwargs:
        assert os.path.exists(kwargs["prompt_file"]), f"Provided Prompt file does not exist {kwargs['prompt_file']}"
        with open(kwargs["prompt_file"], "r") as f:
            user_prompt = '\n'.join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = '\n'.join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
        
    print(f"User prompt:\n{user_prompt}")

    model_id = (
        train_config.output_dir
        if train_config.peft_method is None
        else train_config.model_name
    )
    
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        return_dict=True,
        load_in_8bit=train_config.quantization,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    if train_config.peft_method:
        # Load the Lora model
        model = PeftModel.from_pretrained(model, train_config.output_dir)

    model.eval()

    batch = tokenizer(user_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=max_new_tokens)
        print(f"Model output:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    fire.Fire(main)
