# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

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

def main(model_name, peft_model=None, quantization=False, max_new_tokens=100, prompt_file=None):
    
    if prompt_file is not None:
        assert os.path.exists(prompt_file), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = '\n'.join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = '\n'.join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
        
    print(f"User prompt:\n{user_prompt}")
    
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    if peft_model:
        # Load the Lora model
        model = PeftModel.from_pretrained(model, peft_model)

    model.eval()

    batch = tokenizer(user_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=max_new_tokens)
        print(f"Model output:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    fire.Fire(main)
