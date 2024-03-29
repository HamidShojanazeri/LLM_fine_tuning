# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from auditnlg.safety.exam import safety_scores
import fire
import torch
import os
import sys
from typing import List

from peft import PeftModel, PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM

def perform_safety_check(text, score_threshold, name):
    score = safety_scores(
        data=[{"output": text}], method="Salesforce/safety-flan-t5-small"
    )
    if score[0][0] < score_threshold:
        print(f"{name} failed safety check. Exiting")
        sys.exit(0)

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    min_new_tokens:int=200, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility 
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    max_length: int=0, #The maximum length the generated tokens can have, input prompt+max_new_tokens
    min_length: int=0, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):
    assert safety_score_threshold >= 0.5

    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = "\n".join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    perform_safety_check(user_prompt, safety_score_threshold, "User prompt")

    print(f"User prompt:\n{user_prompt}")
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "bos_token": "</s>",
            "unk_token": "</s>",
            "pad_token": '[PAD]',
        }
    )

    if peft_model:
        print("we are in the peft model!!!!")
        # Load the Lora model
        model = PeftModel.from_pretrained(model, peft_model)

    model.eval()

    batch = tokenizer(user_prompt, return_tensors="pt")
    for k, v in batch.items():
        batch[k]=v.to("cuda:0")

    with torch.no_grad():
        # reference for generate args, https://huggingface.co/docs/transformers/main_classes/text_generation 
        outputs = model.generate(**batch,
                                 max_new_tokens=max_new_tokens,
                                 min_new_tokens=min_new_tokens,
                                 do_sample=do_sample,
                                 top_p=top_p,
                                 temperature=temperature,
                                 max_length=max_length,
                                 min_length=min_length,
                                 use_cache=use_cache,
                                 top_k=top_k,
                                 repetition_penalty=repetition_penalty,
                                 length_penalty=length_penalty,
                                 **kwargs                       
                                 )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    perform_safety_check(output_text, safety_score_threshold, "Model output")

    print(f"Model output:\n{output_text}")


if __name__ == "__main__":
    fire.Fire(main)