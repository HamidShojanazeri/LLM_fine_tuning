import sys
import os 

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from scipy.stats import gmean


from vllm import LLM
from vllm import LLM, SamplingParams
import fire

import random

# def generate_random_input_ids(size, vocab_size):
#     input_ids = [random.randint(0, vocab_size - 1) for _ in range(size)]
#     return input_ids

def byte2gb(x):
    return int(x / 2**30)

def run_benchmark(model_name, prompt_file, max_new_tokens, num_iterations,quantization=True, vLLM=False, tp_size=1):
    
    if prompt_file is not None:
        assert os.path.exists(prompt_file), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = '\n'.join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = '\n'.join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
        
    if vLLM:
        print("we are in the vLLM branch")
        model = LLM(model_name, tensor_parallel_size=tp_size)
        sampling_param = SamplingParams(top_p=0.9, temperature=0.7, max_tokens=max_new_tokens)
    else:
        print("we are in the HF branch")
        
             
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True if quantization else None,
                                                device_map="auto")
        #model.to(torch.bfloat16)
        #model.to("cuda:0")
        
    total_time_per_token = []
 
    for i in range(num_iterations):
        print(f"starting the benchmark iteraion {i}")
       

        with torch.no_grad():
            start_time = time.perf_counter()
            if vLLM:
                outputs = model.generate(user_prompt,sampling_params=sampling_param )
            else:  
                inputs = tokenizer(user_prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to("cuda:0")
                print(f"**** input_ids length **** {input_ids.size()}")
                # model.to("cuda:0")
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                )
            end_time = time.perf_counter()
        print(f"memoy allocated {byte2gb(torch.cuda.memory_allocated())} GB after inference")
        print(f"max memory reseved {byte2gb(torch.cuda.max_memory_reserved())} GB after inference")
        inference_time = (end_time - start_time)*1000
        time_per_token = inference_time / max_new_tokens
        total_time_per_token.append(time_per_token)
    mean = sum(total_time_per_token[1:])/ len(total_time_per_token[1:])
    print(f" time per token list {total_time_per_token}")
    geometric_mean = gmean(total_time_per_token)
    if vLLM:
        print(f"model output:\n {user_prompt} {outputs[0].outputs[0].text}")
    else:
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated text:", generated_text)
    print("Geometric mean time per token: {:.8f} ms".format(geometric_mean))
    print("The mean time per token: {:.8f} ms".format(mean))


if __name__ == "__main__":
    fire.Fire(run_benchmark)
