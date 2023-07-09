import sys
import os 

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from scipy.stats import gmean
from optimum.bettertransformer import BetterTransformer
from utils import clear_gpu_cache, print_model_size
from accelerate import Accelerator

from vllm import LLM
from vllm import LLM, SamplingParams
import fire

import random
import csv

# Function to convert bytes to gigabytes
def byte2gb(x):
    return int(x / 2**30)

# Main function to run the benchmark
def run_benchmark(model_name,
                  prompt_file,
                  max_new_tokens,
                  num_iterations,
                  quantization=False,
                  vLLM=False,
                  tp_size=1,
                  dtype=None,
                  BT=False,
                  profile=False,
                  batch_size=1):
    
    # Check if a prompt file is provided and read from it
    if prompt_file is not None:
        assert os.path.exists(prompt_file), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = '\n'.join(f.readlines())
    # If no prompt file, read from stdin
    elif not sys.stdin.isatty():
        user_prompt = '\n'.join(sys.stdin.readlines())
    else:
        # If no input is provided, exit the program
        print("No user prompt provided. Exiting.")
        sys.exit(1)
        
    #clear GPU cache
    clear_gpu_cache()
    
    # If vLLM is True, use LLM model
    if vLLM:
        print("we are in the vLLM branch")
        model = LLM(model_name, tensor_parallel_size=tp_size)
        sampling_param = SamplingParams(top_p=0.9, temperature=0.7, max_tokens=max_new_tokens)
    else:
        # If vLLM is False, use AutoTokenizer and LlamaForCausalLM
        print("we are in the HF branch")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True if quantization else None,
                                                device_map="auto",
                                                low_cpu_mem_usage=True)
                                                # revision="main",
                                                # offload_folder="offload",
                                                # offload_state_dict=True,
                                                # torch_dtype=torch.float16)
        
        # print_model_size(model)   
            
        if not quantization:
            if dtype is not None and dtype=="bf16":
                model.to(torch.bfloat16)
            elif dtype is not None and dtype=="fp16":
                model.to(torch.float16)
            
            model.to("cuda:0") 
            
        if BT:                                   
            model = BetterTransformer.transform(model)
    total_time_per_token = []
 
    for i in range(num_iterations):
        print(f"starting the benchmark iteraion {i}")
       

        with torch.no_grad():
            if profile:
                torch.cuda.cudart().cudaProfilerStart()
            start_time = time.perf_counter()
            if vLLM:
                prompt_list = [user_prompt for _ in range(batch_size)]
                print("prompt list len *********", len(prompt_list))
                outputs = model.generate(prompt_list,sampling_params=sampling_param )
            else:  
                inputs = tokenizer(user_prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to("cuda:0")
                batch_input_ids = input_ids.expand(batch_size, -1) 
                print("******* batch size is batch_input_ids", batch_input_ids.size())
                outputs = model.generate(
                    batch_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    use_cache=True,
                )
            end_time = time.perf_counter()
            if profile:
                torch.cuda.cudart().cudaProfilerStop()
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
    results = {
        'model_name': model_name,
        'prompt_file_name': prompt_file,
        'max_new_tokens': max_new_tokens,
        'num_iterations': num_iterations,
        'datatype': dtype,
        'BT': BT,
        'batch_size':batch_size,
        'mean_time_per_token': mean,
        'geometric_mean_time_per_token': geometric_mean,
        'vLLM':vLLM,
        'quantization':quantization
    }

      # Save the results to a CSV file
    fields = list(results.keys())
    file_exists = os.path.exists('results.csv')

    with open('results.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)


if __name__ == "__main__":
    fire.Fire(run_benchmark)
