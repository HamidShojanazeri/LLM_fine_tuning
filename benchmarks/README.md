# Benchmarking Inference Solutions

Here, we will use the scripts in this folder to run benchmarks for different infernece solution on llama model to capture `time_per_token` latency.

[vll_benchmark.py](./vllm_benchmark.py) script provides benchmarking code for both [vLLM](https://vllm.ai/), HuggingFace accelerate and Bits&Bytes quantiztion.

To run the benchmarks for vLLM:

```bash
 python vllm_benchmark.py --model_name "/data/home/mreso/LLM_fine_tuning/models/7B/" --prompt_file prompt_2k.txt  --max_new_tokens 100 --num_iterations 30 --vLLM True

```

To run the benchmarks for Accelerate + int8 quantization:

```bash
 python vllm_benchmark.py --model_name "/data/home/mreso/LLM_fine_tuning/models/7B/" --prompt_file prompt_2k.txt  --max_new_tokens 100 --num_iterations 30 --quantization True

```

To run the benchmarks with bf16 or fp16:

```bash
 python vllm_benchmark.py --model_name "/data/home/mreso/LLM_fine_tuning/models/7B/" --prompt_file prompt_2k.txt  --max_new_tokens 100 --num_iterations 30 --dtype bf16

```


To run the benchmarks with bf16 or fp16 and [Accelerated Transformers (BT)](https://pytorch.org/blog/out-of-the-box-acceleration/) make sure to install HF Optimum library from src and run:

```bash
 python vllm_benchmark.py --model_name "/data/home/mreso/LLM_fine_tuning/models/7B/" --prompt_file prompt_2k.txt  --max_new_tokens 100 --num_iterations 30 --dtype bf16 --BT 

```

You can use different promot/context sizes by using [214 tokens](./prompt_small.txt), [1k](./prompt_1k.txt), [2k](./prompt_2k.txt),[3k](./prompt_3k.txt), [4k](./prompt_4k.txt).