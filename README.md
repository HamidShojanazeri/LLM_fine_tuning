## LLM_fine_tuning


### Run the Alpaca dataset with LORA method from PEFT

```bash
python fine_tune.py     --base_model 'decapoda-research/llama-7b-hf'     --data_path 'yahma/alpaca-cleaned'     --output_dir './lora-alpaca'     --batch_size 128     --micro_batch_size 4     --num_epochs 1     --learning_rate 1e-4     --cutoff_len 512     --val_set_size 2000     --lora_r 8     --lora_alpha 16     --lora_dropout 0.05     --lora_target_modules '[q_proj,v_proj]'     --train_on_inputs     --group_by_length
```


## Run with FSDP 

```bash
 torchrun --nnodes 1 --nproc_per_node 4  fsdp_finetuning.py 
 ```
 