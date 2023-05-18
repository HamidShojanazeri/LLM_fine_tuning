import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset
import fire

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def train(
    model_name: str = "decapoda-research/llama-7b-hf",
    dataset_name: str ="Abirate/english_quotes",
    per_device_train_batch_size: int=4,
    gradient_accumulation_steps: int=4,
    warmup_steps: int=100,
    max_steps: int=200,
    learning_rate: float=2e-4,
    fp16: bool=True,
    logging_steps: int=1,
    output_dir: str="llama-lora-outputs",  
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
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
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    print("=====================================",len(data["train"]))
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
