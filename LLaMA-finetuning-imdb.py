import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
import transformers
from transformers import default_data_collator, get_linear_schedule_with_warmup
from datasets import load_dataset
import fire
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType, AdaptionPromptConfig

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
    # dataset_name: str ="Abirate/english_quotes",
    dataset_name: str ="imdb",
    per_device_train_batch_size: int=4,
    gradient_accumulation_steps: int=1,
    warmup_steps: int=100,
    # max_steps: int=200,
    num_epochs: int = 4,
    learning_rate: float=1.4e-4,
    fp16: bool=True,
    logging_steps: int=1,
    output_dir: str="llama-adapter-1-4e-4-bs04-epochs10-4layers",  
):
    model = AutoModelForSequenceClassification.from_pretrained(
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
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)
    
    def format_dataset(dataset):
        dataset = dataset.remove_columns(['text'])
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset
    
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",inference_mode=False
    )
    # config = AdaptionPromptConfig(adapter_len=10,adapter_layers=4, task_type="CAUSAL_LM")
    # config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=30)
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    dataset = load_dataset(dataset_name)
    train_dataset = dataset['train'].train_test_split(test_size=0.2)
    # The split dataset has 'train' and 'test' keys
    train_data = train_dataset['train']
    test_data = train_dataset['test']
    tokenized_train_dataset = train_data.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_data.map(tokenize_function, batched=True)
    tokenized_train_dataset = format_dataset(tokenized_train_dataset)
    tokenized_test_dataset = format_dataset(tokenized_test_dataset)
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

    # data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    # Load the dataset

    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=(len(data["train"]) * num_epochs),
    # )
    
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
        ),
        data_collator=collate_fn,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
