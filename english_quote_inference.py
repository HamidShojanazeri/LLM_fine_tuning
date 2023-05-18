import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# peft_model_id = "ybelkada/opt-6.7b-lora"
peft_model_id = "llama-lora-outputs"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf", return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)

#Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

batch = tokenizer("Two things are infinite: ", return_tensors="pt")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))