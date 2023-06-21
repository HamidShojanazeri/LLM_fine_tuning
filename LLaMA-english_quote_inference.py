import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForSequenceClassification

peft_model_id = "ybelkada/opt-6.7b-lora"
# peft_model_id = "llama-adapter-imdb-4epochs-lr14e4"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf", return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

#Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

batch = tokenizer("The following is a list of companies and the categories they fall into:\n Apple, Facebook, Fedex\n Apple Category:", return_tensors="pt")

# with torch.no_grad():
#         outputs = model(**batch)
#         predictions = outputs.logits.argmax(dim=-1)
#         print(predictions)
    
with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=10, eos_token_id=3)
print("output_tokens", output_tokens)
print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
