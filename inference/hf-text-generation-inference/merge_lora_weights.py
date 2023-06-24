import fire
import torch
from peft import PeftModel
from transformers import AutoCausalLM


def main(**kwargs):
    assert "base_model" in kwargs, "Required argument base_model missing"
    assert "peft_model" in kwargs, "Required argument peft_model missing"
    assert "output_dir" in kwargs, "Required argument output_dir missing"
    
    LORA_WEIGHTS = "arthurangelici/opt-6.7b-lora-caramelo"
        
    model = AutoCausalLM.from_pretrained(
        kwargs["base_model"],
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp", 
    )
        
    model = PeftModel.from_pretrained(
        model, 
        kwargs["peft_model"], 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp", 

    )

    model = model.merge_and_unload()
    model.save_pretrained(kwargs["output_dir"])


if __name__ == "__main__":
    fire.Fire(main)