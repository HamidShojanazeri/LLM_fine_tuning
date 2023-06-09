import pytest

@pytest.fixture
def tokenizer():
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
    tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
                "pad_token": '[PAD]',
            }
        )
    yield tokenizer
        

def test_cnn_dm(tokenizer):
    # from utils.datasets_utils import get_preprocessed_dataset
    from  utils import get_preprocessed_dataset
        
    dataset = get_preprocessed_dataset(tokenizer, "cnn_dailymail", split="train[0:100]")
    
    print(len(next(iter(dataset))["input_ids"]))
    print(tokenizer.decode(next(iter(dataset))["input_ids"]))
    
    
def test_grammar_dataset(tokenizer):
    from grammer_dataset import get_dataset
    
    dataset = get_dataset(tokenizer, csv_name="grammer_dataset/grammar_validation.csv")
    
    print(next(iter(dataset)))
    