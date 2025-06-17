from transformers import GPT2Tokenizer
import os

def get_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_function(tokenizer, max_length=256):
    def _tokenize(examples):
        result = tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        result["labels"] = result["input_ids"].copy()
        return result
    return _tokenize

def tokenize_dataset(dataset, tokenizer, cache_dir="./cache", max_length=256, batch_size=1000, num_proc=None):
    os.makedirs(cache_dir, exist_ok=True)
    return dataset.map(
        tokenize_function(tokenizer, max_length),
        batched=True,
        batch_size=batch_size,
        remove_columns=['text'],
        desc="Tokenizing",
        cache_file_name=os.path.join(cache_dir, "tokenized_dataset.arrow"),
        **({"num_proc": num_proc} if num_proc else {})
    )