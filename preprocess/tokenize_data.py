# tokenize_data.py
from transformers import AutoTokenizer
from .load_data import all_texts
from .clean_text import clean_text

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT2 does not have a pad token by default, set it
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Clean texts
cleaned_texts = [clean_text(t) for t in all_texts]

# Tokenize
tokenized_texts = tokenizer(
    cleaned_texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

print(tokenized_texts['input_ids'].shape)  # check tensor shape
