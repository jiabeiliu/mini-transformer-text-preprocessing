import os
import torch
from tokenize_data import tokenized_texts

# Ensure the folder exists
output_folder = "sample_text_dataset"  # relative to Assign 1 folder
os.makedirs(output_folder, exist_ok=True)

# Save the tokenized sample
output_path = os.path.join(output_folder, "tokenized_sample.pt")
torch.save(tokenized_texts, output_path)

print(f"Tokenized sample saved to {output_path}")
