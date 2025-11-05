import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.input_ids = tokenized_texts["input_ids"]
        self.attention_mask = tokenized_texts["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

# Load tokenized data
from tokenize_data import tokenized_texts

dataset = TextDataset(tokenized_texts)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Test batch
for batch in dataloader:
    print(batch["input_ids"].shape)
    break
