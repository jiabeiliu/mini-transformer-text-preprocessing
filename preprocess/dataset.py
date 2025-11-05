import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    """Simple Dataset that loads tokenized blocks saved with torch.save({'input_ids':..., 'attention_mask':...})

    It expects the saved file to contain tensors of shape (N, block_size).
    """

    def __init__(self, tokenized_path):
        data = torch.load(tokenized_path)
        self.input_ids = data["input_ids"]
        self.attention_mask = data.get("attention_mask")

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx] if self.attention_mask is not None else None,
        }
