import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y
