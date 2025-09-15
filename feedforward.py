import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, n_embd, ff_mult=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd * ff_mult)
        self.fc2 = nn.Linear(n_embd * ff_mult, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))
