import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, n_embd, n_positions):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(n_positions, n_embd)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        return self.tok_emb(idx) + self.pos_emb(pos)
