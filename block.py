import torch.nn as nn
from attention import MultiHeadSelfAttention
from feedforward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_positions, ff_mult=4, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, n_positions, attn_dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, ff_mult, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
