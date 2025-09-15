import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_positions, attn_dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, n_embd * 3, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        mask = torch.tril(torch.ones(n_positions, n_positions))
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(1))

    def forward(self, x):
        b, t, c = x.size()
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :t, :t]
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(2, 1).contiguous().reshape(b, t, c)
        return self.out(y)
