import torch
from attention import MultiHeadSelfAttentionLow
from ffn import FeedForwardLow
from layers import layer_norm

class DecoderBlockLow:
    def __init__(self, n_embd, n_head, n_positions, ff_mult=4, attn_dropout=0.0, dropout=0.0, device='cpu'):
        self.ln1_w = torch.ones(n_embd, requires_grad=True, device=device)
        self.ln1_b = torch.zeros(n_embd, requires_grad=True, device=device)
        self.attn = MultiHeadSelfAttentionLow(n_embd, n_head, n_positions, attn_dropout, device=device)
        self.ln2_w = torch.ones(n_embd, requires_grad=True, device=device)
        self.ln2_b = torch.zeros(n_embd, requires_grad=True, device=device)
        self.ff = FeedForwardLow(n_embd, ff_mult, dropout, device=device)

    def forward(self, x):
        x = x + self.attn.forward(layer_norm(x, self.ln1_w, self.ln1_b))
        x = x + self.ff.forward(layer_norm(x, self.ln2_w, self.ln2_b))
        return x

    def parameters(self):
        params = [self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b]
        params += self.attn.parameters()
        params += self.ff.parameters()
        return params

    def to(self, device):
        self.ln1_w = self.ln1_w.to(device); self.ln1_b = self.ln1_b.to(device)
        self.ln2_w = self.ln2_w.to(device); self.ln2_b = self.ln2_b.to(device)
        self.attn.to(device); self.ff.to(device)
        return self
