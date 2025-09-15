import math, torch, torch.nn.functional as F
from layers import linear

class MultiHeadSelfAttentionLow:
    def __init__(self, n_embd, n_head, n_positions, attn_dropout=0.0, device='cpu'):
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        # correct shapes: Wqkv: (3*n_embd, n_embd), Wout: (n_embd, n_embd)
        self.Wqkv = torch.randn(3 * n_embd, n_embd, requires_grad=True, device=device)
        self.Wout = torch.randn(n_embd, n_embd, requires_grad=True, device=device)
        self.attn_dropout = attn_dropout
        mask = torch.tril(torch.ones(n_positions, n_positions, device=device))
        self.causal_mask = mask.unsqueeze(0).unsqueeze(1)  # 1 x 1 x L x L

    def forward(self, x):
        b, t, c = x.size()
        qkv = linear(x, self.Wqkv).view(b, t, 3, self.n_head, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: b x n_head x t x head_dim
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :t, :t]
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        if self.attn_dropout > 0.0:
            att = F.dropout(att, p=self.attn_dropout, training=True)
        y = torch.matmul(att, v)  # b x n_head x t x head_dim
        y = y.permute(0,2,1,3).contiguous().view(b, t, c)
        return linear(y, self.Wout)

    def parameters(self):
        return [self.Wqkv, self.Wout]

    def to(self, device):
        self.Wqkv = self.Wqkv.to(device)
        self.Wout = self.Wout.to(device)
        self.causal_mask = self.causal_mask.to(device)
        return self
