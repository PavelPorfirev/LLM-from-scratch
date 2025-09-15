import torch, torch.nn.functional as F
from layers import gelu, linear

class FeedForwardLow:
    def __init__(self, n_embd, ff_mult=4, dropout=0.0, device='cpu'):
        self.fc1_W = torch.randn(n_embd * ff_mult, n_embd, requires_grad=True, device=device)
        self.fc1_b = torch.zeros(n_embd * ff_mult, requires_grad=True, device=device)
        self.fc2_W = torch.randn(n_embd, n_embd * ff_mult, requires_grad=True, device=device)
        self.fc2_b = torch.zeros(n_embd, requires_grad=True, device=device)
        self.dropout = dropout

    def forward(self, x):
        x = linear(x, self.fc1_W, self.fc1_b)
        x = gelu(x)
        x = linear(x, self.fc2_W, self.fc2_b)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=True)
        return x

    def parameters(self):
        return [self.fc1_W, self.fc1_b, self.fc2_W, self.fc2_b]

    def to(self, device):
        self.fc1_W = self.fc1_W.to(device); self.fc1_b = self.fc1_b.to(device)
        self.fc2_W = self.fc2_W.to(device); self.fc2_b = self.fc2_b.to(device)
        return self
