import torch

class EmbeddingLow:
    def __init__(self, vocab_size, n_embd, n_positions, device='cpu'):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.tok_emb = torch.randn(vocab_size, n_embd, requires_grad=True, device=device)
        self.pos_emb = torch.randn(n_positions, n_embd, requires_grad=True, device=device)

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device).unsqueeze(0)
        return self.tok_emb[idx] + self.pos_emb[pos]

    def parameters(self):
        return [self.tok_emb, self.pos_emb]

    def to(self, device):
        self.tok_emb = self.tok_emb.to(device)
        self.pos_emb = self.pos_emb.to(device)
        return self
