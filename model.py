import torch
from embeddings import EmbeddingLow
from block import DecoderBlockLow
from layers import layer_norm

class GPTLow:
    def __init__(self, vocab_size, n_positions, n_embd, n_head, n_layer, ff_mult=4, device='cpu'):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.embed = EmbeddingLow(vocab_size, n_embd, n_positions, device=device)
        self.blocks = [DecoderBlockLow(n_embd, n_head, n_positions, ff_mult, device=device) for _ in range(n_layer)]
        self.ln_f_w = torch.ones(n_embd, requires_grad=True, device=device)
        self.ln_f_b = torch.zeros(n_embd, requires_grad=True, device=device)
    def forward(self, idx, targets=None):
        x = self.embed.forward(idx)
        for blk in self.blocks:
            x = blk.forward(x)
        x = layer_norm(x, self.ln_f_w, self.ln_f_b)
        logits = x.matmul(self.embed.tok_emb.t())
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.n_positions else idx[:, -self.n_positions:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def parameters(self):
        params = []
        params += self.embed.parameters()
        for b in self.blocks:
            params += b.parameters()
        params += [self.ln_f_w, self.ln_f_b]
        return params

    def to(self, device):
        self.embed.to(device)
        for i, b in enumerate(self.blocks):
            b.to(device)
            self.blocks[i] = b
        self.ln_f_w = self.ln_f_w.to(device); self.ln_f_b = self.ln_f_b.to(device)
        return self
