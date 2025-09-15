import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import Embedding
from block import DecoderBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd, n_head, n_layer, ff_mult=4):
        super().__init__()
        self.n_positions = n_positions
        self.embed = Embedding(vocab_size, n_embd, n_positions)
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, n_positions, ff_mult) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.embed.tok_emb.weight

    def forward(self, idx, targets=None):
        tok = self.embed(idx)
        x = self.drop(tok)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.n_positions else idx[:, -self.n_positions:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
