import torch
from torch.utils.data import DataLoader
from bpe import BPE
from model import GPTLow
from dataset import CharDataset

def train(
    text,
    vocab_size=200,
    n_positions=128,
    n_embd=256,
    n_head=8,
    n_layer=6,
    block_size=64,
    batch_size=8,
    epochs=1,
    lr=3e-4,
    device='cpu'
):
    bpe = BPE(vocab_size)
    bpe.fit(text)
    ids = bpe.encode(text)
    vocab_size = len(bpe.vocab)
    dataset = CharDataset(ids, block_size)
    if len(dataset) == 0:
        raise ValueError('text too short for block_size')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = GPTLow(vocab_size, n_positions, n_embd, n_head, n_layer, device=device)
    model.to(device)
    params = [p for p in model.parameters() if p is not None and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            _, loss = model.forward(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f'epoch {epoch+1} loss {avg:.4f}')
    return model, bpe
