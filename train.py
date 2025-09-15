import torch
from torch.utils.data import DataLoader
from bpe import BPE
from model import GPT
from dataset import CharDataset

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

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
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    bpe = BPE(vocab_size)
    bpe.fit(text)
    ids = bpe.encode(text)
    vocab_size = len(bpe.vocab)
    dataset = CharDataset(ids, block_size)
    if len(dataset) == 0:
        raise ValueError('text too short for block_size')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = GPT(vocab_size, n_positions, n_embd, n_head, n_layer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    _, loss = model(xb, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                _, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f'epoch {epoch+1} loss {avg:.4f}')
    return model, bpe
