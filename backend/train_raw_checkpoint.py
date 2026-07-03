"""
Train the raw PyTorch Transformer checkpoint used by AttentionForge Studio.

Run from project root:
    python backend/train_raw_checkpoint.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.attentionforge.model import MiniTransformerLM
from backend.attentionforge.tokenizer import DEFAULT_CORPUS, WordTokenizer


def build_training_text() -> str:
    # Repeat the compact corpus so the mini model can overfit enough for a clean demo.
    return (DEFAULT_CORPUS.strip() + "\n") * 80


def make_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def train():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = WordTokenizer.from_text(DEFAULT_CORPUS)
    text = build_training_text()
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    block_size = 32
    batch_size = 32
    steps = 350
    lr = 4e-4

    model = MiniTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=3,
        ff_hidden_dim=256,
        max_len=96,
        dropout=0.05,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"device={device}")
    print(f"vocab={tokenizer.vocab_size} tokens={len(data)} params={sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for step in range(1, steps + 1):
        x, y = make_batch(data, block_size, batch_size, device)
        logits, _ = model(x, causal_mask=True)
        loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step:04d}/{steps} loss={loss.item():.4f}")

    model.eval()
    ckpt_path = ROOT / "backend" / "checkpoints" / "raw_transformer_trained.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 3,
                "ff_hidden_dim": 256,
                "max_len": 96,
                "dropout": 0.0,
            },
            "training": {
                "steps": steps,
                "block_size": block_size,
                "batch_size": batch_size,
                "learning_rate": lr,
                "corpus": "focused AttentionForge Studio corpus repeated 80x",
                "final_loss": float(loss.item()),
            },
        },
        ckpt_path,
    )
    print(f"saved={ckpt_path}")

    for seed in ["attention", "earth", "bottle", "causal mask", "positional encoding"]:
        seed_ids = tokenizer.safe_encode(seed)
        idx = torch.tensor([seed_ids], dtype=torch.long, device=device)
        out = model.generate_raw(idx, max_new_tokens=45, temperature=0.65, top_k=8)[0].cpu().tolist()
        print("\nSEED:", seed)
        print(tokenizer.decode(out))


if __name__ == "__main__":
    train()
