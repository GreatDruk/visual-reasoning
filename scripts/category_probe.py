"""Single-category grounding probe on cached frozen features (train -> val, CPU).

Caption text is replaced by a learned embedding per unique caption string, so the probe
measures whether the cached image features carry the category's signal without the text
encoder or multi-category training in the way. Val captions unseen in train get a shared
UNK id. Usage: uv run python scripts/category_probe.py <feature_dir> <category> [epochs]
"""

import json
import sys

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from src.metrics import evaluate_scores

feat_dir, category = sys.argv[1], sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
torch.manual_seed(0)


def load(split):
    df = pl.read_csv(f"data/{split}.csv").filter(pl.col("category") == category)
    names = json.loads(open(f"{feat_dir}/{split}.json").read())
    pos = {n: i for i, n in enumerate(names)}
    arr = np.load(f"{feat_dir}/{split}.npy", mmap_mode="r")
    idx = np.array([pos[n] for n in df["image"]])
    uniq, inv = np.unique(idx, return_inverse=True)
    x = torch.from_numpy(np.asarray(arr[uniq])).float()
    x = F.avg_pool2d(x, 2) if x.shape[-1] > 12 else x
    return df, x, torch.from_numpy(inv)


tr, xtr, itr = load("train")
va, xva, iva = load("val")
vocab = {c: i + 1 for i, c in enumerate(sorted(set(tr["caption"])))}
ctr = torch.tensor([vocab[c] for c in tr["caption"]])
cva = torch.tensor([vocab.get(c, 0) for c in va["caption"]])
ytr = torch.tensor(tr["label"].to_numpy()).float()
C, H, W = xtr.shape[1:]


class Probe(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.cell = nn.Sequential(nn.Conv2d(C, d, 1), nn.BatchNorm2d(d))
        self.pos = nn.Parameter(torch.randn(1, d, H, W) * 0.02)
        self.q = nn.Embedding(len(vocab) + 1, d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x, c):
        k = (self.cell(x) + self.pos).flatten(2).transpose(1, 2)  # [B, N, d]
        q = self.q(c)
        att = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / k.shape[-1] ** 0.5, -1)
        z = (att.unsqueeze(-1) * k).sum(1)
        return self.out(torch.cat([z, q], -1)).squeeze(-1)


model = Probe()
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.05)
for ep in range(epochs):
    model.train()
    for b in torch.randperm(len(ytr)).split(64):
        loss = F.binary_cross_entropy_with_logits(model(xtr[itr[b]], ctr[b]), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 10 == 0 or ep + 1 == epochs:
        model.eval()
        with torch.no_grad():
            s = torch.cat([model(xva[iva[b]], cva[b]) for b in torch.arange(len(cva)).split(256)]).numpy()
            st = torch.cat([model(xtr[itr[b]], ctr[b]) for b in torch.arange(len(ctr)).split(256)]).numpy()
        m = evaluate_scores(s, va["label"].to_numpy(), va["category"].to_numpy(), va["pair_id"].to_numpy())
        mt = evaluate_scores(st, tr["label"].to_numpy(), tr["category"].to_numpy(), tr["pair_id"].to_numpy())
        print(f"{category} ep {ep + 1}: train auc {mt['auc']:.3f} | val auc {m['auc']:.3f} pair_acc {m['pair_accuracy']:.3f}", flush=True)
