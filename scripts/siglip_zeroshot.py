"""Zero-shot SigLIP2 image-caption scores on val: does the frozen dual encoder already
carry the signal? No fitting; val is only scored. Usage: uv run python scripts/siglip_zeroshot.py"""

import numpy as np
import polars as pl
import torch
import open_clip
from PIL import Image

from src.metrics import evaluate_scores
from src.utils import get_device

NAME, PRETRAINED = "ViT-B-16-SigLIP2-256", "webli"
device = get_device("auto")
model, _, preprocess = open_clip.create_model_and_transforms(NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(NAME)
model = model.to(device).eval()
val = pl.read_csv("data/val.csv")
names = val["image"].unique(maintain_order=True).to_list()
pos = {n: i for i, n in enumerate(names)}
with torch.no_grad():
    img = []
    for i in range(0, len(names), 64):
        x = torch.stack([preprocess(Image.open(f"data/val/{n}").convert("RGB")) for n in names[i:i + 64]]).to(device)
        img.append(model.encode_image(x, normalize=True))
    img = torch.cat(img)
    caps = val["caption"].to_list()
    txt = torch.cat([model.encode_text(tokenizer(caps[i:i + 256]).to(device), normalize=True) for i in range(0, len(caps), 256)])
    rows = torch.tensor([pos[n] for n in val["image"]], device=device)
    # Pretrained SigLIP calibration: sigmoid(scale * cos + bias) is a probability, so accuracy@0.5 is meaningful.
    scores = torch.sigmoid(model.logit_scale.exp() * (img[rows] * txt).sum(-1) + model.logit_bias).cpu().numpy()
m = evaluate_scores(scores, val["label"].to_numpy(), val["category"].to_numpy(), val["pair_id"].to_numpy())
for k, v in m.items():
    print(f"{k:28s} {v:.4f}")
