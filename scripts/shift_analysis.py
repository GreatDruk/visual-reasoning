"""How similar are train, val and test? Text templates + image embeddings.

usage: uv run python scripts/shift_analysis.py [--out DIR] [--device auto]

Text: captions are mapped to templates (colours, shapes, sizes, materials, numbers,
directions replaced by slots); reports per-split template coverage w.r.t. train and
whether the rule-based paraphrases (src/text_aug.py) cover the test templates.

Images (one row per image, no labels used):
  - embeddings: ResNet-50 V2 layer4 global average, SigLIP2 B/16 pooled image embedding,
    and a pixel colour histogram (8 bins x 3 channels) as a model-free reference;
  - adversarial validation: logistic regression, 5-fold out-of-fold ROC-AUC for
    train-vs-val, train-vs-test, val-vs-test, plus a label-permutation null;
  - nearest-neighbour cosine distance to train (train itself: leave-one-out),
    quantiles and a KS test val vs test.

Embeddings are cached in DIR as .npy so the analyses can be rerun without the encoders.
Test images/captions are read here for analysis only (the organisers now allow it);
nothing in this script feeds the training pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import load_images, preprocess
from src.models.image_encoders import TorchvisionEncoder
from src.text_aug import paraphrases
from src.utils import get_device, resolve_path, set_seed

SPLITS = ("train", "val", "test")
SLOTS = {
    "<C>": "blue brown cyan gray green purple red yellow",
    "<S>": "cube cubes cylinder cylinders sphere spheres",
    "<Z>": "large small",
    "<M>": "metal rubber",
    "<N>": "no one two three four five six seven eight nine ten",
    "<LR>": "left right",
    "<FB>": "behind front",
    "<CMP>": "more fewer",
}
SLOT_OF = {w: s for s, ws in SLOTS.items() for w in ws.split()}


def template(caption):
    return " ".join(SLOT_OF.get(w, w) for w in caption.split())


def text_report(dfs):
    train_tpl = set(dfs["train"].caption.map(template))
    para_tpl = {template(p) for c, cat in zip(dfs["train"].caption, dfs["train"].category) for p in paraphrases(c, cat)}
    train_caps = set(dfs["train"].caption)
    vocab = {w for c in dfs["train"].caption for w in c.split()}
    out = {"train_vocab_size": len(vocab), "train_templates": len(train_tpl)}
    for name in ("val", "test"):
        d = dfs[name]
        tpl = d.caption.map(template)
        seen = tpl.isin(train_tpl)
        out[name] = {
            "rows": len(d),
            "templates": int(tpl.nunique()),
            "oov_tokens": int(sum(w not in vocab for c in d.caption for w in c.split())),
            "exact_caption_in_train": round(float(d.caption.isin(train_caps).mean()), 4),
            "rows_template_seen_in_train": round(float(seen.mean()), 4),
            "rows_template_covered_by_paraphrases": round(float(tpl.isin(para_tpl).mean()), 4),
            "unseen_rows_by_category": d[~seen].category.value_counts().to_dict(),
            "unseen_share_by_category": d.assign(u=~seen).groupby("category").u.mean().round(3).to_dict(),
            "top_unseen_templates": tpl[~seen].value_counts().head(15).to_dict(),
        }
    out["category_share"] = pd.DataFrame({n: d.category.value_counts(normalize=True) for n, d in dfs.items()}).round(4).to_dict()
    out["level_share"] = pd.DataFrame({n: d.level.value_counts(normalize=True) for n, d in dfs.items()}).round(4).to_dict()
    out["captions_per_image"] = {n: round(len(d) / d.image.nunique(), 3) for n, d in dfs.items()}
    return out


@torch.no_grad()
def embed(images, fn, device, batch_size=128):
    outs = []
    for s in range(0, len(images), batch_size):
        x = torch.from_numpy(images[s : s + batch_size]).to(device)
        outs.append(fn(x).float().cpu().numpy())
    return np.concatenate(outs)


def compute_embeddings(names, out_dir, device):
    """{encoder: {split: [N, D]}} cached as out_dir/<encoder>_<split>.npy."""
    encoders = ("pixel_hist", "resnet50_l4", "siglip2")
    paths = {(e, s): out_dir / f"{e}_{s}.npy" for e in encoders for s in SPLITS}
    if all(p.exists() for p in paths.values()):
        return {e: {s: np.load(paths[e, s]) for s in SPLITS} for e in encoders}

    images = {s: load_images(f"data/{s}", names[s]) for s in SPLITS}

    def pixel_hist(x):
        q = (x.long() * 8) // 256  # [B, 3, H, W] in 0..7
        return torch.stack([torch.stack([(q[:, c] == b).float().mean((1, 2)) for b in range(8)], 1) for c in range(3)], 1).flatten(1)

    rn = TorchvisionEncoder("resnet50", "IMAGENET1K_V2", "layer4", spatial=False).to(device).eval()

    import open_clip

    clip = open_clip.create_model("ViT-B-16-SigLIP2-256", pretrained="webli").to(device).eval()
    fns = {
        "pixel_hist": pixel_hist,
        "resnet50_l4": lambda x: rn(preprocess(x, (160, 240), "imagenet")),
        "siglip2": lambda x: clip.encode_image(preprocess(x, (256, 256), "siglip"), normalize=True),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for e in encoders:
        result[e] = {}
        for s in SPLITS:
            result[e][s] = embed(images[s], fns[e], device)
            np.save(paths[e, s], result[e][s])
            print(f"{e} {s} {result[e][s].shape}", flush=True)
    return result


def adversarial_auc(a, b, seed, permute=False):
    x = np.concatenate([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    if permute:
        y = np.random.default_rng(seed).permutation(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), p


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def knn_report(emb):
    tr = l2n(emb["train"])
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(tr)
    d = {"train_loo": nn.kneighbors(tr)[0][:, 1]}
    for s in ("val", "test"):
        d[s] = nn.kneighbors(l2n(emb[s]), n_neighbors=1)[0][:, 0]
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    out = {k: {f"q{int(q * 100)}": round(float(np.quantile(v, q)), 5) for q in qs} for k, v in d.items()}
    ks = ks_2samp(d["val"], d["test"])
    out["ks_val_vs_test"] = {"stat": round(float(ks.statistic), 4), "p": float(ks.pvalue)}
    ks = ks_2samp(d["train_loo"], d["test"])
    out["ks_trainloo_vs_test"] = {"stat": round(float(ks.statistic), 4), "p": float(ks.pvalue)}
    return out, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artefacts/shift_analysis"))
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    out_dir = resolve_path(args.out)
    dfs = {s: pd.read_csv(resolve_path(f"data/{s}.csv")) for s in SPLITS}

    report = {"text": text_report(dfs)}
    print(json.dumps(report["text"], indent=1)[:3000], flush=True)

    names = {s: sorted(dfs[s].image.unique()) for s in SPLITS}
    emb = compute_embeddings(names, out_dir, get_device(args.device))

    report["images"] = {}
    for e, by_split in emb.items():
        r = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            auc, prob = adversarial_auc(by_split[a], by_split[b], args.seed)
            r[f"adv_auc_{a}_vs_{b}"] = round(auc, 4)
            if (a, b) == ("train", "test"):
                order = np.argsort(-prob[len(by_split[a]) :])[:20]
                r["most_test_like_test_images"] = [names["test"][i] for i in order]
        r["adv_auc_null_train_vs_test"] = [round(adversarial_auc(by_split["train"], by_split["test"], args.seed + k, True)[0], 4) for k in range(3)]
        r["knn_cosine_to_train"], dists = knn_report(by_split)
        np.savez(out_dir / f"{e}_knn.npz", **dists)
        report["images"][e] = r
        print(e, json.dumps(r, indent=1), flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved", out_dir / "report.json")


if __name__ == "__main__":
    main()
