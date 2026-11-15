"""Small train -> val ablations on existing, strictly read-only feature maps.

No image encoder is constructed and test.csv is never opened. All fits are on CPU
with at most four intra-op threads and one inter-op thread. Results are diagnostics,
not substitutes for a cold offline Kaggle GPU run. Run from the repository root.

Example:
  OMP_NUM_THREADS=4 uv run python scripts/codex_cpu_research.py \
      --routing separate --pool 2 --head attention --epochs 40 --out RUN_DIR
"""

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_key] = "4"

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import re
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from src.metrics import pair_accuracy
from src.models.text_encoders import GRUEncoder, Vocab
from src.text_aug import novel_form, paraphrases


CATEGORIES = ["counting", "position", "spatial", "spatial_counting", "spatial_negation"]
FEATURES = "artefacts/features/clip_ViT-B-16-SigLIP2-256_webli_block-1norm_256x384_siglip"
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def count_input(caption):
    """Factor the requested count out of the visual query (counting category only).

    This extracts an observed caption token, not a label or a scene annotation.
    Singular/plural and is/are are normalized so twins asking different numbers
    use precisely the same visual predicate. There is no scene/program parser.
    """
    match = re.search(r"\bexactly (\w+)\b", caption)
    if not match:
        raise ValueError(f"count readout expects 'exactly N': {caption!r}")
    word = match.group(1)
    number = int(word) if word.isdigit() else NUMBER_WORDS.index(word)
    text = caption[:match.start(1)] + "<count>" + caption[match.end(1):]
    singular = {"objects": "object", "cubes": "cube", "spheres": "sphere", "cylinders": "cylinder", "are": "is"}
    text = " ".join(singular.get(w, w) for w in text.split())
    return text, number


def fingerprint(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def metrics(logits, df):
    y = df["label"].to_numpy()
    cats = df["category"].to_numpy()
    pairs = df["pair_id"].to_numpy()
    result = {
        "auc": float(roc_auc_score(y, logits)),
        "accuracy": float(np.mean((logits > 0) == y)),
        "pair_accuracy": pair_accuracy(logits, y, pairs),
        "per_category": {},
    }
    for category in sorted(set(cats)):
        m = cats == category
        result["per_category"][category] = {
            "rows": int(m.sum()), "pairs": int(m.sum() // 2),
            "auc": float(roc_auc_score(y[m], logits[m])),
            "accuracy": float(np.mean((logits[m] > 0) == y[m])),
            "pair_accuracy": pair_accuracy(logits[m], y[m], pairs[m]),
        }
    result["macro_auc"] = float(np.mean([m["auc"] for m in result["per_category"].values()]))
    return result


class Features:
    def __init__(self, root, split, pool):
        self.path = root / f"{split}.npy"
        self.names = json.loads((root / f"{split}.json").read_text())
        self.positions = {name: i for i, name in enumerate(self.names)}
        self.array = np.load(self.path, mmap_mode="r")
        if self.array.flags.writeable:
            raise RuntimeError("feature map must be read-only")
        self.shape = (self.array.shape[1], self.array.shape[2] // pool, self.array.shape[3] // pool)
        self.pooled = None
        if pool > 1:
            # Only an in-memory view is made: never create/replace a feature cache.
            self.pooled = torch.empty((len(self.names), *self.shape), dtype=torch.float16)
            for start in range(0, len(self.names), 32):
                x = torch.from_numpy(np.array(self.array[start:start + 32], copy=True)).float()
                self.pooled[start:start + len(x)] = F.avg_pool2d(x, pool).half()

    def get(self, row_indices):
        unique, inverse = np.unique(row_indices, return_inverse=True)
        if self.pooled is not None:
            x = self.pooled[torch.from_numpy(unique)].float()
        else:
            x = torch.from_numpy(np.array(self.array[unique], copy=True)).float()
        return x, torch.from_numpy(inverse)


class TextTable:
    def __init__(self, train, val, augment, factor_count=False):
        captions = train["caption"].to_list()
        self.forms = [paraphrases(c, cat) if augment else [c]
                      for c, cat in zip(captions, train["category"])]
        self.vocab = Vocab([count_input(s)[0] if factor_count else s for forms in self.forms for s in forms])
        self.val_captions = val["caption"].to_list()
        self.para_captions = [novel_form(c, cat) for c, cat in zip(self.val_captions, val["category"])]
        all_strings = list(dict.fromkeys([s for fs in self.forms for s in fs] + self.val_captions + self.para_captions))
        ids = {s: i for i, s in enumerate(all_strings)}
        encoded_strings = [count_input(s)[0] for s in all_strings] if factor_count else all_strings
        self.tokens, self.lengths = self.vocab.encode_batch(encoded_strings)
        self.numbers = torch.tensor([count_input(s)[1] for s in all_strings], dtype=torch.float32) if factor_count else None
        self.train_forms = [[ids[s] for s in fs] for fs in self.forms]
        self.val_ids = torch.tensor([ids[s] for s in self.val_captions])
        self.para_ids = torch.tensor([ids[s] for s in self.para_captions])
        self.para_changed = float(np.mean([a != b for a, b in zip(self.val_captions, self.para_captions)]))

    def get(self, ids):
        lengths = self.lengths[ids]
        return self.tokens[ids, :int(lengths.max())], lengths


class GroundingHead(nn.Module):
    """Four learned word queries with normalized attention or extra mass/relations.

    The summary option retains sigmoid mass, its log, existence, attended x/y,
    spatial variance and directional mass relative to each query's location.
    Sigmoid mass counts *cells*, not objects: object size and overlapping receptive
    fields can make this a poor count proxy; the experiment deliberately tests it.
    Complexity is linear in cells (no cell-by-cell attention or detection model).
    """

    def __init__(self, channels, text_dim, height, width, dim, head, queries=4, explicit_coords=True):
        super().__init__()
        self.dim, self.queries, self.head = dim, queries, head
        self.explicit_coords = explicit_coords
        self.cell = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.BatchNorm2d(dim))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
        self.register_buffer("coords", torch.stack([xx.flatten(), yy.flatten()], dim=-1))
        self.xy = nn.Linear(2, dim, bias=False)
        self.pos = nn.Parameter(torch.randn(1, height * width, dim) * 0.02)
        self.word_att = nn.Linear(text_dim, queries)
        self.word_q = nn.Linear(text_dim, dim)
        self.sent_q = nn.Linear(text_dim, queries * dim)
        self.q_norm = nn.LayerNorm(dim)
        self.bias = nn.Linear(text_dim, queries)
        extras = queries * 8 + queries * queries * 4 if head == "summary" else 0
        self.out = nn.Sequential(
            nn.Linear(queries * dim * 3 + extras, 128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 1),
        )

    def forward(self, image, inverse, text):
        sent, words, mask = text
        k = self.cell(image).flatten(2).transpose(1, 2)
        k = k + self.pos
        if self.explicit_coords:
            k = k + self.xy(self.coords)[None]
        k = k[inverse]
        att_words = self.word_att(words).masked_fill(~mask[..., None], -1e4).softmax(1)
        q = self.word_q(torch.einsum("blk,bld->bkd", att_words, words))
        q = self.q_norm(q + self.sent_q(sent).view(-1, self.queries, self.dim))
        scores = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.dim)
        scores = scores + self.bias(sent)[..., None]
        weights = scores.softmax(-1)
        z = torch.einsum("bkn,bnd->bkd", weights, k)
        feats = [z.flatten(1), q.flatten(1), (z * q).flatten(1)]
        if self.head == "summary":
            n = scores.shape[-1]
            p = scores.sigmoid()
            mass = p.sum(-1)
            loc = torch.einsum("bkn,nc->bkc", weights, self.coords)
            var = torch.einsum("bkn,nc->bkc", weights, self.coords.square()) - loc.square()
            dx = self.coords[None, None, :, 0] - loc[..., :1]
            dy = self.coords[None, None, :, 1] - loc[..., 1:2]
            dirs = torch.stack([(-8 * dx).sigmoid(), (8 * dx).sigmoid(),
                                (-8 * dy).sigmoid(), (8 * dy).sigmoid()], dim=2)
            relation = torch.einsum("bkrn,bln->bklr", dirs, p) / n
            feats += [torch.logsumexp(scores, -1) - math.log(n), mass / n,
                      torch.log1p(mass) / math.log1p(n), loc.flatten(1), var.flatten(1),
                      p.amax(-1), relation.flatten(1)]
        return self.out(torch.cat(feats, -1)).squeeze(-1)


class Model(nn.Module):
    def __init__(self, vocab_size, shape, args):
        super().__init__()
        self.text = GRUEncoder(vocab_size, embed_dim=args.dim, hidden_dim=args.dim, dropout=0.1)
        c, h, w = shape
        self.head = DensityHead(c, self.text.out_dim, args.dim, args.pool) if args.head == "density" else GroundingHead(c, self.text.out_dim, h, w, args.dim, args.head, explicit_coords=not args.no_coords)
        self.last_count = None

    def forward(self, x, inverse, tokens, lengths, numbers=None):
        text = self.text(tokens, lengths)
        if isinstance(self.head, DensityHead):
            logits, self.last_count = self.head(x, inverse, text, numbers)
            return logits
        return self.head(x, inverse, text)


class DensityHead(nn.Module):
    """A learned density sum and a number-agreement truth logit.

    No normalized softmax pooling or unrestricted truth MLP can bypass the count.
    Requested N is excluded from the GRU input and used only for agreement.
    Counts/optional regression targets come solely from true train captions.
    """

    def __init__(self, channels, text_dim, dim, pool):
        super().__init__()
        self.cells = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.BatchNorm2d(dim),
                                   nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())
        self.query = nn.Linear(text_dim, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.density_bias = nn.Parameter(torch.tensor(-2.5))
        self.log_cells_per_object = nn.Parameter(torch.tensor(math.log(16 / (pool * pool))))
        self.agreement_bias = nn.Parameter(torch.tensor(2.0))
        self.log_precision = nn.Parameter(torch.tensor(0.0))

    def forward(self, image, inverse, text, numbers):
        cells = self.cells(image).flatten(2)[inverse]
        query = self.query_norm(self.query(text[0]))
        density = (torch.einsum("bd,bdn->bn", query, cells) / math.sqrt(cells.shape[1]) + self.density_bias).sigmoid()
        count = density.sum(-1) / self.log_cells_per_object.exp()
        logits = self.agreement_bias - self.log_precision.exp() * (count - numbers).square()
        return logits, count


@torch.no_grad()
def predict(model, df, features, table, ids, batch_size):
    model.eval()
    image_indices = np.array([features.positions[name] for name in df["image"]])
    result = []
    for start in range(0, len(df), batch_size):
        x, inv = features.get(image_indices[start:start + batch_size])
        tokens, lengths = table.get(ids[start:start + batch_size])
        numbers = table.numbers[ids[start:start + batch_size]] if table.numbers is not None else None
        result.append(model(x, inv, tokens, lengths, numbers).numpy())
    return np.concatenate(result)


def run_fit(train, val, feature_tr, feature_va, args, out_dir):
    start = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    aug_rng = np.random.default_rng(args.seed + 777)
    table = TextTable(train, val, args.augment > 0, factor_count=args.head == "density")
    model = Model(len(table.vocab), feature_tr.shape, args).cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    y = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)
    groups = np.lexsort((-train["label"].to_numpy(), train["pair_id"].to_numpy())).reshape(-1, 2)
    assert torch.all(y[groups[:, 0]] == 1) and torch.all(y[groups[:, 1]] == 0)
    image_indices = np.array([feature_tr.positions[name] for name in train["image"]])
    train_ids = torch.tensor([fs[0] for fs in table.train_forms])
    best, best_state, best_epoch = -float("inf"), None, None
    history, step, fit_sec, eval_sec = [], 0, 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        losses = []
        order = rng.permutation(len(groups))
        if args.balance:
            cats = train["category"].to_numpy()[groups[:, 0]]
            categories = sorted(set(cats))
            order = np.concatenate([rng.choice(np.flatnonzero(cats == c), size=math.ceil(len(groups) / len(categories)), replace=True)
                                    for c in categories])
            rng.shuffle(order)
        for batch in np.array_split(order, math.ceil(len(order) / (args.batch_size // 2))):
            rows = groups[batch].reshape(-1)
            if args.augment > 0:
                ids = torch.tensor([aug_rng.choice(table.train_forms[r]) if aug_rng.random() < args.augment else table.train_forms[r][0] for r in rows])
            else:
                ids = train_ids[rows]
            tokens, lengths = table.get(ids)
            x, inverse = feature_tr.get(image_indices[rows])
            opt.param_groups[0]["lr"] = args.lr * min((step + 1) / args.warmup, 1.0)
            numbers = table.numbers[ids] if table.numbers is not None else None
            logits = model(x, inverse, tokens, lengths, numbers)
            loss = F.binary_cross_entropy_with_logits(logits, y[rows])
            loss = loss + 0.5 * F.softplus(-(logits[0::2] - logits[1::2])).mean()
            if args.count_mse and numbers is not None:
                positive = y[rows] == 1
                loss = loss + args.count_mse * F.mse_loss(model.last_count[positive], numbers[positive])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(float(loss.detach()))
        train_sec = time.monotonic() - epoch_start
        fit_sec += train_sec
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_start = time.monotonic()
            logits = predict(model, val, feature_va, table, table.val_ids, args.batch_size)
            score = metrics(logits, val)
            eval_sec += time.monotonic() - eval_start
            history.append({"epoch": epoch, "step": step, "loss": float(np.mean(losses)),
                            "train_epoch_sec": train_sec, "val": score})
            if score["macro_auc"] > best:
                best = score["macro_auc"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            print(json.dumps({"fit": out_dir.name, **history[-1]}), flush=True)
    last_metrics = history[-1]["val"]
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "categories": sorted(train["category"].unique()), "train_rows": len(train), "val_rows": len(val),
        "params": sum(p.numel() for p in model.parameters()), "vocab": table.vocab.itos,
        "epochs": args.epochs, "steps": step, "best_epoch": best_epoch,
        "selection": "maximum val macro AUC at predeclared eval epochs; one checkpoint per fit",
        "fit_sec": fit_sec, "history_eval_sec": eval_sec,
        "last_val": last_metrics, "val_para_changed_fraction": table.para_changed,
        "train_oov": 0,
        "val_oov_words": sorted({w for c in table.val_captions
                                 for w in table.vocab.split(count_input(c)[0] if args.head == "density" else c)
                                 if w not in table.vocab.stoi}),
        "history": history,
    }
    for split_name, df, features, ids in [("train", train, feature_tr, train_ids),
                                          ("val", val, feature_va, table.val_ids),
                                          ("val_para", val, feature_va, table.para_ids)]:
        logits = predict(model, df, features, table, ids, args.batch_size)
        result[split_name] = metrics(logits, df)
        np.save(out_dir / f"{split_name}_logits.npy", logits)
        np.save(out_dir / f"{split_name}_ids.npy", df["id"].to_numpy())
    if args.image_shuffle:
        # Image permutation is a validation diagnostic, never a training objective.
        names = val["image"].unique(maintain_order=True).to_list()
        permutation = np.random.default_rng(args.seed + 1000).permutation(len(names))
        mapping = {name: names[j] for name, j in zip(names, permutation)}
        shuffled = val.with_columns(pl.Series("image", [mapping[n] for n in val["image"]]))
        logits = predict(model, shuffled, feature_va, table, table.val_ids, args.batch_size)
        result["val_image_shuffle"] = metrics(logits, val)
        np.save(out_dir / "val_image_shuffle_logits.npy", logits)
    if args.save_model:
        torch.save(model.state_dict(), out_dir / "model.pt")
    result["total_sec"] = time.monotonic() - start
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"completed": str(out_dir), "minutes": result["total_sec"] / 60,
                      "val": result["val"], "val_para": result["val_para"]}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path(FEATURES))
    p.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    p.add_argument("--routing", choices=["shared", "separate"], default="separate")
    p.add_argument("--head", choices=["attention", "summary", "density"], default="attention")
    p.add_argument("--count-mse", type=float, default=0.0)
    p.add_argument("--no-coords", action="store_true", help="keep learned grid positions, remove explicit x/y channels")
    p.add_argument("--pool", type=int, choices=[1, 2], default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", type=float, default=0)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--image-shuffle", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.head == "density" and args.categories != ["counting"]:
        p.error("density readout is a counting-only ablation: --categories counting")
    if args.batch_size % 2 or args.warmup < 1 or args.epochs < 1 or args.count_mse < 0 or not 0 <= args.augment <= 1:
        p.error("even batch size, positive epochs/warmup and augmentation in [0, 1] required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    if args.out.exists():
        raise FileExistsError(f"choose a new output directory: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "source.py").write_text(Path(__file__).read_text())
    watched = [args.features / f"{split}.{ext}" for split in ["train", "val"] for ext in ["npy", "json"]]
    before = {str(path): fingerprint(path) for path in watched}
    train = pl.read_csv("data/train.csv").filter(pl.col("category").is_in(args.categories))
    val = pl.read_csv("data/val.csv").filter(pl.col("category").is_in(args.categories))
    features_tr = Features(args.features, "train", args.pool)
    features_va = Features(args.features, "val", args.pool)
    preparation_sec = time.monotonic() - start
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "command": "OMP_NUM_THREADS=4 uv run python " + shlex.join(sys.argv),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": "cpu", "torch_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(),
              "python": platform.python_version(), "torch": torch.__version__,
              "feature_shape": features_tr.shape, "preparation_sec": preparation_sec, "fits": {}}
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2))
    groups = [[c] for c in args.categories] if args.routing == "separate" else [args.categories]
    for cats in groups:
        name = "_".join(cats) if args.routing == "shared" else cats[0]
        tr = train.filter(pl.col("category").is_in(cats))
        va = val.filter(pl.col("category").is_in(cats))
        result["fits"][name] = run_fit(tr, va, features_tr, features_va, args, args.out / name)
        (args.out / "results.json").write_text(json.dumps(result, indent=2))
    after = {str(path): fingerprint(path) for path in watched}
    result["cache_fingerprints_before"] = before
    result["cache_fingerprints_after"] = after
    result["cache_unchanged"] = before == after
    result["total_sec"] = time.monotonic() - start
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    if before != after:
        raise RuntimeError("existing feature files changed during the experiment")


if __name__ == "__main__":
    main()
