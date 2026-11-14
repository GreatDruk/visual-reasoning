"""Caption tables, image/feature sources and batch iteration."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import torch

from .text_aug import novel_form, paraphrases, swap_left_right
from .utils import resolve_path


@dataclass
class Split:
    """Caption rows of one split. `img_idx` indexes the split's image source."""

    name: str
    ids: np.ndarray
    image_names: list
    img_idx: np.ndarray
    captions: list
    categories: np.ndarray
    labels: np.ndarray | None = None
    pair_ids: np.ndarray | None = None
    source: object = field(default=None, repr=False)

    def __len__(self):
        return len(self.captions)


def load_split(name, csv_path):
    """Read a csv; images are numbered in order of first appearance."""
    df = pl.read_csv(resolve_path(csv_path))
    if "label" in df.columns:
        if "pair_id" not in df.columns:
            raise ValueError(f"{name}: labelled rows require pair_id")
        for key, pair in df.group_by("pair_id", maintain_order=True):
            if pair.height != 2 or pair["image"].n_unique() != 1 or set(pair["label"].to_list()) != {0, 1} or key[0] is None:
                raise ValueError(f"{name}: invalid labelled pair {key[0]!r}: expected two rows, one image, labels {{0, 1}}")
    image_names = df["image"].unique(maintain_order=True).to_list()
    pos = {n: i for i, n in enumerate(image_names)}
    return Split(
        name=name,
        ids=df["id"].to_numpy(),
        image_names=image_names,
        img_idx=np.array([pos[n] for n in df["image"]], dtype=np.int64),
        captions=df["caption"].str.to_lowercase().to_list(),
        categories=df["category"].to_numpy(),
        labels=df["label"].to_numpy() if "label" in df.columns else None,
        pair_ids=df["pair_id"].to_numpy() if "pair_id" in df.columns else None,
    )


def subset(split, mask):
    """Select rows; compact image indexing only before attaching an image source."""
    rows = np.nonzero(mask)[0]
    pick = lambda v: None if v is None else v[rows]
    image_names, img_idx = split.image_names, split.img_idx[rows]
    if split.source is None:
        used, img_idx = np.unique(img_idx, return_inverse=True)
        image_names = [split.image_names[i] for i in used]
    return Split(
        name=split.name,
        ids=split.ids[rows],
        image_names=image_names,
        img_idx=img_idx,
        captions=[split.captions[r] for r in rows],
        categories=split.categories[rows],
        labels=pick(split.labels),
        pair_ids=pick(split.pair_ids),
        source=split.source,
    )


def paraphrased(split, name=None):
    """Rule-based paraphrasing diagnostic; shares rules with training augmentation."""
    captions = [novel_form(c, cat) for c, cat in zip(split.captions, split.categories)]
    return Split(**{**split.__dict__, "name": name or f"{split.name}_para", "captions": captions})


def concat_splits(a, b, name):
    """Concatenate two labelled splits; image sources are chained with ConcatSource."""
    offset = len(a.source)
    pair_offset = int(a.pair_ids.max()) + 1
    return Split(
        name=name,
        ids=np.concatenate([a.ids, b.ids]),
        image_names=a.image_names + b.image_names,
        img_idx=np.concatenate([a.img_idx, b.img_idx + offset]),
        captions=a.captions + b.captions,
        categories=np.concatenate([a.categories, b.categories]),
        labels=np.concatenate([a.labels, b.labels]),
        pair_ids=np.concatenate([a.pair_ids, b.pair_ids + pair_offset]),
        source=ConcatSource([a.source, b.source]),
    )


class ArraySource:
    """Per-image arrays (cached float16 feature maps, or uint8 images) with an optional
    pre-flipped copy for horizontal-flip augmentation."""

    def __init__(self, array, flipped=None):
        self.array, self.flipped = array, flipped

    def __len__(self):
        return len(self.array)

    def get(self, idx, flip=None):
        out = np.asarray(self.array[idx])
        if flip is not None and flip.any():
            if self.flipped is None:
                raise ValueError("flip requested but no flipped copy is cached")
            out = out.copy()
            out[flip] = np.asarray(self.flipped[idx[flip]])
        return torch.from_numpy(np.ascontiguousarray(out))


class ConcatSource:
    def __init__(self, sources):
        self.sources = sources
        self.offsets = np.cumsum([0] + [len(s) for s in sources])

    def __len__(self):
        return int(self.offsets[-1])

    def get(self, idx, flip=None):
        flip = np.zeros(len(idx), bool) if flip is None else flip
        parts, order = [], []
        for k, s in enumerate(self.sources):
            m = (idx >= self.offsets[k]) & (idx < self.offsets[k + 1])
            if m.any():
                parts.append(s.get(idx[m] - self.offsets[k], flip[m]))
                order.append(np.nonzero(m)[0])
        out = torch.cat(parts)
        inv = np.empty(len(idx), np.int64)
        inv[np.concatenate(order)] = np.arange(len(idx))
        return out[torch.from_numpy(inv)]


def pair_groups(split):
    """Row indices grouped by pair_id: array [P, 2]."""
    order = np.argsort(split.pair_ids, kind="stable")
    rows = order.reshape(-1, 2)
    return rows


def sample_pairs_by_category(split, pairs_per_category, seed):
    """Choose complete, category-balanced pairs before attaching image sources."""
    if (isinstance(pairs_per_category, (bool, np.bool_))
            or not isinstance(pairs_per_category, (int, np.integer)) or pairs_per_category <= 0):
        raise ValueError(f"{split.name}: pairs_per_category must be a positive integer")
    if split.labels is None or split.pair_ids is None:
        raise ValueError(f"{split.name}: labels and pair_ids are required")
    _, counts = np.unique(split.pair_ids, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError(f"{split.name}: expected complete pairs")
    groups = pair_groups(split)
    categories = split.categories[groups]
    mixed = categories[:, 0] != categories[:, 1]
    if mixed.any():
        pair_id = split.pair_ids[groups[np.flatnonzero(mixed)[0], 0]]
        raise ValueError(f"{split.name}: mixed categories in pair_id {pair_id}")
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(split), dtype=bool)
    for category in sorted(set(categories[:, 0])):
        available = groups[categories[:, 0] == category]
        if len(available) < pairs_per_category:
            raise ValueError(
                f"{split.name}: category {category}: requested {pairs_per_category}, available {len(available)}"
            )
        selected = rng.choice(len(available), size=pairs_per_category, replace=False)
        mask[available[selected].reshape(-1)] = True
    return subset(split, mask)


def train_batches(split, batch_size, rng, aug, pair_sampler=True):
    """Yield training batches as dicts.

    With pair_sampler both twins of a pair land in the same batch and receive the same
    augmentation (flip, paraphrase form), so the pair loss compares like with like.
    aug: {"hflip_swap_lr": p, "reorder_clauses_p": p}.
    """
    if pair_sampler:
        groups = pair_groups(split)
        groups = groups[rng.permutation(len(groups))]
        per_batch = max(1, batch_size // 2)
        chunks = [groups[i : i + per_batch] for i in range(0, len(groups), per_batch)]
    else:
        perm = rng.permutation(len(split))
        chunks = [perm[i : i + batch_size, None] for i in range(0, len(perm), batch_size)]

    p_flip = aug.get("hflip_swap_lr", 0.0)
    p_para = aug.get("reorder_clauses_p", 0.0)
    for chunk in chunks:
        g, k = chunk.shape
        flip = np.repeat(rng.random(g) < p_flip, k)
        para = np.repeat(rng.random(g) < p_para, k)
        form_u = np.repeat(rng.random(g), k)
        rows = chunk.reshape(-1)
        captions = []
        for r, f, p, u in zip(rows, flip, para, form_u):
            c = split.captions[r]
            if p:
                forms = paraphrases(c, split.categories[r])
                c = forms[int(u * len(forms))]
            if f:
                c = swap_left_right(c)
            captions.append(c)
        batch = {
            "rows": rows,
            "img_idx": split.img_idx[rows],
            "flip": flip,
            "captions": captions,
            "labels": torch.from_numpy(split.labels[rows]),
        }
        if pair_sampler:
            lab = split.labels[rows].reshape(g, k)
            base = np.arange(g) * 2
            true_pos = base + np.argmax(lab, axis=1)
            false_pos = base + np.argmin(lab, axis=1)
            batch["pair_index"] = torch.from_numpy(np.stack([true_pos, false_pos], 1))
        yield batch


def eval_batches(split, batch_size):
    for i in range(0, len(split), batch_size):
        rows = np.arange(i, min(i + batch_size, len(split)))
        yield {
            "rows": rows,
            "img_idx": split.img_idx[rows],
            "flip": np.zeros(len(rows), bool),
            "captions": [split.captions[r] for r in rows],
        }


def save_submission(ids, probs, path, target_col="label"):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": ids, target_col: probs}).write_csv(path)
    return path
