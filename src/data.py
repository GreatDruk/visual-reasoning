"""Scene batches keep paired captions together and separate fit from evaluation."""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from PIL import Image

from .programs import AND, EXACT, EXIST, FEWER, MORE, NONE, WORD_INDEX, parse_program
from .runtime import Progress
from .text import canonical_form


def encode_programs(programs, device):
    def words(key):
        return torch.tensor(
            [
                [WORD_INDEX[w] for w in getattr(p, key)] + [0] * (4 - len(getattr(p, key)))
                for p in programs
            ],
            dtype=torch.long,
            device=device,
        )

    return dict(
        a=words("a"),
        b=words("b"),
        ref=words("ref"),
        ops=torch.tensor([p.op for p in programs], device=device),
        numbers=torch.tensor([p.number for p in programs], device=device),
        relations=torch.tensor([p.relation for p in programs], device=device),
    )


def reflected(program):
    relation = {1: 2, 2: 1, 5: 6, 6: 5}.get(program.relation, program.relation)
    return replace(program, relation=relation)


def implications(program, truth):
    if program.relation:
        return (
            [(program.a, 1)]
            if truth and (program.op == EXIST or (program.op == EXACT and program.number > 0))
            else []
        )
    if program.op == EXIST:
        return [(program.a, truth)]
    if program.op == NONE:
        return [(program.a, 1 - truth)]
    if not truth:
        return []
    if program.op == EXACT:
        return [(program.a, int(program.number > 0))]
    if program.op == AND:
        return [(program.a, 1), (program.b, 1)]
    if program.op in (MORE, FEWER):
        return [(program.a if program.op == MORE else program.b, 1)]
    return []


class Scenes:
    def __init__(self, root, split, limit=None):
        self.root = Path(root)
        self.split = split
        self.image_read_seconds = 0.0
        self._rgb = None
        self._encoded = {}
        frame = pd.read_csv(self.root / f"{split}.csv")
        if not frame.id.is_unique:
            raise ValueError("Duplicate row IDs")
        if limit is not None:
            frame = frame[frame.image.isin(frame.image.unique()[:limit])]
        self.frame = frame.reset_index(drop=True)
        self.scene, self.names = pd.factorize(self.frame.image, sort=False)
        self.names = self.names.to_numpy(dtype=str)
        self.rows = [np.flatnonzero(self.scene == i) for i in range(len(self.names))]
        captions = [canonical_form(r.caption, r.category) for r in self.frame.itertuples()]
        self.programs = [parse_program(c) for c in captions]
        if any(p is None for p in self.programs):
            raise ValueError(
                f"Unsupported captions: {[c for c, p in zip(captions, self.programs) if p is None][:3]}"
            )
        self.pairs = np.array(
            [
                rows
                for rows in self.frame.groupby(["image", "category"], sort=False).indices.values()
            ],
            dtype=np.int64,
        )
        if (
            self.pairs.ndim != 2
            or self.pairs.shape[1] != 2
            or len(self.pairs) * 2 != len(self.frame)
        ):
            raise ValueError("Exactly two captions per image/category are required")
        self.label = self.frame.label.to_numpy(dtype=np.float32) if "label" in frame else None
        if self.label is not None and not np.all(self.label[self.pairs].sum(1) == 1):
            raise ValueError("Every labelled pair must contain one true statement")
        self.partner = np.empty(len(self.frame), dtype=np.int64)
        a, b = self.pairs.T
        self.partner[a] = b
        self.partner[b] = a
        self.aux = []
        for rows in self.rows:
            facts = set()
            if split == "train":
                for j in rows:
                    for words, y in implications(self.programs[j], int(self.label[j])):
                        if words and y:
                            facts.add(words)
                            facts.update((w,) for w in words)
            self.aux.append(sorted(facts))
        self._aux_words = []
        self._aux_rows = []
        for facts in self.aux:
            first = len(self._aux_words)
            self._aux_words.extend(
                [[WORD_INDEX[w] for w in words] + [0] * (4 - len(words)) for words in facts]
            )
            self._aux_rows.append(np.arange(first, len(self._aux_words)))
        self._pairs_local = [
            np.searchsorted(rows, self.pairs[np.isin(self.pairs[:, 0], rows)]) for rows in self.rows
        ]

    def prepare_images(self, cache_root, device):
        """Decode immutable JPEGs once; keep the compact uint8 scene bank on the GPU."""
        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            (str(self.root.resolve()) + self.split + "\n".join(self.names)).encode()
        ).hexdigest()[:20]
        path = cache_root / (key + ".npy")
        progress = Progress(f"{self.split}/images", len(self.names), device)
        with FileLock(str(path) + ".lock"):
            if not path.exists():

                def read(name):
                    with Image.open(self.root / self.split / name) as image:
                        return np.array(image.convert("RGB"))

                pending = path.with_suffix(".tmp.npy")
                bank = np.lib.format.open_memmap(
                    pending, mode="w+", dtype=np.uint8, shape=(len(self.names), 160, 240, 3)
                )
                started = time.perf_counter()
                with ThreadPoolExecutor(2) as pool:
                    for i, pixels in enumerate(pool.map(read, self.names)):
                        if pixels.shape != (160, 240, 3):
                            raise ValueError("Unexpected scene resolution")
                        bank[i] = pixels
                        progress.update(i + 1)
                self.image_read_seconds += time.perf_counter() - started
                bank.flush()
                bank._mmap.close()
                pending.replace(path)
        bank = np.load(path, mmap_mode="r")
        self._rgb = torch.empty(bank.shape, dtype=torch.uint8, device=device)
        for start in range(0, len(bank), 512):
            self._rgb[start : start + 512].copy_(
                torch.from_numpy(np.array(bank[start : start + 512], copy=True)).to(device)
            )
        bank._mmap.close()
        progress.update(
            len(self.names), force=True, resident_gib=round(self._rgb.numel() / 2**30, 3)
        )

    def images(self, indices, device):
        if self._rgb is not None:
            pixels = self._rgb[torch.as_tensor(indices, device=self._rgb.device)]
            return (
                pixels.to(device)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255)
                .contiguous(memory_format=torch.channels_last)
            )
        started = time.perf_counter()

        def read(i):
            with Image.open(self.root / self.split / self.names[i]) as im:
                return np.array(im.convert("RGB"))

        with ThreadPoolExecutor(3) as pool:
            pixels = np.stack(list(pool.map(read, indices)))
        self.image_read_seconds += time.perf_counter() - started
        return (
            torch.from_numpy(pixels)
            .to(device)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255)
            .contiguous(memory_format=torch.channels_last)
        )

    def batch(self, indices, device, flips=None):
        rows = np.concatenate([self.rows[i] for i in indices])
        lengths = np.array([len(self.rows[i]) for i in indices])
        scenes = np.repeat(np.arange(len(indices)), lengths)
        offsets = np.r_[0, np.cumsum(lengths)[:-1]]
        pairs = np.concatenate(
            [self._pairs_local[n] + offset for n, offset in zip(indices, offsets)]
        )
        key = str(device)
        if key not in self._encoded:
            self._encoded[key] = (
                encode_programs(self.programs, device),
                torch.tensor(self._aux_words, dtype=torch.long, device=device).reshape(-1, 4),
                None if self.label is None else torch.as_tensor(self.label, device=device),
            )
        encoded, words, labels = self._encoded[key]
        row_tensor = torch.as_tensor(rows, device=device)
        programs = {k: v[row_tensor] for k, v in encoded.items()}
        if flips is not None:
            mask = torch.as_tensor(flips, device=device)[torch.as_tensor(scenes, device=device)]
            relation = programs["relations"]
            reverse = torch.tensor([0, 2, 1, 3, 4, 6, 5], device=device)
            programs["relations"] = torch.where(mask, reverse[relation], relation)
        aux_indices = np.concatenate([self._aux_rows[n] for n in indices]).astype(np.int64)
        aux_scenes = np.repeat(np.arange(len(indices)), [len(self._aux_rows[n]) for n in indices])
        aux = (
            None
            if not len(aux_indices)
            else (
                torch.as_tensor(aux_scenes, device=device),
                words[torch.as_tensor(aux_indices, device=device)],
            )
        )
        return dict(
            rows=rows,
            scenes=torch.as_tensor(scenes, device=device),
            programs=programs,
            pairs=torch.as_tensor(pairs, device=device),
            aux=aux,
            label=None if labels is None else labels[row_tensor],
            category=self.frame.category.to_numpy()[rows],
        )


class Teacher:
    def __init__(self, path, scenes):
        if scenes.split != "train":
            raise ValueError("Teacher targets may only be opened for train")
        with np.load(path, allow_pickle=False) as z:
            names = list(z["names"])
            idx = np.array([names.index(n) for n in scenes.names])
            self.fields = {k: z[k][idx].copy() for k in z.files if k != "names"}
        self.names = scenes.names
        self._targets = None

    def prepare_targets(self, device, levels=1, soft=False, weak_only=False):
        from .losses import dense_targets

        self._targets = []
        progress = Progress("train/targets", 2 * len(self.names), device)
        for level in range(levels):
            h, w = 40 // 2**level, 60 // 2**level
            bank = [
                torch.empty((2, len(self.names), h, w), dtype=torch.bool, device=device),
                torch.empty((2, len(self.names), 4, h, w), device=device),
                torch.empty((2, len(self.names), 4, h, w), dtype=torch.int8, device=device),
                torch.empty((2, len(self.names), h, w), device=device) if soft else None,
            ]
            for flip in (0, 1):
                for start in range(0, len(self.names), 128):
                    idx = np.arange(start, min(start + 128, len(self.names)))
                    target = dense_targets(
                        self.batch(idx, np.full(len(idx), flip, bool)),
                        (h, w),
                        4 * 2**level,
                        torch.device("cpu"),
                        level,
                        levels > 1,
                        weak_only=weak_only,
                    )
                    for dest, value in zip(bank, target):
                        if dest is not None:
                            dest[flip, start : start + len(idx)].copy_(
                                value.to(device=device, dtype=dest.dtype)
                            )
                    if level == 0:
                        progress.update(flip * len(self.names) + start + len(idx))
            self._targets.append(bank)
        progress.update(2 * len(self.names), force=True, levels=levels)

    def targets(self, indices, flips):
        device = self._targets[0][0].device
        i = torch.as_tensor(indices, device=device)
        f = torch.as_tensor(flips, dtype=torch.long, device=device)
        return [
            (
                bank[0][f, i].float(),
                bank[1][f, i],
                bank[2][f, i].long(),
                bank[3][f, i] if bank[3] is not None else torch.zeros((), device=device),
            )
            for bank in self._targets
        ]

    def batch(self, indices, flips):
        parts = []
        for i, flip in zip(indices, flips):
            p = self.fields["p"][i]
            valid = self.fields["valid"][i]
            box = self.fields["boxes"][i].copy()
            if flip:
                box[:, 0] = 1 - box[:, 0] - box[:, 2]
            attrs = [self.fields[k][i] for k in ("size", "color", "material", "shape")]
            labels = np.stack([np.where(a.max(-1) >= 0.68, a.argmax(-1), -100) for a in attrs], -1)
            parts.append((box, p, valid, labels))
        return parts

    def crop_labels(self, indices, boxes):
        result = np.full((len(indices), 4), -100, np.int64)
        for j, (i, b) in enumerate(zip(indices, boxes)):
            f = self.fields
            ref = f["boxes"][i]
            inter = (
                (np.minimum(b[:2] + b[2:], ref[:, :2] + ref[:, 2:]) - np.maximum(b[:2], ref[:, :2]))
                .clip(0)
                .prod(-1)
            )
            iou = inter / (b[2:].prod() + ref[:, 2:].prod(-1) - inter).clip(1e-8)
            iou = np.where(f["valid"][i] & (f["p"][i] > 0.86), iou, 0)
            if iou.max() < 0.55:
                continue
            k = iou.argmax()
            for g, key in enumerate(("size", "color", "material", "shape")):
                prob = f[key][i, k]
                if prob.max() >= 0.8:
                    result[j, g] = prob.argmax()
        return result
