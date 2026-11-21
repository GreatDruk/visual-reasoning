"""Scene batches keep paired captions together and separate fit from evaluation."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .programs import AND, EXACT, EXIST, FEWER, MORE, NONE, WORD_INDEX, parse_program
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

    def images(self, indices, device):
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
        lookup = {n: i for i, n in enumerate(indices)}
        scenes = np.array([lookup[i] for i in self.scene[rows]])
        pos = {r: j for j, r in enumerate(rows)}
        pairs = np.array([[pos[r], pos[self.partner[r]]] for r in rows if r < self.partner[r]])
        programs = [
            reflected(p) if flips is not None and flips[i] else p
            for p, i in zip((self.programs[r] for r in rows), scenes)
        ]
        aux_scenes = []
        aux_words = []
        for i, n in enumerate(indices):
            for words in self.aux[n]:
                aux_scenes.append(i)
                aux_words.append([WORD_INDEX[w] for w in words] + [0] * (4 - len(words)))
        aux = (
            None
            if not aux_words
            else (
                torch.tensor(aux_scenes, device=device),
                torch.tensor(aux_words, device=device),
            )
        )
        return dict(
            rows=rows,
            scenes=torch.as_tensor(scenes, device=device),
            programs=encode_programs(programs, device),
            pairs=torch.as_tensor(pairs, device=device),
            aux=aux,
            label=None if self.label is None else torch.as_tensor(self.label[rows], device=device),
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
