"""Train-only dataset audit for the FiLM scaling investigation.

Reads data/train.csv plus the archived medium_w9 fit ids (verification only).
No val/test access, no model fitting or evaluation; CPU-only data analysis.

Run: .venv/bin/python scripts/film_scaling_audit.py
Outputs:
  artefacts/film_scaling/audit.json                 full audit (Russian summary inside)
  artefacts/film_scaling/subsets/original_1152.json successful medium_w9 subset (anchor)
  artefacts/film_scaling/subsets/nested_1152.json   identical to the anchor
  artefacts/film_scaling/subsets/nested_4608.json   category-balanced extension
  artefacts/film_scaling/subsets/nested_9216.json   category-balanced extension
  artefacts/film_scaling/stage1_dataset.md          Russian method/results/limits report
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import polars as pl

from src.data import load_split, sample_pairs_by_category

# Deterministic, transparent lexicon normalisation for structural caption
# templates. Only attribute-value words are collapsed into slots; every
# relation, quantifier and logical operator stays literal. The lexicon is
# justified from train.csv caption samples only (CLEVR-style scene grammar).
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"]
COLOR_WORDS = ["red", "blue", "green", "brown", "purple", "cyan", "yellow", "gray", "grey"]
SHAPE_WORDS = ["cube", "sphere", "cylinder"]
SIZE_WORDS = ["large", "small", "big"]
MATERIAL_WORDS = ["rubber", "metal", "metallic"]
LEXICON = {}
for _slot, _words in (("NUM", NUMBER_WORDS), ("COLOR", COLOR_WORDS), ("SHAPE", SHAPE_WORDS),
                      ("SIZE", SIZE_WORDS), ("MATERIAL", MATERIAL_WORDS)):
    for _w in _words:
        LEXICON[_w] = _slot
TOKEN_RE = re.compile(r"[a-z]+|\d+")


def caption_tokens(caption):
    return TOKEN_RE.findall(caption.lower())


def normalize_caption(caption):
    """Caption -> structural template: attribute values become slots, rest stays literal."""
    out = []
    for tok in caption_tokens(caption):
        if tok in LEXICON:
            out.append(LEXICON[tok])
        elif tok.isdigit():
            out.append("NUM")
        elif tok.endswith("s") and tok[:-1] in LEXICON:  # regular plurals of lexicon words
            out.append(LEXICON[tok[:-1]])
        else:
            out.append(tok)
    return " ".join(out)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy_bits(counter):
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counter.values())


def length_stats(lengths):
    arr = np.asarray(lengths, dtype=np.int64)
    hist = Counter(int(x) for x in arr)
    q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": int(arr.min()), "max": int(arr.max()), "mean": float(arr.mean()),
        "std": float(arr.std()), "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
        "histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def template_stats(templates, categories):
    per_cat = {}
    cats = sorted(set(categories))
    for cat in cats:
        counts = Counter(t for t, c in zip(templates, categories) if c == cat)
        top = counts.most_common(5)
        per_cat[cat] = {
            "captions": sum(counts.values()), "unique_templates": len(counts),
            "templates_per_1000_captions": 1000.0 * len(counts) / sum(counts.values()),
            "entropy_bits": entropy_bits(counts),
            "top_templates": [{"template": t, "count": n} for t, n in top],
        }
    counts = Counter(templates)
    return {
        "unique_templates": len(counts),
        "entropy_bits": entropy_bits(counts),
        "effective_templates": 2.0 ** entropy_bits(counts),
        "top_templates": [{"template": t, "count": n} for t, n in counts.most_common(10)],
        "per_category": per_cat,
    }


def caption_label_statistics(captions, labels):
    """In-sample caption-only ceiling from label counts, not a trained model."""
    counts = defaultdict(Counter)
    for caption, label in zip(captions, labels):
        counts[caption][int(label)] += 1
    rows = len(labels)
    ambiguous = [values for values in counts.values() if len(values) > 1]
    return {
        "singleton_captions": sum(sum(values.values()) == 1 for values in counts.values()),
        "label_ambiguous_captions": len(ambiguous),
        "rows_on_label_ambiguous_captions": sum(sum(values.values()) for values in ambiguous),
        "caption_only_accuracy_ceiling": sum(max(values.values()) for values in counts.values()) / rows,
        "caption_label_conditional_entropy_bits": sum(
            -sum(n * math.log2(n / sum(values.values())) for n in values.values())
            for values in counts.values()) / rows,
    }


def describe(rows):
    """rows: list of dicts with id, image, caption, level, category, pair_id, label."""
    ids = [r["id"] for r in rows]
    pair_counts = Counter(r["pair_id"] for r in rows)
    images = Counter(r["image"] for r in rows)
    pairs_per_image = Counter()
    pair_image = {}
    for r in rows:
        pair_image.setdefault(r["pair_id"], r["image"])
    for image in pair_image.values():
        pairs_per_image[image] += 1
    labels = Counter(r["label"] for r in rows)
    cats = Counter(r["category"] for r in rows)
    levels = Counter(r["level"] for r in rows)
    strata = Counter((r["category"], r["level"]) for r in rows)
    lengths = [len(caption_tokens(r["caption"])) for r in rows]
    captions = [r["caption"] for r in rows]
    templates = [normalize_caption(c) for c in captions]
    return {
        "rows": len(rows), "pairs": len(pair_counts),
        "unique_images": len(images),
        "unique_captions": len(set(captions)),
        "duplicate_caption_rows": len(captions) - len(set(captions)),
        "label_counts": {str(k): labels[k] for k in sorted(labels)},
        "category_counts": {k: cats[k] for k in sorted(cats)},
        "level_counts": {str(k): levels[k] for k in sorted(levels)},
        "strata_counts": {f"{c}|{l}": strata[(c, l)] for (c, l) in sorted(strata)},
        "caption_length_tokens": length_stats(lengths),
        "pairs_per_image_histogram": {str(k): pairs_per_image[k] for k in sorted(set(pairs_per_image.values()))},
        "pairs_per_image_mean": float(np.mean(list(pairs_per_image.values()))),
        "pairs_per_image_max": max(pairs_per_image.values()),
        "rows_per_image_max": max(images.values()),
        "vocabulary": sorted(set(w for c in captions for w in caption_tokens(c))),
        "caption_label_statistics": caption_label_statistics(captions, [r["label"] for r in rows]),
        "templates": templates,
    }


def contradiction_audit(row_list):
    """Exact-duplicate and contradiction counts; caption text never leaves this function."""
    by_ic = defaultdict(list)
    for r in row_list:
        by_ic[(r["image"], r["caption"])].append(r)
    dup = [v for v in by_ic.values() if len(v) > 1]
    conflict = [v for v in dup if len({r["label"] for r in v}) > 1]
    by_pair = defaultdict(list)
    for r in row_list:
        by_pair[r["pair_id"]].append(r)
    sig_img, sig_cap = Counter(), Counter()
    self_identical = []
    for pid, rs in by_pair.items():
        caps = sorted(r["caption"] for r in rs)
        if caps[0] == caps[-1]:
            self_identical.append(pid)
        sig_img[(rs[0]["image"], tuple(caps))] += 1
        sig_cap[tuple(caps)] += 1
    sig_img_dup = {k: v for k, v in sig_img.items() if v > 1}
    sig_cap_dup = {k: v for k, v in sig_cap.items() if v > 1}
    return {
        "rows": len(row_list), "pairs": len(by_pair),
        "duplicate_image_caption_groups": len(dup),
        "duplicate_image_caption_rows": sum(len(v) for v in dup),
        "conflicting_label_groups": len(conflict),
        "conflicting_label_rows": sum(len(v) for v in conflict),
        "conflicting_examples_head": [
            {"image": v[0]["image"],
             "caption_sha256_12": hashlib.sha256(v[0]["caption"].encode()).hexdigest()[:12],
             "ids": sorted(r["id"] for r in v),
             "pair_ids": sorted({r["pair_id"] for r in v}),
             "labels": sorted(r["label"] for r in v)}
            for v in conflict[:5]],
        "pair_self_identical_caption_pairs": len(self_identical),
        "pair_self_identical_pair_ids_head": sorted(self_identical)[:20],
        "duplicate_pair_signatures_with_image_groups": len(sig_img_dup),
        "duplicate_pair_signatures_with_image_pairs": sum(sig_img_dup.values()),
        "duplicate_pair_signatures_captions_only_groups": len(sig_cap_dup),
        "duplicate_pair_signatures_captions_only_pairs": sum(sig_cap_dup.values()),
    }


def hamilton(quota, capacities, key_order):
    """Largest-remainder (largest deficit) allocation; ties broken by key_order position."""
    alloc = {k: min(int(math.floor(quota[k])), capacities[k]) for k in key_order}
    remaining = int(round(sum(quota.values()))) - sum(alloc.values())
    while remaining > 0:
        candidates = [k for k in key_order if alloc[k] < capacities[k]]
        if not candidates:
            raise SystemExit("hamilton: allocation not attainable within capacities")
        best = max(candidates, key=lambda k: (quota[k] - alloc[k], -key_order.index(k)))
        alloc[best] += 1
        remaining -= 1
    return alloc


def main():
    parser = argparse.ArgumentParser(description="Train-only dataset audit for FiLM scaling.")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--reference-ids", default="artefacts/film_collapse/medium_w9/train_fit_ids.npy")
    parser.add_argument("--out-dir", default="artefacts/film_scaling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-category", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs=3, default=[1152, 4608, 9216])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    subsets_dir = out_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(args.train_csv)
    rows = df.select(["id", "image", "caption", "level", "category", "pair_id", "label"]).to_dicts()
    by_id = {r["id"]: r for r in rows}
    csv_sha = sha256_file(args.train_csv)

    # ---- reconstruct the successful medium_w9 subset exactly -----------------
    split = load_split("train", args.train_csv)
    anchor = sample_pairs_by_category(split, args.pairs_per_category, args.seed)
    anchor_ids = [int(i) for i in anchor.ids]
    anchor_rows = [by_id[i] for i in anchor_ids]

    reference = np.load(args.reference_ids)
    ref_ids = [int(i) for i in reference]
    reconstruction = {
        "method": f"sample_pairs_by_category(load_split('train', {args.train_csv!r}), "
                  f"{args.pairs_per_category}, {args.seed})",
        "rows": len(anchor_ids), "pairs": len(anchor_ids) // 2,
        "reference_path": args.reference_ids,
        "reference_rows": len(ref_ids),
        "reference_sha256": sha256_file(args.reference_ids),
        "set_equal": sorted(anchor_ids) == sorted(ref_ids),
        "order_equal": anchor_ids == ref_ids,
        "anchor_ids_sha256": hashlib.sha256(np.asarray(anchor_ids, np.int64).tobytes()).hexdigest(),
    }
    if not reconstruction["set_equal"]:
        raise SystemExit("anchor reconstruction does not match train_fit_ids.npy")
    # exact caption fidelity: reconstructed captions must equal the csv rows by id
    caption_mismatch = [int(i) for i, c in zip(anchor.ids, anchor.captions)
                        if by_id[int(i)]["caption"] != c]
    reconstruction["caption_exact_match_by_id"] = not caption_mismatch
    reconstruction["caption_mismatch_ids_head"] = caption_mismatch[:20]
    reconstruction["anchor_ids_in_csv_order"] = anchor_ids == sorted(anchor_ids)
    anchor_pair_ids = sorted({by_id[i]["pair_id"] for i in anchor_ids})
    reconstruction["anchor_pair_ids_all_two_rows"] = all(
        sum(1 for r in rows if r["pair_id"] == pid) == 2 for pid in anchor_pair_ids)

    # ---- pair and stratum tables -------------------------------------------
    pairs = {}
    for r in rows:
        p = pairs.setdefault(r["pair_id"], {"pair_id": r["pair_id"], "ids": [], "image": r["image"],
                                            "category": r["category"], "level": r["level"],
                                            "labels": set()})
        p["ids"].append(r["id"])
        p["labels"].add(r["label"])
        if p["image"] != r["image"]:
            raise SystemExit(f"pair {r['pair_id']} spans multiple images")
    for p in pairs.values():
        p["ids"].sort()
        if len(p["ids"]) != 2 or p["labels"] != {0, 1}:
            raise SystemExit(f"pair {p['pair_id']} malformed: ids={p['ids']} labels={p['labels']}")
    strata_pairs = {}
    for p in pairs.values():
        strata_pairs.setdefault((p["category"], p["level"]), []).append(p["pair_id"])
    for key in strata_pairs:
        strata_pairs[key].sort()
    strata_order = sorted(strata_pairs)
    full_pair_counts = {f"{c}|{l}": len(strata_pairs[(c, l)]) for (c, l) in strata_order}

    anchor_pairs_by_cat = {}
    for pid in anchor_pair_ids:
        anchor_pairs_by_cat.setdefault(pairs[pid]["category"], []).append(pid)
    for cat in anchor_pairs_by_cat:
        anchor_pairs_by_cat[cat].sort()

    # ---- nested category-balanced extension --------------------------------
    # Per category the deterministic pair order is: anchor pairs (ascending
    # pair_id) followed by the remaining pairs (ascending pair_id) permuted by
    # np.random.default_rng(seed) over categories in sorted order. Prefixes of
    # this order give nested subsets by construction.
    rng = np.random.default_rng(args.seed)
    category_order = sorted(set(p["category"] for p in pairs.values()))
    extended_order = {}
    for cat in category_order:
        all_pids = sorted(pid for pid, p in pairs.items() if p["category"] == cat)
        anchor_pids = anchor_pairs_by_cat.get(cat, [])
        anchor_set = set(anchor_pids)
        remaining = [pid for pid in all_pids if pid not in anchor_set]
        perm = rng.permutation(len(remaining))
        extended_order[cat] = list(anchor_pids) + [remaining[i] for i in perm]

    if len(args.sizes) < 1 or args.sizes[0] != len(anchor_ids):
        raise SystemExit(f"first nested size must be the anchor size {len(anchor_ids)}")
    per_cat_request = {size: size // (2 * len(category_order)) for size in args.sizes}
    if any(2 * len(category_order) * n != size for size, n in per_cat_request.items()):
        raise SystemExit(f"sizes must be divisible by {2 * len(category_order)} rows")

    nested_ids = {}
    nested_manifests = {}
    feasibility = {}
    for size in args.sizes:
        want = per_cat_request[size]
        chosen_pairs, capped = [], []
        for cat in category_order:
            avail = len(extended_order[cat])
            take = min(want, avail)
            if take < want:
                capped.append({"category": cat, "requested_pairs": want, "available_pairs": avail})
            chosen_pairs.extend(extended_order[cat][:take])
        feasibility[size] = {
            "pairs_per_category_requested": want,
            "available_pairs_per_category": {cat: len(extended_order[cat]) for cat in category_order},
            "capped": capped, "feasible": not capped,
        }
        ids_sorted = sorted(pid for pid in chosen_pairs for pid in pairs[pid]["ids"])
        nested_ids[size] = ids_sorted
        manifest_rows = [by_id[i] for i in ids_sorted]
        stat = describe(manifest_rows)
        manifest = {
            "name": f"nested_{size}",
            "requested_rows": size,
            "actual_rows": stat["rows"],
            "seed": args.seed,
            "ids": ids_sorted,
            "pair_ids": sorted(chosen_pairs),
            "pair_count": len(chosen_pairs),
            "pairs_per_category": want,
            "strategy": ("anchor category-balanced nested extension: "
                         f"{args.pairs_per_category} pairs/category (medium_w9 anchor) "
                         f"-> {want} pairs/category; pair order = anchor pairs then "
                         f"default_rng({args.seed}) permutation of remaining pairs per category"),
            "category_counts": stat["category_counts"],
            "level_counts": stat["level_counts"],
            "strata_counts": stat["strata_counts"],
            "unique_images": stat["unique_images"],
            "label_counts": stat["label_counts"],
            "train_csv": args.train_csv,
            "train_csv_sha256": csv_sha,
            "anchor": "subsets/original_1152.json",
        }
        nested_manifests[size] = (manifest, stat)

    # anchor manifest (always written) + nested_1152 must be identical
    anchor_stat = describe(anchor_rows)
    original_manifest = {
        "name": "original_1152",
        "requested_rows": len(anchor_ids),
        "actual_rows": len(anchor_ids),
        "seed": args.seed,
        "ids": anchor_ids,
        "pair_ids": anchor_pair_ids,
        "pair_count": len(anchor_pair_ids),
        "pairs_per_category": args.pairs_per_category,
        "strategy": ("exact reconstruction of the successful medium_w9 fit subset: "
                     f"sample_pairs_by_category(train, {args.pairs_per_category}, {args.seed}); "
                     f"verified against {args.reference_ids}"),
        "category_counts": anchor_stat["category_counts"],
        "level_counts": anchor_stat["level_counts"],
        "strata_counts": anchor_stat["strata_counts"],
        "unique_images": anchor_stat["unique_images"],
        "label_counts": anchor_stat["label_counts"],
        "train_csv": args.train_csv,
        "train_csv_sha256": csv_sha,
        "verified_against": {
            "path": args.reference_ids,
            "set_equal": reconstruction["set_equal"],
            "order_equal": reconstruction["order_equal"],
        },
        "anchor": None,
    }

    # ---- verification ------------------------------------------------------
    def verify(manifest):
        sel = [by_id[i] for i in manifest["ids"]]
        pair_counts = Counter(r["pair_id"] for r in sel)
        labels = Counter(r["label"] for r in sel)
        return {
            "rows_match_ids": len(sel) == manifest["actual_rows"] == manifest["requested_rows"],
            "pair_integrity": all(v == 2 for v in pair_counts.values()),
            "pairs_complete": len(pair_counts) == manifest["pair_count"],
            "label_balance": {str(k): labels[k] for k in sorted(labels)},
            "label_balanced_50_50": labels[0] == labels[1],
            "strata_pair_counts": {f"{c}|{l}": sum(1 for pid in pair_counts if pairs[pid]["category"] == c and pairs[pid]["level"] == l)
                                   for (c, l) in strata_order},
        }

    verifications = {"original_1152": verify(original_manifest)}
    for size in args.sizes:
        verifications[f"nested_{size}"] = verify(nested_manifests[size][0])
    s1152, s4608, s9216 = (set(nested_ids[s]) for s in args.sizes)
    nesting = {
        "nested_1152_equals_original": s1152 == set(anchor_ids) and nested_ids[args.sizes[0]] == anchor_ids,
        "nested_1152_subset_of_nested_4608": s1152 <= s4608,
        "nested_4608_subset_of_nested_9216": s4608 <= s9216,
    }

    # ---- full vs subset audit ---------------------------------------------
    full_stat = describe(rows)
    full_vocab = set(full_stat["vocabulary"])
    full_templates = set(full_stat["templates"])
    row_categories = [r["category"] for r in rows]
    full_cat_templates = {cat: {t for t, c in zip(full_stat["templates"], row_categories) if c == cat}
                          for cat in sorted(full_stat["category_counts"])}
    vocab_word_counts = Counter(w for r in rows for w in caption_tokens(r["caption"]))

    def subset_vs_full(stat, row_list):
        vocab = set(stat["vocabulary"])
        templates = set(stat["templates"])
        cat_templates = {}
        for cat in sorted(stat["category_counts"]):
            cat_templates[cat] = set(t for t, r in zip(stat["templates"], row_list) if r["category"] == cat)
        missing = {w: vocab_word_counts[w] for w in sorted(full_vocab - vocab)}
        stratum_dev = {}
        for (c, l) in strata_order:
            key = f"{c}|{l}"
            full_p = full_pair_counts[key] / full_stat["pairs"]
            sub_p = stat["strata_counts"].get(key, 0) / stat["rows"] if stat["rows"] else 0.0
            stratum_dev[key] = {
                "full_pair_share": full_p, "subset_row_share": sub_p,
                "deviation_pp": 100.0 * (sub_p - full_p),
                "ratio_subset_over_full": (sub_p / full_p) if full_p else None,
            }
        tv = 0.5 * sum(abs(d["subset_row_share"] - d["full_pair_share"]) for d in stratum_dev.values())
        return {
            "vocabulary_covered": len(vocab), "vocabulary_full": len(full_vocab),
            "vocabulary_coverage": len(vocab) / len(full_vocab),
            "vocabulary_occurrence_coverage": 1.0 - sum(missing.values()) / sum(vocab_word_counts.values()),
            "vocabulary_missing_from_subset": missing,
            "templates_unique": len(templates),
            "template_coverage_of_full_inventory": len(templates & full_templates) / len(full_templates),
            "all_subset_templates_in_full": templates <= full_templates,
            "per_category_template_coverage": {
                cat: {"subset_templates": len(cat_templates[cat]), "full_templates": len(full_cat_templates[cat]),
                      "coverage": len(cat_templates[cat] & full_cat_templates[cat]) / len(full_cat_templates[cat])}
                for cat in sorted(cat_templates)
            },
            "stratum_proportion_deviation": stratum_dev,
            "stratum_tv_distance_from_full": tv,
            "max_abs_deviation_pp": max(abs(d["deviation_pp"]) for d in stratum_dev.values()),
        }

    comparisons = {
        "original_1152": subset_vs_full(anchor_stat, anchor_rows),
        **{f"nested_{size}": subset_vs_full(nested_manifests[size][1], [by_id[i] for i in nested_ids[size]])
           for size in args.sizes},
    }

    # image reuse across the full train and the subsets
    def image_set(id_list):
        return {by_id[i]["image"] for i in id_list}

    anchor_images = image_set(anchor_ids)
    nested_images = {size: image_set(nested_ids[size]) for size in args.sizes}
    image_reuse = {
        "full_unique_images": full_stat["unique_images"],
        "full_pairs_per_image_mean": full_stat["pairs_per_image_mean"],
        "full_pairs_per_image_histogram": full_stat["pairs_per_image_histogram"],
        "original_1152_unique_images": anchor_stat["unique_images"],
        "original_1152_image_share_of_full": anchor_stat["unique_images"] / full_stat["unique_images"],
        "original_1152_pairs_per_image_mean": anchor_stat["pairs_per_image_mean"],
        "original_1152_pairs_per_image_histogram": anchor_stat["pairs_per_image_histogram"],
        "nested_unique_images": {f"nested_{s}": len(nested_images[s]) for s in args.sizes},
        "nested_pairs_per_image_mean": {f"nested_{s}": nested_manifests[s][1]["pairs_per_image_mean"] for s in args.sizes},
        "images_nested_1152_subset_of_4608": nested_images[args.sizes[0]] <= nested_images[args.sizes[1]],
        "images_nested_4608_subset_of_9216": nested_images[args.sizes[1]] <= nested_images[args.sizes[2]],
        "shared_images_1152_vs_4608": len(nested_images[args.sizes[0]] & nested_images[args.sizes[1]]),
        "shared_images_4608_vs_9216": len(nested_images[args.sizes[1]] & nested_images[args.sizes[2]]),
        "anchor_images_equals_nested_1152": anchor_images == nested_images[args.sizes[0]],
    }

    # exact duplicates / label contradictions (learnability constraint, no parser)
    contradictions = {
        "full_train": contradiction_audit(rows),
        "original_1152": contradiction_audit(anchor_rows),
        **{f"nested_{size}": contradiction_audit([by_id[i] for i in nested_ids[size]]) for size in args.sizes},
    }

    # full-proportional reference allocation (Hamilton, largest deficit) --
    # reported for transparency alongside the anchor-strategy manifests.
    full_proportional_reference = {}
    for size in args.sizes:
        target_pairs = size // 2
        quota = {key: target_pairs * full_pair_counts[key] / full_stat["pairs"] for key in full_pair_counts}
        caps = {key: full_pair_counts[key] for key in full_pair_counts}
        alloc = hamilton(quota, caps, [f"{c}|{l}" for (c, l) in strata_order])
        full_proportional_reference[str(size)] = {
            "target_pairs": target_pairs,
            "quota": {k: round(quota[k], 4) for k in quota},
            "allocated_pairs": alloc,
            "attainable": sum(alloc.values()) == target_pairs and all(alloc[k] <= caps[k] for k in alloc),
        }

    # ---- write manifests ---------------------------------------------------
    def write_json(path, obj):
        path.write_text(json.dumps(obj, indent=2, allow_nan=False))

    write_json(subsets_dir / "original_1152.json", original_manifest)
    for size in args.sizes:
        write_json(subsets_dir / f"nested_{size}.json", nested_manifests[size][0])

    # ---- audit.json --------------------------------------------------------
    def clean(stat):
        out = {k: v for k, v in stat.items() if k not in ("vocabulary", "templates")}
        return out

    audit = {
        "generated_by": "scripts/film_scaling_audit.py",
        "argv": sys.argv,
        "reproducibility": {
            "seed": args.seed, "pairs_per_category": args.pairs_per_category,
            "requested_sizes": args.sizes,
            "train_csv": args.train_csv, "train_csv_sha256": csv_sha,
            "reference_ids": args.reference_ids,
            "python": platform.python_version(),
            "numpy": np.__version__, "polars": pl.__version__,
            "extension_pair_order": ("per category (sorted): anchor pair_ids ascending, then remaining "
                                     f"pair_ids ascending permuted by np.random.default_rng({args.seed}), "
                                     "categories iterated in sorted order"),
            "template_normalisation": {
                "method": "deterministic lexicon substitution, no parser/model fitting",
                "slots": {"NUM": NUMBER_WORDS, "COLOR": COLOR_WORDS, "SHAPE": SHAPE_WORDS,
                          "SIZE": SIZE_WORDS, "MATERIAL": MATERIAL_WORDS},
                "rules": ["lowercase, tokenise [a-z]+|\\d+",
                          "lexicon word or its regular plural -> slot token",
                          "digit runs -> NUM",
                          "all other tokens (relations, quantifiers, logical operators, "
                          "function words) stay literal"],
            },
        },
        "reconstruction": reconstruction,
        "full_train": {**clean(full_stat),
                       "vocabulary_size": len(full_vocab),
                       "vocabulary_counts": dict(sorted(vocab_word_counts.items())),
                       "templates": template_stats(full_stat["templates"], [r["category"] for r in rows]),
                       "contradictions": contradictions["full_train"],
                       "caption_lowercase_already": all(r["caption"] == r["caption"].lower() for r in rows)},
        "original_1152": {**clean(anchor_stat),
                          "vocabulary_size": len(set(anchor_stat["vocabulary"])),
                          "templates": template_stats(anchor_stat["templates"], [r["category"] for r in anchor_rows]),
                          "contradictions": contradictions["original_1152"]},
        "subset_strategy": {
            "decision": "anchor category-balanced nested extension (parent direction)",
            "rationale": ("medium_w9 anchor uses 64 pairs/category (uniform 1/9 per category-level "
                          "stratum; category determines level, so 9 joint strata). nested_1152 is "
                          "byte-identical to the anchor so existing checkpoints stay comparable; "
                          "larger sets extend with the same fixed original strata proportions. "
                          "Deviation from the full train distribution is reported below and is NOT "
                          "removed: full proportions are skewed (binding ~17.3% vs spatial_negation "
                          "~4.2% of pairs)."),
            "stratum_proportions_fixed": "original anchor: 64 pairs per category-level stratum (1/9 each)",
            "deviation_from_full_distribution": comparisons["original_1152"]["stratum_proportion_deviation"],
            "tv_distance_from_full": comparisons["original_1152"]["stratum_tv_distance_from_full"],
            "max_abs_deviation_pp": comparisons["original_1152"]["max_abs_deviation_pp"],
            "full_proportional_reference_hamilton": full_proportional_reference,
            "feasibility": feasibility,
        },
        "comparisons_subset_vs_full": comparisons,
        "image_reuse": image_reuse,
        "nested_manifests": {
            f"nested_{size}": {"manifest": str(subsets_dir / f"nested_{size}.json"),
                               "contradictions": contradictions[f"nested_{size}"],
                               **clean(nested_manifests[size][1])}
            for size in args.sizes
        },
        "verification": {
            "per_manifest": verifications,
            "nesting": nesting,
            "requested_vs_actual_rows": {f"nested_{size}": {"requested": size,
                                                            "actual": nested_manifests[size][0]["actual_rows"]}
                                         for size in args.sizes},
            "all_checks_passed": (
                all(v["pair_integrity"] and v["label_balanced_50_50"] and v["rows_match_ids"]
                    for v in verifications.values())
                and all(nesting.values())
                and all(feasibility[size]["feasible"] for size in args.sizes)
            ),
        },
        "limits": [
            "Only data/train.csv is read; val/test are not accessed anywhere in this script.",
            "No model is trained or evaluated here; this is a CPU-only data audit.",
            "Templates come from a deterministic lexicon normalisation of surface words; they "
            "capture construction diversity of phrasing, not scene semantics.",
            "The anchor subset is category-balanced (uniform over 9 category-level strata), which "
            "deliberately deviates from the full train stratum proportions to stay comparable with "
            "the successful medium_w9 run; deviation numbers are reported, not corrected.",
            "train_fit_ids.npy is used only to verify the reconstructed anchor ids.",
        ],
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2))

    # ---- Russian report ----------------------------------------------------
    def fmt_dev_table():
        lines = ["| Стратум (категория \\| уровень) | Доля парами в train | Доля строк в подмножестве | Отклонение, п.п. | Отношение |",
                 "|---|---|---|---|---|"]
        for key, d in comparisons["original_1152"]["stratum_proportion_deviation"].items():
            cat, lvl = key.split("|")
            lines.append(f"| {cat} (ур. {lvl}) | {100*d['full_pair_share']:.2f}% | {100*d['subset_row_share']:.2f}% "
                         f"| {d['deviation_pp']:+.2f} | {d['ratio_subset_over_full']:.2f} |")
        return "\n".join(lines)

    def fmt_manifest_table():
        lines = ["| Манифест | Строк | Пар | Пар/категория | Уникальных изображений | Баланс меток |",
                 "|---|---|---|---|---|---|"]
        lines.append(f"| original_1152 | {original_manifest['actual_rows']} | {original_manifest['pair_count']} "
                     f"| {args.pairs_per_category} | {original_manifest['unique_images']} | 50/50 |")
        for size in args.sizes:
            m = nested_manifests[size][0]
            lines.append(f"| nested_{size} | {m['actual_rows']} | {m['pair_count']} "
                         f"| {m['pairs_per_category']} | {m['unique_images']} | 50/50 |")
        return "\n".join(lines)

    full_len = full_stat["caption_length_tokens"]
    anc_len = anchor_stat["caption_length_tokens"]
    min_stratum_key, min_stratum_n = min(((f"{c}|{l}", len(v)) for (c, l), v in strata_pairs.items()),
                                         key=lambda kv: kv[1])
    ts_full = audit["full_train"]["templates"]
    ts_anc = audit["original_1152"]["templates"]
    comp = comparisons["original_1152"]
    md = f"""# Этап 1. Аудит датасета для масштабирования FiLM

Отчёт сгенерирован скриптом `scripts/film_scaling_audit.py` (запуск один раз, CPU).
Детальные числа: `{out_dir}/audit.json`, манифесты подмножеств: `{subsets_dir}/`.

## Метод

- Читается **только** `data/train.csv` ({full_stat['rows']} строк, {full_stat['pairs']} пар, {full_stat['unique_images']} изображений). Val/test не читаются; моделей не обучается и не оценивается.
- Успешное подмножество `medium_w9` восстановлено точно: `sample_pairs_by_category(train, {args.pairs_per_category}, {args.seed})` -> {reconstruction['rows']} строк; идентификаторы сверены с `{args.reference_ids}` (совпадение множества: **{reconstruction['set_equal']}**, порядка: **{reconstruction['order_equal']}**).
- Стратификация: пересечение category × level. В train каждая категория принадлежит ровно одному уровню, поэтому 9 совместных страт. Каждая пара целиком внутри одной страты и несёт метки {{0, 1}} — сохранение пар автоматически даёт баланс классов 50/50.
- Шаблоны конструкций считаются детерминированной нормализацией лексиконом (без парсера/модели): слова-значения атрибутов (числа, цвета, формы, размеры, материалы) заменяются на слоты NUM/COLOR/SHAPE/SIZE/MATERIAL, все остальные слова (отношения, кванторы, логические операторы: *no, exactly, more, fewer, than, as many, left, right* и т.п.) остаются как есть. Лексикон выведен из примеров `train.csv` (CLEVR-подобная грамматика сцен).
- Вложенные подмножества строятся по парам (пара не режется), размеры делятся нацело: {args.sizes[0]} / {args.sizes[1]} / {args.sizes[2]} строк = {args.sizes[0]//2} / {args.sizes[1]//2} / {args.sizes[2]//2} пар.

## Результаты

### Полный train

- Строк: {full_stat['rows']}, пар: {full_stat['pairs']}, уникальных изображений: {full_stat['unique_images']}, уникальных подписей: {full_stat['unique_captions']} (повторных вхождений caption: {full_stat['duplicate_caption_rows']}).
- Пар на изображение: среднее {full_stat['pairs_per_image_mean']:.3f}, максимум {full_stat['pairs_per_image_max']} (гистограмма в audit.json).
- Противоречия и дубликаты: конфликтующие метки у (image, caption) — **{contradictions['full_train']['conflicting_label_groups']}** групп ({contradictions['full_train']['conflicting_label_rows']} строк); идентичные подписи внутри пары — **{contradictions['full_train']['pair_self_identical_caption_pairs']}** пар. Это проверка точных входных конфликтов, не семантической корректности меток и не общей обучаемости. Дубликаты (image, caption) — {contradictions['full_train']['duplicate_image_caption_groups']} групп; повторяющиеся сигнатуры пар с изображением — {contradictions['full_train']['duplicate_pair_signatures_with_image_groups']} групп, только по подписям — {contradictions['full_train']['duplicate_pair_signatures_captions_only_groups']} групп. Детали без сырых подписей — в audit.json (`contradictions`).
- Баланс классов: {full_stat['label_counts']}.
- Длина подписи (токенов): среднее {full_len['mean']:.2f} ± {full_len['std']:.2f}, медиана {full_len['median']:.0f}, min {full_len['min']}, max {full_len['max']}.
- Словарь train: {len(full_vocab)} уникальных токенов.
- Шаблоны: {ts_full['unique_templates']} уникальных, энтропия {ts_full['entropy_bits']:.2f} бит, эффективное число конструкций {ts_full['effective_templates']:.1f}.

### Оригинальное подмножество (anchor, original_1152.json)

- Строк: {anchor_stat['rows']}, пар: {anchor_stat['pairs']}, уникальных изображений: {anchor_stat['unique_images']}, уникальных подписей: {anchor_stat['unique_captions']} (дубликатов: {anchor_stat['duplicate_caption_rows']}).
- Пар на изображение: среднее {anchor_stat['pairs_per_image_mean']:.3f}, максимум {anchor_stat['pairs_per_image_max']}.
- Переиспользование изображений: {anchor_stat['unique_images']} уникальных изображений ({100*image_reuse['original_1152_image_share_of_full']:.1f}% от train), среднее пар на изображение в train {full_stat['pairs_per_image_mean']:.3f} против {anchor_stat['pairs_per_image_mean']:.3f} в подмножестве; подробные гистограммы в audit.json (`image_reuse`).
- Противоречия внутри подмножества: конфликтующие (image, caption) — {contradictions['original_1152']['conflicting_label_groups']} групп, пар с идентичными подписями — {contradictions['original_1152']['pair_self_identical_caption_pairs']}, дубликаты сигнатур пар (с изображением) — {contradictions['original_1152']['duplicate_pair_signatures_with_image_groups']} групп.
- Длина подписи: среднее {anc_len['mean']:.2f} ± {anc_len['std']:.2f}, медиана {anc_len['median']:.0f} — близко к полному train ({full_len['mean']:.2f} ± {full_len['std']:.2f}).
- Словарь: покрыто {comp['vocabulary_covered']} из {comp['vocabulary_full']} токенов ({100*comp['vocabulary_coverage']:.1f}%), по встречаемости {100*comp['vocabulary_occurrence_coverage']:.1f}%; отсутствующие токены перечислены в audit.json.
- Шаблоны: {ts_anc['unique_templates']} уникальных; инвентарь train покрыт на {100*comp['template_coverage_of_full_inventory']:.1f}%; все шаблоны подмножества присутствуют в train: {comp['all_subset_templates_in_full']}. Энтропия {ts_anc['entropy_bits']:.2f} бит.
- Полнота по категориям (шаблоны): {json.dumps({k: f"{100*v['coverage']:.0f}%" for k, v in comp['per_category_template_coverage'].items()}, ensure_ascii=False)}.

### Представительность: отклонение от полного распределения

Anchor категориально сбалансирован (64 пары на категорию = 11.11% на страту), тогда как полный train скошен. Отклонение **сознательное** (нужно для совместимости с успешным medium_w9), числа:

{fmt_dev_table()}

- Total variation distance от полного распределения: **{comp['stratum_tv_distance_from_full']:.4f}**; максимальное отклонение: **{comp['max_abs_deviation_pp']:.2f} п.п.**
- Для сравнения: аллокация, пропорциональная полному train (гамильтоново/наибольший недобор), приведена в audit.json (`subset_strategy.full_proportional_reference_hamilton`) и достижима для всех трёх размеров: {json.dumps({str(s): full_proportional_reference[str(s)]["attainable"] for s in args.sizes})}.

### Вложенные манифесты (nested scaling)

{fmt_manifest_table()}

- Стратегия: пары категорий идут в фиксированном порядке — anchor-пары, затем перемешанный остаток (`default_rng({args.seed})`); меньшее подмножество — префикс большего. Вложенность проверена: 1152 ⊆ 4608 ⊆ 9216: **{nesting['nested_1152_subset_of_nested_4608'] and nesting['nested_4608_subset_of_nested_9216']}**; nested_1152 идентичен anchor: **{nesting['nested_1152_equals_original']}**.
- Целостность пар, баланс меток 50/50, соответствие requested == actual проверены для всех манифестов: **{audit['verification']['all_checks_passed']}**.
- Достижимость: минимум пар в страте — {min_stratum_n} ({min_stratum_key}), требуется максимум {args.sizes[-1] // 2 // len(category_order)} на категорию; запас достаточный.
- Противоречия по размерам (конфликтующие (image, caption) групп / пары с идентичными подписями): {json.dumps({f"nested_{s}": [contradictions[f"nested_{s}"]['conflicting_label_groups'], contradictions[f"nested_{s}"]['pair_self_identical_caption_pairs']] for s in args.sizes}, ensure_ascii=False)}.
- Схема манифеста: `{{requested_rows, actual_rows, seed, ids, pair_ids, pair_count, pairs_per_category, strategy, category_counts, level_counts, strata_counts, unique_images, label_counts, train_csv, train_csv_sha256, anchor, ...}}`.

## Ограничения

- Аудит использует только train; распределения val/test не проверялись (по правилам и не должны влиять на дизайн подмножеств).
- Шаблоны — поверхностная нормализация лексиконом: отражают разнообразие формулировок, а не семантику сцены; не являются обученным парсером.
- Anchor подмножество намеренно отклоняется от полного распределения страт (см. таблицу): результаты на нём нельзя напрямую интерпретировать как оценку на полном train без учёта этого сдвига.
- `train_fit_ids.npy` использован только для сверки идентификаторов; сырые подписи в отчёты не выносятся (только шаблоны и счётчики).
- Анализ детерминирован: seed {args.seed}, фиксированные порядки обхода; повторный запуск воспроизводит те же файлы.

## Воспроизводимость

```
.venv/bin/python scripts/film_scaling_audit.py --train-csv data/train.csv \\
    --reference-ids artefacts/film_collapse/medium_w9/train_fit_ids.npy \\
    --out-dir {out_dir} --seed {args.seed} --pairs-per-category {args.pairs_per_category} \\
    --sizes {' '.join(str(s) for s in args.sizes)}
```

- SHA-256 `data/train.csv`: `{csv_sha}`
- SHA-256 `train_fit_ids.npy`: `{reconstruction['reference_sha256']}`
- Python {platform.python_version()}, numpy {np.__version__}, polars {pl.__version__}
"""
    (out_dir / "stage1_dataset.md").write_text(md)

    # ---- console compressed findings --------------------------------------
    findings = {
        "train_rows": full_stat["rows"], "train_pairs": full_stat["pairs"],
        "train_images": full_stat["unique_images"],
        "anchor_reconstruction": {"rows": reconstruction["rows"],
                                  "set_equal": reconstruction["set_equal"],
                                  "order_equal": reconstruction["order_equal"]},
        "anchor_images": anchor_stat["unique_images"],
        "anchor_pairs_per_image_mean": round(anchor_stat["pairs_per_image_mean"], 3),
        "vocab_coverage": round(comparisons["original_1152"]["vocabulary_coverage"], 4),
        "template_coverage": round(comparisons["original_1152"]["template_coverage_of_full_inventory"], 4),
        "templates_full": ts_full["unique_templates"], "templates_anchor": ts_anc["unique_templates"],
        "contradictions_full": {
            "conflicting_image_caption_groups": contradictions["full_train"]["conflicting_label_groups"],
            "conflicting_rows": contradictions["full_train"]["conflicting_label_rows"],
            "pair_self_identical_caption_pairs": contradictions["full_train"]["pair_self_identical_caption_pairs"],
            "dup_pair_signature_with_image_groups": contradictions["full_train"]["duplicate_pair_signatures_with_image_groups"],
            "dup_pair_signature_caption_only_groups": contradictions["full_train"]["duplicate_pair_signatures_captions_only_groups"],
        },
        "stratum_tv_distance": round(comparisons["original_1152"]["stratum_tv_distance_from_full"], 4),
        "max_stratum_deviation_pp": round(comparisons["original_1152"]["max_abs_deviation_pp"], 2),
        "nested_actual_rows": {f"nested_{s}": nested_manifests[s][0]["actual_rows"] for s in args.sizes},
        "nesting_ok": nesting, "all_checks_passed": audit["verification"]["all_checks_passed"],
        "outputs": [str(out_dir / "audit.json"), str(out_dir / "stage1_dataset.md")]
                    + [str(subsets_dir / f) for f in sorted(p.name for p in subsets_dir.glob("*.json"))],
    }
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
