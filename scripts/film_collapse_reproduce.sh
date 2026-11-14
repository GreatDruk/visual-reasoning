#!/usr/bin/env bash
# Последовательное воспроизведение; только новые output paths, никаких val/test.
set -euo pipefail
root="${1:?Укажите новый каталог результатов относительно корня repo}"
if [ -e "$root" ]; then echo "Каталог уже существует: $root" >&2; exit 1; fi
mkdir -p "$root"
run() {
    local name="$1"; shift
    .venv/bin/python -m src.train experiment=film_trace "name_exp=debug_collapse_${name}" "paths.output_dir=$root/$name" "$@" > "$root/${name}.log" 2>&1
    .venv/bin/python scripts/film_collapse_measure.py "$root/$name" "$root/${name}_measurements.json" --full-train > "$root/${name}_measure.log" 2>&1
}
# Исходная partial ветка уже сохранена; повтор нужен только при воспроизведении с нуля.
run partial
run frozen model.image_encoder.unfreeze_last_n_blocks=0
run pair loss.pair_weight=0.5
run small_w9 data.train_pairs_per_category=16 train.epochs=120 train.scheduler.warmup_steps=9 train.time_budget_min=15 train.eval.train=true train.diagnostics.every_steps=270
run medium_w9 data.train_pairs_per_category=64 train.epochs=30 train.scheduler.warmup_steps=9 train.time_budget_min=15 train.eval.train=true train.diagnostics.every_steps=270
run medium_w938 data.train_pairs_per_category=64 train.epochs=30 train.scheduler.warmup_steps=938 train.time_budget_min=15 train.eval.train=true train.diagnostics.every_steps=270
