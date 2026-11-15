#!/usr/bin/env bash
# Воспроизведение train-only исследования в новом каталоге; MPS jobs строго последовательны.
set -euo pipefail
root="${1:?Укажите новый каталог результатов относительно корня repo}"
if [ -e "$root" ]; then echo "Каталог уже существует: $root" >&2; exit 1; fi
mkdir -p "$root"
.venv/bin/python scripts/film_scaling_audit.py --out-dir "$root" > "$root/audit.log" 2>&1
run() {
    local name="$1"; shift
    .venv/bin/python scripts/film_scaling_run.py "$@" "$root/$name" > "$root/$name.log" 2>&1
}
run existing_memorized evaluate --start reference
run shared_initial evaluate --start initial --subset "$root/subsets/nested_4608.json"
run scaling_4608 run --subset "$root/subsets/nested_4608.json" --start initial \
    --steps 1080 --snapshots 0,270,540,1080 --schedule cosine --horizon 1080 --warmup 9 \
    --baseline-dir "$root/shared_initial"
run warm_original run --subset "$root/subsets/original_1152.json" --start memorized \
    --steps 540 --snapshots 0,50,270,540 --schedule constant --horizon 540 --warmup 0 \
    --baseline-dir "$root/existing_memorized"
run warm_full run --start memorized \
    --steps 540 --snapshots 0,50,270,540 --schedule constant --horizon 540 --warmup 0 \
    --baseline-dir "$root/existing_memorized"
run initial_full run --start initial \
    --steps 540 --snapshots 0,50,270,540 --schedule constant --horizon 540 --warmup 0 \
    --baseline-dir "$root/shared_initial"
.venv/bin/python scripts/film_scaling_gradients.py "$root/adaptive_gradients" \
    --checkpoint "memorized:$root/warm_full/checkpoint_step0000.pt" \
    --checkpoint "after50:$root/warm_full/checkpoint_step0050.pt" \
    --checkpoint "after540:$root/warm_full/checkpoint_step0540.pt" > "$root/adaptive_gradients.log" 2>&1
.venv/bin/python scripts/film_scaling_receipt.py --root "$root" > "$root/verification.log" 2>&1
