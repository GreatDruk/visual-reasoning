"""Read-only verifier for the train-only constant-LR long FiLM run (4320 steps).

Verifies one new run directory (default artefacts/film_long_training) against the
scaling_4608 baseline without any new inference: saved logits and measurements are
recomputed with the runner's own quality(), the pair batch sampler is replayed from
the seed, and checkpoint RNG/optimizer/BN/gradient state is compared with the
baseline step-0 capture, the baseline verified receipt and its 4 saved measurements.
Old fits are never re-verified. The receipt is written exclusively ('x'); a failed or
nonfinite run is reported, never claimed successful.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
import polars as pl
import torch

from film_scaling_run import dump, quality, state_hash
from src.data import load_split, pair_groups, sample_pairs_by_category, subset, train_batches
from src.diagnostics import prediction_summary

PLANNED_STEPS = (0, 1080, 2160, 4320)
FITTED_ROWS = 4608
PROBE_PAIRS = 16
PROBE_ROWS = 32
STOP_OK = ('max_steps', 'time_budget')
STOP_ALL = STOP_OK + ('nonfinite_loss', 'nonfinite_gradients')
REASONS = ('total_preclip_gradient_norm_above_100',
           'fixed_36_row_probe_auc_drop_at_least_0.20; noisy_diagnostic_not_full_quality',
           'preclip_module_gradient_norm_above_100')
ALLOWED_CONFIG_DIFFS = (
    'investigation', 'name_exp', 'paths.output_dir',
    'train.scheduler', 'train.time_budget_min', 'train.snapshot_reserve_sec',
    'train.anomaly_snapshot_gradient_norm',
    'train.max_steps', 'train.epochs', 'train.snapshot_steps',
    'train.diagnostics.every_steps', 'train.diagnostics.capture_steps',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    """Copy json.dump(allow_nan=False)-safe: numpy scalars unwrapped, nonfinite floats None."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def assert_close_tree(expected, actual, path='$', abs_tol=1e-7, rel_tol=1e-9):
    """Assert two nested json-ish structures agree; numbers within tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(expected) == set(actual), \
            (path, sorted(expected), sorted(actual) if isinstance(actual, dict) else actual)
        for key in expected:
            assert_close_tree(expected[key], actual[key], f'{path}.{key}', abs_tol, rel_tol)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(expected) == len(actual), (path, len(expected), len(actual))
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_close_tree(left, right, f'{path}[{index}]', abs_tol, rel_tol)
    elif isinstance(expected, bool) or expected is None or isinstance(actual, bool) or actual is None:
        assert expected == actual, (path, expected, actual)
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        assert math.isclose(float(expected), float(actual), abs_tol=abs_tol, rel_tol=rel_tol), (path, expected, actual)
    else:
        assert expected == actual, (path, expected, actual)


def lambda_factor(cfg, index):
    """LambdaLR multiplier at scheduler index (== last_epoch after that many steps)."""
    warmup = cfg.train.scheduler.warmup_steps
    return (index + 1) / warmup if index < warmup else 1.0


def expected_lr(cfg, name, index):
    base = cfg.train.optimizer.lr if name == 'task' else cfg.train.optimizer.image_encoder_lr
    return base * lambda_factor(cfg, index)


def probe_orders(split, seed, pairs):
    """The runner's fixed probe: ids plus the image derangement and twin-swap orders."""
    probe = sample_pairs_by_category(split, pairs, seed)
    indices = np.arange(len(probe))
    order = next(np.roll(indices, shift) for shift in range(1, len(probe))
                 if np.all(probe.img_idx[np.roll(indices, shift)] != probe.img_idx))
    caption_order = indices.copy()
    groups = pair_groups(probe)
    caption_order[groups[:, 0]], caption_order[groups[:, 1]] = groups[:, 1], groups[:, 0]
    return probe, order, caption_order


def replay(cfg, fitted, achieved, state_steps, id_steps):
    """Re-run the Trainer batch stream exactly; record ids, rng states and cursors."""
    rng = np.random.default_rng(cfg.seed)
    aug = {'hflip_swap_lr': cfg.aug.image.hflip_swap_lr, 'reorder_clauses_p': cfg.aug.text.reorder_clauses_p}
    order, first_1080 = hashlib.sha256(), None
    seen = np.zeros(len(fitted), np.int64)
    ids_at, rng_at, cursor_at, rows_at, epoch_at = {}, {}, {}, {}, {}
    epoch_starts = {}
    rows = step = 0
    steps_per_epoch = math.ceil(len(fitted) / cfg.train.batch_size)
    if 0 in state_steps:
        rng_at[0] = copy.deepcopy(rng.bit_generator.state)
    if achieved == 0:
        return {'order_sha256': hashlib.sha256().hexdigest(), 'first_1080_sha256': None, 'seen': seen,
                'ids_at': {}, 'rng_at': {}, 'cursor_at': {}, 'rows_at': {}, 'epoch_at': {},
                'epoch_starts': epoch_starts, 'rows': 0, 'steps_per_epoch': steps_per_epoch}
    for epoch in range(1, cfg.train.epochs + 1):
        epoch_starts[epoch] = copy.deepcopy(rng.bit_generator.state)
        cursor = 0
        for batch in train_batches(fitted, cfg.train.batch_size, rng, aug, cfg.train.pair_sampler):
            step += 1
            cursor += 1
            ids = fitted.ids[batch['rows']]
            order.update(ids.astype('<i8').tobytes())
            seen[batch['rows']] += 1
            rows += len(ids)
            if step in id_steps:
                ids_at[step] = ids.tolist()
            if step in state_steps:
                rng_at[step] = copy.deepcopy(rng.bit_generator.state)
                cursor_at[step], rows_at[step], epoch_at[step] = cursor, rows, epoch
            if step == 1080:
                first_1080 = order.hexdigest()
            if step == achieved:
                break
        if step == achieved:
            break
    assert step == achieved, f'replay reached {step} batches, run claims {achieved} steps'
    return {'order_sha256': order.hexdigest(), 'first_1080_sha256': first_1080, 'seen': seen,
            'ids_at': ids_at, 'rng_at': rng_at, 'cursor_at': cursor_at, 'rows_at': rows_at,
            'epoch_at': epoch_at, 'epoch_starts': epoch_starts, 'rows': rows,
            'steps_per_epoch': steps_per_epoch}


def replay_cursor(cfg, fitted, epoch_start_state, cursor):
    """Advance one epoch's generator from epoch_start_sampler_rng to the saved cursor."""
    rng = np.random.default_rng(cfg.seed)
    rng.bit_generator.state = copy.deepcopy(epoch_start_state)
    aug = {'hflip_swap_lr': cfg.aug.image.hflip_swap_lr, 'reorder_clauses_p': cfg.aug.text.reorder_clauses_p}
    index = 0
    for index, _ in enumerate(train_batches(fitted, cfg.train.batch_size, rng, aug, cfg.train.pair_sampler), 1):
        if index == cursor:
            break
    assert index == cursor, (index, cursor)
    return rng.bit_generator.state


def bn_group(state, prefix):
    mean = {k: v for k, v in state.items() if k.startswith(prefix) and k.endswith('running_mean')}
    var = {k: v for k, v in state.items() if k.startswith(prefix) and k.endswith('running_var')}
    counters = {k: int(v) for k, v in state.items() if k.startswith(prefix) and k.endswith('num_batches_tracked')}
    return mean, var, counters


def l2_total(tensors):
    return math.sqrt(sum(float(t.float().norm()) ** 2 for t in tensors))


def bn_record(state, previous):
    out = {}
    for group, prefix in (('cnn', 'image_encoder.'), ('fusion', 'fusion.')):
        mean, var, counters = bn_group(state, prefix)
        item = {'buffers': len(counters),
                'running_mean_l2': l2_total(mean.values()),
                'running_var_l2': l2_total(var.values()),
                'counters_min': min(counters.values()) if counters else None,
                'counters_max': max(counters.values()) if counters else None}
        if previous is not None:
            pmean, pvar, _ = bn_group(previous, prefix)
            item['delta_running_mean_l2'] = l2_total([v - pmean[k] for k, v in mean.items()])
            item['delta_running_var_l2'] = l2_total([v - pvar[k] for k, v in var.items()])
        out[group] = item
    return out


def compact(group):
    return {'rows': group['rows'], **group['metrics'],
            'bce': group['summary']['bce'],
            'pair_accuracy': group['summary']['pair_accuracy'],
            'margin_mean': group['summary']['signed_pair_margin']['mean'],
            'logit_std': group['summary']['logits']['std']}


def diff_paths(left, right, prefix=''):
    out = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f'{prefix}.{key}' if prefix else key
            if key not in left or key not in right:
                out.append(path)
            else:
                out.extend(diff_paths(left[key], right[key], path))
    elif left != right:
        out.append(prefix)
    return out


def load_baseline_receipt(baseline):
    multi = baseline.parent / 'verified_receipt.json'
    if multi.exists():
        runs = json.loads(multi.read_text()).get('runs', {})
        if baseline.name in runs:
            return runs[baseline.name]
    single = baseline.parent / f'{baseline.name}_verified.json'
    assert single.exists(), f'no baseline verified receipt next to {baseline}'
    return json.loads(single.read_text())


def replicate_reasons(cfg, checkpoints, observed, probe_events, achieved):
    """Replay the runner's snapshot trigger bookkeeping (probe best, alert norms)."""
    triggers = set(int(s) for s in cfg.train.snapshot_steps) | {int(cfg.train.max_steps)}
    alert = {s for s, ck in checkpoints.items()
             if ck['total_gradient_norm_before_clip'] is not None and ck['total_gradient_norm_before_clip'] > 100}
    fires = [(0, 'start')]
    for step in sorted((triggers | alert) - {0}):
        if step <= achieved:
            fires.append((step, 'loop'))
    if achieved not in {step for step, _ in fires}:
        fires.append((achieved, 'final'))
    by_step = {}
    for row in probe_events:
        by_step.setdefault(row['step'], []).append(row)
    best, expected = 0., {}
    for step, kind in fires:
        allowed = {'start': ('initial',), 'loop': ('step',), 'final': ('step', 'epoch_end')}[kind]
        pool = [row for row in by_step.get(step, []) if row['reason'] in allowed]
        probe_auc = pool[-1]['metrics']['auc'] if pool else None
        reasons = []
        total = checkpoints.get(step, {}).get('total_gradient_norm_before_clip')
        if total is not None and total > 100:
            reasons.append(REASONS[0])
        if probe_auc is not None and best - probe_auc >= .20:
            reasons.append(REASONS[1])
        update = observed.get(step)
        if update is not None and any(value['norm'] is not None and value['norm'] > 100
                                      for value in update['gradients_before_clip'].values()):
            reasons.append(REASONS[2])
        if probe_auc is not None:
            best = max(best, probe_auc)
        expected[step] = reasons
    return expected


def verify_run(run, baseline, full, levels_full, inputs_manifest):
    run, baseline = Path(run), Path(baseline)
    cfg = OmegaConf.load(run / 'config.yaml')
    base_cfg = OmegaConf.load(baseline / 'config.yaml')
    invocation = json.loads((run / 'invocation.json').read_text())
    result = json.loads((run / 'results.json').read_text())
    receipt_base = load_baseline_receipt(baseline)

    # Train-only scope.
    assert not cfg.data.train_on_val and not cfg.train.eval.holdout and not cfg.train.predict_test
    assert not cfg.train.eval.train and cfg.data.train_csv == 'data/train.csv'

    # Ids and vocabulary: identical to the baseline fit.
    np.testing.assert_array_equal(np.load(run / 'train_ids.npy'), full.ids)
    np.testing.assert_array_equal(np.load(run / 'train_ids.npy'), np.load(baseline / 'train_ids.npy'))
    fitted_ids = np.load(run / 'fitted_ids.npy')
    assert fitted_ids.shape == (FITTED_ROWS,)
    np.testing.assert_array_equal(fitted_ids, np.load(baseline / 'fitted_ids.npy'))
    fitted = subset(full, np.isin(full.ids, fitted_ids))
    np.testing.assert_array_equal(fitted.ids, fitted_ids)
    assert json.loads((run / 'vocab.json').read_text()) == json.loads((baseline / 'vocab.json').read_text())
    levels = levels_full[np.isin(full.ids, fitted_ids)]

    # Config: the same recipe; only schedule/budgets/length/provenance may differ.
    planned = list(PLANNED_STEPS)
    assert cfg.train.max_steps == PLANNED_STEPS[-1] == 4320
    assert cfg.train.scheduler.name == 'constant'
    assert cfg.train.scheduler.warmup_steps == 9
    assert cfg.train.scheduler.total_steps == 4320
    assert cfg.train.time_budget_min == 29
    assert cfg.train.batch_size == 32 and cfg.train.pair_sampler
    assert cfg.train.epochs == math.ceil(4320 / math.ceil(len(fitted) / cfg.train.batch_size))
    triggers = sorted(set(planned) | set(range(270, cfg.train.max_steps + 1, 270)))
    assert [int(s) for s in cfg.train.snapshot_steps] == triggers
    assert [int(s) for s in cfg.train.diagnostics.capture_steps] == triggers
    assert cfg.train.diagnostics.every_steps == 270
    assert cfg.train.get('anomaly_snapshot_gradient_norm') == 100.
    diffs = diff_paths(OmegaConf.to_container(base_cfg, resolve=True), OmegaConf.to_container(cfg, resolve=True))
    bad = [path for path in diffs
           if not any(path == allowed or path.startswith(allowed + '.') for allowed in ALLOWED_CONFIG_DIFFS)]
    assert not bad, ('config recipe drift versus baseline', bad)

    # Invocation matches the contract run.
    assert invocation['action'] == 'run' and invocation['start'] == 'initial'
    assert invocation['fitted_only'] and invocation['anomaly_snapshots']
    assert invocation['steps'] == 4320 and invocation['horizon'] == 4320 and invocation['warmup'] == 9
    assert invocation['schedule'] == 'constant'
    assert [int(s) for s in invocation['snapshots'].split(',')] == planned
    assert math.isclose(invocation['task_lr'], 3e-4, rel_tol=1e-12)
    assert Path(invocation['subset']).name == 'nested_4608.json'
    assert Path(invocation['reference']).name == 'medium_w9'
    assert Path(invocation['baseline_dir']).name == baseline.name

    # Provenance hashes must match the files as they are now (never git state).
    assert cfg.investigation.train_csv_sha256 == sha256(cfg.data.train_csv)
    sources = dict(cfg.investigation.source_sha256)
    assert sources
    for path, digest in sources.items():
        assert sha256(path) == digest, ('source file changed since the run', path)
    assert cfg.investigation.initial_state_sha256 == receipt_base['initial_state_sha256']
    assert cfg.investigation.start_state_sha256 == cfg.investigation.initial_state_sha256

    protected = 0
    if inputs_manifest is not None and Path(inputs_manifest).exists():
        manifest = json.loads(Path(inputs_manifest).read_text())
        for path, digest in manifest['sha256'].items():
            assert sha256(path) == digest, ('protected input changed', path)
        protected = len(manifest['sha256'])

    # Results shape and budgets.
    stop = result['stop_reason']
    assert stop in STOP_ALL
    clean = stop in STOP_OK
    achieved = result['optimizer_steps']
    assert result['fitted_rows'] == FITTED_ROWS
    assert math.isclose(result['epochs_equivalent'], result['processed_rows'] / FITTED_ROWS, rel_tol=1e-12)
    assert 0 < result['wall_sec'] <= 30 * 60, ('wall budget 30 min exceeded', result['wall_sec'])
    if stop == 'max_steps':
        assert achieved == 4320
    else:
        assert achieved < 4320, 'max_steps reached but stop_reason is not max_steps'
    failure = None
    if (run / 'failure.json').exists():
        failure = json.loads((run / 'failure.json').read_text())
        assert not clean, 'failure.json on a clean run'
        assert failure['reason'] == stop and failure['step'] == achieved, failure
    saved = [int(s) for s in result['snapshot_steps']]
    assert saved == sorted(set(saved))
    measurement_steps = [int(s) for s in result['measurement_steps']]
    assert set(measurement_steps) <= set(saved)
    if clean:
        assert measurement_steps == saved
    missing = [s for s in planned if s not in saved]
    if stop == 'max_steps':
        assert not missing, ('planned snapshots missing at max_steps', missing)
    else:
        assert missing == [s for s in planned if s > achieved], ('unexpected missing snapshots', missing)
        assert PLANNED_STEPS[-1] in missing, 'time_budget stop must not pretend 4320 was reached'
    if clean:
        assert achieved in saved, 'final snapshot missing'

    # Diagnostics trace and sampler replay up to the achieved steps.
    trace = [json.loads(line) for line in (run / 'diagnostics.jsonl').read_text().splitlines()]
    observed = {row['step']: row for row in trace if row.get('event') == 'optimizer_step'}
    probe_events = [row for row in trace if row.get('event') == 'probe']
    assert all(0 < s <= achieved for s in observed)
    history_steps = {int(row['step']) for row in result['history']}
    rep = replay(cfg, fitted, achieved, set(saved) | history_steps, set(observed))
    assert rep['rows'] == result['processed_rows']
    assert rep['steps_per_epoch'] == 144
    for row in result['history']:
        step = int(row['step'])
        assert row['processed_rows'] == rep['rows_at'].get(step, 0)
        for name in ('task', 'image_encoder'):
            assert math.isclose(row[f'lr/{name}'], expected_lr(cfg, name, step), rel_tol=1e-12, abs_tol=1e-15)
    complete_epochs = sum(row['completed_epoch'] for row in result['history'])

    for step, row in observed.items():
        assert row['ids'] == rep['ids_at'][step], (step, 'batch order mismatch')
        for name in ('task', 'image_encoder'):
            assert math.isclose(row['lr_used'][name], expected_lr(cfg, name, step - 1), rel_tol=1e-12, abs_tol=1e-15)
            assert math.isclose(row['lr_after'][name], expected_lr(cfg, name, step), rel_tol=1e-12, abs_tol=1e-15)
        if clean:
            assert all(value['finite'] for value in row['gradients_before_clip'].values())
            for name, bn in row['batchnorm'].items():
                assert bn['training'] == (not name.startswith('image_encoder')), (step, name)
                if name.startswith('fusion'):
                    assert bn['batches_tracked'] == row['step'], (step, name)
            if step in planned and step > 0:
                assert all(value['norm'] is not None and value['norm'] > 0
                           for value in row['gradients_before_clip'].values())
                assert all(value['norm'] is not None and value['norm'] > 0 for value in row['updates'].values())

    # The first 1080 batches must be the baseline's exact batch order.
    baseline_trace = [json.loads(line) for line in (baseline / 'diagnostics.jsonl').read_text().splitlines()]
    baseline_captures = {row['step']: row for row in baseline_trace
                         if row.get('event') == 'optimizer_step' and row['step'] <= achieved}
    for step, row in baseline_captures.items():
        assert rep['ids_at'][step] == row['ids'], (step, 'baseline capture order mismatch')
    if achieved >= 1080:
        assert rep['first_1080_sha256'] == receipt_base['batch_order_sha256'], 'first 1080 batches differ from baseline'

    # Fixed probes: ids and permutations must be constant and match the baseline.
    probe, image_order, caption_order = probe_orders(fitted, cfg.seed, PROBE_PAIRS)
    probe_ids = probe.ids.tolist()
    dp_probe, dp_image, dp_caption = probe_orders(fitted, cfg.seed, 2)
    diagnostic_probe = json.loads((run / 'diagnostic_probe.json').read_text())
    baseline_probe = json.loads((baseline / 'measurement_step0000.json').read_text())['probes']['fitted']
    assert baseline_probe['ids'] == probe_ids
    assert baseline_probe['image_order'] == image_order.tolist()
    assert baseline_probe['caption_order'] == caption_order.tolist()
    assert diagnostic_probe['ids'] == dp_probe.ids.tolist()
    assert diagnostic_probe['image_order'] == dp_image.tolist()
    assert diagnostic_probe['caption_order'] == dp_caption.tolist()
    assert diagnostic_probe['initial_state_sha256'] == cfg.investigation.initial_state_sha256
    assert diagnostic_probe['fitted_rows'] == FITTED_ROWS
    baseline_dp = json.loads((baseline / 'diagnostic_probe.json').read_text())
    assert diagnostic_probe['ids'] == baseline_dp['ids']
    assert diagnostic_probe['image_order'] == baseline_dp['image_order']
    assert diagnostic_probe['caption_order'] == baseline_dp['caption_order']

    # Checkpoints: state, RNG, sampler cursors, optimizer/scheduler counters, LR, BN.
    checkpoints, snapshot_rows = {}, []
    initial_bn_state = None
    previous_state = None
    gradient_probe_ids = None
    for step in saved:
        ck = torch.load(run / f'checkpoint_step{step:04d}.pt', map_location='cpu', weights_only=True)
        checkpoints[step] = ck
        required = {'model', 'optimizer', 'scheduler', 'step', 'epoch', 'sampler_rng',
                    'epoch_start_sampler_rng', 'batches_completed_in_epoch', 'processed_rows',
                    'stop_reason', 'torch_rng', 'python_rng', 'numpy_rng', 'mps_rng', 'cuda_rng',
                    'scaler', 'optimizer_observation', 'diagnostic_reasons', 'lr_next',
                    'total_gradient_norm_before_clip'}
        assert required <= set(ck), (step, sorted(required - set(ck)))
        assert ck['step'] == step
        if str(cfg.investigation.device) != 'cuda':
            assert ck['scaler'] is None and ck['cuda_rng'] is None, (step, 'scaler/cuda_rng on non-cuda device')
        epoch = 0 if step == 0 else math.ceil(step / rep['steps_per_epoch'])
        if clean or step == 0:
            # A corrupt-update failure can leave cursor/rng one batch past the stopped
            # step; only intact runs prove exact replay consistency per checkpoint.
            assert ck['epoch'] == epoch
            assert ck['batches_completed_in_epoch'] == (0 if step == 0 else step - (epoch - 1) * rep['steps_per_epoch'])
            assert ck['processed_rows'] == (0 if step == 0 else rep['rows_at'][step])
            assert ck['sampler_rng'] == rep['rng_at'][step]
            if step == 0:
                assert ck['epoch_start_sampler_rng'] is None
            else:
                assert ck['epoch_start_sampler_rng'] == rep['epoch_starts'][epoch]
                assert ck['batches_completed_in_epoch'] == rep['cursor_at'][step]
                # The saved cursor must land exactly on sampler_rng, one batch short of it never does.
                assert replay_cursor(cfg, fitted, ck['epoch_start_sampler_rng'],
                                     ck['batches_completed_in_epoch']) == ck['sampler_rng']
        assert ck['scheduler']['last_epoch'] == step
        states = ck['optimizer']['state']
        if step == 0:
            assert not states, 'step 0 must start with fresh AdamW moments'
            counters = (None, None)
        else:
            counters = (min(int(v['step']) for v in states.values()),
                        max(int(v['step']) for v in states.values()))
            assert counters == (step, step), ('optimizer step counters', step, counters)
        lrs = {group['name']: group['lr'] for group in ck['optimizer']['param_groups']}
        assert set(lrs) == {'task', 'image_encoder'}
        for name, lr in lrs.items():
            assert math.isclose(lr, expected_lr(cfg, name, step), rel_tol=1e-12, abs_tol=1e-15)
        assert set(ck['lr_next']) == {'task', 'image_encoder'}
        for name, lr in ck['lr_next'].items():
            assert math.isclose(lr, expected_lr(cfg, name, step), rel_tol=1e-12, abs_tol=1e-15)
        observation = ck['optimizer_observation']
        if step == 0:
            assert observation is None
        elif step in observed:
            assert_close_tree(observed[step], observation, f'checkpoint[{step}].optimizer_observation',
                              abs_tol=1e-12, rel_tol=1e-12)
        else:
            assert observation is None, (step, 'observation without a captured trace row')
        if clean and step in planned and step > 0:
            assert step in observed and observation is not None, ('planned snapshot lacks exact trace row', step)
            assert_close_tree(ck['lr_next'], observation['lr_after'], f'checkpoint[{step}].lr_next',
                              abs_tol=1e-15, rel_tol=1e-12)
        if clean:
            assert ck['stop_reason'] == ('epochs' if step != achieved else stop), (step, ck['stop_reason'])
        else:
            assert ck['stop_reason'] in STOP_ALL

        # CNN BatchNorm buffers are frozen (eval-mode backbone); fusion counts training batches.
        state = ck['model']
        if step == 0:
            initial_bn_state = state
            assert state_hash(state) == cfg.investigation.initial_state_sha256
        cnn_mean, cnn_var, cnn_counters = bn_group(state, 'image_encoder.')
        for key, tensor in {**cnn_mean, **cnn_var}.items():
            assert torch.equal(tensor, initial_bn_state[key]), (step, key, 'CNN BN buffer changed')
        for key, value in cnn_counters.items():
            assert value == int(initial_bn_state[key]), (step, key, 'CNN BN counter changed')
        bn = bn_record(state, previous_state)
        previous_state = state
        if clean:
            assert bn['fusion']['counters_min'] == bn['fusion']['counters_max'] == step, \
                (step, 'fusion batches_tracked != optimizer steps', bn['fusion'])

        # Gradient probe: fixed 32 fitted rows of the same probe, eval-mode read-only grads.
        gradient_probe = ck.get('gradient_probe')
        if clean:
            assert gradient_probe is not None and set(gradient_probe) == {'ids', 'mode', 'gradients'}
            assert gradient_probe['mode'] == 'eval/no_dropout; read-only autograd.grad'
            assert gradient_probe['ids'] == probe_ids[:PROBE_ROWS]
            assert len(set(gradient_probe['ids'])) == PROBE_ROWS
            assert set(gradient_probe['ids']) <= set(fitted_ids.tolist())
            if gradient_probe_ids is None:
                gradient_probe_ids = gradient_probe['ids']
            assert gradient_probe['ids'] == gradient_probe_ids, (step, 'gradient probe rows moved')
            assert set(gradient_probe['gradients']) == {'text_encoder', 'fusion', 'image_encoder'}
            for name, value in gradient_probe['gradients'].items():
                assert set(value) == {'norm', 'rms', 'finite'}, (step, name)
                assert value['finite'] and math.isfinite(value['norm']) and value['norm'] >= 0
                assert 0 <= value['rms'] <= value['norm'], (step, name, value)
                if step in planned and step > 0:
                    assert value['norm'] > 0 and value['rms'] > 0, (step, name, 'zero probe gradient')
        elif gradient_probe is not None:
            assert set(gradient_probe) == {'ids', 'mode', 'gradients'}

        total_norm = ck['total_gradient_norm_before_clip']
        assert total_norm is None or isinstance(total_norm, float)
        if step == 0:
            assert total_norm is None
        elif clean:
            assert math.isfinite(total_norm)
            if step in planned and step > 0:
                assert total_norm > 0, (step, 'zero total preclip gradient norm')

        # Saved logits and measurements: metrics recomputed, including category/level.
        measurement_path = run / f'measurement_step{step:04d}.json'
        logits_path = run / f'logits_step{step:04d}.npy'
        if measurement_path.exists():
            measurement = json.loads(measurement_path.read_text())
            logits = np.load(logits_path)
            assert measurement['step'] == step
            assert set(measurement['groups']) == {'fitted'} and set(measurement['probes']) == {'fitted'}
            assert logits.shape == (FITTED_ROWS,) and np.isfinite(logits).all()
            assert_close_tree(quality(logits, fitted, levels), measurement['groups']['fitted'],
                              f'measurement[{step}].groups.fitted')
            probe_row = measurement['probes']['fitted']
            assert probe_row['ids'] == probe_ids
            assert probe_row['image_order'] == image_order.tolist()
            assert probe_row['caption_order'] == caption_order.tolist()
            for key in ('preactivation', 'activation', 'logits', 'image_logits', 'caption_logits',
                        'image_delta', 'caption_delta', 'summary'):
                assert key in probe_row, (step, key)
            plogits = np.asarray(probe_row['logits'], dtype=np.float32)
            ilogits = np.asarray(probe_row['image_logits'], dtype=np.float32)
            clogits = np.asarray(probe_row['caption_logits'], dtype=np.float32)
            assert_close_tree(prediction_summary(plogits, probe.labels, probe.pair_ids), probe_row['summary'],
                              f'measurement[{step}].probes.fitted.summary')
            assert math.isclose(float(np.abs(plogits - ilogits).mean()), probe_row['image_delta'],
                                rel_tol=1e-5, abs_tol=1e-7)
            assert math.isclose(float(np.abs(plogits - clogits).mean()), probe_row['caption_delta'],
                                rel_tol=1e-5, abs_tol=1e-7)
            positions = {row_id: i for i, row_id in enumerate(fitted_ids)}
            expected_logits = logits[[positions[row_id] for row_id in probe_row['ids']]]
            np.testing.assert_allclose(plogits, expected_logits, rtol=1e-4, atol=2e-4)
            metrics = compact(measurement['groups']['fitted'])
        else:
            assert not clean, ('missing measurement at saved snapshot', step)
            measurement, logits, metrics = None, None, None

        snapshot_rows.append({
            'step': step, 'status': 'saved', 'epoch': ck['epoch'], 'processed_rows': ck['processed_rows'],
            'stop_reason': ck['stop_reason'], 'scheduler_last_epoch': ck['scheduler']['last_epoch'],
            'optimizer_step_counters': {'min': counters[0], 'max': counters[1]},
            'lr_next': safe(ck['lr_next']), 'total_gradient_norm_before_clip': safe(total_norm),
            'optimizer_observation': 'exact_trace_row' if observation is not None else None,
            'diagnostic_reasons': list(ck['diagnostic_reasons']),
            'bn': bn,
            'gradients_updates_trace': ({'lr_used': observation['lr_used'],
                                         'gradients_before_clip': observation['gradients_before_clip'],
                                         'updates': observation['updates']} if observation is not None else None),
            'gradient_probe': safe(gradient_probe),
            'metrics': safe(metrics),
        })
    for step in missing:
        snapshot_rows.append({'step': step, 'status': 'missing', 'reason': stop if not clean else 'time_budget'})
    snapshot_rows.sort(key=lambda row: row['step'])

    # Trigger bookkeeping: each kept anomaly snapshot documents a real trigger.
    expected_reasons = replicate_reasons(cfg, checkpoints, observed, probe_events, achieved)
    for step, ck in checkpoints.items():
        if clean:
            assert list(ck['diagnostic_reasons']) == expected_reasons.get(step, []), \
                (step, ck['diagnostic_reasons'], expected_reasons.get(step))
    if clean:
        for step, reasons in expected_reasons.items():
            assert step in checkpoints or reasons == [], ('unsaved fire with trigger reasons', step, reasons)

    # Final weights must be the final checkpoint's weights.
    model_state = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    final_hash = state_hash(checkpoints[saved[-1]]['model'])
    if clean:
        assert saved[-1] == achieved
        assert state_hash(model_state) == final_hash, 'model.pt differs from the final checkpoint'

    # Baseline step-0 capture: identical initialization and RNG streams.
    baseline0 = torch.load(baseline / 'checkpoint_step0000.pt', map_location='cpu', weights_only=True)
    zero = checkpoints[0]
    assert state_hash(zero['model']) == state_hash(baseline0['model'])
    assert torch.equal(zero['torch_rng'], baseline0['torch_rng'])
    assert zero['python_rng'] == baseline0['python_rng']
    assert zero['numpy_rng'] == baseline0['numpy_rng']
    assert zero['sampler_rng'] == baseline0['sampler_rng']
    assert zero['optimizer']['param_groups'] == baseline0['optimizer']['param_groups']
    assert zero['scheduler'] == baseline0['scheduler']
    if zero['mps_rng'] is None:
        assert baseline0['mps_rng'] is None
    else:
        assert torch.equal(zero['mps_rng'], baseline0['mps_rng'])
    assert (zero['cuda_rng'] is None) == (baseline0.get('cuda_rng') is None)

    # Baseline quality curve from the 4 saved measurements; no old fit is re-run.
    baseline_curve = {}
    for step in (0, 270, 540, 1080):
        measurement = json.loads((baseline / f'measurement_step{step:04d}.json').read_text())
        baseline_curve[step] = safe(compact(measurement['groups']['fitted']))
    quality_curve = {row['step']: row['metrics'] for row in snapshot_rows if row.get('metrics')}

    # Fitted/logit comparison at shared steps, purely from saved artefacts.
    full_positions = {row_id: i for i, row_id in enumerate(full.ids)}
    comparison = {}
    for step in sorted(set(quality_curve) & set(baseline_curve)):
        run_row, base_row = quality_curve[step], baseline_curve[step]
        deltas = {key: run_row[key] - base_row[key] for key in run_row
                  if key != 'rows' and isinstance(run_row[key], (int, float))
                  and isinstance(base_row.get(key), (int, float))}
        run_logits = np.load(run / f'logits_step{step:04d}.npy')
        base_logits = np.load(baseline / f'logits_step{step:04d}.npy')
        selected = base_logits[[full_positions[row_id] for row_id in fitted_ids]]
        close = bool(np.allclose(run_logits, selected, rtol=1e-4, atol=2e-4))
        entry = {'run': run_row, 'baseline': base_row, 'metric_deltas': safe(deltas),
                 'logits_max_abs_diff': float(np.abs(run_logits - selected).max()),
                 'logits_allclose_rtol1e-4_atol2e-4': close}
        if step == 0:
            assert close, 'step-0 logits disagree with the baseline initial forward (misalignment?)'
        comparison[step] = entry

    status = 'verified_' + stop if clean else 'failed_' + stop
    return {
        'data_scope': 'train only', 'run': str(run), 'baseline': str(baseline), 'status': status,
        'verified': bool(clean), 'stop_reason': stop, 'failure': failure,
        'steps': achieved, 'planned_steps': PLANNED_STEPS[-1],
        'processed_rows': result['processed_rows'], 'epochs_equivalent': result['epochs_equivalent'],
        'complete_epochs': complete_epochs, 'fit_min': result['fit_min'], 'wall_sec': result['wall_sec'],
        'planned_snapshot_steps': planned, 'saved_snapshot_steps': saved,
        'missing_snapshots': {str(s): (stop if not clean else 'time_budget') for s in missing},
        'batch_order_sha256': rep['order_sha256'],
        'first_1080_batch_order_sha256': rep['first_1080_sha256'],
        'baseline_batch_order_sha256': receipt_base['batch_order_sha256'],
        'baseline_captured_batches_matched': len(baseline_captures),
        'captured_optimizer_batches_verified': len(observed),
        'exposures_per_row': {'min': int(rep['seen'].min()), 'max': int(rep['seen'].max()),
                              'mean': float(rep['seen'].mean()), 'unseen_rows': int((rep['seen'] == 0).sum())},
        'initial_state_sha256': cfg.investigation.initial_state_sha256,
        'start_state_sha256': cfg.investigation.start_state_sha256,
        'final_state_sha256': final_hash,
        'gradient_probe_ids': gradient_probe_ids,
        'snapshots': snapshot_rows,
        'baseline_quality_curve': baseline_curve,
        'quality_curve': safe(quality_curve),
        'baseline_comparison': safe(comparison),
        'config_diff_paths': diffs,
        'source_sha256_verified': len(sources),
        'protected_inputs_verified': protected,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True, help='new run directory (e.g. artefacts/film_long_training)')
    parser.add_argument('--baseline', type=Path, default=Path('artefacts/film_scaling/scaling_4608'))
    parser.add_argument('--output', type=Path, help='receipt json (default: RUN/verified_receipt.json)')
    parser.add_argument('--inputs', type=Path, default=Path('artefacts/film_long_training_inputs.json'))
    args = parser.parse_args()
    full = load_split('train', 'data/train.csv')
    levels = pl.read_csv('data/train.csv')['level'].to_numpy()
    receipt = verify_run(args.run, args.baseline, full, levels, args.inputs)
    output = args.output or args.run / 'verified_receipt.json'
    dump(output, safe(receipt))
    print(json.dumps({'output': str(output), 'status': receipt['status'], 'steps': receipt['steps'],
                      'saved_snapshots': receipt['saved_snapshot_steps'],
                      'protected_inputs_verified': receipt['protected_inputs_verified']}), flush=True)
    assert receipt['verified'], receipt['status']


if __name__ == '__main__':
    main()
