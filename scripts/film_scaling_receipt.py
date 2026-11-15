"""Verify train-only scaling artefacts and summarize measurements without fitting."""
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
import torch

from film_scaling_run import dump, state_hash
from src.data import load_split, subset, train_batches
from src.diagnostics import prediction_summary
from src.metrics import evaluate_scores


def summary(measurement):
    result = {'step': measurement['step'], 'groups': {}, 'probes': {}}
    for name, row in measurement['groups'].items():
        result['groups'][name] = {'rows': row['rows'], **row['metrics'],
                                 'bce': row['summary']['bce'],
                                 'pair_accuracy': row['summary']['pair_accuracy'],
                                 'margin_mean': row['summary']['signed_pair_margin']['mean']}
    for name, row in measurement['probes'].items():
        scale = row['summary']['logits']['std']
        result['probes'][name] = {'image_delta': row['image_delta'], 'caption_delta': row['caption_delta'],
                                 'logit_std': scale,
                                 'image_delta_over_logit_std': row['image_delta'] / scale if scale else None,
                                 'caption_delta_over_logit_std': row['caption_delta'] / scale if scale else None,
                                 'readout_negative_fraction': row['preactivation']['negative_fraction'],
                                 'readout_zero_fraction': row['activation']['zero_fraction'],
                                 'readout_all_zero_units': row['activation']['all_zero_units']}
    return result


def verify_run(path, full):
    cfg = OmegaConf.load(path / 'config.yaml')
    result = json.loads((path / 'results.json').read_text())
    assert not cfg.data.train_on_val and not cfg.train.eval.holdout and not cfg.train.predict_test
    assert result['stop_reason'] == 'max_steps'
    assert result['optimizer_steps'] == cfg.train.max_steps
    np.testing.assert_array_equal(np.load(path / 'train_ids.npy'), full.ids)
    fitted_ids = np.load(path / 'fitted_ids.npy')
    fitted = subset(full, np.isin(full.ids, fitted_ids))
    np.testing.assert_array_equal(fitted.ids, fitted_ids)
    assert result['fitted_rows'] == len(fitted)
    wanted = set(result['snapshot_steps'])
    trace = [json.loads(line) for line in (path / 'diagnostics.jsonl').read_text().splitlines()]
    observed = {row['step']: row for row in trace if row['event'] == 'optimizer_step'}
    rng = np.random.default_rng(cfg.seed)
    order_hash, seen, rng_states = hashlib.sha256(), np.zeros(len(fitted), dtype=np.int64), {0: copy.deepcopy(rng.bit_generator.state)}
    rows, step = 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        for batch in train_batches(fitted, cfg.train.batch_size, rng,
                                   {'hflip_swap_lr': 0., 'reorder_clauses_p': 0.}, cfg.train.pair_sampler):
            step += 1
            ids = fitted.ids[batch['rows']]
            order_hash.update(ids.astype('<i8').tobytes())
            seen[batch['rows']] += 1
            rows += len(ids)
            if step in wanted:
                rng_states[step] = copy.deepcopy(rng.bit_generator.state)
            if step in observed:
                row = observed[step]
                assert row['ids'] == ids.tolist(), (path, step, 'batch order mismatch')
                absolute = step - 1
                warmup = cfg.train.scheduler.warmup_steps
                if absolute < warmup:
                    factor = (absolute + 1) / warmup
                elif cfg.train.scheduler.name == 'cosine':
                    factor = .5 * (1 + math.cos(math.pi * (absolute - warmup) /
                                               max(1, cfg.train.scheduler.total_steps - warmup)))
                else:
                    factor = 1.
                for group, base in [('task', cfg.train.optimizer.lr), ('image_encoder', cfg.train.optimizer.image_encoder_lr)]:
                    assert math.isclose(row['lr_used'][group], base * factor, rel_tol=1e-12, abs_tol=1e-15)
                assert all(value['finite'] for value in row['gradients_before_clip'].values())
                assert all(not value['training'] if name.startswith('image_encoder') else value['training']
                           for name, value in row['batchnorm'].items())
            if step == cfg.train.max_steps:
                break
        if step == cfg.train.max_steps:
            break
    assert rows == result['processed_rows']
    snapshots = []
    final_state_hash = None
    for step in result['snapshot_steps']:
        saved = torch.load(path / f'checkpoint_step{step:04d}.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == step
        assert saved['sampler_rng'] == rng_states[step]
        if step == 0:
            assert state_hash(saved['model']) == cfg.investigation.start_state_sha256
            assert not saved['optimizer']['state'], 'restart must use fresh AdamW moments'
        if step == cfg.train.max_steps:
            final_state_hash = state_hash(saved['model'])
            assert all(int(value['step']) == step for value in saved['optimizer']['state'].values())
        measurement = json.loads((path / f'measurement_step{step:04d}.json').read_text())
        logits = np.load(path / f'logits_step{step:04d}.npy')
        assert np.isfinite(logits).all() and logits.shape == (len(full),)
        computed = prediction_summary(logits, full.labels, full.pair_ids)
        metrics = evaluate_scores(torch.from_numpy(logits).sigmoid().numpy(), full.labels, full.categories, full.pair_ids)
        assert math.isclose(computed['bce'], measurement['groups']['full']['summary']['bce'], abs_tol=1e-7)
        assert math.isclose(metrics['auc'], measurement['groups']['full']['metrics']['auc'], abs_tol=1e-12)
        assert math.isclose(computed['pair_accuracy'], measurement['groups']['full']['summary']['pair_accuracy'], abs_tol=1e-7)
        snapshots.append(summary(measurement))
    assert state_hash(torch.load(path / 'model.pt', map_location='cpu', weights_only=True)) == final_state_hash
    exposure_quality = {}
    if len(fitted) == len(full):
        original_ids = np.load('artefacts/film_collapse/medium_w9/train_fit_ids.npy')
        original_mask = np.isin(full.ids, original_ids)
        for domain, domain_mask in (('original', original_mask), ('rest', ~original_mask)):
            for exposure, exposure_mask in (('seen_this_run', seen > 0), ('unseen_this_run', seen == 0)):
                mask = domain_mask & exposure_mask
                if not mask.any():
                    continue
                values = prediction_summary(logits[mask], full.labels[mask], full.pair_ids[mask])
                auc = evaluate_scores(torch.from_numpy(logits[mask]).sigmoid().numpy(), full.labels[mask],
                                      full.categories[mask], full.pair_ids[mask])['auc']
                exposure_quality[f'{domain}_{exposure}'] = {
                    'rows': int(mask.sum()), 'auc': auc, 'bce': values['bce'],
                    'pair_accuracy': values['pair_accuracy'],
                    'margin_mean': values['signed_pair_margin']['mean']}
    return {'path': str(path), 'initial_state_sha256': cfg.investigation.initial_state_sha256,
            'start_state_sha256': cfg.investigation.start_state_sha256,
            'steps': result['optimizer_steps'], 'rows': len(fitted), 'processed_rows': rows,
            'epochs_equivalent': result['epochs_equivalent'], 'fit_min': result['fit_min'],
            'complete_epochs': sum(row['completed_epoch'] for row in result['history']),
            'batch_order_sha256': order_hash.hexdigest(), 'captured_optimizer_batches_verified': len(observed),
            'exposures_per_row': {'min': int(seen.min()), 'max': int(seen.max()), 'mean': float(seen.mean()),
                                  'unseen_rows': int((seen == 0).sum())},
            'snapshots': snapshots,
            'final_quality_by_exposure': exposure_quality,
            'selected_gradients_updates': {str(step): {key: row[key] for key in
                                            ('lr_used', 'gradients_before_clip', 'updates')}
                                          for step, row in observed.items() if step in wanted}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('artefacts/film_scaling'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    full = load_split('train', 'data/train.csv')
    result = {'data_scope': 'train only', 'runs': {}}
    for name in ('scaling_4608', 'scaling_9216', 'warm_original', 'warm_full', 'initial_full'):
        path = args.root / name
        if not path.exists():
            continue
        result['runs'][name] = verify_run(path, full)
        print(name, json.dumps(result['runs'][name]['snapshots'][-1]['groups']), flush=True)
    a, b, c = (args.root / name for name in ('warm_original', 'warm_full', 'initial_full'))
    if all(path.exists() for path in (a, b, c)):
        zero = [torch.load(path / 'checkpoint_step0000.pt', map_location='cpu', weights_only=True) for path in (a, b, c)]
        assert state_hash(zero[0]['model']) == state_hash(zero[1]['model'])
        assert state_hash(zero[2]['model']) == result['runs']['initial_full']['initial_state_sha256']
        assert zero[0]['scheduler'] == zero[1]['scheduler'] == zero[2]['scheduler']
        for checkpoint in zero[1:]:
            assert torch.equal(zero[0]['torch_rng'], checkpoint['torch_rng'])
            if zero[0]['mps_rng'] is None:
                assert checkpoint['mps_rng'] is None
            else:
                assert torch.equal(zero[0]['mps_rng'], checkpoint['mps_rng'])
            assert zero[0]['sampler_rng'] == checkpoint['sampler_rng']
            assert zero[0]['optimizer']['param_groups'] == checkpoint['optimizer']['param_groups']
        assert result['runs']['warm_full']['batch_order_sha256'] == result['runs']['initial_full']['batch_order_sha256']
        result['warm_start_controls_verified'] = True
    dump(args.output or args.root / 'verified_receipt.json', result)


if __name__ == '__main__':
    main()
