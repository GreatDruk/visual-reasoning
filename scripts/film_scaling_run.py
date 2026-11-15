"""Train-only FiLM scaling experiments using the existing Hydra recipe and Trainer.

All outputs are exclusive new directories. No validation/test split is loaded.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
import polars as pl
import torch

from src.data import load_split, pair_groups, sample_pairs_by_category, subset
from src.diagnostics import prediction_summary
from src.metrics import evaluate_scores
from src.models.model import VisualReasoningModel
from src.models.text_encoders import Vocab
from src.train import _attach_sources, instantiate
from src.training import Trainer
from src.utils import get_device, set_seed

REFERENCE = Path('artefacts/film_collapse/medium_w9')


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@contextmanager
def observation(trainer):
    """Keep evaluation and gradient measurements outside the training RNG stream."""
    modes = {module: module.training for module in trainer.model.modules()}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if trainer.device.type == 'mps' else None
    cuda_state = torch.cuda.get_rng_state_all() if trainer.device.type == 'cuda' else None
    try:
        yield
    finally:
        for module, mode in modes.items():
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def build(reference, device):
    cfg = OmegaConf.load(reference / 'config.yaml')
    # These paths remain provenance, never inputs to this investigation.
    assert not cfg.data.train_on_val and not cfg.train.eval.holdout and not cfg.train.predict_test
    set_seed(cfg.seed)
    full = load_split('train', cfg.data.train_csv)
    original = sample_pairs_by_category(full, 64, cfg.seed)
    np.testing.assert_array_equal(original.ids, np.load(REFERENCE / 'train_fit_ids.npy'))
    encoder, channels, mode = _attach_sources(cfg, {'train': full}, device, False)
    saved = json.loads((reference / 'vocab.json').read_text())
    vocab = Vocab([], lowercase=saved['lowercase'])
    vocab.itos = saved['itos']
    vocab.stoi = {word: i for i, word in enumerate(vocab.itos)}
    text = instantiate(cfg.model.text_encoder, vocab_size=len(vocab))
    fusion = instantiate(cfg.model.fusion, in_channels=channels, text_dim=text.out_dim, _convert_='all')
    model = VisualReasoningModel(text, fusion, encoder, spatial=True)
    manifest = json.loads((reference / 'diagnostic_probe.json').read_text())
    initial_hash = state_hash(model.state_dict())
    assert initial_hash == manifest['initial_state_sha256'], 'initial weights differ from reference'
    # Subsets now share the attached full image source and original image indices.
    original = subset(full, np.isin(full.ids, original.ids))
    rest = subset(full, ~np.isin(full.ids, original.ids))
    trainer = Trainer(model, instantiate(cfg.loss), vocab, cfg.train, device, mode,
                      cfg.data.image_size, cfg.data.normalize, cfg.seed)
    return cfg, full, original, rest, trainer, initial_hash


def quality(logits, split, levels):
    scores = torch.from_numpy(logits).sigmoid().numpy()
    result: dict = {'rows': len(split), 'metrics': evaluate_scores(scores, split.labels, split.categories, split.pair_ids),
              'summary': prediction_summary(logits, split.labels, split.pair_ids)}
    for axis, values in (('category', split.categories), ('level', levels)):
        result[axis] = {}
        for value in sorted(set(values.tolist())):
            mask = values == value
            result[axis][str(value)] = {
                'rows': int(mask.sum()),
                'metrics': evaluate_scores(scores[mask], split.labels[mask], split.categories[mask], split.pair_ids[mask]),
                'summary': prediction_summary(logits[mask], split.labels[mask], split.pair_ids[mask])}
    return result


def sensitivity(trainer, split):
    # Fixed stratified 288-row probes; every group is training data, not holdout.
    probe = sample_pairs_by_category(split, 16, trainer.seed)
    indices = np.arange(len(probe))
    order = next(np.roll(indices, shift) for shift in range(1, len(probe))
                 if np.all(probe.img_idx[np.roll(indices, shift)] != probe.img_idx))
    caption_order = indices.copy()
    groups = pair_groups(probe)
    caption_order[groups[:, 0]], caption_order[groups[:, 1]] = groups[:, 1], groups[:, 0]
    activations = {}
    def capture(name):
        def hook(module, args, output):
            # Clone before the following in-place ReLU can modify this output.
            activations.setdefault(name, []).append(output.detach().cpu().clone())
        return hook
    handles = [trainer.model.fusion.readout.mlp[i].register_forward_hook(capture(name))
               for i, name in ((0, 'preactivation'), (1, 'activation'))]
    try:
        logits = trainer.predict(probe)
    finally:
        for handle in handles:
            handle.remove()
    image_logits = trainer.predict(replace(probe, img_idx=probe.img_idx[order]))
    text_logits = trainer.predict(replace(probe, captions=[probe.captions[i] for i in caption_order]))
    result = {'ids': probe.ids.tolist(), 'image_order': order.tolist(), 'caption_order': caption_order.tolist(),
              'logits': logits.tolist(), 'image_logits': image_logits.tolist(), 'caption_logits': text_logits.tolist(),
              'image_delta': float(np.abs(logits - image_logits).mean()),
              'caption_delta': float(np.abs(logits - text_logits).mean()),
              'summary': prediction_summary(logits, probe.labels, probe.pair_ids)}
    for name, parts in activations.items():
        x = torch.cat(parts)
        result[name] = {'mean': float(x.mean()), 'std': float(x.std(unbiased=False)),
                        'negative_fraction': float((x < 0).float().mean()),
                        'zero_fraction': float((x == 0).float().mean()),
                        'all_zero_units': int((x == 0).all(0).sum()),
                        'p05_p50_p95': torch.quantile(x.flatten(), torch.tensor([.05, .5, .95])).tolist()}
    return result


def measure(trainer, full, original, rest, fitted, output, step, cached_logits=None):
    started = time.monotonic()
    with observation(trainer):
        logits = trainer.predict(full) if cached_logits is None else np.load(cached_logits)
        assert logits.shape == (len(full),)
        assert np.isfinite(logits).all()
        np.save(output / f'logits_step{step:04d}.npy', logits)
        levels = pl.read_csv('data/train.csv')['level'].to_numpy()
        result = {'step': step, 'groups': {}}
        for name, split in (('full', full), ('original', original), ('rest', rest), ('fitted', fitted)):
            mask = np.isin(full.ids, split.ids)
            result['groups'][name] = quality(logits[mask], split, levels[mask])
        result['probes'] = {name: sensitivity(trainer, split) for name, split in
                            (('original', original), ('rest', rest), ('fitted', fitted))}
        if cached_logits is not None:
            positions = {row_id: i for i, row_id in enumerate(full.ids)}
            probe = result['probes']['original']
            expected = logits[[positions[row_id] for row_id in probe['ids']]]
            np.testing.assert_allclose(probe['logits'], expected, rtol=1e-4, atol=2e-4)
            result['reused_initial_logits'] = str(cached_logits)
    result['measurement_sec'] = time.monotonic() - started
    dump(output / f'measurement_step{step:04d}.json', result)
    print(json.dumps({'step': step, 'sec': result['measurement_sec'],
                     **{name: {'auc': row['metrics']['auc'], 'bce': row['summary']['bce'],
                               'pair_accuracy': row['summary']['pair_accuracy'],
                               'margin': row['summary']['signed_pair_margin']['mean']}
                        for name, row in result['groups'].items()},
                     'image_delta': {name: row['image_delta'] for name, row in result['probes'].items()}},
                    allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('evaluate', 'run'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--subset', type=Path)
    parser.add_argument('--baseline-dir', type=Path, help='Reuse step0 logits only after exact state/ids checks')
    parser.add_argument('--start', choices=('initial', 'memorized', 'reference'), default='initial')
    parser.add_argument('--steps', type=int, default=1080)
    parser.add_argument('--snapshots', default='0,270,540,1080')
    parser.add_argument('--schedule', choices=('cosine', 'constant'), default='cosine')
    parser.add_argument('--horizon', type=int, default=1080)
    parser.add_argument('--warmup', type=int, default=9)
    parser.add_argument('--task-lr', type=float, default=3e-4)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'invocation.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    device = get_device(args.device)
    cfg, full, original, rest, trainer, initial_hash = build(args.reference, device)
    if args.subset:
        manifest = json.loads(args.subset.read_text())
        fitted = subset(full, np.isin(full.ids, manifest['ids']))
        assert len(fitted) == manifest['actual_rows']
    else:
        fitted = full
    if args.start != 'initial':
        path = REFERENCE if args.start == 'memorized' else args.reference
        trainer.model.load_state_dict(torch.load(path / 'model.pt', map_location=device, weights_only=True))
    start_hash = state_hash(trainer.model.state_dict())
    np.save(args.output / 'train_ids.npy', full.ids)
    np.save(args.output / 'fitted_ids.npy', fitted.ids)
    dump(args.output / 'vocab.json', {'itos': trainer.vocab.itos, 'lowercase': trainer.vocab.lowercase})
    cfg.data.train_pairs_per_category = None
    cfg.name_exp = f'debug_scaling_{args.output.name}'
    cfg.paths.output_dir = str(args.output)
    cfg.train.early_stopping = None
    cfg.train.eval.train = False
    cfg.train.epochs = math.ceil(args.steps / math.ceil(len(fitted) / cfg.train.batch_size))
    cfg.train.time_budget_min = 29
    cfg.train.max_steps = args.steps
    cfg.train.snapshot_steps = [int(s) for s in args.snapshots.split(',')]
    cfg.train.scheduler.name = args.schedule
    cfg.train.scheduler.warmup_steps = args.warmup
    cfg.train.scheduler.total_steps = args.horizon
    cfg.train.optimizer.lr = args.task_lr
    cfg.train.optimizer.image_encoder_lr = args.task_lr / 10
    cfg.train.diagnostics.every_steps = 50 if args.steps <= 540 else 270
    cfg.train.diagnostics.channel_stats = True
    cfg.train.diagnostics.capture_steps = list(cfg.train.snapshot_steps)
    cfg.investigation = {'reference': str(args.reference), 'subset_manifest': str(args.subset) if args.subset else None,
                         'start': args.start, 'initial_state_sha256': initial_hash, 'start_state_sha256': start_hash,
                         'optimizer_state': 'reset_identically_all_branches; old checkpoint contains model only',
                         'scheduler_state': 'reset_to_explicit_absolute_step_trajectory',
                         'batch_sampler': 'whole pairs; numpy default_rng seed42; no augmentation',
                         'measurements': 'full train only; original/rest are not holdout',
                         'torch_version': str(torch.__version__), 'device': str(device)}
    cfg.investigation.vocab_source = str(args.reference / 'vocab.json')
    cfg.investigation.oov_tokens = sorted({w for c in full.captions for w in trainer.vocab.split(c)
                                          if w not in trainer.vocab.stoi})
    cfg.investigation.oov_rows = sum(any(w not in trainer.vocab.stoi for w in trainer.vocab.split(c))
                                     for c in full.captions)
    cfg.investigation.train_csv_sha256 = hashlib.sha256(Path(cfg.data.train_csv).read_bytes()).hexdigest()
    if args.subset:
        assert manifest['train_csv_sha256'] == cfg.investigation.train_csv_sha256
    cfg.investigation.source_sha256 = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in ('src/training.py', 'src/diagnostics.py', 'src/data.py', 'scripts/film_scaling_run.py')}
    OmegaConf.save(cfg, args.output / 'config.yaml', resolve=True)
    cached_logits = None
    if args.baseline_dir is not None:
        baseline_cfg = OmegaConf.load(args.baseline_dir / 'config.yaml')
        assert baseline_cfg.investigation.start_state_sha256 == start_hash
        np.testing.assert_array_equal(np.load(args.baseline_dir / 'train_ids.npy'), full.ids)
        cached_logits = args.baseline_dir / 'logits_step0000.npy'
    trainer.cfg = cfg.train
    trainer.diagnostics_dir = args.output
    if args.action == 'evaluate':
        measure(trainer, full, original, rest, fitted, args.output, 0)
        return
    # Keep the reference's post-initialization RNG stream; observations preserve it.
    rows = []
    def snapshot(current, optimizer, scheduler, epoch, step):
        with observation(current):
            state = {name: tensor.detach().cpu().clone() for name, tensor in current.model.state_dict().items()}
            numpy_rng = np.random.get_state()
            torch.save({'model': state, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'step': step, 'epoch': epoch, 'sampler_rng': current.rng.bit_generator.state,
                        'torch_rng': torch.get_rng_state(),
                        'python_rng': random.getstate(),
                        'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                        'mps_rng': torch.mps.get_rng_state() if device.type == 'mps' else None},
                       args.output / f'checkpoint_step{step:04d}.pt')
            rows.append(measure(current, full, original, rest, fitted, args.output, step,
                                cached_logits=cached_logits if step == 0 else None))
    started = time.monotonic()
    history = trainer.fit(fitted, {}, {'hflip_swap_lr': 0., 'reorder_clauses_p': 0.}, step_callback=snapshot)
    elapsed = time.monotonic() - started
    torch.save(trainer.model.state_dict(), args.output / 'model.pt')
    dump(args.output / 'results.json', {'history': history, 'optimizer_steps': trainer.optimizer_steps,
                                      'processed_rows': trainer.processed_rows,
                                      'stop_reason': trainer.stop_reason, 'fit_min': elapsed / 60,
                                      'snapshot_steps': [row['step'] for row in rows],
                                      'fitted_rows': len(fitted),
                                      'epochs_equivalent': trainer.processed_rows / len(fitted)})
    print(json.dumps({'output': str(args.output), 'fit_min': elapsed / 60, 'steps': trainer.optimizer_steps}), flush=True)


if __name__ == '__main__':
    main()
