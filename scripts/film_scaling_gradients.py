"""Read-only category gradients on train; run only after measured forgetting.

Uses the existing Trainer forward/loss. No optimizer steps or source checkpoint writes.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn

from film_scaling_run import REFERENCE, build, dump, observation, state_hash
from src.data import eval_batches, pair_groups, sample_pairs_by_category, subset
from src.diagnostics import prediction_summary
from src.utils import get_device


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(torch.dot(a, b)) / denominator if denominator else None


def matrix_summary(vectors, categories):
    matrix = [[cosine(vectors[a], vectors[b]) for b in categories] for a in categories]
    off = [matrix[i][j] for i in range(len(categories)) for j in range(i + 1, len(categories))
           if matrix[i][j] is not None]
    return {'categories': categories, 'cosine': matrix, 'valid_off_diagonal_pairs': len(off),
            'off_diagonal_mean': float(np.mean(off)) if off else None,
            'off_diagonal_min': float(np.min(off)) if off else None,
            'off_diagonal_negative_fraction': float(np.mean(np.asarray(off) < 0)) if off else None}


def _gradient_vector(grads):
    """Cast once on CPU: long float32 reductions can yield cosine(v, v) > 1."""
    return torch.cat([gradient.detach().cpu().flatten() for gradient in grads]).double()


def bn_counterfactual(trainer, base, current, original, rest_probe, first):
    """Separate changed weights from BN running statistics; no optimizer updates."""
    bn_keys = {f'{name}.{field}' for name, module in trainer.model.named_modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)
               for field, _ in module.named_buffers(recurse=False)}
    cases = [('base_weights_base_bn', base, base)] if first else [
        ('base_weights_current_bn', base, current),
        ('current_weights_base_bn', current, base),
        ('current_weights_current_bn', current, current)]
    result = {}
    with observation(trainer):
        for name, weights, statistics in cases:
            mixed = {key: statistics[key] if key in bn_keys else value for key, value in weights.items()}
            trainer.model.load_state_dict(mixed)
            result[name] = {}
            for domain, split in (('original', original), ('rest_probe', rest_probe)):
                metrics, logits = trainer.evaluate(split)
                values = prediction_summary(logits, split.labels, split.pair_ids)
                result[name][domain] = {'auc': metrics['auc'], 'bce': values['bce'],
                                        'pair_accuracy': values['pair_accuracy'],
                                        'margin_mean': values['signed_pair_margin']['mean']}
    trainer.model.load_state_dict(current)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', action='append', required=True, help='LABEL:PATH; model.pt or new snapshot')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = get_device(args.device)
    cfg, full, original, rest, trainer, initial_hash = build(REFERENCE, device)
    dump(args.output / 'procedure.json', {
        'reference_config': str(REFERENCE / 'config.yaml'), 'checkpoints': args.checkpoint,
        'seed': cfg.seed, 'device': str(device), 'optimizer_updates': 0,
        'dropout': 'off', 'gradient_bn_modes': ['eval', 'train_bn_dropout_off'],
        'pairs_per_category_per_domain': 32, 'pairs_per_batch': 16, 'replicates': 2,
        'statistics_dtype': 'float64 on CPU; model forward/backward remain float32',
        'bn_counterfactual_reference': 'first --checkpoint argument',
        'data_scope': 'train only', 'interpretation': 'local gradient alignment, not a causal interference intervention'})
    categories = sorted(set(full.categories))
    # Two disjoint 16-pair batches for each category/domain, fixed across all states.
    batches = OrderedDict()
    for domain, split in (('original', original), ('rest', rest)):
        selected = sample_pairs_by_category(split, 32, cfg.seed)
        for category in categories:
            cat = subset(selected, selected.categories == category)
            groups = pair_groups(cat)
            for replicate in range(2):
                mask = np.zeros(len(cat), dtype=bool)
                mask[groups[replicate * 16:(replicate + 1) * 16].reshape(-1)] = True
                batches[(domain, category, replicate)] = subset(cat, mask)
    parameters = OrderedDict()
    for name, parameter in trainer.model.named_parameters():
        if parameter.requires_grad:
            group = ('image_encoder' if name.startswith('image_encoder.') else
                     'text_encoder' if name.startswith('text_encoder.') else
                     'readout' if name.startswith('fusion.readout.') else 'film_trunk')
            parameters.setdefault(group, []).append(parameter)
    flat_parameters = [p for group in parameters.values() for p in group]
    result = {'initial_state_sha256': initial_hash, 'seed': cfg.seed, 'device': str(device),
              'torch_version': str(torch.__version__),
              'method': 'BCE gradients, no clipping/optimizer; two disjoint 32-row batches/category/domain; '
                        'dropout off; both checkpoint-eval BN and training FiLM BN; CNN BN always eval; '
                        'buffers reset before every batch; raw gradients averaged before cosine',
              'batch_ids': {'|'.join(map(str, key)): value.ids.tolist() for key, value in batches.items()},
              'states': {}}
    started = time.monotonic()
    base_state = None
    rest_probe = sample_pairs_by_category(rest, 16, cfg.seed)
    for specification in args.checkpoint:
        label, path = specification.split(':', 1)
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = state['model'] if 'model' in state else state
        first = base_state is None
        if first:
            base_state = state
        trainer.model.load_state_dict(state)
        counterfactual = bn_counterfactual(trainer, base_state, state, original, rest_probe, first)
        buffers = {name: value.detach().clone() for name, value in trainer.model.named_buffers()}
        state_result = {'path': path, 'state_sha256': state_hash(state), 'modes': {},
                        'bn_running_stats_counterfactual': counterfactual}
        print(json.dumps({'state': label, 'bn_counterfactual': counterfactual}, allow_nan=False), flush=True)
        for mode in ('eval', 'train_bn_dropout_off'):
            gradients, losses = {}, {}
            for key, split in batches.items():
                with observation(trainer):
                    trainer.model.train(mode != 'eval')
                    for module in trainer.model.modules():
                        if isinstance(module, nn.Dropout):
                            module.eval()
                    for name, buffer in trainer.model.named_buffers():
                        buffer.copy_(buffers[name])
                    batch = next(eval_batches(split, len(split)))
                    logits = trainer._forward(split, batch)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, torch.as_tensor(split.labels, device=device, dtype=torch.float32))
                    grads = torch.autograd.grad(loss, flat_parameters)
                    offset, grouped = 0, {}
                    for group, params in parameters.items():
                        grouped[group] = _gradient_vector(grads[offset:offset + len(params)])
                        assert torch.isfinite(grouped[group]).all()
                        offset += len(params)
                    gradients[key] = grouped
                    losses['|'.join(map(str, key))] = float(loss.detach())
            mode_result = {'batch_losses': losses, 'groups': {}}
            for group in parameters:
                means = {domain: {category: (gradients[(domain, category, 0)][group] +
                                             gradients[(domain, category, 1)][group]) / 2
                                  for category in categories} for domain in ('original', 'rest')}
                # Full-data pair sampler weights categories by frequency, not uniformly.
                pooled = {}
                for domain, split in (('original', original), ('rest', rest)):
                    pooled[domain] = sum(means[domain][category] * float(np.mean(split.categories == category))
                                         for category in categories)
                mode_result['groups'][group] = {
                    'rest_between_categories': matrix_summary(means['rest'], categories),
                    'original_between_categories': matrix_summary(means['original'], categories),
                    'original_rest_pooled_cosine': cosine(pooled['original'], pooled['rest']),
                    'original_pooled_norm': float(pooled['original'].norm()),
                    'rest_pooled_norm': float(pooled['rest'].norm()),
                    'original_rest_same_category_cosine': {
                        category: cosine(means['original'][category], means['rest'][category]) for category in categories},
                    'rest_replicate_cosine': {
                        category: cosine(gradients[('rest', category, 0)][group], gradients[('rest', category, 1)][group])
                        for category in categories}}
            state_result['modes'][mode] = mode_result
            print(json.dumps({'state': label, 'mode': mode,
                              'groups': {name: {k: row[k] for k in ('original_rest_pooled_cosine',
                                                                   'original_pooled_norm', 'rest_pooled_norm')}
                                         for name, row in mode_result['groups'].items()}}, allow_nan=False), flush=True)
        result['states'][label] = state_result
        dump(args.output / f'{label}.json', state_result)
    result['elapsed_sec'] = time.monotonic() - started
    dump(args.output / 'gradient_interference.json', result)


if __name__ == '__main__':
    main()
