"""Read-only train measurements for the FiLM collapse investigation.

Run: .venv/bin/python scripts/film_collapse_measure.py RUN_DIR OUTPUT.json
No fitting, validation/test access, or checkpoint writes.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
import torch

from src.data import load_split, pair_groups, sample_pairs_by_category
from src.diagnostics import prediction_summary
from src.models.model import VisualReasoningModel
from src.models.text_encoders import Vocab
from src.train import _attach_sources, instantiate
from src.training import Trainer
from src.utils import get_device, set_seed


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--full-train', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = OmegaConf.load(args.run / 'config.yaml')
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    split = load_split('train', cfg.data.train_csv)
    if cfg.data.train_pairs_per_category is not None:
        split = sample_pairs_by_category(split, cfg.data.train_pairs_per_category, cfg.seed)
    if not args.full_train:
        split = sample_pairs_by_category(split, min(64, cfg.data.train_pairs_per_category or 64), cfg.seed)
    cfg.train.cache_features = False
    encoder, channels, mode = _attach_sources(cfg, {'train': split}, device, False)
    saved_vocab = json.loads((args.run / 'vocab.json').read_text())
    vocab = Vocab([], lowercase=saved_vocab['lowercase'])
    vocab.itos = saved_vocab['itos']
    vocab.stoi = {word: i for i, word in enumerate(vocab.itos)}
    text = instantiate(cfg.model.text_encoder, vocab_size=len(vocab))
    fusion = instantiate(cfg.model.fusion, in_channels=channels, text_dim=text.out_dim, _convert_='all')
    model = VisualReasoningModel(text, fusion, encoder, spatial=True)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    manifest = json.loads((args.run / 'diagnostic_probe.json').read_text())
    assert state_hash(initial) == manifest['initial_state_sha256'], 'initial state reconstruction mismatch'
    trained = torch.load(args.run / 'model.pt', map_location='cpu', weights_only=True)
    changes = {}
    for prefix in ('image_encoder.body.layer3.4.', 'image_encoder.body.layer3.5.', 'image_encoder.', 'text_encoder.', 'fusion.'):
        keys = [name for name, _ in model.named_parameters() if name.startswith(prefix)]
        delta = sum(float((trained[k].double() - initial[k].double()).square().sum()) for k in keys) ** .5
        scale = sum(float(initial[k].double().square().sum()) for k in keys) ** .5
        changes[prefix] = {'delta_norm': delta, 'relative_norm': delta / scale if scale else None,
                           'changed_tensors': sum(not torch.equal(initial[k], trained[k]) for k in keys), 'tensors': len(keys)}
    trainer = Trainer(model, instantiate(cfg.loss), vocab, cfg.train, device, mode,
                      cfg.data.image_size, cfg.data.normalize, cfg.seed)
    probe = sample_pairs_by_category(split, min(64, cfg.data.train_pairs_per_category or 64), cfg.seed)
    indices = np.arange(len(probe))
    image_order = next(np.roll(indices, shift) for shift in range(1, len(probe))
                       if np.all(probe.img_idx[np.roll(indices, shift)] != probe.img_idx))
    caption_order = indices.copy()
    groups = pair_groups(probe)
    caption_order[groups[:, 0]], caption_order[groups[:, 1]] = groups[:, 1], groups[:, 0]
    result: dict = {'run': str(args.run), 'device': str(device), 'initial_state_sha256': state_hash(initial),
              'parameter_changes': changes, 'probe_ids': probe.ids.tolist(), 'labels': probe.labels.tolist(),
              'pair_ids': probe.pair_ids.tolist(), 'categories': probe.categories.tolist(),
              'image_order': image_order.tolist(), 'caption_order': caption_order.tolist(), 'states': {}}
    for name, state in (('initial', initial), ('trained', trained)):
        trainer.model.load_state_dict(state)
        preactivations = []
        handle = trainer.model.fusion.readout.mlp[0].register_forward_hook(
            lambda module, inputs, output, values=preactivations: values.append(output.detach().cpu().clone()))
        try:
            logits = trainer.predict(probe)
        finally:
            handle.remove()
        image_logits = trainer.predict(replace(probe, img_idx=probe.img_idx[image_order]))
        text_logits = trainer.predict(replace(probe, captions=[probe.captions[i] for i in caption_order]))
        pre = torch.cat(preactivations).numpy()
        result['states'][name] = {
            'metrics': prediction_summary(logits, probe.labels, probe.pair_ids),
            'auc_metrics': trainer.evaluate(probe)[0],
            'logits': logits.tolist(), 'image_shuffled_logits': image_logits.tolist(),
            'caption_swapped_logits': text_logits.tolist(),
            'image_delta': float(np.abs(logits-image_logits).mean()),
            'caption_delta': float(np.abs(logits-text_logits).mean()),
            'preactivation': {'mean': float(pre.mean()), 'std': float(pre.std()),
                             'negative_fraction': float((pre < 0).mean()),
                             'p05_p50_p95': np.quantile(pre, [.05, .5, .95]).tolist()},
        }
        if args.full_train and name == 'trained':
            metrics, full_logits = trainer.evaluate(split)
            result['full_train'] = {'metrics': metrics, 'summary': prediction_summary(full_logits, split.labels, split.pair_ids),
                                    'ids': split.ids.tolist(), 'logits': full_logits.tolist()}
    result['trace_probes'] = []
    for line in (args.run / 'diagnostics.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['event'] == 'probe':
            result['trace_probes'].append({
                'step': row['step'], 'reason': row['reason'], 'metrics': row['metrics'],
                'summary': prediction_summary(row['logits'], manifest['labels'], manifest['pair_ids']),
                'image_delta': row['image_delta'], 'caption_delta': row['caption_delta'],
                'preactivation': row['activations']['readout_linear'],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output), 'parameter_changes': changes,
                      'trained_probe': {k: v for k, v in result['states']['trained'].items() if 'logits' not in k},
                      'full_train': result.get('full_train', {}).get('metrics')}, indent=2))


if __name__ == '__main__':
    main()
