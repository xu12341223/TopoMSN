"""Evaluate fixed validation-selected checkpoints and export six metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from config import Config
from data import channel_names, load_subject_labels, open_dataset
from model import TopoMSN
from utils import METRICS, evaluate, loader, select_device, source_digest, write_json


def test_fold(root, checkpoint, device):
    # Only load checkpoints produced by this package, not untrusted pickle files.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['format_version'] != 1 or saved['source_sha256'] != source_digest():
        raise ValueError('Checkpoint protocol/source differs from the current package')
    cfg = Config(**saved['config'])
    _, _, labels = load_subject_labels(root)
    if labels != {int(k): v for k, v in saved['labels'].items()}:
        raise ValueError('Dataset subject IDs/labels differ from training')
    if channel_names(root, cfg) != saved['channel_names']:
        raise ValueError('Channel order differs from training')
    subjects = saved['split']['test']
    dataset = open_dataset(root, subjects, labels, cfg)
    channels = len(saved['channel_names'])
    prior = torch.zeros(5, channels, channels)
    model = TopoMSN(cfg, saved['positions'], saved['region_ids'], prior, prior.bool())
    model.load_state_dict(saved['state_dict'], strict=True)
    model.to(device)
    metrics, targets, probabilities = evaluate(model, loader(dataset, cfg), device)
    directory = checkpoint.parent / 'test'
    directory.mkdir(exist_ok=False)
    np.savez_compressed(directory / 'predictions.npz', targets=targets,
        probabilities=probabilities, predictions=probabilities.argmax(axis=1),
        subject_ids=np.array([r[0] for r in dataset.records]),
        segment_indices=np.array([r[1] for r in dataset.records]))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(directory / 'metrics.json', dict(dataset=cfg.dataset,
        fold=saved['split']['fold_index'], checkpoint_sha256=digest,
        unit='segment', positive_class=1 if cfg.num_class == 2 else None,
        class_names=cfg.spec['classes'], num_segments=len(targets), metrics=metrics))
    print(json.dumps(dict(fold=saved['split']['fold_index'], **metrics)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', default='auto')
    options = parser.parse_args()
    manifest = json.loads((options.run / 'splits.json').read_text())
    folds = manifest['folds']
    selected = [s for s in folds if options.fold in (0, s['fold_index'])]
    if not selected:
        parser.error('Unknown fold index')
    checkpoints = [options.run / f'fold_{s["fold_index"]}' / 'best.pt' for s in selected]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if (checkpoint.parent / 'test').exists():
            raise FileExistsError(f'Test already exported: {checkpoint.parent / "test"}')
    device = select_device(options.device)
    for checkpoint in checkpoints:
        test_fold(options.data_root, checkpoint, device)
    metrics_files = [options.run / f'fold_{s["fold_index"]}' / 'test/metrics.json' for s in folds]
    if all(path.exists() for path in metrics_files):
        results = [json.loads(path.read_text()) for path in metrics_files]
        summary = {}
        for key in METRICS:
            values = [row['metrics'][key] for row in results]
            summary[key] = (None if any(v is None for v in values) else
                            dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1))))
        write_json(options.run / 'summary.json', dict(folds=len(folds), std_ddof=1, metrics=summary))


if __name__ == '__main__':
    main()
