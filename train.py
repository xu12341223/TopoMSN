"""Subject-disjoint cross-validation. Test features are not accessed here."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import Config, DATASETS
from data import (channel_names, estimate_training_group_prior, montage,
                  open_dataset, subject_splits)
from model import TopoMSN, make_geometric_prior, make_stable_support
from utils import (checkpoint_save, evaluate, loader, select_device, set_seed,
                   source_digest, write_json)


def train_fold(root, output, cfg, split, labels, names, device):
    fold = split['fold_index']
    set_seed(cfg.seed + fold)
    directory = output / f'fold_{fold}'
    directory.mkdir()
    train = open_dataset(root, split['train'], labels, cfg)
    validation = open_dataset(root, split['validation'], labels, cfg)
    positions, region_ids = montage(names)
    with torch.no_grad():
        prior, reliability, count = estimate_training_group_prior(train, cfg, device)
    geometry = make_geometric_prior(positions).to(device)
    support = make_stable_support(reliability, geometry, cfg.stable_support_ratio,
                                  cfg.spatial_neighbors)
    model = TopoMSN(cfg, positions, region_ids, prior.cpu(), support.cpu()).to(device)
    temporal, other = [], []
    for name, parameter in model.named_parameters():
        (temporal if name.startswith('eeg_encoder.') else other).append(parameter)
    optimizer = torch.optim.AdamW([
        dict(params=temporal, lr=cfg.learning_rate),
        dict(params=other, lr=cfg.graph_learning_rate)], weight_decay=cfg.weight_decay)
    counts = np.bincount(train.targets, minlength=cfg.num_class)
    if np.any(counts == 0):
        raise ValueError('Missing class in training segments')
    weights = counts.sum() / counts
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    train_loader, val_loader = loader(train, cfg, True), loader(validation, cfg)
    best, stale, history = -1.0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, examples = 0.0, 0
        for eeg, context, target in train_loader:
            eeg, context, target = eeg.to(device), context.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(eeg, context, return_aux=True)
            loss = (criterion(logits, target)
                    + cfg.lambda_topology_aux * criterion(aux['topology_logits'], target)
                    + cfg.lambda_prior * aux['prior_distance']
                    + cfg.lambda_geometry * aux['geometry_residual_mse'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Non-finite gradient')
            optimizer.step()
            loss_sum += loss.item() * len(target)
            examples += len(target)
        metrics, _, _ = evaluate(model, val_loader, device)
        history.append(dict(epoch=epoch, training_loss=loss_sum / examples, validation=metrics))
        write_json(directory / 'history.json', history)
        print(f'fold={fold} epoch={epoch} loss={loss_sum/examples:.5f} '
              f'val_macro_f1={metrics["macro_f1"]:.5f}', flush=True)
        if metrics['macro_f1'] > best:
            best, stale = metrics['macro_f1'], 0
            checkpoint_save(directory / 'best.pt', dict(
                format_version=1, config=cfg.to_dict(), split=split,
                labels={str(k): int(v) for k, v in labels.items()},
                channel_names=names, positions=torch.tensor(positions),
                region_ids=torch.tensor(region_ids), source_sha256=source_digest(),
                epoch=epoch, validation=metrics, prior_context_count=count,
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}))
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    write_json(directory / 'status.json', dict(state='trained', best_validation_macro_f1=best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path,
                        help='Reuse the original subject split JSON rather than creating new folds')
    parser.add_argument('--fold', type=int, default=0, help='0: all folds; otherwise one fold index')
    parser.add_argument('--device', default='auto')
    for name in ('epochs', 'patience', 'batch_size', 'num_workers', 'seed'):
        parser.add_argument('--' + name.replace('_', '-'), type=int)
    options = parser.parse_args()
    settings = {key: getattr(options, key) for key in
                ('epochs', 'patience', 'batch_size', 'num_workers', 'seed')
                if getattr(options, key) is not None}
    cfg = Config(dataset=options.dataset, **settings)
    if min(cfg.epochs, cfg.patience, cfg.batch_size) < 1 or cfg.num_workers < 0:
        parser.error('Epochs, patience and batch size must be positive; workers cannot be negative')
    folds, labels = subject_splits(options.data_root, cfg, options.split_manifest)
    selected = [s for s in folds if options.fold in (0, s['fold_index'])]
    if not selected:
        parser.error('Unknown fold index')
    names = channel_names(options.data_root, cfg)
    montage(names)
    options.output.mkdir(parents=True, exist_ok=False)
    write_json(options.output / 'config.json', cfg.to_dict())
    write_json(options.output / 'splits.json', dict(dataset=cfg.dataset, folds=folds))
    device = select_device(options.device)
    for split in selected:
        train_fold(options.data_root, options.output, cfg, split, labels, names, device)


if __name__ == '__main__':
    main()
