"""Shared reproducibility, evaluation and checkpoint helpers."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader

METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'auroc', 'auprc')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checkpoint_save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def select_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def loader(dataset, cfg, shuffle=False):
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, drop_last=False,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(cfg.seed))


def classification_metrics(targets, probabilities):
    classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predicted, labels=np.arange(classes), average='macro', zero_division=0)
    result = dict(accuracy=float(accuracy_score(targets, predicted)),
                  macro_precision=float(precision), macro_recall=float(recall), macro_f1=float(f1))
    if set(np.unique(targets)) != set(range(classes)):
        result.update(auroc=None, auprc=None)
    elif classes == 2:
        result.update(auroc=float(roc_auc_score(targets, probabilities[:, 1])),
                      auprc=float(average_precision_score(targets, probabilities[:, 1])))
    else:
        one_hot = np.eye(classes)[targets]
        result.update(auroc=float(roc_auc_score(one_hot, probabilities, average='macro')),
                      auprc=float(average_precision_score(one_hot, probabilities, average='macro')))
    return result


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    probabilities, targets = [], []
    for eeg, context, target in batches:
        logits = model(eeg.to(device), context.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite evaluation logits')
        probabilities.append(logits.softmax(dim=-1).cpu().numpy())
        targets.append(target.numpy())
    if not probabilities:
        raise ValueError('Empty evaluation set')
    probabilities, targets = np.concatenate(probabilities), np.concatenate(targets)
    return classification_metrics(targets, probabilities), targets, probabilities
