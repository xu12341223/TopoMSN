"""Unified prepared-data interface for four EEG datasets."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from config import STANDARD_BANDS, REGION_NAMES
from model import FrequencyDecomposer, absolute_correlation
import json
from sklearn.model_selection import StratifiedKFold, train_test_split

def feature_path(root_path, subject_id):
    feature_dir = Path(root_path) / 'Feature'
    candidates = (feature_dir / f'feature_{int(subject_id)}.npy', feature_dir / f'feature_{int(subject_id):02d}.npy', feature_dir / f'feature_{int(subject_id):03d}.npy')
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No feature file for subject {subject_id}: {candidates}')

def load_subject_labels(root_path):
    labels = np.load(Path(root_path) / 'Label' / 'label.npy')
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Labels and subject IDs must be finite integers')
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f'Expected label.npy with [label, subject_id], got {labels.shape}')
    subject_ids = labels[:, 1].astype(int)
    subject_y = labels[:, 0].astype(int)
    if len(np.unique(subject_ids)) != len(subject_ids):
        raise ValueError('label.npy contains duplicate subject IDs')
    return (subject_ids, subject_y, dict(zip(subject_ids.tolist(), subject_y.tolist())))

def normalize_segment(segment):
    segment = np.asarray(segment, dtype=np.float32)
    mean = segment.mean(axis=0, keepdims=True)
    std = segment.std(axis=0, keepdims=True)
    return (segment - mean) / np.maximum(std, 1e-06)

class CausalTopologyDataset(Dataset):
    """Lazily returns a current EEG segment and its causal topology context."""

    def __init__(self, root_path, subject_ids, label_map, context_segments=2):
        if context_segments < 1:
            raise ValueError('topo_context_segments must be at least 1')
        self.context_segments = int(context_segments)
        self.arrays = {}
        self.records = []
        self.targets = []
        self.seq_len = None
        self.num_channels = None
        for subject_id in subject_ids:
            subject_id = int(subject_id)
            array = np.load(feature_path(root_path, subject_id), mmap_mode='r')
            if array.ndim != 3:
                raise ValueError(f'Subject {subject_id}: expected [segments, time, channels], got {array.shape}')
            if len(array) < self.context_segments:
                raise ValueError(f'Subject {subject_id}: {len(array)} segments cannot provide a {self.context_segments}-segment topology context')
            if self.seq_len is None:
                (self.seq_len, self.num_channels) = (int(array.shape[1]), int(array.shape[2]))
            if array.shape[1:] != (self.seq_len, self.num_channels):
                raise ValueError(f'Subject {subject_id}: inconsistent segment shape {array.shape[1:]}')
            self.arrays[subject_id] = array
            label = int(label_map[subject_id])
            for current_index in range(self.context_segments - 1, len(array)):
                self.records.append((subject_id, current_index))
                self.targets.append(label)
        self.targets = np.asarray(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        (subject_id, current_index) = self.records[index]
        array = self.arrays[subject_id]
        start = current_index - self.context_segments + 1
        current = normalize_segment(array[current_index])
        context = np.asarray(array[start:current_index + 1], dtype=np.float32)
        context = context.reshape(-1, context.shape[-1])
        context = normalize_segment(context)
        if not np.isfinite(current).all() or not np.isfinite(context).all():
            raise ValueError(f'Non-finite data for subject {subject_id}, segment {current_index}')
        return (torch.from_numpy(current).float(), torch.from_numpy(context).float(), torch.tensor(self.targets[index], dtype=torch.long))

def make_standard_positions(channel_names):
    """Return standard_1005 coordinates in the converter's channel order."""
    try:
        import mne
    except ImportError as exc:
        raise ImportError('MNE is required once at startup to obtain standard electrode coordinates. Install it with `pip install mne`.') from exc
    montage = mne.channels.make_standard_montage('standard_1005')
    positions = montage.get_positions()['ch_pos']
    canonical = {name.lower(): value for (name, value) in positions.items()}
    missing = [name for name in channel_names if name.lower() not in canonical]
    if missing:
        raise ValueError(f'standard_1005 has no coordinates for channels: {missing}')
    xyz = np.stack([canonical[name.lower()] for name in channel_names]).astype(np.float32)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(xyz, axis=1).max()
    return xyz / max(float(scale), 1e-06)

def make_region_ids(channel_names):
    """Map 10-5 channel names to five reproducible anatomical scalp regions."""
    region_ids = []
    for name in channel_names:
        upper = name.upper()
        if upper.startswith(('FT', 'TP', 'T')):
            region = 'temporal'
        elif upper.startswith(('PO', 'O')):
            region = 'occipital'
        elif upper.startswith(('CP', 'P')):
            region = 'parietal'
        elif upper.startswith('C'):
            region = 'central'
        elif upper.startswith(('FP', 'AF', 'FC', 'F')):
            region = 'frontal'
        else:
            raise ValueError(f'Cannot assign channel {name!r} to a scalp region')
        region_ids.append(REGION_NAMES.index(region))
    counts = np.bincount(region_ids, minlength=len(REGION_NAMES))
    if np.any(counts == 0):
        missing = [REGION_NAMES[index] for (index, count) in enumerate(counts) if count == 0]
        raise ValueError(f'The channel montage has empty regions: {missing}')
    return np.asarray(region_ids, dtype=np.int64)

@torch.no_grad()
def estimate_training_group_prior(dataset, args, device):
    """Estimate subject-balanced PCC mean and cross-subject reliability."""
    indices_by_subject = {}
    for (index, (subject_id, _)) in enumerate(dataset.records):
        indices_by_subject.setdefault(int(subject_id), []).append(index)
    selected = []
    selected_subject_slots = []
    per_subject = int(args.prior_segments_per_subject)
    subject_ids = sorted(indices_by_subject)
    for (subject_slot, subject_id) in enumerate(subject_ids):
        indices = indices_by_subject[subject_id]
        if per_subject <= 0 or len(indices) <= per_subject:
            subject_indices = indices
        else:
            positions = np.linspace(0, len(indices) - 1, per_subject).round().astype(int)
            subject_indices = [indices[position] for position in positions]
        selected.extend(subject_indices)
        selected_subject_slots.extend([subject_slot] * len(subject_indices))
    loader = DataLoader(Subset(dataset, selected), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    decomposer = FrequencyDecomposer(args.sfreq).to(device)
    subject_sums = torch.zeros(len(subject_ids), len(STANDARD_BANDS), dataset.num_channels, dataset.num_channels, device=device)
    subject_counts = torch.zeros(len(subject_ids), device=device)
    offset = 0
    for (_, context, _) in loader:
        context = context.to(device, non_blocking=True)
        correlation = absolute_correlation(decomposer(context))
        batch_size = correlation.size(0)
        slots = torch.as_tensor(selected_subject_slots[offset:offset + batch_size], device=device, dtype=torch.long)
        subject_sums.index_add_(0, slots, correlation)
        subject_counts.index_add_(0, slots, torch.ones_like(slots, dtype=torch.float32))
        offset += batch_size
    if offset == 0 or (subject_counts == 0).any():
        raise RuntimeError('No training segments were available for the group prior')
    subject_priors = subject_sums / subject_counts[:, None, None, None]
    prior = subject_priors.mean(dim=0)
    subject_std = subject_priors.std(dim=0, unbiased=False)
    reliability = prior * torch.exp(-subject_std / prior.clamp_min(0.001))
    eye = torch.eye(dataset.num_channels, device=device, dtype=torch.bool)
    prior = prior.masked_fill(eye[None], 0.0)
    reliability = reliability.masked_fill(eye[None], 0.0)
    return (prior, reliability, len(selected))


def channel_names(root, cfg):
    path = Path(root) / 'preprocess_config.json'
    names = cfg.spec['channels']
    if path.exists():
        metadata = json.loads(path.read_text())
        supplied = metadata.get('channel_names', metadata.get('channels'))
        if supplied is not None:
            if not isinstance(supplied, list) or not all(isinstance(x, str) for x in supplied):
                raise ValueError('Metadata channel order must be a list of channel names')
            names = supplied
        if float(metadata.get('sfreq', cfg.sfreq)) != cfg.sfreq:
            raise ValueError('Sampling frequency does not match the configured protocol')
    if len(names) != len(cfg.spec['channels']) or len(set(names)) != len(names):
        raise ValueError('Wrong channel count or duplicate channel names')
    return names


def montage(names):
    aliases = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    canonical = [aliases.get(name, name) for name in names]
    return make_standard_positions(canonical), make_region_ids(canonical)


def open_dataset(root, subjects, labels, cfg):
    dataset = CausalTopologyDataset(root, subjects, labels, cfg.spec['context_segments'])
    expected = (cfg.spec['seq_len'], len(cfg.spec['channels']))
    if (dataset.seq_len, dataset.num_channels) != expected:
        raise ValueError(f'Expected segment shape {expected}, got '
                         f'{(dataset.seq_len, dataset.num_channels)}')
    return dataset


def subject_splits(root, cfg, manifest=None):
    ids, targets, labels = load_subject_labels(root)
    if set(targets.tolist()) != set(range(cfg.num_class)):
        raise ValueError('Dataset labels must match the configured classes')
    if manifest:
        document = json.loads(Path(manifest).read_text())
        if isinstance(document, dict) and document.get('dataset', cfg.dataset) != cfg.dataset:
            raise ValueError('Split manifest is for a different dataset')
        folds = document.get('folds', document) if isinstance(document, dict) else document
    else:
        if np.bincount(targets).min() < cfg.folds:
            raise ValueError('Not enough subjects per class for stratified folds')
        folds = []
        splitter = StratifiedKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
        for index, (development, test) in enumerate(splitter.split(ids, targets), 1):
            train, val = train_test_split(ids[development], test_size=cfg.val_ratio,
                stratify=targets[development], random_state=cfg.seed + index)
            folds.append(dict(fold_index=index, train=train.tolist(),
                              validation=val.tolist(), test=ids[test].tolist()))
    if len(folds) != cfg.folds:
        raise ValueError(f'Expected {cfg.folds} folds')
    known, all_test, indices = set(ids.tolist()), [], set()
    for ordinal, split in enumerate(folds, 1):
        split['fold_index'] = int(split.get('fold_index', ordinal))
        if split['fold_index'] in indices:
            raise ValueError('Duplicate fold index')
        indices.add(split['fold_index'])
        groups = []
        for key in ('train', 'validation', 'test'):
            raw = split[key]
            if any(isinstance(x, bool) or int(x) != x for x in raw):
                raise ValueError('Subject IDs must be integers')
            group = set(raw)
            if not group or len(group) != len(raw) or not group <= known:
                raise ValueError(f'Invalid subjects in {key}')
            if set(labels[x] for x in group) != set(range(cfg.num_class)):
                raise ValueError(f'Missing class in {key}')
            groups.append(group)
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Subject leakage between train/validation/test')
        if set.union(*groups) != known:
            raise ValueError('Fold does not cover all subjects')
        all_test.extend(split['test'])
    if set(all_test) != known or len(all_test) != len(known):
        raise ValueError('Each subject must occur in exactly one test fold')
    return folds, labels


if __name__ == '__main__':
    import argparse
    from config import Config, DATASETS
    parser = argparse.ArgumentParser(description='Validate prepared EEG data; no files are changed')
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    options = parser.parse_args()
    config = Config(dataset=options.dataset)
    subjects, _, label_map = load_subject_labels(options.data_root)
    names = channel_names(options.data_root, config)
    montage(names)
    data = open_dataset(options.data_root, subjects, label_map, config)
    for index in range(len(data)):
        data[index]
    print(json.dumps(dict(dataset=config.dataset, subjects=len(subjects),
                          usable_segments=len(data), channels=names)))
