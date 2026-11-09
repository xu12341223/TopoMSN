"""Shared dataset and network configuration."""
from dataclasses import asdict, dataclass
DS004584_CHANNELS = ['AF3', 'AF4', 'AF7', 'AF8', 'AFz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'FT10', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO7', 'PO8', 'POz', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']

STANDARD_BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))

REGION_NAMES = ('frontal', 'central', 'temporal', 'parietal', 'occipital')
ADFTD_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

STROKE_MCA_CHANNELS = ['O2', 'Oz', 'O1', 'PO8', 'PO6', 'PO4', 'POz', 'PO3', 'PO5', 'PO7', 'P8', 'P6', 'P4', 'P2', 'Pz', 'P1', 'P3', 'P5', 'P7', 'TP8', 'CP6', 'CP4', 'CP2', 'CPz', 'CP1', 'CP3', 'CP5', 'TP7', 'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8']

APAVA_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'C4', 'T4', 'T5', 'P3', 'P4', 'T6', 'O1', 'O2']

DATASETS = {
    'ds004584': dict(folder='DS004584', channels=DS004584_CHANNELS, seq_len=512,
                    context_segments=2, classes=['HC', 'PD']),
    'adftd': dict(folder='ADFTD', channels=ADFTD_CHANNELS, seq_len=256,
                 context_segments=4, classes=['HC', 'FTD', 'AD']),
    'apava': dict(folder='APAVA', channels=APAVA_CHANNELS, seq_len=256,
                 context_segments=4, classes=['HC', 'AD']),
    'stroke_mca': dict(folder='STROKE_MCA', channels=STROKE_MCA_CHANNELS, seq_len=512,
                      context_segments=2, classes=['Control', 'Stroke']),
}


@dataclass
class Config:
    dataset: str = 'ds004584'
    sfreq: float = 256.0
    d_model: int = 64
    dropout: float = 0.3
    e_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    patch_len_list: str = '16,32,64'
    stride_list: str = '8,16,32'
    node_dim: int = 32
    graph_hidden: int = 64
    graph_layers: int = 2
    cheb_order: int = 3
    geometry_hidden: int = 32
    stable_support_ratio: float = 0.15
    spatial_neighbors: int = 4
    prior_segments_per_subject: int = 16
    top_ratio: float = 0.25
    graph_temperature: float = 0.7
    topo_scale: float = 0.2
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-5
    graph_learning_rate: float = 5e-5
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    lambda_topology_aux: float = 0.1
    lambda_prior: float = 1e-3
    lambda_geometry: float = 1e-4
    seed: int = 41
    folds: int = 5
    val_ratio: float = 0.2

    @property
    def spec(self):
        return DATASETS[self.dataset]

    @property
    def num_class(self):
        return len(self.spec['classes'])

    def to_dict(self):
        return asdict(self)
