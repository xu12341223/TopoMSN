"""TopoMSN main network; core operators retained from the experiment code."""
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from config import STANDARD_BANDS, REGION_NAMES

# The classes below retain the temporal encoder used by the local experiments.

class SimplePatchEmbedding(nn.Module):

    def __init__(self, enc_in, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.enc_in = enc_in
        self.proj = nn.Linear(patch_len * enc_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        (B, T, C) = x.shape
        if T < self.patch_len:
            pad_len = self.patch_len - T
            x = F.pad(x, (0, 0, 0, pad_len))
            T = x.shape[1]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        if patches.shape[-1] == self.patch_len:
            patches = patches.permute(0, 1, 3, 2)
        (B, N, P, C) = patches.shape
        patches = patches.reshape(B, N, P * C)
        out = self.proj(patches)
        out = self.dropout(out)
        return out

class MultiPatchEmbedding(nn.Module):

    def __init__(self, enc_in, patch_len_list, stride_list, d_model, dropout=0.1):
        super().__init__()
        if len(patch_len_list) != len(stride_list):
            raise ValueError('patch_len_list and stride_list must have the same length')
        self.embeddings = nn.ModuleList([SimplePatchEmbedding(enc_in, patch_len, stride, d_model, dropout) for (patch_len, stride) in zip(patch_len_list, stride_list)])

    def forward(self, x):
        return [embedding(x) for embedding in self.embeddings]

class TemporalEncoder(nn.Module):
    """Multi-scale patch encoder retained from the local temporal baseline."""

    def __init__(self, configs):
        super(TemporalEncoder, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.use_multi_patch = getattr(configs, 'use_multi_patch', False)
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.dropout_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.num_class = configs.num_class
        if self.use_multi_patch:
            patch_len_list = getattr(configs, 'patch_len_list', [8, 16, 32])
            if isinstance(patch_len_list, str):
                patch_len_list = [int(x) for x in patch_len_list.split(',') if x]
            stride_list = getattr(configs, 'stride_list', None)
            if stride_list is None:
                stride_list = [max(1, p // 2) for p in patch_len_list]
            elif isinstance(stride_list, str):
                stride_list = [int(x) for x in stride_list.split(',') if x]
            self.patch_len_list = patch_len_list
            self.stride_list = stride_list
            self.enc_embedding = MultiPatchEmbedding(enc_in=self.enc_in, patch_len_list=self.patch_len_list, stride_list=self.stride_list, d_model=self.d_model, dropout=self.dropout_rate)
            self.pos_embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, max(1, (self.seq_len - patch_len) // stride + 1), self.d_model)) for (patch_len, stride) in zip(self.patch_len_list, self.stride_list)])
            self.encoders = nn.ModuleList([self._make_encoder() for _ in self.patch_len_list])
            self.scale_fusion = nn.Sequential(nn.Linear(self.d_model * len(self.patch_len_list), self.d_model), nn.LayerNorm(self.d_model), nn.GELU(), nn.Dropout(self.dropout_rate))
        else:
            self.enc_embedding = SimplePatchEmbedding(enc_in=self.enc_in, patch_len=self.patch_len, stride=self.stride, d_model=self.d_model, dropout=self.dropout_rate)
            patch_num = max(1, (self.seq_len - self.patch_len) // self.stride + 1)
            self.pos_embedding = nn.Parameter(torch.randn(1, patch_num, self.d_model))
            self.encoder = self._make_encoder()
        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.d_model, self.num_class)

    def _make_encoder(self):
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=self.dropout_rate, activation='gelu', batch_first=True, norm_first=True)
        return nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

    def forward_features(self, x_enc, x_mark_enc=None):
        if self.use_multi_patch:
            scale_features = []
            for (x, pos_embedding, encoder) in zip(self.enc_embedding(x_enc), self.pos_embeddings, self.encoders):
                N = x.shape[1]
                x = x + pos_embedding[:, :N, :]
                x = encoder(x)
                scale_features.append(x.mean(dim=1))
            x = self.scale_fusion(torch.cat(scale_features, dim=1))
            x = self.act(x)
            x = self.dropout(x)
            return x
        x = self.enc_embedding(x_enc)
        N = x.shape[1]
        x = x + self.pos_embedding[:, :N, :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.act(x)
        x = self.dropout(x)
        return x

    def classification(self, x_enc, x_mark_enc=None):
        x = self.forward_features(x_enc, x_mark_enc)
        out = self.classifier(x)
        return out

    def forward(self, x_enc):
        return self.classification(x_enc)

def make_geometric_prior(positions, sigma=0.35):
    positions = torch.as_tensor(positions, dtype=torch.float32)
    distance2 = torch.cdist(positions, positions).pow(2)
    adjacency = torch.exp(-distance2 / (2.0 * sigma * sigma))
    adjacency.fill_diagonal_(0.0)
    return adjacency

class FrequencyDecomposer(nn.Module):

    def __init__(self, sampling_rate, bands=STANDARD_BANDS):
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.bands = tuple(bands)

    def forward(self, signal):
        length = signal.size(1)
        spectrum = torch.fft.rfft(signal, dim=1)
        frequencies = torch.fft.rfftfreq(length, d=1.0 / self.sampling_rate, device=signal.device)
        masks = torch.stack([(frequencies >= low) & (frequencies < high) for (low, high) in self.bands]).to(spectrum.dtype)
        band_spectrum = spectrum[:, None, :, :] * masks[None, :, :, None]
        return torch.fft.irfft(band_spectrum, n=length, dim=2)

class ChannelTemporalEncoder(nn.Module):

    def __init__(self, out_dim, dropout):
        super().__init__()
        middle = max(8, out_dim // 2)
        self.encoder = nn.Sequential(nn.Conv1d(1, middle, kernel_size=15, stride=4, padding=7, bias=False), nn.BatchNorm1d(middle), nn.GELU(), nn.Conv1d(middle, out_dim, kernel_size=9, stride=4, padding=4, bias=False), nn.BatchNorm1d(out_dim), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(1), nn.Dropout(dropout))

    def forward(self, band_signal):
        (batch, bands, time_steps, channels) = band_signal.shape
        sequences = band_signal.permute(0, 1, 3, 2).reshape(-1, 1, time_steps)
        encoded = self.encoder(sequences)
        return encoded.view(batch, bands, channels, -1)

def signed_correlation(band_signal, eps=1e-06):
    centered = band_signal - band_signal.mean(dim=2, keepdim=True)
    scale = centered.square().sum(dim=2).clamp_min(eps).sqrt()
    covariance = torch.einsum('bktc,bktd->bkcd', centered, centered)
    correlation = covariance / (scale.unsqueeze(-1) * scale.unsqueeze(-2)).clamp_min(eps)
    correlation = correlation.clamp(-1.0, 1.0)
    eye = torch.eye(correlation.size(-1), device=correlation.device, dtype=torch.bool)
    return correlation.masked_fill(eye[None, None], 0.0)

def absolute_correlation(band_signal, eps=1e-06):
    return signed_correlation(band_signal, eps=eps).abs()

def symmetric_global_topk(adjacency, ratio, rank_by_abs=False):
    """Keep exactly the strongest ratio of undirected edges in each graph."""
    channels = adjacency.size(-1)
    if ratio >= 1.0:
        return adjacency
    (rows, columns) = torch.triu_indices(channels, channels, offset=1, device=adjacency.device)
    edge_values = adjacency[..., rows, columns]
    keep = max(1, min(edge_values.size(-1), int(math.ceil(edge_values.size(-1) * ratio))))
    ranking_values = edge_values.abs() if rank_by_abs else edge_values
    indices = ranking_values.topk(keep, dim=-1).indices
    mask = torch.zeros_like(edge_values).scatter_(-1, indices, 1.0)
    retained = edge_values * mask
    sparse = torch.zeros_like(adjacency)
    sparse[..., rows, columns] = retained
    sparse[..., columns, rows] = retained
    return sparse

def make_stable_support(reliability, geometric_prior, functional_ratio, spatial_k):
    """Union of reliable training-fold edges and bidirectional spatial kNN."""
    functional = symmetric_global_topk(reliability, functional_ratio) > 0
    channels = geometric_prior.size(0)
    spatial_k = max(1, min(int(spatial_k), channels - 1))
    spatial = geometric_prior.clone()
    spatial.fill_diagonal_(-1.0)
    neighbors = spatial.topk(spatial_k, dim=-1).indices
    spatial_mask = torch.zeros_like(spatial, dtype=torch.bool)
    spatial_mask.scatter_(1, neighbors, True)
    spatial_mask = spatial_mask | spatial_mask.transpose(0, 1)
    eye = torch.eye(channels, device=reliability.device, dtype=torch.bool)
    return (functional | spatial_mask[None]).masked_fill(eye[None], False)

class PairwiseGeometryBias(nn.Module):
    """Refine a reliable RBF prior with coordinate-conditioned pair features."""

    def __init__(self, positions, geometric_prior, hidden_dim):
        super().__init__()
        self.register_buffer('positions', torch.as_tensor(positions, dtype=torch.float32))
        self.register_buffer('geometric_prior', geometric_prior)
        self.network = nn.Sequential(nn.Linear(14, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self):
        channels = self.positions.size(0)
        source = self.positions[:, None].expand(channels, channels, -1)
        target = self.positions[None, :].expand(channels, channels, -1)
        difference = source - target
        distance = difference.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        dot_product = (source * target).sum(dim=-1, keepdim=True)
        pair_features = torch.cat([source, target, difference, difference.abs(), distance, dot_product], dim=-1)
        correction = self.network(pair_features).squeeze(-1)
        correction = 0.5 * (correction + correction.transpose(0, 1))
        base = self.geometric_prior.clamp(0.0001, 1.0 - 0.0001)
        geometry = torch.sigmoid(torch.logit(base) + torch.sigmoid(self.residual_scale) * correction)
        eye = torch.eye(geometry.size(0), device=geometry.device, dtype=torch.bool)
        geometry = geometry.masked_fill(eye, 0.0)
        return (geometry, correction)

class ChebyshevGraphConv(nn.Module):
    """Residual Chebyshev message passing retained from the validated V2 path."""

    def __init__(self, in_dim, out_dim, order, dropout):
        super().__init__()
        if order < 1:
            raise ValueError('cheb_order must be at least 1')
        self.order = order
        self.filters = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(order + 1)])
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, operator):
        polynomials = [features]
        if self.order >= 1:
            polynomials.append(torch.matmul(operator, features))
        for _ in range(2, self.order + 1):
            polynomials.append(2.0 * torch.matmul(operator, polynomials[-1]) - polynomials[-2])
        output = sum((layer(value) for (layer, value) in zip(self.filters, polynomials)))
        output = self.dropout(F.gelu(output))
        return self.norm(output + self.residual(features))

def chebyshev_operator(adjacency):
    degree = adjacency.abs().sum(dim=-1).clamp_min(1e-06)
    inv_sqrt = degree.rsqrt()
    normalized = inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)
    return -normalized

class GatedResidualFusion(nn.Module):

    def __init__(self, dim, dropout):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, anchor, residual, scale):
        gate = self.gate(torch.cat([anchor, residual], dim=-1))
        return (self.norm(anchor + scale * gate * residual), gate)


class AdaptiveConnectivity(nn.Module):
    """Per-sample, per-band fusion of proximity, PCC, QK and training priors."""

    def __init__(self, cfg, positions, geometry, group_prior, support):
        super().__init__()
        self.register_buffer('group_prior', group_prior)
        self.register_buffer('stable_support', support.bool())
        self.geometry = PairwiseGeometryBias(positions, geometry, cfg.geometry_hidden)
        self.query = nn.Linear(cfg.node_dim, cfg.node_dim, bias=False)
        self.key = nn.Linear(cfg.node_dim, cfg.node_dim, bias=False)
        self.source_gate = nn.Sequential(nn.Linear(cfg.node_dim, cfg.node_dim),
                                         nn.GELU(), nn.Linear(cfg.node_dim, 4))
        nn.init.zeros_(self.source_gate[-1].weight)
        with torch.no_grad():
            self.source_gate[-1].bias.copy_(torch.tensor([.15, .65, .05, .15]).log())
        self.temperature = cfg.graph_temperature
        self.top_ratio = cfg.top_ratio

    def forward(self, features, correlation):
        batch, bands, channels, _ = features.shape
        geometry, correction = self.geometry()
        spatial = geometry[None, None].expand(batch, bands, -1, -1)
        group = (self.group_prior * self.stable_support)[None].expand(batch, -1, -1, -1)
        query = F.normalize(self.query(features), dim=-1)
        key = F.normalize(self.key(features), dim=-1)
        similarity = torch.matmul(query, key.transpose(-1, -2))
        similarity = .5 * (similarity + similarity.transpose(-1, -2))
        qk = torch.sigmoid(similarity / max(self.temperature, 1e-3))
        eye = torch.eye(channels, dtype=torch.bool, device=features.device)
        qk = qk.masked_fill(eye, 0)
        gates = self.source_gate(features.mean(dim=2)).softmax(dim=-1)
        sources = torch.stack([spatial, correlation, qk, group], dim=-1)
        adjacency = (sources * gates[:, :, None, None]).sum(dim=-1)
        adjacency = .5 * (adjacency + adjacency.transpose(-1, -2))
        adjacency = symmetric_global_topk(adjacency.masked_fill(eye, 0), self.top_ratio)
        return adjacency, {
            'adjacency': adjacency, 'source_gates': gates,
            'prior_distance': F.mse_loss(qk[..., ~eye], correlation.detach()[..., ~eye]),
            'geometry_residual_mse': correction.square().mean(),
        }


class HierarchicalReadout(nn.Module):
    def __init__(self, cfg, region_ids):
        super().__init__()
        hidden = cfg.graph_hidden
        self.register_buffer('membership', F.one_hot(
            torch.as_tensor(region_ids, dtype=torch.long), len(REGION_NAMES)).float())
        self.channel_score = nn.Linear(hidden, 1)
        self.assignment_score = nn.Linear(hidden, 1)
        self.region_graph = ChebyshevGraphConv(hidden, hidden, cfg.cheb_order, cfg.dropout)
        self.region_score = nn.Linear(hidden, 1)
        self.hierarchy_gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                            nn.Linear(hidden, hidden), nn.Sigmoid())
        self.hierarchy_scale_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, features, adjacency):
        valid = self.membership.bool()[None, None]
        scores = self.assignment_score(features).expand(-1, -1, -1, len(REGION_NAMES))
        assignment = scores.masked_fill(~valid, -1e4).softmax(dim=2) * valid
        regions = torch.einsum('bkcr,bkch->bkrh', assignment, features)
        region_adj = torch.einsum('bkcr,bkcd,bkds->bkrs', assignment, adjacency, assignment)
        regions = self.region_graph(regions, chebyshev_operator(region_adj))
        channel = (self.channel_score(features).softmax(dim=2) * features).sum(dim=2)
        region = (self.region_score(regions).softmax(dim=2) * regions).sum(dim=2)
        gate = self.hierarchy_gate(torch.cat([channel, region], dim=-1))
        gate = gate * self.hierarchy_scale_logit.sigmoid()
        return (1 - gate) * channel + gate * region


class TopologyEncoder(nn.Module):
    def __init__(self, cfg, positions, region_ids, geometry, group_prior, support):
        super().__init__()
        self.decomposer = FrequencyDecomposer(cfg.sfreq)
        self.node_encoder = ChannelTemporalEncoder(cfg.node_dim, cfg.dropout)
        self.register_buffer('electrode_positions', torch.as_tensor(positions).float())
        self.position_encoder = nn.Sequential(nn.Linear(3, cfg.node_dim), nn.GELU(),
                                               nn.Linear(cfg.node_dim, cfg.node_dim))
        self.position_scale = nn.Parameter(torch.tensor(.1))
        self.graph_learner = AdaptiveConnectivity(cfg, positions, geometry, group_prior, support)
        self.graph_layers = nn.ModuleList([
            ChebyshevGraphConv(cfg.node_dim if index == 0 else cfg.graph_hidden,
                               cfg.graph_hidden, cfg.cheb_order, cfg.dropout)
            for index in range(cfg.graph_layers)])
        self.hierarchical_readout = HierarchicalReadout(cfg, region_ids)
        layer = nn.TransformerEncoderLayer(cfg.graph_hidden, cfg.n_heads, cfg.d_ff,
                    cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.band_embedding = nn.Parameter(torch.randn(1, len(STANDARD_BANDS), cfg.graph_hidden) * .02)
        self.band_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.band_score = nn.Linear(cfg.graph_hidden, 1)
        self.output = nn.Sequential(nn.Linear(cfg.graph_hidden, cfg.d_model),
                                   nn.LayerNorm(cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout))

    def forward(self, context):
        bands = self.decomposer(context)
        features = self.node_encoder(bands)
        features = features + self.position_scale * self.position_encoder(self.electrode_positions)[None, None]
        adjacency, aux = self.graph_learner(features, absolute_correlation(bands))
        operator = chebyshev_operator(adjacency)
        for layer in self.graph_layers:
            features = layer(features, operator)
        tokens = self.hierarchical_readout(features, adjacency)
        tokens = self.band_encoder(tokens + self.band_embedding)
        weights = self.band_score(tokens).softmax(dim=1)
        return self.output((weights * tokens).sum(dim=1)), aux


class TopoMSN(nn.Module):
    def __init__(self, cfg, positions, region_ids, group_prior, support):
        super().__init__()
        if not 0 < cfg.top_ratio <= 1:
            raise ValueError('top_ratio must be in (0, 1]')
        self.topo_scale = cfg.topo_scale
        self.topo_scale_logit = nn.Parameter(torch.tensor(-3.0))
        temporal = SimpleNamespace(task_name='classification', pred_len=0,
            enc_in=len(cfg.spec['channels']), seq_len=cfg.spec['seq_len'],
            d_model=cfg.d_model, dropout=cfg.dropout, e_layers=cfg.e_layers,
            n_heads=cfg.n_heads, d_ff=cfg.d_ff, num_class=cfg.num_class,
            use_multi_patch=True, patch_len_list=cfg.patch_len_list, stride_list=cfg.stride_list)
        self.eeg_encoder = TemporalEncoder(temporal)
        geometry = make_geometric_prior(positions)
        self.topology_encoder = TopologyEncoder(cfg, positions, region_ids, geometry, group_prior, support)
        self.fusion = GatedResidualFusion(cfg.d_model, cfg.dropout)
        self.topology_classifier = nn.Linear(cfg.d_model, cfg.num_class)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_class)

    def forward(self, eeg, context, return_aux=False):
        temporal = self.eeg_encoder.forward_features(eeg)
        topology, aux = self.topology_encoder(context)
        scale = self.topo_scale * self.topo_scale_logit.sigmoid()
        fused, gate = self.fusion(temporal, topology, scale)
        logits = self.classifier(fused)
        if return_aux:
            aux.update(topology_logits=self.topology_classifier(topology),
                       fusion_gate=gate, topology_scale=scale)
            return logits, aux
        return logits
