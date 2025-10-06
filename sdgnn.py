# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced SDGNN with Motif-based Higher-order Learning
Improvements: adaptive sampling, loss scheduling, caching, early stopping, better logging
"""
from collections import Counter
import itertools
import os
import sys
import time
import math
import random
import subprocess
import logging
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import torch_sparse AFTER scipy to avoid circular import
try:
    import torch_sparse
except ImportError:
    print("Warning: torch_sparse not available. Some functionality may be limited.")
    torch_sparse = None

from common import DATASET_NUM_DIC
from fea_extra import FeaExtra
from logistic_function import logistic_embedding
import networkx as nx

# ==================== Utility Classes ====================

class TrainingLogger:
    """Enhanced logging for motif-based training"""
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
        self.metrics = defaultdict(list)
        
        # Setup file logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, message, level='info'):
        getattr(self.logger, level)(message)
    
    def log_epoch(self, epoch, metrics_dict):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
        
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in metrics_dict.items()])
        self.log(f"Epoch {epoch}: {metrics_str}")
    
    def save_metrics(self):
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def plot_metrics(self, save_path=None):
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import os

            if save_path is None:
                save_path = os.path.join(self.log_dir, 'metrics.png')

            metrics_to_plot = [
                'loss', 'edge_loss', 'motif_loss', 'triad_loss',
                'accuracy','f1_micro','f1_macro','f1_weighted','auc','eval_time'
            ]
            available_metrics = [m for m in metrics_to_plot if m in self.metrics and self.metrics[m]]

            if not available_metrics:
                return

            n_metrics = len(available_metrics)
            fig, axes = plt.subplots((n_metrics + 1) // 2, 2, figsize=(14, 4 * ((n_metrics + 1) // 2)))
            if n_metrics == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for idx, metric in enumerate(available_metrics):
                values = self.metrics[metric]

                # pick epochs of the same length as values
                epochs = self.metrics.get('epoch', list(range(len(values))))
                if len(epochs) != len(values):
                    epochs = list(range(len(values)))   # fallback: just use range

                axes[idx].plot(epochs, values, linewidth=2)
                axes[idx].set_xlabel('Epoch', fontsize=10)
                axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                axes[idx].set_title(metric.replace('_', ' ').title())
                axes[idx].grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(len(available_metrics), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            self.log(f"Metrics plot saved to {save_path}")
        except Exception as e:
            self.log(f"Could not plot metrics: {e}", level='warning')



class EarlyStopping:
    """Stop training when validation metric stops improving"""
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'max' for metrics to maximize (accuracy), 'min' for loss
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MotifLossScheduler:
    """Gradually introduce motif losses during training"""
    def __init__(self, alpha_max, beta_max, warmup_epochs=10, schedule_type='linear'):
        self.alpha_max = alpha_max
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
    
    def get_weights(self, epoch):
        if epoch < self.warmup_epochs:
            ratio = epoch / self.warmup_epochs
            if self.schedule_type == 'cosine':
                ratio = (1 - math.cos(ratio * math.pi)) / 2
            return self.alpha_max * ratio, self.beta_max * ratio
        return self.alpha_max, self.beta_max


class NeighborhoodCache:
    """Cache sampled neighborhoods to reduce redundant sampling"""
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        self.hits = 0
        self.misses = 0
    
    def get_or_sample(self, cache_key, neigh_set, max_neighbors, sampler_fn):
        """
        Get cached neighbors or sample new ones.
        
        Args:
            cache_key: Unique identifier for this cache entry (e.g., "node_123_pos")
            neigh_set: The set of neighbors to sample from
            max_neighbors: Maximum number of neighbors to sample
            sampler_fn: Function that takes (neigh_set, max_neighbors) and returns sampled set
        """
        if cache_key in self.cache:
            self.hits += 1
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]
        
        self.misses += 1
        # Sample and cache
        neighbors = sampler_fn(neigh_set, max_neighbors)
        
        if len(self.cache) >= self.max_size:
            # Evict least accessed
            if self.access_count:
                min_key = min(self.access_count, key=self.access_count.get)
                del self.cache[min_key]
                del self.access_count[min_key]
        
        self.cache[cache_key] = neighbors
        self.access_count[cache_key] = 1
        return neighbors
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
    
    def clear(self):
        self.cache.clear()
        self.access_count.clear()
        self.hits = 0
        self.misses = 0


# ==================== Argument Parser ====================

parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, default='cpu', help='Devices')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', default='bitcoin_alpha', help='Dataset')
parser.add_argument('--dim', type=int, default=20, help='Embedding dimension')
parser.add_argument('--fea_dim', type=int, default=20, help='Feature embedding dimension')
parser.add_argument('--batch_size', type=int, default=500, help='Batch size')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
parser.add_argument('--k', default=1, help='Folder k')
parser.add_argument('--agg', default='attention', choices=['mean', 'attention'], help='Aggregator choose')
parser.add_argument('--alpha', type=float, default=0.01, help='Motif Laplacian weight')
parser.add_argument('--beta', type=float, default=0.5, help='Triad reconstruction weight')
parser.add_argument('--max_pos_per_u', type=int, default=50, help='Max positive neighbors per node')
parser.add_argument('--max_neg_per_u', type=int, default=50, help='Max negative neighbors per node')
parser.add_argument('--eval_every', type=int, default=5, help='Evaluation frequency')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs for loss scheduling')
parser.add_argument('--use_cache', action='store_true', help='Use neighborhood caching')
parser.add_argument('--lr_schedule', action='store_true', help='Use learning rate scheduling')
parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping')
parser.add_argument('--triad_samples', type=int, default=500, help='Number of start nodes for triad sampling')
parser.add_argument('--triad_prob', type=float, default=0.6, help='Probability for ESU depth expansion')
parser.add_argument('--motif_scale', type=float, default=1.0, help='Scale factor for motif weights')
parser.add_argument('--motif_loss_type', type=str, default='l2', choices=['l2', 'cosine', 'contrastive','logistic'], 
                    help='Type of motif loss: l2 (default), cosine similarity, or contrastive')
parser.add_argument('--motif_gamma', type=float, default=0.2,
                    help='Weight for motif message passing channel in the encoder')
parser.add_argument('--use_motif_channels', action='store_true',
                    help='Enable motif-aware message passing (Option B).')
args = parser.parse_args()

OUTPUT_DIR = f'./embeddings/sdgnn-{args.agg}'
LOG_DIR = f'./logs/sdgnn-{args.agg}'

for directory in [OUTPUT_DIR, LOG_DIR, 'embeddings', 'logs']:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Global constants
NUM_NODE = DATASET_NUM_DIC[args.dataset]
WEIGHT_DECAY = args.weight_decay
NODE_FEAT_SIZE = args.fea_dim
EMBEDDING_SIZE1 = args.dim
DEVICES = torch.device(args.devices)
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DROUPOUT = args.dropout
K = args.k

# ==================== Motif Sampling Functions ====================



def signed_bce_loss(z_a, z_b, sign):
    score = torch.dot(z_a, z_b)
    return F.softplus(-sign * score)

log_vars = nn.Parameter(torch.zeros(3))  # edge, motif, triad

def weighted_loss(edge_loss, motif_loss, triad_loss):
    losses = torch.stack([edge_loss, motif_loss, triad_loss])
    weighted = torch.sum(torch.exp(-log_vars) * losses + log_vars)
    return weighted

def signed_motif_loss(embeddings, motif_dict, unique_nodes_set, unique_nodes_dict,
                      loss_type="logistic", max_pairs=10000, device="cpu", weighting="edge"):
    total_loss = 0.0
    total_edges = 0
    motif_losses = []

    for key, A in motif_dict.items():
        coo = A.tocoo()
        rows, cols, vals = coo.row, coo.col, coo.data
        if len(rows) == 0:
            continue

        # Restrict to batch nodes only
        mask = [(r in unique_nodes_set and c in unique_nodes_set) for r,c in zip(rows, cols)]
        mask = np.array(mask, dtype=bool)
        if not mask.any():
            continue

        rows, cols, vals = rows[mask], cols[mask], vals[mask]

        if len(rows) > max_pairs:
            idx = np.random.choice(len(rows), max_pairs, replace=False)
            rows, cols, vals = rows[idx], cols[idx], vals[idx]

        bi = [unique_nodes_dict[r] for r in rows]
        bj = [unique_nodes_dict[c] for c in cols]

        zi, zj = embeddings[bi], embeddings[bj]
        V = torch.tensor(vals, dtype=torch.float32, device=device)
        V = V / (V.mean() + 1e-8)


        s = -torch.ones(len(rows), device=device) if "out_in_neg" in key else torch.ones(len(rows), device=device)
        scores = (zi * zj).sum(dim=1)

        if loss_type == "logistic":
            edge_loss = F.softplus(-s * scores) * V
        elif loss_type == "l2":
            edge_loss = ((zi - zj).pow(2).sum(dim=1)) * V
        else:  # cosine
            cos = F.cosine_similarity(zi, zj)
            edge_loss = (1 - s * cos) * V

        if weighting == "edge":
            total_loss += edge_loss.sum()
            total_edges += edge_loss.numel()
        else:  # weighting == "motif"
            motif_losses.append(edge_loss.mean())

    if weighting == "edge":
        return (total_loss / (total_edges + 1e-8)) if total_edges > 0 else torch.tensor(0.0, device=device)
    else:
        return motif_losses

def build_signed_digraph_from_edgelist(filename):
    """Build NetworkX DiGraph with signed edges"""
    G = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                u, v, s = int(parts[0]), int(parts[1]), int(parts[2])
                G.add_edge(u, v, sign=s)
    return G
def triad_margin_loss(z_u, z_v, sign, m=1.0, m0=0.1):
    score = (z_u * z_v).sum()
    if sign == 1:   # positive edge
        return F.relu(m - score)
    elif sign == -1:  # negative edge
        return F.relu(m + score)
    else:  # missing edge
        return F.relu(torch.abs(score) - m0)
    
def rand_esu_triads(G: nx.DiGraph, p_depth=0.8, num_start_nodes=None, rng=None):
    """
    Probabilistic ESU-style sampling for k=3 (triads).
    Returns triads as dicts with 'nodes' and 'edges'.
    """
    if rng is None:
        rng = random.Random(0)
    
    nodes = list(G)
    if num_start_nodes is not None and num_start_nodes < len(nodes):
        rng.shuffle(nodes)
        nodes = nodes[:num_start_nodes]
    
    triads = []
    for v in nodes:
        ext1 = list(set(G.successors(v)) | set(G.predecessors(v)))
        ext1 = [u for u in ext1 if u > v]
        
        for u in ext1:
            if rng.random() > p_depth:
                continue
            
            neigh_u = set(G.successors(u)) | set(G.predecessors(u))
            neigh_v = set(G.successors(v)) | set(G.predecessors(v))
            ext2 = sorted((neigh_u | neigh_v) - {v, u})
            
            for w in ext2:
                if w <= u:
                    continue

                # Collect edges among triad nodes
                edges = []
                for a, b in [(v,u), (u,w), (w,v)]:
                    if G.has_edge(a, b):
                        sign = G[a][b].get("sign", 1)  # default +1
                        edges.append((a, b, sign))
                
                triads.append({
                    "nodes": (v, u, w),
                    "edges": edges
                })
    
    return triads

def triad_edge_sign_dir(G: nx.DiGraph, a, b):
    """Return edge information between nodes a and b"""
    has_ab = G.has_edge(a, b)
    has_ba = G.has_edge(b, a)
    s_ab = G[a][b]['sign'] if has_ab else 0
    s_ba = G[b][a]['sign'] if has_ba else 0
    return has_ab, s_ab, has_ba, s_ba

def build_signed_directional_motif_matrices(G, triads, num_nodes):
    out_in_pos = defaultdict(float)
    out_in_neg = defaultdict(float)
    co_out = defaultdict(float)
    co_in = defaultdict(float)

    for triad in triads:
        nodes = triad["nodes"]
        edges = triad["edges"]

        # (1) Signed out→in edges
        for (a, b, s) in edges:
            if s == 1:
                out_in_pos[(a, b)] += 1
            elif s == -1:
                out_in_neg[(a, b)] += 1

        # (2) Co-out motifs: do u,v both regulate the same target w?
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                # find common successors (targets) in the original graph
                common_targets = set(G.successors(ni)) & set(G.successors(nj))
                for t in common_targets:
                    co_out[(ni, nj)] += 1
                    co_out[(nj, ni)] += 1

        # (3) Co-in motifs: do u,v both receive input from the same source w?
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                # find common predecessors (sources) in the original graph
                common_sources = set(G.predecessors(ni)) & set(G.predecessors(nj))
                for s in common_sources:
                    co_in[(ni, nj)] += 1
                    co_in[(nj, ni)] += 1

    def to_csr(dic):
        if not dic:
            return sp.csr_matrix((num_nodes, num_nodes))
        rows, cols, vals = zip(*[(i, j, v) for (i, j), v in dic.items()])
        return sp.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))

    return {
        "out_in_pos": to_csr(out_in_pos),
        "out_in_neg": to_csr(out_in_neg),
        "co_out": to_csr(co_out),
        "co_in": to_csr(co_in),
    }

def build_motif_adjacency_and_instances(G, triads, type_weights=None):
    
    rows, cols, vals = [], [], []
    triad_records = []
    for triad in triads:
        v, u, w = triad["nodes"]
        weight = 1.0
        
        edges = []
        for (a, b) in [(v,u), (u,v), (v,w), (w,v), (u,w), (w,u)]:
            has_ab, s_ab, _, _ = triad_edge_sign_dir(G, a, b)
            if has_ab:
                edges.append((a, b, s_ab))
        
        triad_records.append({"nodes": (v,u,w), "edges": edges, "weight": weight})
        
        for (a,b) in [(v,u), (v,w), (u,w)]:
            rows += [a, b]
            cols += [b, a]
            vals += [weight, weight]
    
    max_node = max(G.nodes()) if G.nodes() else 0
    n = max_node + 1
    A_mot = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A_mot, triad_records

# ==================== Neural Network Modules ====================

class Encoder(nn.Module):
    """Encode features to embeddings (now supports per-channel weights)"""
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggs, channel_weights=None):
        super(Encoder, self).__init__()
        
        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggs = aggs
        self.embed_dim = embed_dim

        # weights for each adjacency channel (len == len(adj_lists))
        if channel_weights is None:
            channel_weights = [1.0] * len(adj_lists)
        assert len(channel_weights) == len(adj_lists), "channel_weights length must match adj_lists"
        self.channel_weights = [float(w) for w in channel_weights]
        
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)
        
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        
        self.nonlinear_layer = nn.Sequential(
            nn.Linear((len(adj_lists) + 1) * feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, embed_dim)
        )
        self.nonlinear_layer.apply(init_weights)
    
    def forward(self, nodes):
        # Convert to list
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.detach().cpu().numpy().tolist()
        elif isinstance(nodes, np.ndarray):
            nodes = nodes.tolist()
        
        # aggregate per channel and apply per-channel weight
        neigh_feats = []
        for (adj, agg, ch_w, ch_idx) in zip(self.adj_lists, self.aggs, self.channel_weights, range(len(self.adj_lists))):
            nf = agg(nodes, adj, ch_idx)
            if ch_w != 1.0:
                nf = nf * ch_w
            neigh_feats.append(nf)

        self_feats = self.features(torch.LongTensor(nodes).to(DEVICES))
        combined = torch.cat([self_feats] + neigh_feats, 1)
        combined = self.nonlinear_layer(combined)
        return combined


class AttentionAggregator(nn.Module):
    """Attention-based neighborhood aggregation"""
    def __init__(self, features, in_dim, out_dim, node_num, dropout_rate=DROUPOUT, slope_ratio=0.1):
        super(AttentionAggregator, self).__init__()
        
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))
        nn.init.kaiming_normal_(self.a.data)
        
        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)
    
    def forward(self, nodes, adj, ind):
        # Ensure nodes is a list or numpy array, not a tensor
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.cpu().numpy().tolist()
        node_pku = np.array(nodes)
        edges = np.array(adj[node_pku, :].nonzero()).T
        edges[:, 0] = node_pku[edges[:, 0]]
        
        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))
        batch_node_num = len(unique_nodes_list)
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)
        
        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]
        
        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        new_embeddings = self.out_linear_layer(self.features(n2))
        
        original_node_edge = np.array([self.unique_nodes_dict[nodes], 
                                      self.unique_nodes_dict[nodes]]).T
        edges = np.vstack((edges, original_node_edge))
        edges = torch.LongTensor(edges).to(DEVICES)
        
        edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], 
                             new_embeddings[edges[:, 1], :]), dim=1)
        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", 
                                                      [edge_h_2, self.a]), 
                                        self.slope_ratio))
        indices = edges
        
        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0],
                                        torch.Size([batch_node_num, batch_node_num]), 
                                        device=DEVICES)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))
        results = torch.sparse.mm(matrix, new_embeddings)
        output_emb = results.div(row_sum)
        
        return output_emb[self.unique_nodes_dict[nodes]]


class MeanAggregator(nn.Module):
    """Mean-based neighborhood aggregation"""
    def __init__(self, features, in_dim, out_dim, node_num):
        super(MeanAggregator, self).__init__()
        
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_linear_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim)
        )
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)
    
    def forward(self, nodes, adj, ind):
        # Ensure nodes is a list or numpy array, not a tensor
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.cpu().numpy().tolist()
        mask = [1, 1, 0, 0]
        node_tmp = np.array(nodes)
        edges = np.array(adj[node_tmp, :].nonzero()).T
        edges[:, 0] = node_tmp[edges[:, 0]]
        
        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))
        batch_node_num = len(unique_nodes_list)
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)
        
        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]
        
        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        new_embeddings = self.out_linear_layer(self.features(n2))
        edges = torch.LongTensor(edges).to(DEVICES)
        
        values = torch.where(edges[:, 0] == edges[:, 1], 
                           torch.FloatTensor([mask[ind]]).to(DEVICES),
                           torch.FloatTensor([1]).to(DEVICES))
        
        matrix = torch.sparse_coo_tensor(edges.t(), values,
                                        torch.Size([batch_node_num, batch_node_num]),
                                        device=DEVICES)
        row_sum = torch.spmm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(DEVICES), row_sum)
        
        results = torch.spmm(matrix, new_embeddings)
        output_emb = results.div(row_sum)
        
        return output_emb[self.unique_nodes_dict[nodes]]


class SDGNN(nn.Module):
    """Signed Directed GNN with Motif Learning"""
    def __init__(self, enc):
        super(SDGNN, self).__init__()
        self.enc = enc
        self.score_function1 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE1, 1),
            nn.Sigmoid()
        )
        self.score_function2 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE1, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(EMBEDDING_SIZE1 * 2, 1)
    
    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds
    
    def criterion_motif(self, nodes, motif_dict, triad_records, triads_by_node,
                        pos_adj_dict, neg_adj_dict,
                        alpha=1.0, beta=1.0,
                        max_triads_per_batch=1000,
                        max_pairs_from_A_mot=20000,
                        max_pos_per_u=50, max_neg_per_u=50,
                        cache=None, debug=False):
        """
        Enhanced criterion with signed directional motif loss and triad loss.
        """
        device = next(self.parameters()).device

        # Helper for neighbor sampling
        def _sample_neighbors(node_id, neigh_set, k, is_pos):
            if cache is not None:
                cache_key = f"{node_id}_{'pos' if is_pos else 'neg'}"
                return cache.get_or_sample(
                    cache_key,
                    neigh_set,
                    k,
                    lambda ns, k: set(random.sample(list(ns), min(k, len(ns)))) if ns else set()
                )
            neigh = list(neigh_set) if not isinstance(neigh_set, set) else list(neigh_set)
            if len(neigh) > k:
                return set(random.sample(neigh, k))
            return set(neigh)

        # --- Sample neighbors
        pos_neighbors_list, neg_neighbors_list = [], []
        for u in nodes:
            pos_neighbors_list.append(_sample_neighbors(u, pos_adj_dict[u], max_pos_per_u, True))
            neg_neighbors_list.append(_sample_neighbors(u, neg_adj_dict[u], max_neg_per_u, False))

        unique_nodes_set = set(nodes)
        for s in pos_neighbors_list: unique_nodes_set |= s
        for s in neg_neighbors_list: unique_nodes_set |= s
        unique_nodes_list = list(unique_nodes_set)
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}

        Z = self.enc(unique_nodes_list)

        # Initialize losses
        edge_loss = torch.tensor(0.0, device=device)
        motif_loss = torch.tensor(0.0, device=device)
        triad_loss = torch.tensor(0.0, device=device)

        # --- Edge Loss
        edge_losses = []
        for i, u in enumerate(nodes):
            u_idx = unique_nodes_dict[u]
            z_u = Z[u_idx]
            for v in pos_neighbors_list[i]:
                idx = unique_nodes_dict.get(v)
                if idx is not None:
                    edge_losses.append(signed_bce_loss(z_u, Z[idx], sign=1.0))
            for v in neg_neighbors_list[i]:
                idx = unique_nodes_dict.get(v)
                if idx is not None:
                    edge_losses.append(signed_bce_loss(z_u, Z[idx], sign=-1.0))
        if edge_losses:
            edge_loss = torch.stack(edge_losses).mean()

        # --- Motif Loss
        if alpha > 0.0 and motif_dict is not None:
            motif_loss = signed_motif_loss(
                Z, motif_dict, unique_nodes_set, unique_nodes_dict,
                loss_type="logistic",
                max_pairs=max_pairs_from_A_mot,
                device=device,
                weighting="edge"
            )
            motif_loss = alpha * motif_loss

        # --- Triad Loss (margin-based)
        triad_losses = []
        if beta > 0.0 and triads_by_node is not None and triad_records is not None:
            triad_idx_pool = []
            for u in nodes:
                triad_idx_pool.extend(triads_by_node.get(u, []))
            if triad_idx_pool:
                triad_idx_pool = list(set(triad_idx_pool))
                for t_idx in triad_idx_pool:
                    rec = triad_records[t_idx]
                    a,b,c = rec["nodes"]
                    observed = {(x,y): s for (x,y,s) in rec["edges"]}

                    triad_pairs = [(a,b),(b,c),(c,a)]  # 3 directed edges
                    # positives: only those that exist
                    for (x,y) in triad_pairs:
                        ia = unique_nodes_dict.get(x)
                        ib = unique_nodes_dict.get(y)
                        if ia is None or ib is None: 
                            continue
                        s_xy = observed.get((x,y), 0)
                        if s_xy != 0:
                            triad_losses.append(signed_bce_loss(Z[ia], Z[ib], sign=float(s_xy)))

                    # negatives: sample at most one missing per pair
                    for (x,y) in triad_pairs:
                        if (x,y) in observed: 
                            continue
                        ia = unique_nodes_dict.get(x)
                        ib = unique_nodes_dict.get(y)
                        if ia is None or ib is None:
                            continue
                        score = (Z[ia] * Z[ib]).sum()
                        # hinge to push toward 0 with small margin
                        triad_losses.append(F.relu(torch.abs(score) - 0.2))

        if triad_losses:
            triad_loss = beta * torch.stack(triad_losses).mean()

        # --- Adaptive weighting
        # define self.log_vars if not already done
        if not hasattr(self, "log_vars"):
            self.log_vars = nn.Parameter(torch.zeros(3, device=device))

        losses = torch.stack([edge_loss, motif_loss, triad_loss])
        total_loss = torch.sum(torch.exp(-self.log_vars) * losses + self.log_vars)

        # Track dict
        loss_dict = {
            'edge': edge_loss.item(),
            'motif': motif_loss.item(),
            'triad': triad_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


# ==================== Data Loading ====================

def load_data2(filename=''):
    """Load signed directed graph data"""
    adj_lists1 = defaultdict(set)
    adj_lists1_1 = defaultdict(set)
    adj_lists1_2 = defaultdict(set)
    adj_lists2 = defaultdict(set)
    adj_lists2_1 = defaultdict(set)
    adj_lists2_2 = defaultdict(set)
    adj_lists3 = defaultdict(set)
    
    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if len(info) < 3:
                continue
            person1 = int(info[0])
            person2 = int(info[1])
            v = int(info[2])
            adj_lists3[person2].add(person1)
            adj_lists3[person1].add(person2)
            
            if v == 1:
                adj_lists1[person1].add(person2)
                adj_lists1[person2].add(person1)
                adj_lists1_1[person1].add(person2)
                adj_lists1_2[person2].add(person1)
            else:
                adj_lists2[person1].add(person2)
                adj_lists2[person2].add(person1)
                adj_lists2_1[person1].add(person2)
                adj_lists2_2[person2].add(person1)
    
    return adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3


def read_emb(num_nodes, fpath):
    """Read embeddings from file"""
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                dim = int(line.split()[1])
                embeddings = np.random.rand(num_nodes, dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                assert len(emb) == dim
                embeddings[int(node)] = np.array(emb)
    return embeddings


# ==================== Main Training Function ====================

def run(dataset, k):
    """Enhanced training loop with signed directional motif loss, triad loss,
    and motif-aware message passing (Option B)."""
    num_nodes = DATASET_NUM_DIC[dataset] + 3
   
    # Initialize logger
    logger = TrainingLogger(log_dir=LOG_DIR)
    logger.log(f"Starting training for dataset: {dataset}, fold: {k}")
    logger.log(f"Configuration: {vars(args)}")

    t0 = time.time()

    # Load data (signed, directed)
    filename = f'./experiment-data/{dataset}-train-{k}.edgelist'
    adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3 = load_data2(filename)
    logger.log(f"Data loaded in {time.time() - t0:.2f}s")

    # -----------------------------------------------------------------------------
    # Signed directional motifs (global) + triad records (local)
    # -----------------------------------------------------------------------------
    use_motif_losses = (args.alpha > 0.0) or (args.beta > 0.0)
    use_motif_channels = getattr(args, "use_motif_channels", True)  # allow toggle; default ON
    motif_gamma = float(getattr(args, "motif_gamma", 0.2))          # weight for motif channels

    motif_dict = None
    triad_records = None
    triads_by_node = None

    if use_motif_losses or use_motif_channels:
        t_motif = time.time()
        logger.log("Building motif structures...")

        # Build NX graph, sample triads
        G = build_signed_digraph_from_edgelist(filename)
        triads = rand_esu_triads(G, p_depth=args.triad_prob, num_start_nodes=args.triad_samples)
        logger.log(f"Sampled {len(triads)} triads")

        # Build triad records (local instances) and a global motif adjacency view
        try:
            A_mot, triad_records = build_motif_adjacency_and_instances(G, triads)
            print(f"[DEBUG] Number of triads sampled: {len(triads)}")
            print(f"[DEBUG] Number of triad_records: {len(triad_records)}")
            print(f"[DEBUG] Motif adjacency nnz (edges): {A_mot.nnz}")

            edge_counts = [len(rec["edges"]) for rec in triad_records]
            print(f"[DEBUG] Triad edge distribution: "
                  f"min={min(edge_counts)}, max={max(edge_counts)}, "
                  f"mean={np.mean(edge_counts):.2f}")
            print(f"[DEBUG] Count by edge size: {dict(Counter(edge_counts))}")
        except Exception:
            triad_records = []
            for t in triads:
                nodes = list(t)
                edges = []
                for a in nodes:
                    for b in nodes:
                        if a == b:
                            continue
                        has_ab, s_ab, has_ba, s_ba = triad_edge_sign_dir(G, a, b)
                        if has_ab:
                            edges.append((a, b, s_ab))
                triad_records.append({"nodes": nodes, "edges": edges, "weight": 1.0})

        # Index triads by node for quick batch pooling
        triads_by_node = defaultdict(list)
        for idx, rec in enumerate(triad_records):
            for n in rec["nodes"]:
                triads_by_node[n].append(idx)

        motifs_per_node = [len(triads_by_node[n]) for n in triads_by_node]
        print(f"[DEBUG] Motifs per node: mean={np.mean(motifs_per_node):.2f}, "
              f"max={np.max(motifs_per_node)}, min={np.min(motifs_per_node)}")

        # Build signed, directional motif adjacencies (global views)
        motif_dict = build_signed_directional_motif_matrices(G, triads, num_nodes)

        # Normalize/scale & clip each matrix
        def _normalize_row_stochastic(A, name):
            if A.nnz == 0:
                return A
            A = A.tocsr()
            row_sum = np.asarray(A.sum(axis=1)).ravel() + 1e-8
            inv = 1.0 / row_sum
            D_inv = sp.diags(inv)
            A = D_inv @ A
            # cap to [0,1] to avoid huge weights in losses
            A = A.tocoo(copy=True)
            A.data = np.clip(A.data, 0.0, 1.0)
            return A.tocsr()

        motif_dict = {key: _normalize_row_stochastic(A, key) for key, A in motif_dict.items()}


        if getattr(args, "motif_scale", 1.0) != 1.0:
            for key in motif_dict:
                motif_dict[key] = motif_dict[key] * args.motif_scale
            logger.log(f"Applied motif scale factor: {args.motif_scale}")

        # Diagnostics
        for key, A in motif_dict.items():
            density = (A.nnz / (A.shape[0] * A.shape[1])) if A.shape[0] > 0 else 0.0
            rng = (float(A.data.min()), float(A.data.max())) if A.nnz else (0.0, 0.0)
            print(f"[DEBUG] {key}: nnz={A.nnz}, density={A.nnz / (NUM_NODE**2):.6f}")
            logger.log(f"[motif:{key}] shape={A.shape}, nnz={A.nnz}, density={density:.6f}, range={rng}")
 
        logger.log(f"Triads indexed for {len(triads_by_node)} unique nodes")
        logger.log(f"Motif structures built in {time.time() - t_motif:.2f}s")

    features = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    features.weight.requires_grad = True
    nn.init.xavier_uniform_(features.weight)
    features = features.to(DEVICES)

    # --- helper: build & sanitize sparse adjs -----------------------------------
    def _to_csr(adj_list):
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        if not edges:
            return sp.csr_matrix((num_nodes, num_nodes))
        edges = np.array(edges, dtype=np.int64)
        return sp.csr_matrix((np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
                            shape=(num_nodes, num_nodes))

    def _sanitize_adj(A, n):
        """Clip any out-of-range indices and force exact (n, n) shape."""
        if A is None:
            return sp.csr_matrix((n, n))
        coo = A.tocoo()
        if coo.nnz == 0:
            return sp.csr_matrix((n, n))
        mask = (coo.row >= 0) & (coo.row < n) & (coo.col >= 0) & (coo.col < n)
        if not np.all(mask):
            bad = int((~mask).sum())
            print(f"[WARN] Clipped {bad} invalid edges from adj matrix.")
        return sp.csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=(n, n))

    # Build base (edge) channels and sanitize
    base_adj_lists = [_sanitize_adj(_to_csr(x), num_nodes)
                    for x in [adj_lists1_1, adj_lists1_2, adj_lists2_1, adj_lists2_2]]

    # ---- Motif channels (Option B): add as extra message-passing adjacencies ----
    all_adj_lists = list(base_adj_lists)
    if getattr(args, "use_motif_channels", True) and motif_dict is not None:
        def _scaled(A, s):
            A = _sanitize_adj(A, num_nodes)
            if A.nnz == 0 or s == 1.0:
                return A * s
            A = A.tocsr(copy=True)
            A.data *= s
            return A

        motif_keys = ['out_in_pos', 'out_in_neg']
        motif_dict = {k: v for k, v in motif_dict.items() if k in motif_keys}

        for key in motif_keys:
            A = motif_dict.get(key, sp.csr_matrix((num_nodes, num_nodes)))
            all_adj_lists.append(_scaled(A, float(getattr(args, "motif_gamma", 0.2))))

        # quick debug: confirm max indices are < num_nodes
        for idx, A in enumerate(all_adj_lists):
            if A.nnz:
                coo = A.tocoo()
                mx = max(int(coo.row.max()), int(coo.col.max()))
                if mx >= num_nodes:
                    print(f"[ERR] Channel {idx} has max idx {mx} >= num_nodes {num_nodes}")
        logger.log(f"Motif channels added with gamma={getattr(args, 'motif_gamma', 0.2)} "
                f"(total channels={len(all_adj_lists)})")
    else:
        logger.log("Motif channels disabled; using edge-only message passing.")

    # Aggregators / Encoders (two-hop) over ALL channels (edge + motif)
    aggregator = MeanAggregator if args.agg == 'mean' else AttentionAggregator
    aggs_all = [aggregator(features, NODE_FEAT_SIZE, NODE_FEAT_SIZE, num_nodes) for _ in all_adj_lists]
    enc1 = Encoder(features, NODE_FEAT_SIZE, EMBEDDING_SIZE1, all_adj_lists, aggs_all).to(DEVICES)

    aggs2_all = [aggregator(lambda n: enc1(n), EMBEDDING_SIZE1, EMBEDDING_SIZE1, num_nodes) for _ in all_adj_lists]
    enc2 = Encoder(lambda n: enc1(n), EMBEDDING_SIZE1, EMBEDDING_SIZE1, all_adj_lists, aggs2_all)
    model = SDGNN(enc2).to(DEVICES)

    # Optimizer / schedulers / early stopping
    params = list({id(p): p for p in list(model.parameters()) + list(enc1.parameters())}.values())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if args.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, threshold_mode='rel'
        )

    loss_scheduler = MotifLossScheduler(
        alpha_max=args.alpha, beta_max=args.beta, warmup_epochs=args.warmup_epochs, schedule_type='cosine'
    )
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4, mode='max')

    cache = NeighborhoodCache(max_size=10000) if args.use_cache else None

    logger.log(f"Total initialization time: {time.time() - t0:.2f}s")
    logger.log("Starting training...")

    best_auc, best_epoch = 0.0, 0

    # -----------------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------------
    for epoch in range(EPOCHS + 2):
        epoch_start = time.time()

        # ------------------------ Evaluation ------------------------
        if epoch % args.eval_every == 0:
            model.eval()
            eval_start = time.time()

            # Export embeddings
            all_embedding = np.zeros((NUM_NODE, EMBEDDING_SIZE1))
            for i in range(0, NUM_NODE, BATCH_SIZE):
                with torch.no_grad():
                    embed = model.forward(list(range(i, min(i + BATCH_SIZE, NUM_NODE))))
                    all_embedding[i:min(i + BATCH_SIZE, NUM_NODE)] = embed.data.cpu().numpy()

            fpath = os.path.join(OUTPUT_DIR, f'embedding-{dataset}-{k}-{epoch}.npy')
            np.save(fpath, all_embedding)

            # Diagnostics: signed motif "energy" (sampled)
            try:
                if motif_dict is not None:
                    def _signed_energy(A, sign_label, sample_cap=20000):
                        coo = A.tocoo()
                        if coo.nnz == 0:
                            return float('nan')
                        idx = np.arange(coo.nnz)
                        if coo.nnz > sample_cap:
                            idx = np.random.choice(coo.nnz, sample_cap, replace=False)
                        i_idx = coo.row[idx]
                        j_idx = coo.col[idx]
                        Zi = torch.from_numpy(all_embedding[i_idx]).float()
                        Zj = torch.from_numpy(all_embedding[j_idx]).float()
                        scores = (Zi * Zj).sum(dim=1)
                        s = torch.full_like(scores, fill_value=1.0) if sign_label > 0 else torch.full_like(scores, fill_value=-1.0)
                        return float((F.softplus(-s * scores)).mean().item())

                    e_pos = _signed_energy(motif_dict.get("out_in_pos", sp.csr_matrix((num_nodes, num_nodes))), +1)
                    e_neg = _signed_energy(motif_dict.get("out_in_neg", sp.csr_matrix((num_nodes, num_nodes))), -1)
                    logger.log(f"motif_energy_pos={e_pos:.6f}, motif_energy_neg={e_neg:.6f}")
            except Exception as e:
                logger.log(f"Motif energy diag error: {e}", level='warning')

            # Edge-dot stats
            try:
                def _edges_from_adj(adj_dict):
                    return [(u, v) for u, nbrs in adj_dict.items() for v in nbrs
                            if 0 <= u < NUM_NODE and 0 <= v < NUM_NODE]

                pos_edges_all = _edges_from_adj(adj_lists1)  # positive edges (dict of sets)
                neg_edges_all = _edges_from_adj(adj_lists2)  # negative edges

                import random
                sample_cap = 20000
                pos_edges_sample = random.sample(pos_edges_all, min(sample_cap, len(pos_edges_all))) if pos_edges_all else []
                neg_edges_sample = random.sample(neg_edges_all, min(sample_cap, len(neg_edges_all))) if neg_edges_all else []

                def edge_dot_stats(edges):
                    if not edges:
                        return float('nan'), float('nan')
                    i_idx = np.fromiter((u for u, _ in edges), dtype=np.int64, count=len(edges))
                    j_idx = np.fromiter((v for _, v in edges), dtype=np.int64, count=len(edges))
                    Zi = torch.from_numpy(all_embedding[i_idx]).float()
                    Zj = torch.from_numpy(all_embedding[j_idx]).float()
                    s = (Zi * Zj).sum(dim=1)
                    return s.mean().item(), s.std().item()

                mu_pos, sd_pos = edge_dot_stats(pos_edges_sample)
                mu_neg, sd_neg = edge_dot_stats(neg_edges_sample)
                logger.log(f"dot_pos_mu={mu_pos:.3f}±{sd_pos:.3f}, dot_neg_mu={mu_neg:.3f}±{sd_neg:.3f}")
            except Exception as e:
                logger.log(f"Edge dot diag error: {e}", level='warning')

            # Downstream evaluation (logistic baseline)
            val_auc = None
            try:
                pos_ratio, accuracy, f1_micro, f1_macro, f1_weighted, auc_score = logistic_embedding(
                    k=k, dataset=dataset, epoch=epoch, dirname=OUTPUT_DIR
                )
                val_auc = auc_score
                eval_metrics = {
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'f1_micro': f1_micro,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'auc': auc_score,
                    'eval_time': time.time() - eval_start
                }
                logger.log_epoch(epoch, eval_metrics)

                if auc_score > best_auc:
                    best_auc, best_epoch = auc_score, epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'auc': auc_score,
                    }, os.path.join(OUTPUT_DIR, f'best_model-{dataset}-{k}.pt'))

                if early_stopping(auc_score, epoch):
                    logger.log(f"Early stopping triggered at epoch {epoch}")
                    logger.log(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
                    break
            except Exception as e:
                logger.log(f"Evaluation error: {e}", level='warning')

            if scheduler is not None and val_auc is not None:
                scheduler.step(val_auc)

            model.train()

        # ------------------------ Training ------------------------
        alpha_curr, beta_curr = loss_scheduler.get_weights(epoch)

        train_start = time.time()
        total_losses, edge_losses, motif_losses, triad_losses = [], [], [], []

        nodes_pku = np.random.permutation(NUM_NODE).tolist()
        num_batches = NUM_NODE // BATCH_SIZE

        for batch in range(num_batches):
            optimizer.zero_grad()
            b_index = batch * BATCH_SIZE
            e_index = (batch + 1) * BATCH_SIZE
            nodes_batch = nodes_pku[b_index:e_index]

            loss, batch_loss_dict = model.criterion_motif(
                nodes=nodes_batch,
                motif_dict=motif_dict,           # used by motif/triad losses
                triad_records=triad_records,
                triads_by_node=triads_by_node,
                pos_adj_dict=adj_lists1,
                neg_adj_dict=adj_lists2,
                alpha=alpha_curr,
                beta=beta_curr,
                max_triads_per_batch=1000,
                max_pairs_from_A_mot=10000,
                max_pos_per_u=args.max_pos_per_u,
                max_neg_per_u=args.max_neg_per_u,
                cache=cache
            )
            logger.log(f"Motif batch_loss_dict={batch_loss_dict['motif']:.6f}, "
                       f"Triad loss={batch_loss_dict['triad']:.6f}, "
                       f"Edge loss={batch_loss_dict['edge']:.6f}")
            total_losses.append(loss.item())
            edge_losses.append(batch_loss_dict['edge'])
            motif_losses.append(batch_loss_dict['motif'])
            triad_losses.append(batch_loss_dict['triad'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
            optimizer.step()

        train_time = time.time() - train_start
        epoch_time = time.time() - epoch_start

        # Log training metrics
        train_metrics = {
            'epoch': epoch,
            'loss': np.mean(total_losses) if total_losses else 0.0,
            'edge_loss': np.mean(edge_losses) if edge_losses else 0.0,
            'motif_loss': np.mean(motif_losses) if motif_losses else 0.0,
            'triad_loss': np.mean(triad_losses) if triad_losses else 0.0,
            'alpha': alpha_curr,
            'beta': beta_curr,
            'train_time': train_time,
            'epoch_time': epoch_time,
            'lr': optimizer.param_groups[0]['lr']
        }

        if 0 < train_metrics['motif_loss'] < 1e-4:
            logger.log(f"  Note: Motif loss is {train_metrics['motif_loss']:.8f} (very small, may show as 0.0000)")

        if cache is not None:
            train_metrics.update({f'cache_{k}': v for k, v in cache.get_stats().items()})

        logger.log_epoch(epoch, train_metrics)

        if scheduler is not None and epoch % args.eval_every == 0:
            scheduler.step(np.mean(total_losses) if total_losses else 0.0)

    # -----------------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------------
    logger.log("=" * 80)
    logger.log(f"Training completed!")
    logger.log(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    logger.log(f"Total training time: {time.time() - t0:.2f}s")
    if cache is not None:
        logger.log(f"Cache statistics: {cache.get_stats()}")
    logger.save_metrics()
    logger.plot_metrics()
    logger.log("=" * 80)


def main():
    """Main entry point"""
    logger = TrainingLogger(log_dir=LOG_DIR)
    logger.log("=" * 80)
    logger.log("SDGNN with Motif-based Higher-order Learning (Enhanced)")
    logger.log("=" * 80)
    logger.log(f"NUM_NODE: {NUM_NODE}")
    logger.log(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
    logger.log(f"NODE_FEAT_SIZE: {NODE_FEAT_SIZE}")
    logger.log(f"EMBEDDING_SIZE: {EMBEDDING_SIZE1}")
    logger.log(f"LEARNING_RATE: {LEARNING_RATE}")
    logger.log(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.log(f"EPOCHS: {EPOCHS}")
    logger.log(f"DROPOUT: {DROUPOUT}")
    logger.log(f"AGGREGATOR: {args.agg}")
    logger.log(f"ALPHA: {args.alpha}")
    logger.log(f"BETA: {args.beta}")
    logger.log(f"DEVICE: {DEVICES}")
    logger.log("=" * 80)
    
    dataset = args.dataset
    try:
        run(dataset=dataset, k=1)
    except KeyboardInterrupt:
        logger.log("Training interrupted by user", level='warning')
    except Exception as e:
        logger.log(f"Training failed with error: {e}", level='error')
        import traceback
        logger.log(traceback.format_exc(), level='error')
        raise


if __name__ == "__main__":
    main()