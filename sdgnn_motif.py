# sdgnn_motif.py
# -*- coding: utf-8 -*-
import os, time, math, random, argparse, json
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATASET_NUM_DIC
from fea_extra import FeaExtra
from logistic_function import logistic_embedding
from motif_utils import (
    build_signed_directed_motif_adj,
    combine_motif_set, spectrally_normalize, scipy_csr_to_torch_sparse
)

# ------------------------- Args -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--dataset', default='bitcoin_alpha')
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--fea_dim', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--k', default=1)
parser.add_argument('--agg', default='attention', choices=['mean','attention'])
parser.add_argument('--bpr_weight', type=float, default=1.0)
parser.add_argument('--sign_bce_weight', type=float, default=0.5)
parser.add_argument('--eval_every', type=int, default=2)
parser.add_argument('--log_dir', type=str, default='./logs')
args = parser.parse_args()

# --------------------- Globals --------------------------
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
DEV = torch.device(args.devices)
EPOCHS = args.epochs
BATCH = args.batch_size
EMB = args.dim
FEA = args.fea_dim
LR = args.lr
WD = args.weight_decay
K = args.k
DATASET = args.dataset
OUTDIR = f'./embeddings/sdgnn-motif'
LOGDIR = args.log_dir
os.makedirs('embeddings', exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

# ------------------ SDGNN (edge encoder) ----------------
class Encoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_mats, aggs):
        super().__init__()
        self.features = features
        self.feat_dim = feature_dim
        self.adj_mats = adj_mats
        self.aggs = nn.ModuleList(aggs)
        self.proj = nn.Sequential(
            nn.Linear((len(adj_mats)+1) * feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, embed_dim)
        )
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, nodes):
        if isinstance(nodes, list) or isinstance(nodes, np.ndarray):
            nidx = torch.as_tensor(nodes, dtype=torch.long, device=DEV)
        else:
            nidx = nodes.to(DEV)
        neigh_feats = [agg(nidx, adj, i) for i, (adj, agg) in enumerate(zip(self.adj_mats, self.aggs))]
        self_feats = self.features(nidx)
        X = torch.cat([self_feats] + neigh_feats, dim=1)
        return self.proj(X)

class AttentionAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, num_nodes, dropout=0.0, slope=0.1):
        super().__init__()
        self.features = features; self.in_dim = in_dim; self.out_dim = out_dim
        self.a = nn.Parameter(torch.empty(out_dim*2, 1)); nn.init.kaiming_normal_(self.a)
        self.out_linear = nn.Linear(in_dim, out_dim)
        self.unique_nodes = np.zeros(num_nodes, dtype=np.int32)
        self.dropout = nn.Dropout(dropout); self.leaky = nn.LeakyReLU(slope)

    def forward(self, nodes, adj_csr, ind):
        if torch.is_tensor(nodes):
            nodes = nodes.detach().cpu().numpy()
        elif isinstance(nodes, list):
            nodes = np.array(nodes, dtype=np.int64)
        rows, cols = _csr_rows_cols(adj_csr, nodes)
        unique = np.unique(np.concatenate([nodes, rows, cols]))
        self.unique_nodes[unique] = np.arange(len(unique))
        r = np.vectorize(lambda x: self.unique_nodes[x])(rows)
        c = np.vectorize(lambda x: self.unique_nodes[x])(cols)
        nu = torch.as_tensor(unique, dtype=torch.long, device=DEV)
        new_emb = self.out_linear(self.features(nu))
        eij = torch.cat([new_emb[r,:], new_emb[c,:]], dim=1)
        e = torch.exp(self.leaky(eij @ self.a)).squeeze(-1)
        idx = torch.stack([torch.as_tensor(r, device=DEV), torch.as_tensor(c, device=DEV)], dim=0)
        mat = torch.sparse_coo_tensor(idx, e, (len(unique), len(unique)), device=DEV).coalesce()
        one = torch.ones((len(unique),1), device=DEV)
        denom = torch.sparse.mm(mat, one)
        out = torch.sparse.mm(mat, new_emb) / (denom + 1e-8)
        ret = out[self.unique_nodes[nodes], :]
        return ret

def _csr_rows_cols(adj_csr, nodes):
    if torch.is_tensor(nodes):
        nodes = nodes.detach().cpu().numpy()
    elif isinstance(nodes, list):
        nodes = np.array(nodes, dtype=np.int64)

    rows, cols = [], []
    for u in nodes:
        start, end = adj_csr.indptr[u], adj_csr.indptr[u + 1]
        cols_u = adj_csr.indices[start:end]
        if len(cols_u) == 0:
            continue
        rows_u = np.full(cols_u.shape, u, dtype=np.int64)
        rows.append(rows_u)
        cols.append(cols_u)

    if rows:
        return np.concatenate(rows), np.concatenate(cols)
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

class MeanAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, num_nodes):
        super().__init__()
        self.features = features; self.in_dim=in_dim; self.out_dim=out_dim
        self.out_linear = nn.Sequential(nn.Linear(in_dim,out_dim), nn.Tanh(), nn.Linear(out_dim,out_dim))
        self.unique_nodes = np.zeros(num_nodes, dtype=np.int32)

    def forward(self, nodes, adj_csr, ind):
        if torch.is_tensor(nodes):
            nodes = nodes.detach().cpu().numpy()
        elif isinstance(nodes, list):
            nodes = np.array(nodes, dtype=np.int64)
        rows, cols = _csr_rows_cols(adj_csr, nodes)
        unique = np.unique(np.concatenate([nodes, rows, cols]))
        self.unique_nodes[unique] = np.arange(len(unique))
        r = np.vectorize(lambda x: self.unique_nodes[x])(rows)
        c = np.vectorize(lambda x: self.unique_nodes[x])(cols)
        nu = torch.as_tensor(unique, dtype=torch.long, device=DEV)
        new_emb = self.out_linear(self.features(nu))
        values = torch.ones(len(r), device=DEV)
        idx = torch.stack([torch.as_tensor(r, device=DEV), torch.as_tensor(c, device=DEV)], dim=0)
        mat = torch.sparse_coo_tensor(idx, values, (len(unique), len(unique)), device=DEV).coalesce()
        one = torch.ones((len(unique),1), device=DEV)
        denom = torch.sparse.mm(mat, one)
        denom[denom==0] = 1.0
        out = torch.sparse.mm(mat, new_emb) / denom
        return out[self.unique_nodes[nodes], :]

# ----------------- Motif propagation + fusion -----------------
class MotifPropagate(nn.Module):
    def __init__(self, Mnorm_torch):
        super().__init__()
        self.M = Mnorm_torch
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, Z):
        return self.alpha * torch.sparse.mm(self.M, Z)

class MotifFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim*2, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, Z1, Z2):
        h = torch.relu(self.fc(torch.cat([Z1, Z2], dim=1)))
        beta = torch.softmax(self.gate(h), dim=1)
        return beta * Z1 + (1.0 - beta) * Z2

# ----------------- Final model: SDM-GNN -----------------
class SDM_GNN(nn.Module):
    def __init__(self, base_enc: Encoder, motif_prop: MotifPropagate, dim: int):
        super().__init__()
        self.enc = base_enc
        self.mprop = motif_prop
        self.fuse = MotifFusion(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )

    def forward_batch(self, batch_nodes, Z_motif_cache):
        """Forward pass for a batch of nodes using cached motif features."""
        if isinstance(batch_nodes, list) or isinstance(batch_nodes, np.ndarray):
            batch_idx = torch.as_tensor(batch_nodes, dtype=torch.long, device=DEV)
        else:
            batch_idx = batch_nodes.to(DEV)
        
        # Edge encoder (with gradients)
        Z_edge_batch = self.enc(batch_idx)
        
        # Get cached motif features
        Z_motif_batch = Z_motif_cache[batch_idx]
        
        # Fuse and final FFN
        Z_fused = self.fuse(Z_edge_batch, Z_motif_batch)
        Z_final = self.ffn(Z_fused)
        
        return Z_final, batch_idx

    def precompute_motif_features(self, num_nodes):
        """Pre-compute motif-enhanced features for all nodes (no grad)."""
        with torch.no_grad():
            idx = torch.arange(num_nodes, dtype=torch.long, device=DEV)
            Z_edge_all = self.enc(idx).detach()
            Z_motif_all = self.mprop(Z_edge_all)
        return Z_motif_all

# ----------------- Losses -----------------
def bpr_loss(pos_scores, neg_scores):
    return F.softplus(neg_scores - pos_scores).mean()

def sign_bce_loss(scores, labels,pos_weight):
    return F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)

# ----------------- Data loading & helpers -----------------
def load_edgelist_signed(filename):
    pos_out = defaultdict(list); pos_in = defaultdict(list)
    neg_out = defaultdict(list); neg_in = defaultdict(list)
    with open(filename) as fp:
        for line in fp:
            u, v, s = line.strip().split()
            u = int(u); v = int(v); s = int(s)
            if s == 1:
                pos_out[u].append(v); pos_in[v].append(u)
            else:
                neg_out[u].append(v); neg_in[v].append(u)
    return pos_in, pos_out, neg_in, neg_out

def adj_from_dict(adj_dicts, N):
    edges = []
    for D in adj_dicts:
        for u, nbrs in D.items():
            for v in nbrs:
                edges.append((u, v))
    if not edges:
        return sp.csr_matrix((N, N), dtype=np.float32)
    e = np.array(edges, dtype=np.int64)
    return sp.csr_matrix((np.ones(len(e), dtype=np.float32), (e[:,0], e[:,1])), shape=(N, N))

def sample_negative(u, N, pos_set, max_tries=100):
    """Sample negative with max retries to prevent infinite loops."""
    for _ in range(max_tries):
        v = np.random.randint(0, N)
        if v != u and v not in pos_set:
            return v
    # Fallback: find any valid negative
    candidates = set(range(N)) - pos_set - {u}
    if candidates:
        return np.random.choice(list(candidates))
    return None  # No valid negatives

# ----------------- Metrics Logger -----------------
class MetricsLogger:
    def __init__(self, log_dir, dataset, k):
        self.log_dir = log_dir
        self.dataset = dataset
        self.k = k
        self.metrics = {
            'epoch': [],
            'loss_total': [],
            'loss_bpr': [],
            'loss_bce': [],
            'time': [],
            'pos_ratio': [],
            'accuracy': [],
            'f1_macro': [],
            'f1_negative': [],
            'f1_positive': [],
            'auc': []
        }
    
    def log_train(self, epoch, loss_total, loss_bpr, loss_bce, elapsed):
        self.metrics['epoch'].append(epoch)
        self.metrics['loss_total'].append(loss_total)
        self.metrics['loss_bpr'].append(loss_bpr)
        self.metrics['loss_bce'].append(loss_bce)
        self.metrics['time'].append(elapsed)
    
    def log_eval(self, pos_ratio, acc, f1_0, f1_1, f1_2, auc):
        # Store most recent eval metrics (align with last logged epoch)
        self.metrics['pos_ratio'].append(pos_ratio)
        self.metrics['accuracy'].append(acc)
        self.metrics['f1_macro'].append(f1_2)
        self.metrics['f1_negative'].append(f1_0)
        self.metrics['f1_positive'].append(f1_1)
        self.metrics['auc'].append(auc)
    
    def save(self):
        filepath = os.path.join(self.log_dir, f'metrics_{self.dataset}_k{self.k}.json')
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def print_summary(self, epoch):
        print(f"\nEpoch {epoch:03d} Summary:")
        print(f"  Loss Total: {self.metrics['loss_total'][-1]:.4f}")
        print(f"  Loss BPR:   {self.metrics['loss_bpr'][-1]:.4f}")
        print(f"  Loss BCE:   {self.metrics['loss_bce'][-1]:.4f}")
        print(f"  Time:       {self.metrics['time'][-1]:.2f}s")
        if self.metrics['accuracy']:
            print(f"  Accuracy:   {self.metrics['accuracy'][-1]:.4f}")
            print(f"  AUC:        {self.metrics['auc'][-1]:.4f}")
            print(f"  F1-Macro:   {self.metrics['f1_macro'][-1]:.4f}")

# ----------------- Training -----------------
def run(dataset, k):
    N = DATASET_NUM_DIC[dataset] + 3
    path = f'./experiment-data/{dataset}-train-{k}.edgelist'
    pos_in, pos_out, neg_in, neg_out = load_edgelist_signed(path)

    # Initialize logger
    logger = MetricsLogger(LOGDIR, dataset, k)

    # Base feature table
    features = nn.Embedding(N, FEA, device=DEV)
    features.weight.requires_grad_(True)

    # Four adjacencies
    A_list = [pos_out, pos_in, neg_out, neg_in]
    A_csr = [adj_from_dict([A_list[i]], N) for i in range(4)]

    # Aggregators
    Agg = AttentionAggregator if args.agg == 'attention' else MeanAggregator
    aggs = [Agg(features, FEA, FEA, N) for _ in range(4)]

    # Build encoder
    enc = Encoder(features, FEA, EMB, A_csr, aggs).to(DEV)

    # Motif matrices
    motif_csr_dict = build_signed_directed_motif_adj(
        pos_in=pos_in, pos_out=pos_out, neg_in=neg_in, neg_out=neg_out, num_nodes=N
    )
    Msum = combine_motif_set(motif_csr_dict, weights=None)
    Mnorm = spectrally_normalize(Msum)
    M_torch = scipy_csr_to_torch_sparse(Mnorm, DEV)
    mprop = MotifPropagate(M_torch).to(DEV)

    # Model
    model = SDM_GNN(enc, mprop, EMB).to(DEV)

    # Optimizer with gradient clipping
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # Prepare edges
    pos_edges = [(u, v, 1) for u, nbrs in pos_out.items() for v in nbrs]
    neg_edges = [(u, v, 0) for u, nbrs in neg_out.items() for v in nbrs]
    all_edges = pos_edges + neg_edges

    num_pos = len(pos_edges)
    num_neg = len(neg_edges)
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=DEV, dtype=torch.float32)

    # Positive sets for negative sampling
    pos_sets = defaultdict(set)
    for u, v, _ in all_edges:
        pos_sets[u].add(v)

    print(f"\nTraining on {len(all_edges)} edges ({len(pos_edges)} pos, {len(neg_edges)} neg)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(EPOCHS + 1):
        model.train()
        np.random.shuffle(all_edges)
        
        t0 = time.time()
        
        # ============ PHASE 1: Pre-compute motif features ============
        Z_motif_cache = model.precompute_motif_features(N)
        
        # ============ PHASE 2: Mini-batch training ============
        total_loss = 0.0
        total_bpr = 0.0
        total_bce = 0.0
        num_batches = 0
        
        for i in range(0, len(all_edges), BATCH):
            batch = all_edges[i:i+BATCH]
            opt.zero_grad()
            
            # Get unique nodes in batch
            batch_nodes = set([u for (u,v,y) in batch] + [v for (u,v,y) in batch])
            negatives = []
            for (u,v,y) in batch:
                if y==1:
                    vn = sample_negative(u, N, pos_sets[u])
                    if vn is not None:
                        negatives.append((u, vn))
                        batch_nodes.add(vn)
            batch_nodes = list(batch_nodes)
            
            # Forward pass for batch
            Z_batch, batch_idx = model.forward_batch(batch_nodes, Z_motif_cache)
            
            # Create mapping: node_id -> position in Z_batch
            node_to_idx = {int(n.item()): i for i, n in enumerate(batch_idx)}
            
                        # use indices (all should exist now)
            pos_pairs = [(u,v) for (u,v,y) in batch if y==1]
            pos_s = [(Z_batch[node_to_idx[u]] * Z_batch[node_to_idx[v]]).sum()
                    for (u,v) in pos_pairs]
            neg_s = [(Z_batch[node_to_idx[u]] * Z_batch[node_to_idx[vn]]).sum()
                    for (u,vn) in negatives]
            
            if pos_s:
                pos_scores = torch.stack(pos_s)
                neg_scores = torch.stack(neg_s)
                loss_bpr = bpr_loss(pos_scores, neg_scores)
            else:
                loss_bpr = torch.tensor(0.0, device=DEV)
            
            # Sign BCE Loss
            u_list = [node_to_idx[u] for (u,v,y) in batch]
            v_list = [node_to_idx[v] for (u,v,y) in batch]
            lbl = torch.tensor([y for (_,_,y) in batch], device=DEV, dtype=torch.float32)
            
            u_emb = Z_batch[u_list]
            v_emb = Z_batch[v_list]
            logits = (u_emb * v_emb).sum(dim=1, keepdim=True)
            loss_bce = sign_bce_loss(logits, lbl.view(-1,1),pos_weight)
            
            # Total loss
            loss = args.bpr_weight * loss_bpr + args.sign_bce_weight * loss_bce
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            
            total_loss += loss.item()
            total_bpr += loss_bpr.item()
            total_bce += loss_bce.item()
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_bpr = total_bpr / num_batches
        avg_bce = total_bce / num_batches
        elapsed = time.time() - t0
        
        # Log training metrics
        logger.log_train(epoch, avg_loss, avg_bpr, avg_bce, elapsed)
        
        # Evaluation
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Final embeddings for evaluation
                Z_motif_final = model.precompute_motif_features(N)
                idx = torch.arange(N, dtype=torch.long, device=DEV)
                Z_edge_final = model.enc(idx)
                Z_fused_final = model.fuse(Z_edge_final, Z_motif_final)
                Z_final = model.ffn(Z_fused_final).detach().cpu().numpy()
            
            # Save embeddings
            np.save(os.path.join(OUTDIR, f'embedding-{dataset}-{k}-{epoch}.npy'), Z_final)
            
            # Evaluate
            pos_ratio, acc, f1_0, f1_1, f1_2, auc = logistic_embedding(
                k=k, dataset=dataset, epoch=epoch, dirname=OUTDIR
            )
            
            # Log eval metrics
            logger.log_eval(pos_ratio, acc, f1_0, f1_1, f1_2, auc)
            logger.print_summary(epoch)
        else:
            print(f'Epoch {epoch:03d} | Loss: {avg_loss:.4f} (BPR: {avg_bpr:.4f}, BCE: {avg_bce:.4f}) | Time: {elapsed:.2f}s')
    
    # Save metrics to file
    logger.save()

def main():
    global dataset
    dataset = DATASET
    print(f'Device: {DEV}')
    print(f'Dataset: {dataset}')
    run(dataset=dataset, k=K)

if __name__ == "__main__":
    main()