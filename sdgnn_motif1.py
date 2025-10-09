# sdgnn_motif.py
# -*- coding: utf-8 -*-
import os, time, math, random, argparse, json
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score

from common import DATASET_NUM_DIC
from fea_extra import FeaExtra
from logistic_function import logistic_embedding
from motif_utils1 import (
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
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--fea_dim', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--k', default=1)
parser.add_argument('--agg', default='attention', choices=['mean','attention'])
parser.add_argument('--bpr_weight', type=float, default=1.0)
parser.add_argument('--sign_bce_weight', type=float, default=1)
parser.add_argument('--eval_every', type=int, default=2)
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--disable_motif', action='store_true',
                    help='Bypass motif propagation and fusion (pure base encoder).')

parser.add_argument('--lr_sched', choices=['plateau','step','cosine','none'], default='plateau')
parser.add_argument('--lr_factor', type=float, default=0.5)       # LR decay factor
parser.add_argument('--lr_patience', type=int, default=3)         # epochs (eval cycles) with no improvement
parser.add_argument('--early_stop_patience', type=int, default=5)
parser.add_argument('--min_delta', type=float, default=1e-4)      # min AUC gain to count as improvement
parser.add_argument('--step_size', type=int, default=10)          # for StepLR
parser.add_argument('--cosine_tmax', type=int, default=50)        # for CosineAnnealingLR
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
LOGDIR = args.log_dir
OUTDIR = f'{LOGDIR}/embeddings'
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
    def __init__(self, features, in_dim, out_dim, num_nodes, dropout=args.dropout, slope=0.1):
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
def _get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

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
    def __init__(self, base_enc: Encoder, motif_prop: MotifPropagate, dim: int, disable_motif: bool = False):
        super().__init__()
        self.enc = base_enc
        self.mprop = motif_prop
        self.disable_motif = disable_motif
        self.fuse = MotifFusion(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )
        self.edge_scorer = nn.Bilinear(dim, dim, 1)

    def forward_batch(self, batch_nodes, Z_motif_cache):
        if isinstance(batch_nodes, (list, np.ndarray)):
            batch_idx = torch.as_tensor(batch_nodes, dtype=torch.long, device=DEV)
        else:
            batch_idx = batch_nodes.to(DEV)

        Z_edge_batch  = self.enc(batch_idx)                 # with grad

        if self.disable_motif:
            Z_final = self.ffn(Z_edge_batch)
            return Z_final, batch_idx

        Z_motif_batch = self.mprop.alpha * Z_motif_cache[batch_idx]  # alpha learns

        Z_fused = self.fuse(Z_edge_batch, Z_motif_batch)
        Z_final = self.ffn(Z_fused)
        return Z_final, batch_idx

    def precompute_motif_features(self, num_nodes):
        with torch.no_grad():
            idx = torch.arange(num_nodes, dtype=torch.long, device=DEV)
            Z_edge_all = self.enc(idx)
            S = torch.sparse.mm(self.mprop.M, Z_edge_all)
        return S

# ----------------- Losses -----------------
def bpr_loss(pos_scores, neg_scores):
    return F.softplus(neg_scores - pos_scores).mean()
def infer_num_nodes_from_edgelists(dataset, k):
    max_id = -1
    for split in ['train', 'test']:
        f = f'./experiment-data/{dataset}-{split}-{k}.edgelist'
        if os.path.exists(f):
            with open(f) as fp:
                for line in fp:
                    u, v, *_ = line.strip().split()
                    u = int(u); v = int(v)
                    if u > max_id: max_id = u
                    if v > max_id: max_id = v
    if max_id < 0:
        raise RuntimeError(f'No edges found for {dataset}, fold {k}. Check your paths.')
    return max_id + 1
# NEW: per-sample weighting (upweight negatives)
def sign_bce_loss(scores, labels, w_pos=1.0, w_neg=1.0):
    """
    scores: [B,1] logits, labels: [B,1] in {0,1}
    """
    bce = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    weights = w_pos * labels + w_neg * (1 - labels)
    return (weights * bce).mean()

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
    """Random negative (fallback for hard-neg)."""
    for _ in range(max_tries):
        v = np.random.randint(0, N)
        if v != u and v not in pos_set:
            return v
    candidates = set(range(N)) - pos_set - {u}
    if candidates:
        return np.random.choice(list(candidates))
    return None

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
        self.metrics['pos_ratio'].append(pos_ratio)
        self.metrics['accuracy'].append(acc)
        self.metrics['f1_macro'].append(f1_2)
        self.metrics['f1_negative'].append(f1_0)
        self.metrics['f1_positive'].append(f1_1)
        self.metrics['auc'].append(auc)
    
    def save(self):
        filepath = os.path.join(self.log_dir, f'metrics_motif_{self.dataset}_k{self.k}.json')
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
    N = infer_num_nodes_from_edgelists(dataset, k)
    print(f"Inferred number of nodes N={N} from edgelists.")
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
    model = SDM_GNN(enc, mprop, EMB, disable_motif=args.disable_motif).to(DEV)

    # Optimizer with gradient clipping
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    if args.lr_sched == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=args.lr_factor, patience=args.lr_patience)
    elif args.lr_sched == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.lr_factor)
    elif args.lr_sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.cosine_tmax)
    else:
        scheduler = None

    # Prepare edges
    pos_edges = [(u, v, 1) for u, nbrs in pos_out.items() for v in nbrs]
    neg_edges = [(u, v, 0) for u, nbrs in neg_out.items() for v in nbrs]
    all_edges = pos_edges + neg_edges

    pos_nbrs = defaultdict(set)
    neg_nbrs = defaultdict(set)
    for (u, v, _) in pos_edges:
        pos_nbrs[u].add(v)
    for (u, v, _) in neg_edges:
        neg_nbrs[u].add(v)


    # Positive sets for negatives / hard-negative filtering
    pos_sets = defaultdict(set)
    for u, v, _ in all_edges:
        pos_sets[u].add(v)

    print(f"\nTraining on {len(all_edges)} edges ({len(pos_edges)} pos, {len(neg_edges)} neg)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Balanced batch sampler (1:1 pos/neg with replacement) ---
    pos_only = [(u,v,1) for (u,v,_) in pos_edges]
    neg_only = [(u,v,0) for (u,v,_) in neg_edges]
    
    def next_batch(B):
        bp = B // 2
        bn = B - bp
        P = random.choices(pos_only, k=min(bp, len(pos_only))) if len(pos_only) >= bp else random.choices(pos_only, k=bp)
        Nn = random.choices(neg_only, k=bn)  # oversample negs if needed
        batch = P + Nn
        random.shuffle(batch)
        return batch
    

    USE_BALANCED_BATCH = True  # you already have a balanced sampler

    if USE_BALANCED_BATCH:
        w_pos, w_neg = 1.0, 1.0   # sampler handles class balance
    else:
        # fallback (your original weighting)
        w_pos = 1.0
        w_neg = len(pos_edges) / max(1, len(neg_edges))



    best_auc = -float('inf')
    best_epoch = -1
    no_improve = 0
    best_model_path = os.path.join(OUTDIR, f'best-model-{dataset}-k{k}.pt')
    best_embed_path = os.path.join(OUTDIR, f'best-embedding-{dataset}-{k}.npy')

    for epoch in range(EPOCHS + 1):
        model.train()
        t0 = time.time()

        # ============ PHASE 1: Pre-compute motif features ============
        Z_motif_cache = None if args.disable_motif else model.precompute_motif_features(N)


        # ============ PHASE 2: Mini-batch training ============
        total_loss = 0.0
        total_bpr = 0.0
        total_bce = 0.0
        num_batches = 0

        num_steps = max(1, len(all_edges) // BATCH)
        for _ in range(num_steps):
            batch = next_batch(BATCH)
            opt.zero_grad()

            # Build batch node set and random negatives for fallback
            batch_nodes = set([u for (u,v,_) in batch] + [v for (_,v,_) in batch])
            rand_neg_by_u = {}
            for (u,v,y) in batch:
                if y == 1:
                    vn = sample_negative(u, N, pos_sets[u])
                    if vn is not None:
                        rand_neg_by_u[u] = vn
                        batch_nodes.add(vn)
            batch_nodes = list(batch_nodes)

            # Forward pass for batch nodes
            Z_batch, batch_idx = model.forward_batch(batch_nodes, Z_motif_cache)
            node_to_idx = {int(n.item()): i for i, n in enumerate(batch_idx)}

            # --- BPR with HARD negatives in-batch ---
            pos_pairs = [(u,v) for (u,v,y) in batch if y==1]
            pos_s, neg_s = [], []

            # Precompute candidate matrix for hard negs
            cand_nodes = batch_nodes
            cand_idx = torch.as_tensor([node_to_idx[n] for n in cand_nodes], device=DEV)
            Z_cand = Z_batch[cand_idx]  # [C, d]

            for (u,v) in pos_pairs:
                ui = node_to_idx[u]
                vi = node_to_idx[v]

                # score(u, all candidates)
                u_rep = Z_batch[ui].unsqueeze(0).expand(Z_cand.size(0), -1)
                cand_scores = model.edge_scorer(u_rep, Z_cand).squeeze(-1)  # [C]

                forbid = torch.zeros_like(cand_scores, dtype=torch.bool)
                forbid[cand_idx == ui] = True
                forbid[cand_idx == vi] = True

                # forbid only positive neighbors (let observed negatives be candidates)
                if u in pos_nbrs:
                    for vv in pos_nbrs[u]:
                        if vv in node_to_idx:
                            forbid[cand_idx == node_to_idx[vv]] = True

                cand_scores = cand_scores.masked_fill(forbid, float('-inf'))

                # pick hardest remaining
                order = torch.argsort(cand_scores, descending=True).tolist()
                vn = None
                for r in order:
                    vv = cand_nodes[r]
                    if vv is not None:
                        vn = vv
                        break
                if vn is None:
                    vn = rand_neg_by_u.get(u, None)
                    if vn is None or vn not in node_to_idx:
                        continue  # no neg found; skip this pair

                vni = node_to_idx[vn]
                s_pos = model.edge_scorer(Z_batch[ui].unsqueeze(0), Z_batch[vi].unsqueeze(0)).squeeze()
                s_neg = model.edge_scorer(Z_batch[ui].unsqueeze(0), Z_batch[vni].unsqueeze(0)).squeeze()
                pos_s.append(s_pos)
                neg_s.append(s_neg)

            if pos_s:
                pos_scores = torch.stack(pos_s)
                neg_scores = torch.stack(neg_s)
                loss_bpr = bpr_loss(pos_scores, neg_scores)
            else:
                loss_bpr = torch.tensor(0.0, device=DEV)

            # --- Sign BCE on observed edges (balanced weights) ---
            u_list = [node_to_idx[u] for (u,_,_) in batch]
            v_list = [node_to_idx[v] for (_,v,_) in batch]
            lbl = torch.tensor([y for (_,_,y) in batch], device=DEV, dtype=torch.float32).view(-1,1)

            u_emb = Z_batch[u_list]
            v_emb = Z_batch[v_list]
            logits = model.edge_scorer(u_emb, v_emb)  # [B,1]
            loss_bce = sign_bce_loss(logits, lbl, w_pos=w_pos, w_neg=w_neg)

            # --- Total loss ---
            loss = args.bpr_weight * loss_bpr + args.sign_bce_weight * loss_bce
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.item())
            total_bpr += float(loss_bpr.item())
            total_bce += float(loss_bce.item())
            num_batches += 1

        # Averages
        avg_loss = total_loss / num_batches
        avg_bpr = total_bpr / num_batches
        avg_bce = total_bce / num_batches
        elapsed = time.time() - t0

        # Log training metrics + alpha
        logger.log_train(epoch, avg_loss, avg_bpr, avg_bce, elapsed)
        alpha_val = model.mprop.alpha.item() if not args.disable_motif else float('nan')
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} (BPR: {avg_bpr:.4f}, BCE: {avg_bce:.4f}) "
            f"| alpha={alpha_val:.3f} | Time: {elapsed:.2f}s")
        
       

        # Evaluation
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                idx = torch.arange(N, dtype=torch.long, device=DEV)
                Z_edge_final = model.enc(idx)
                if args.disable_motif:
                    Z_final = model.ffn(Z_edge_final).cpu().numpy()
                else:
                    Z_motif_base  = model.precompute_motif_features(N)
                    Z_motif_final = model.mprop.alpha * Z_motif_base
                    Z_fused_final = model.fuse(Z_edge_final, Z_motif_final)
                    Z_final       = model.ffn(Z_fused_final).cpu().numpy()

                Z_tensor = torch.from_numpy(Z_final).to(DEV)

                test_pairs, test_labels = [], []
                test_path = f'./experiment-data/{dataset}-test-{k}.edgelist'
                with open(test_path) as fp:
                    for line in fp:
                        i, j, s = line.split()
                        test_pairs.append((int(i), int(j)))
                        test_labels.append(1 if int(s) == 1 else 0)
                test_labels = np.array(test_labels, dtype=int)

                u = torch.as_tensor([i for (i, _) in test_pairs], device=DEV, dtype=torch.long)
                v = torch.as_tensor([j for (_, j) in test_pairs], device=DEV, dtype=torch.long)

                logits = model.edge_scorer(Z_tensor[u], Z_tensor[v]).squeeze(-1)
                prob   = torch.sigmoid(logits).cpu().numpy()

            roc  = roc_auc_score(test_labels, prob)
            aupr = average_precision_score(test_labels, prob)
            print(f"  (internal) ROC-AUC={roc:.4f}, PR-AUC={aupr:.4f}")
            # Save embeddings
            np.save(os.path.join(OUTDIR, f'embedding-{dataset}-{k}-{epoch}.npy'), Z_final)

            # Evaluate (ensure your logistic probe uses Hadamard/L1/L2 + scaling)
            pos_ratio, acc, f1_0, f1_1, f1_2, auc = logistic_embedding(
                k=k, dataset=dataset, epoch=epoch, dirname=OUTDIR
            )

            logger.log_eval(pos_ratio, acc, f1_0, f1_1, f1_2, auc)
            logger.print_summary(epoch)

            if scheduler:
                if args.lr_sched == 'plateau':
                    scheduler.step(auc)   # monitors validation AUC
                else:
                    scheduler.step()      # step/cosine advance each epoch

            # --- Early stopping on AUC ---
            if auc > best_auc + args.min_delta:
                best_auc = auc
                best_epoch = epoch
                no_improve = 0
                # Save best weights + embeddings
                torch.save(
                    {'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'auc': auc,
                    'alpha': float(model.mprop.alpha.data.item()),
                    'lr': _get_lr(opt)},
                    best_model_path
                )
                # We already have Z_final here
                np.save(best_embed_path, Z_final)
                print(f"[BEST] AUC improved to {best_auc:.4f} @ epoch {best_epoch}. Saved best model/embeddings.")
            else:
                no_improve += 1
                print(f"No AUC improvement for {no_improve} evals (best {best_auc:.4f} @ {best_epoch}).")
                if no_improve >= args.early_stop_patience:
                    print(f"Early stopping triggered (patience={args.early_stop_patience}). "
                        f"Best AUC={best_auc:.4f} at epoch {best_epoch}.")
                    break
        else:
            if scheduler and args.lr_sched in ('step', 'cosine'):
                scheduler.step()
            print(f"  (lr={_get_lr(opt):.2e})")

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
