# motif_utils.py
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch

###############################################################################
# Signed–Directed motif adjacency construction (16 triad types)
# Uses FeaExtra's pos_in/pos_out/neg_in/neg_out lists to assemble M_{u,v}.
###############################################################################

def _to_csr_from_triplets(triplets, shape):
    if not triplets:
        return sp.csr_matrix(shape, dtype=np.float32)
    rows, cols, vals = zip(*triplets)
    return sp.csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)

def _row_normalize_csr(mat: sp.csr_matrix) -> sp.csr_matrix:
    # Asymmetric (row) normalize to preserve directionality
    mat = mat.tocsr(copy=True)
    row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
    row_sum[row_sum == 0.0] = 1.0
    inv = 1.0 / row_sum
    Dinv = sp.diags(inv, format="csr")
    return Dinv.dot(mat)

def scipy_csr_to_torch_sparse(mat: sp.csr_matrix, device: torch.device):
    mat = mat.tocoo()
    indices = torch.tensor([mat.row, mat.col], dtype=torch.long, device=device)
    values = torch.tensor(mat.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape), device=device).coalesce()

def build_signed_directed_motif_adj(pos_in, pos_out, neg_in, neg_out, num_nodes):
    """
    Returns dict[str] -> scipy CSR, each N x N:
      'ff_++','ff_+-','ff_-+','ff_--'      # u->k, k->v (feed-forward chain)
      'coout_++','coout_+-','coout_-+','coout_--'  # u->k and v->k (co-outgoing)
      'chain_++','chain_+-','chain_-+','chain_--'  # k->u->v (relay)
      'coin_++','coin_+-','coin_-+','coin_--'      # k->u and k->v (co-incoming)
    Counts how often (u,v) co-occur in a triad instance of each signed/directed type.
    """
    N = num_nodes
    M = {name: [] for name in [
        'ff_++','ff_+-','ff_-+','ff_--',
        'coout_++','coout_+-','coout_-+','coout_--',
        'chain_++','chain_+-','chain_-+','chain_--',
        'coin_++','coin_+-','coin_-+','coin_--'
    ]}

    # Feed-forward (u -> k -> v)
    for u in range(N):
        # u --(+)-> k
        for k in pos_out.get(u, []):
            # k --(+)-> v
            for v in pos_out.get(k, []):
                M['ff_++'].append((u, v, 1.0))
            # k --(-)-> v
            for v in neg_out.get(k, []):
                M['ff_+-'].append((u, v, 1.0))
        # u --(-)-> k
        for k in neg_out.get(u, []):
            # k --(+)-> v
            for v in pos_out.get(k, []):
                M['ff_-+'].append((u, v, 1.0))
            # k --(-)-> v
            for v in neg_out.get(k, []):
                M['ff_--'].append((u, v, 1.0))

    # Co-outgoing siblings (u -> k) and (v -> k)  (fill at (u,v), keep direction u->v)
    # Here we "credit" u->v if both point to same k with specified signs.
    for k in range(N):
        # + +
        pv = pos_in.get(k, [])
        for i in range(len(pv)):
            ui = pv[i]
            for j in range(len(pv)):
                vj = pv[j]
                M['coout_++'].append((ui, vj, 1.0))
        # + -
        nv = neg_in.get(k, [])
        for ui in pv:
            for vj in nv:
                M['coout_+-'].append((ui, vj, 1.0))
        # - +
        for ui in nv:
            for vj in pv:
                M['coout_-+'].append((ui, vj, 1.0))
        # - -
        for i in range(len(nv)):
            ui = nv[i]
            for j in range(len(nv)):
                vj = nv[j]
                M['coout_--'].append((ui, vj, 1.0))

    # Chain (k -> u -> v): credit (u,v) if k->u and u->v with specific signs
    for u in range(N):
        # k --(+)-> u
        for k in pos_in.get(u, []):
            # u --(+)-> v
            for v in pos_out.get(u, []):
                M['chain_++'].append((u, v, 1.0))
            # u --(-)-> v
            for v in neg_out.get(u, []):
                M['chain_+-'].append((u, v, 1.0))
        # k --(-)-> u
        for k in neg_in.get(u, []):
            # u --(+)-> v
            for v in pos_out.get(u, []):
                M['chain_-+'].append((u, v, 1.0))
            # u --(-)-> v
            for v in neg_out.get(u, []):
                M['chain_--'].append((u, v, 1.0))

    # Co-incoming parents (k -> u) and (k -> v): credit (u,v)
    for k in range(N):
        # + +
        pv = pos_out.get(k, [])
        for i in range(len(pv)):
            ui = pv[i]
            for j in range(len(pv)):
                vj = pv[j]
                M['coin_++'].append((ui, vj, 1.0))
        # + -
        nv = neg_out.get(k, [])
        for ui in pv:
            for vj in nv:
                M['coin_+-'].append((ui, vj, 1.0))
        # - +
        for ui in nv:
            for vj in pv:
                M['coin_-+'].append((ui, vj, 1.0))
        # - -
        for i in range(len(nv)):
            ui = nv[i]
            for j in range(len(nv)):
                vj = nv[j]
                M['coin_--'].append((ui, vj, 1.0))

    # Build CSR and normalize rows
    out = {}
    for name, triplets in M.items():
        csr = _to_csr_from_triplets(triplets, (N, N))
        out[name] = _row_normalize_csr(csr)
    return out

def combine_motif_set(motif_dict: dict, weights: dict = None) -> sp.csr_matrix:
    """
    Weighted sum of motif matrices: M_sum = (Σ w_i * M_i) / (Σ w_i)
    weights: dict[name] -> float, default 1.0
    """
    mats = []
    wsum = 0.0
    for name, mat in motif_dict.items():
        w = 1.0 if (weights is None or name not in weights) else float(weights[name])
        if w <= 0.0:
            continue
        mats.append(mat.multiply(w))
        wsum += w
    if not mats:
        raise ValueError("No motif matrices to combine.")
    M = mats[0]
    for i in range(1, len(mats)):
        M = M + mats[i]
    if wsum > 0:
        M = M * (1.0 / wsum)
    return M

def csr_power_iteration_max_singular(mat: sp.csr_matrix, iters=20):
    # crude σ_max estimate (L2 norm via power iteration)
    # operate on mat.T @ mat for largest eigenvalue
    x = np.random.randn(mat.shape[1]).astype(np.float32)
    x /= np.linalg.norm(x) + 1e-8
    for _ in range(iters):
        y = mat.dot(x)
        z = mat.T.dot(y)
        nrm = np.linalg.norm(z) + 1e-8
        x = z / nrm
    sigma = np.sqrt(np.linalg.norm(mat.dot(x))**2)
    return float(sigma) if sigma > 0 else 1.0

def spectrally_normalize(Msum: sp.csr_matrix) -> sp.csr_matrix:
    sigma = csr_power_iteration_max_singular(Msum, iters=20)
    if sigma <= 0:
        sigma = 1.0
    return Msum * (1.0 / sigma)
