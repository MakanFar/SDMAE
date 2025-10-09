
# -*- coding: utf-8 -*-
"""
prepare_folds_stratified.py
----------------------------
Usage:
    python prepare_folds_stratified.py --dataset bitcoin_alpha --num_folds 5 --seed 42

Description:
    Reads a simplified 3-column edgelist (u v s) from ./experiment-data/{dataset}.edgelist
    and creates stratified K-fold splits preserving positive/negative ratios.
    Writes:
        ./experiment-data/{dataset}-train-{k}.edgelist
        ./experiment-data/{dataset}-test-{k}.edgelist
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

def load_edges_labels(path):
    edges = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            u, v, s = parts[:3]
            edges.append(line)
            labels.append(1 if int(float(s)) > 0 else 0)
    return np.array(edges), np.array(labels)

def main():
    ap = argparse.ArgumentParser(description='Prepare stratified K-fold splits for signed network edgelist (u v s).')
    ap.add_argument('--dataset', required=True, help='Dataset name; expects ./experiment-data/{dataset}.edgelist')
    ap.add_argument('--dataset_path', default='./experiment-data', help='Folder containing the edgelist')
    ap.add_argument('--num_folds', type=int, default=5, help='Number of folds')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    args = ap.parse_args()

    os.makedirs(args.dataset_path, exist_ok=True)
    full_path = os.path.join(args.dataset_path, f"{args.dataset}.edgelist")
    if not os.path.exists(full_path):
        raise FileNotFoundError(full_path)

    edges, labels = load_edges_labels(full_path)
    print(f"Loaded {len(edges)} edges from {full_path}")
    print(f"Positive: {labels.sum()}, Negative: {len(labels)-labels.sum()}\n")

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for fold, (tr, te) in enumerate(skf.split(edges, labels), start=1):
        tr_path = os.path.join(args.dataset_path, f"{args.dataset}-train-{fold}.edgelist")
        te_path = os.path.join(args.dataset_path, f"{args.dataset}-test-{fold}.edgelist")
        np.savetxt(tr_path, edges[tr], fmt='%s')
        np.savetxt(te_path, edges[te], fmt='%s')

        # Stats
        pos_tr = sum(1 for e in edges[tr] if e.strip().endswith(' 1'))
        neg_tr = len(tr) - pos_tr
        pos_te = sum(1 for e in edges[te] if e.strip().endswith(' 1'))
        neg_te = len(te) - pos_te
        print(f"Fold {fold}: Train={len(tr)} ({pos_tr}+/{neg_tr}-), Test={len(te)} ({pos_te}+/{neg_te}-) -> {tr_path}, {te_path}")

    print("\n✅ Stratified folds prepared successfully.")

if __name__ == '__main__':
    main()
