#!/usr/bin/env bash
set -euo pipefail
PY="/c/Users/Makan/anaconda3/envs/torchcuda/python.exe"
[ -x "$PY" ] || { echo "Not found: $PY"; exit 1; }
"$PY" -V


# Common args (add lr/wd/etc if you want)
BASE="--dataset bitcoin_alpha --devices cuda --epochs 100 --batch_size 500 --eval_every 2"

# A0: Base encoder only (no motif)
for k in 1 2 3 4 5; do
  outdir="logs/bitcoin_alpha/abl/A0_base_nomotif/k$k"
  mkdir -p "$outdir"
  $PY sdgnn_motif.py $BASE --k $k --agg attention --disable_motif \
    --log_dir "$outdir" | tee "$outdir/train.log"
done

# A1: Full model (motif on)
for k in 1 2 3 4 5; do
  outdir="logs/bitcoin_alpha/abl/A1_full/k$k"
  mkdir -p "$outdir"
  $PY sdgnn_motif.py $BASE --k $k --agg attention \
    --log_dir "$outdir" | tee "$outdir/train.log"
done
