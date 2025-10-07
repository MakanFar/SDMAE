#!/bin/bash
#SBATCH --job-name=sdgnn_train
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Load environment
module load conda
module load cuda/12.6.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grn

echo "Running on node $(hostname)"
echo "Checking GPU access via PyTorch..."
python -c "import torch; print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU');"

# Run your code
python sdgnn_motif.py \
  --devices cuda:0 \
  --dataset bitcoin_alpha \
  --epochs 100 \
  --agg attention \
  --k 1 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --bpr_weight 0.5 \
  --sign_bce_weight 1.0 \
  --eval_every 2 
