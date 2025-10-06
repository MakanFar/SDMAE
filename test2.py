import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open('./logs/metrics_bitcoin_alpha_k1.json') as f:
    metrics = json.load(f)

# Extract eval epochs (only epochs where evaluation happened)
eval_epochs = [e for i, e in enumerate(metrics['epoch']) 
               if i < len(metrics['accuracy']) and metrics['accuracy'][i] is not None]

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Training Metrics - Bitcoin Alpha', fontsize=16, fontweight='bold')

# ==================== Row 1: Training Losses ====================
# Total Loss
axes[0, 0].plot(metrics['epoch'], metrics['loss_total'], 'b-', linewidth=2, label='Total Loss')
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Loss', fontsize=11)
axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# BPR Loss
axes[0, 1].plot(metrics['epoch'], metrics['loss_bpr'], 'r-', linewidth=2, label='BPR Loss')
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Loss', fontsize=11)
axes[0, 1].set_title('BPR (Ranking) Loss', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# BCE Loss
axes[0, 2].plot(metrics['epoch'], metrics['loss_bce'], 'g-', linewidth=2, label='BCE Loss')
axes[0, 2].set_xlabel('Epoch', fontsize=11)
axes[0, 2].set_ylabel('Loss', fontsize=11)
axes[0, 2].set_title('BCE (Sign Prediction) Loss', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# ==================== Row 2: Evaluation Metrics ====================
# Accuracy
axes[1, 0].plot(eval_epochs, metrics['accuracy'], 'b-o', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('Accuracy', fontsize=11)
axes[1, 0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1])

# AUC
axes[1, 1].plot(eval_epochs, metrics['auc'], 'r-o', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('AUC', fontsize=11)
axes[1, 1].set_title('AUC-ROC Score', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1])

# F1 Scores
axes[1, 2].plot(eval_epochs, metrics['f1_macro'], 'purple', linewidth=2, marker='o', 
                markersize=4, label='F1-Macro')
axes[1, 2].plot(eval_epochs, metrics['f1_positive'], 'g', linewidth=2, marker='s', 
                markersize=4, label='F1-Positive')
axes[1, 2].plot(eval_epochs, metrics['f1_negative'], 'orange', linewidth=2, marker='^', 
                markersize=4, label='F1-Negative')
axes[1, 2].set_xlabel('Epoch', fontsize=11)
axes[1, 2].set_ylabel('F1 Score', fontsize=11)
axes[1, 2].set_title('F1 Scores', fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend(fontsize=9)
axes[1, 2].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("Saved training_curves.png")

# ==================== Additional: Combined Loss Plot ====================
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(metrics['epoch'], metrics['loss_total'], 'b-', linewidth=2.5, label='Total Loss', alpha=0.8)
ax.plot(metrics['epoch'], metrics['loss_bpr'], 'r--', linewidth=2, label='BPR Loss', alpha=0.7)
ax.plot(metrics['epoch'], metrics['loss_bce'], 'g--', linewidth=2, label='BCE Loss', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('Training Loss Components', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_components.png', dpi=300, bbox_inches='tight')
print("Saved loss_components.png")

# ==================== Print Summary Statistics ====================
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Total Epochs: {len(metrics['epoch'])}")
print(f"Evaluation Points: {len(eval_epochs)}")
print(f"\nFinal Metrics (Epoch {eval_epochs[-1]}):")
print(f"  Accuracy:      {metrics['accuracy'][-1]:.4f}")
print(f"  AUC:           {metrics['auc'][-1]:.4f}")
print(f"  F1-Macro:      {metrics['f1_macro'][-1]:.4f}")
print(f"  F1-Positive:   {metrics['f1_positive'][-1]:.4f}")
print(f"  F1-Negative:   {metrics['f1_negative'][-1]:.4f}")
print(f"\nFinal Losses (Epoch {metrics['epoch'][-1]}):")
print(f"  Total Loss:    {metrics['loss_total'][-1]:.4f}")
print(f"  BPR Loss:      {metrics['loss_bpr'][-1]:.4f}")
print(f"  BCE Loss:      {metrics['loss_bce'][-1]:.4f}")
print(f"\nBest Performance:")
best_acc_idx = np.argmax(metrics['accuracy'])
best_auc_idx = np.argmax(metrics['auc'])
print(f"  Best Accuracy: {metrics['accuracy'][best_acc_idx]:.4f} (Epoch {eval_epochs[best_acc_idx]})")
print(f"  Best AUC:      {metrics['auc'][best_auc_idx]:.4f} (Epoch {eval_epochs[best_auc_idx]})")
print(f"\nTotal Training Time: {sum(metrics['time']):.2f}s ({sum(metrics['time'])/60:.2f}min)")
print(f"Avg Time per Epoch: {np.mean(metrics['time']):.2f}s")
print("="*60)

plt.show()