import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open('./logs/bitcoin_alpha_mc/metrics_motif_bitcoin_alpha_k1.json') as f:
    metrics = json.load(f)

# Extract eval epochs (only epochs where evaluation happened)
eval_epochs = [e for i, e in enumerate(metrics['epoch']) 
               if i < len(metrics['accuracy']) and metrics['accuracy'][i] is not None]

# Helper function for dynamic y-axis
def dynamic_ylim(ax, data, margin=0.02):
    """Set y-limits based on first recorded metric value and data range."""
    valid = [d for d in data if d is not None]
    if len(valid) == 0:
        return
    ymin = valid[0]
    ymax = max(valid)
    yrange = ymax - ymin if ymax != ymin else ymax * 0.05
    ax.set_ylim([ymin - margin * yrange, ymax + margin * yrange])

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

# BCE Loss
axes[0, 2].plot(metrics['epoch'], metrics['loss_bce'], 'g-', linewidth=2, label='BCE Loss')
axes[0, 2].set_xlabel('Epoch', fontsize=11)
axes[0, 2].set_ylabel('Loss', fontsize=11)
axes[0, 2].set_title('BCE (Sign Prediction) Loss', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# ==================== Row 2: Evaluation Metrics ====================
# Accuracy (dynamic y-axis)
axes[1, 0].plot(eval_epochs, metrics['accuracy'], 'b-o', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('Accuracy', fontsize=11)
axes[1, 0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
dynamic_ylim(axes[1, 0], metrics['accuracy'])

# AUC-ROC (dynamic y-axis)
axes[1, 1].plot(eval_epochs, metrics['auc'], 'r-o', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('AUC-ROC', fontsize=11)
axes[1, 1].set_title('AUC-ROC Score', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
dynamic_ylim(axes[1, 1], metrics['auc'])

# AUPR (dynamic y-axis, new plot)
if 'aupr' in metrics:
    aupr_data = metrics['aupr']
else:
    aupr_data = [None] * len(eval_epochs)
    print("⚠️  Warning: 'aupr' key not found in metrics file.")

axes[1, 2].plot(eval_epochs, aupr_data, 'm-o', linewidth=2, markersize=4)
axes[1, 2].set_xlabel('Epoch', fontsize=11)
axes[1, 2].set_ylabel('AUPR', fontsize=11)
axes[1, 2].set_title('Area Under Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
dynamic_ylim(axes[1, 2], aupr_data)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Saved training_curves.png")

# ==================== Combined Loss Plot ====================
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(metrics['epoch'], metrics['loss_total'], 'b-', linewidth=2.5, label='Total Loss', alpha=0.8)
ax.plot(metrics['epoch'], metrics['loss_bce'], 'g--', linewidth=2, label='BCE Loss', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('Training Loss Components', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_components.png', dpi=300, bbox_inches='tight')
print("✅ Saved loss_components.png")

# ==================== Print Summary ====================
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Total Epochs: {len(metrics['epoch'])}")
print(f"Evaluation Points: {len(eval_epochs)}")

if 'aupr' in metrics:
    print(f"\nFinal Metrics (Epoch {eval_epochs[-1]}):")
    print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
    print(f"  AUC-ROC:  {metrics['auc'][-1]:.4f}")
    print(f"  AUPR:     {metrics['aupr'][-1]:.4f}")
else:
    print("\nFinal Metrics (no AUPR available):")
    print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
    print(f"  AUC-ROC:  {metrics['auc'][-1]:.4f}")

print(f"\nFinal Losses (Epoch {metrics['epoch'][-1]}):")
print(f"  Total Loss: {metrics['loss_total'][-1]:.4f}")
print(f"  BCE Loss:   {metrics['loss_bce'][-1]:.4f}")

best_acc_idx = np.argmax(metrics['accuracy'])
best_auc_idx = np.argmax(metrics['auc'])
print(f"\nBest Accuracy: {metrics['accuracy'][best_acc_idx]:.4f} (Epoch {eval_epochs[best_acc_idx]})")
print(f"Best AUC-ROC:  {metrics['auc'][best_auc_idx]:.4f} (Epoch {eval_epochs[best_auc_idx]})")

if 'aupr' in metrics:
    best_aupr_idx = np.argmax(metrics['aupr'])
    print(f"Best AUPR:     {metrics['aupr'][best_aupr_idx]:.4f} (Epoch {eval_epochs[best_aupr_idx]})")

print(f"\nTotal Training Time: {sum(metrics['time']):.2f}s ({sum(metrics['time'])/60:.2f}min)")
print(f"Avg Time per Epoch: {np.mean(metrics['time']):.2f}s")
print("="*60)

plt.show()
