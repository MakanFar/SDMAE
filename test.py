import json
import matplotlib.pyplot as plt
import os

def plot_training_curves(data):
    """
    Plot training performance and loss components from a training log dictionary.
    """

    # ---- Plot Loss Breakdown ----
    plt.figure(figsize=(12, 6))
    if "loss" in data:
        plt.plot(range(len(data["loss"])), data["loss"], label="Total Loss", color="red", alpha=0.7)
    if "edge_loss" in data:
        plt.plot(range(len(data["edge_loss"])), data["edge_loss"], label="Edge Loss", color="blue")
    if "motif_loss" in data:
        plt.plot(range(len(data["motif_loss"])), data["motif_loss"], label="Motif Loss", color="green")
    if "triad_loss" in data:
        plt.plot(range(len(data["triad_loss"])), data["triad_loss"], label="Triad Loss", color="purple")

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Loss Components Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Plot Performance Metrics ----
    plt.figure(figsize=(12, 6))
    if "accuracy" in data:
        plt.plot(range(len(data["accuracy"])), data["accuracy"], label="Accuracy", marker="o")
    if "f1_micro" in data:
        plt.plot(range(len(data["f1_micro"])), data["f1_micro"], label="F1 Micro", marker="x")
    if "f1_macro" in data:
        plt.plot(range(len(data["f1_macro"])), data["f1_macro"], label="F1 Macro", marker="s")
    if "f1_weighted" in data:
        plt.plot(range(len(data["f1_weighted"])), data["f1_weighted"], label="F1 Weighted", marker="^")
    if "auc" in data:
        plt.plot(range(len(data["auc"])), data["auc"], label="AUC", marker="d")

    plt.xlabel("Evaluation Step")
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Plot Cache Efficiency ----
    if "cache_hit_rate" in data:
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(data["cache_hit_rate"])), data["cache_hit_rate"], label="Cache Hit Rate", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Hit Rate")
        plt.title("Cache Hit Rate Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ---- Plot Training Time ----
    if "train_time" in data and "epoch_time" in data:
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(data["train_time"])), data["train_time"], label="Train Time", color="teal")
        plt.plot(range(len(data["epoch_time"])), data["epoch_time"], label="Epoch Time", color="brown")
        plt.xlabel("Epoch")
        plt.ylabel("Seconds")
        plt.title("Training and Epoch Times")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    log_file = os.path.join("logs\sdgnn-attention", "metrics.json")

    with open(log_file, "r") as f:
        metrics = json.load(f)

    plot_training_curves(metrics)
