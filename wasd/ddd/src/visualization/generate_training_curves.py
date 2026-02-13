"""
Generate training curve comparison plots from model result histories.
Outputs to project_reports/ensemble_eval.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("reports/ensemble_eval/plots")
MODELS = {
    "EfficientNetV2-S": Path("models/results/efficientnetv2_3class_augmented_results.json"),
    "MaxViT-Tiny": Path("models/results/maxvit_3class_augmented_results.json"),
    "Swin Transformer": Path("models/results/swin_3class_augmented_results.json"),
}


def load_history(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data.get("history")


def plot_metric(metric_name, title, output_path):
    plt.figure(figsize=(7, 5))
    for model_name, path in MODELS.items():
        history = load_history(path)
        if not history or metric_name not in history:
            continue
        plt.plot(history[metric_name], label=model_name, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric("train_loss", "Training Loss", RESULTS_DIR / "training_loss_comparison.png")
    plot_metric("val_loss", "Validation Loss", RESULTS_DIR / "val_loss_comparison.png")
    plot_metric("train_acc", "Training Accuracy", RESULTS_DIR / "training_acc_comparison.png")
    plot_metric("val_acc", "Validation Accuracy", RESULTS_DIR / "val_acc_comparison.png")
    print("✅ Training curve comparison plots saved to project_reports/ensemble_eval")


if __name__ == "__main__":
    main()
