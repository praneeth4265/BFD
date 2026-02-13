"""
Generate Precision-Recall and calibration curves from saved probabilities.
Requires: reports/ensemble_eval/data/ensemble_probabilities.npz
Outputs PR and calibration plots to project_reports/ensemble_eval/
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]
COLOR_CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c']
MODEL_COLORS = {
    "convnextv2": "#1f77b4",
    "efficientnetv2": "#ff7f0e",
    "maxvit": "#2ca02c",
    "swin": "#d62728",
    "ensemble": "#9467bd",
}
MODEL_LINESTYLES = {
    "convnextv2": "-",
    "efficientnetv2": "-",
    "maxvit": "-",
    "swin": "-",
    "ensemble": "-",
}
MODEL_MARKERS = {
    "convnextv2": "o",
    "efficientnetv2": "s",
    "maxvit": "^",
    "swin": "D",
    "ensemble": "P",
}
MODEL_LINEWIDTHS = {
    "convnextv2": 3.0,
    "efficientnetv2": 2.5,
    "maxvit": 2.0,
    "swin": 2.5,
    "ensemble": 3.0,
}
MODEL_ZORDERS = {
    "convnextv2": 2,
    "efficientnetv2": 3,
    "maxvit": 4,
    "swin": 5,
    "ensemble": 6,
}

RESULTS_DIR = Path("reports/ensemble_eval/plots")
PROB_PATH = RESULTS_DIR / "ensemble_probabilities.npz"


def plot_pr_all_models_class(class_idx, class_name, y_true, probs_by_model, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y_true_bin = (y_true == class_idx).astype(int)
    for model_key, y_prob in probs_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:, class_idx])
        ap = average_precision_score(y_true_bin, y_prob[:, class_idx])
        label = f"{model_key} (AP={ap:.4f})"
        ls = MODEL_LINESTYLES[model_key]
        zord = MODEL_ZORDERS[model_key]
        lw = MODEL_LINEWIDTHS[model_key]
        if zoom:
            # In zoom view, use markers at intervals so overlapping lines are all visible
            marker = MODEL_MARKERS[model_key]
            # Subsample for marker placement to avoid clutter
            n_pts = len(recall)
            markevery = max(1, n_pts // 20)
            ax.plot(
                recall, precision,
                color=MODEL_COLORS[model_key], label=label,
                linewidth=lw, linestyle=ls, marker=marker,
                markersize=7, markevery=markevery,
                zorder=zord, alpha=0.9,
            )
        else:
            ax.plot(
                recall, precision,
                color=MODEL_COLORS[model_key], label=label,
                linewidth=lw, linestyle=ls, zorder=zord,
            )
    if zoom:
        ax.set_xlim(0.95, 1.005)
        ax.set_ylim(0.95, 1.005)
        ax.set_title(f"All Models PR (Precision-Recall) — {class_name} (Zoomed)")
    else:
        ax.set_title(f"All Models PR (Precision-Recall) — {class_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.grid(alpha=0.2)
    ax.margins(x=0.02, y=0.02)
    fig.subplots_adjust(right=0.76)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(model_name, y_true, y_prob, output_path):
    plt.figure(figsize=(7, 6))
    for idx, cls in enumerate(CLASSES):
        y_true_bin = (y_true == idx).astype(int)
        prob_true, prob_pred = calibration_curve(
            y_true_bin, y_prob[:, idx], n_bins=10, strategy="quantile"
        )
        plt.plot(prob_pred, prob_true, marker="o", color=COLOR_CYCLE[idx], label=cls)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title(f"{model_name} Calibration Curves")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    if not PROB_PATH.exists():
        raise FileNotFoundError(f"Missing probabilities: {PROB_PATH}")

    data = np.load(PROB_PATH)
    y_true = data["y_true"]

    model_keys = [
        "convnextv2",
        "efficientnetv2",
        "maxvit",
        "swin",
        "ensemble",
    ]

    name_map = {
        "convnextv2": "ConvNeXt V2",
        "efficientnetv2": "EfficientNetV2-S",
        "maxvit": "MaxViT-Tiny",
        "swin": "Swin Transformer",
        "ensemble": "Ensemble",
    }

    probs_by_model = {}
    for key in model_keys:
        if key == "ensemble":
            y_prob = data["probs_ensemble"]
        else:
            y_prob = data[f"probs_{key}"]
        probs_by_model[key] = y_prob

    # PR plots generated in combined per-class section below

    print("✅ PR plots saved to project_reports/ensemble_eval")

    # Combined per-class PR plots (all models on one chart)
    for idx, cls in enumerate(CLASSES):
        combined_path = RESULTS_DIR / f"pr_all_models_{cls}.png"
        combined_zoom_path = RESULTS_DIR / f"pr_all_models_{cls}_zoom.png"
        plot_pr_all_models_class(idx, cls, y_true, probs_by_model, combined_path, zoom=False)
        plot_pr_all_models_class(idx, cls, y_true, probs_by_model, combined_zoom_path, zoom=True)


if __name__ == "__main__":
    main()
