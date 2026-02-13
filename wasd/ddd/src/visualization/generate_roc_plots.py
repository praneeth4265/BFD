"""
Regenerate ROC curve plots from existing ensemble_results.json with distinct colors.
Uses saved ROC curve data so it avoids re-running model inference.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_PATH = Path("reports/ensemble_eval/data/ensemble_results.json")
OUTPUT_DIR = Path("reports/ensemble_eval/plots")

COLOR_CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c']


def plot_roc_curves(roc_data, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for (class_name, curve), color in zip(roc_data.items(), COLOR_CYCLE):
        ax.plot(
            curve['fpr'],
            curve['tpr'],
            label=class_name,
            color=color,
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_roc_zoom(roc_data, title, output_path, x_max=0.05, y_min=0.95):
    fig, ax = plt.subplots(figsize=(7, 6))
    for (class_name, curve), color in zip(roc_data.items(), COLOR_CYCLE):
        ax.plot(
            curve['fpr'],
            curve['tpr'],
            label=class_name,
            color=color,
            linewidth=2,
        )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, 1.0)
    ax.set_title(f"{title} (Zoomed)")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    with RESULTS_PATH.open() as f:
        data = json.load(f)

    models = data['models']
    for key, metrics in models.items():
        output_path = OUTPUT_DIR / f"roc_{key}.png"
        zoom_path = OUTPUT_DIR / f"roc_{key}_zoom.png"
        plot_roc_curves(metrics['roc_curves'], f"{metrics['model']} ROC Curves", output_path)
        plot_roc_zoom(metrics['roc_curves'], f"{metrics['model']} ROC Curves", zoom_path)

    ensemble_path = OUTPUT_DIR / "roc_ensemble.png"
    ensemble_zoom_path = OUTPUT_DIR / "roc_ensemble_zoom.png"
    plot_roc_curves(data['ensemble']['roc_curves'], "Ensemble ROC Curves", ensemble_path)
    plot_roc_zoom(data['ensemble']['roc_curves'], "Ensemble ROC Curves", ensemble_zoom_path)

    print("✅ ROC plots regenerated with distinct colors.")


if __name__ == "__main__":
    main()
