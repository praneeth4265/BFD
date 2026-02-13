"""
Generate detailed per-model PDF reports from ensemble evaluation results.

Outputs (under reports/ensemble_eval/pdfs/):
- convnextv2_detailed_report.pdf
- efficientnetv2_detailed_report.pdf
- maxvit_detailed_report.pdf
- swin_detailed_report.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_PATH = Path("reports/ensemble_eval/data/ensemble_results.json")
OUTPUT_DIR = Path("reports/ensemble_eval/pdfs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "convnextv2": "ConvNeXt V2 Base",
    "efficientnetv2": "EfficientNetV2-S",
    "maxvit": "MaxViT-Tiny",
    "swin": "Swin Transformer (Tiny)",
}


def load_results() -> Dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results: {RESULTS_PATH}")
    with RESULTS_PATH.open() as f:
        return json.load(f)


def add_image(ax, image_path: Path, title: str) -> None:
    ax.axis("off")
    if image_path.exists():
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
    else:
        ax.text(0.5, 0.5, f"Missing: {image_path.name}", ha="center", va="center")
        ax.set_title(title, fontsize=10)


def summary_page(model_key: str, model_metrics: Dict, classes: List[str]) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    display_name = MODEL_DISPLAY.get(model_key, model_metrics.get("model", model_key))

    fig.text(0.5, 0.96, f"{display_name} — Detailed Report", ha="center", fontsize=16)
    fig.text(0.5, 0.93, "Test set evaluation", ha="center", fontsize=10)

    metrics_lines = [
        f"Accuracy: {model_metrics['accuracy']:.5f}",
        f"Precision (macro): {model_metrics['precision_macro']:.5f}",
        f"Recall (macro): {model_metrics['recall_macro']:.5f}",
        f"F1 (macro): {model_metrics['f1_macro']:.5f}",
        f"AUC-ROC (macro): {model_metrics['auc_roc_macro']:.5f}",
    ]
    fig.text(0.05, 0.86, "Overall Metrics", fontsize=12)
    fig.text(0.05, 0.82, "\n".join(metrics_lines), fontsize=10)

    report = model_metrics["classification_report"]
    table_data = []
    for cls in classes:
        row = report[cls]
        table_data.append([
            cls,
            f"{row['precision']:.5f}",
            f"{row['recall']:.5f}",
            f"{row['f1-score']:.5f}",
            f"{int(row['support'])}",
        ])

    col_labels = ["Class", "Precision", "Recall", "F1", "Support"]
    ax = fig.add_axes([0.05, 0.45, 0.9, 0.32])
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    notes = (
        "Artifacts included on next page:\n"
        "- Confusion matrix\n"
        "- ROC curve (full + zoom)\n"
        "- Calibration curve"
    )
    fig.text(0.05, 0.36, notes, fontsize=9)

    return fig


def plots_page(model_key: str, display_name: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    fig.suptitle(f"{display_name} — Diagnostics", fontsize=14)

    base_dir = Path("reports/ensemble_eval/plots")
    add_image(axes[0, 0], base_dir / f"confusion_matrix_{model_key}.png", "Confusion Matrix")
    add_image(axes[0, 1], base_dir / f"roc_{model_key}.png", "ROC Curves")
    add_image(axes[1, 0], base_dir / f"roc_{model_key}_zoom.png", "ROC Curves (Zoom)")
    add_image(axes[1, 1], base_dir / f"calibration_{model_key}.png", "Calibration Curve")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def generate_reports() -> None:
    data = load_results()
    classes = data["classes"]

    for model_key, metrics in data["models"].items():
        display_name = MODEL_DISPLAY.get(model_key, metrics.get("model", model_key))
        output_path = OUTPUT_DIR / f"{model_key}_detailed_report.pdf"

        with PdfPages(output_path) as pdf:
            fig1 = summary_page(model_key, metrics, classes)
            pdf.savefig(fig1)
            plt.close(fig1)

            fig2 = plots_page(model_key, display_name)
            pdf.savefig(fig2)
            plt.close(fig2)

    print(f"✅ Per-model reports saved to {OUTPUT_DIR}")


def main() -> None:
    generate_reports()


if __name__ == "__main__":
    main()
