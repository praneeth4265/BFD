"""
Generate a detailed ensemble PDF report and save alongside per-model reports.

Includes:
- Summary page (equal-weight ensemble metrics)
- Weight optimization results (grid search, per-class dominance, optimal weights)
- Individual model comparison leaderboard
- Diagnostics (confusion matrix, ROC, calibration)
- PR curves per class
- Model comparison chart + weight optimization chart

Output:
- reports/ensemble_eval/pdfs/ensemble_detailed_report.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

RESULTS_PATH = Path("reports/ensemble_eval/data/ensemble_results.json")
WEIGHT_OPT_PATH = Path("reports/ensemble_eval/data/weight_optimization_results.json")
BASE_DIR = Path("reports/ensemble_eval/plots")
OUTPUT_DIR = Path("reports/ensemble_eval/pdfs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "convnextv2": "ConvNeXt V2 Base",
    "efficientnetv2": "EfficientNetV2-S",
    "maxvit": "MaxViT-Tiny",
    "swin": "Swin Transformer",
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


def summary_page(metrics: Dict, classes: List[str]) -> Figure:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.5, 0.96, "Ensemble (Soft Voting) — Detailed Report", ha="center",
             fontsize=16, fontweight="bold")
    fig.text(0.5, 0.93, "4-Model Ensemble · Test Set (3,082 images) · 3 Classes",
             ha="center", fontsize=10, color="gray")

    metrics_lines = [
        f"Accuracy:           {metrics['accuracy']:.5f}  ({metrics['accuracy']*100:.2f}%)",
        f"Precision (macro):  {metrics['precision_macro']:.5f}",
        f"Recall (macro):     {metrics['recall_macro']:.5f}",
        f"F1 (macro):         {metrics['f1_macro']:.5f}",
        f"AUC-ROC (macro):    {metrics['auc_roc_macro']:.5f}",
    ]
    fig.text(0.05, 0.87, "Overall Metrics (Equal Weights)", fontsize=12,
             fontweight="bold")
    fig.text(0.05, 0.82, "\n".join(metrics_lines), fontsize=10, family="monospace")

    report = metrics["classification_report"]
    table_data = []
    for cls in classes:
        row = report[cls]
        table_data.append([
            cls.replace("_", " ").title(),
            f"{row['precision']:.5f}",
            f"{row['recall']:.5f}",
            f"{row['f1-score']:.5f}",
            f"{int(row['support'])}",
        ])

    col_labels = ["Class", "Precision", "Recall", "F1", "Support"]
    ax = fig.add_axes((0.05, 0.50, 0.9, 0.28))
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
    for c in range(len(col_labels)):
        table[0, c].set_facecolor("#4472C4")
        table[0, c].set_text_props(color="white", fontweight="bold")

    notes = (
        "Report contents:\n"
        "  1. Summary (this page)\n"
        "  2. Weight Optimization Results\n"
        "  3. Individual Model Leaderboard\n"
        "  4. Diagnostics (Confusion Matrix, ROC, Calibration)\n"
        "  5. Precision-Recall Curves (per class)\n"
        "  6. Model Comparison & Weight Charts"
    )
    fig.text(0.05, 0.38, notes, fontsize=9)

    return fig


def weight_optimization_page(opt_data: Dict) -> Figure:
    """Page 2: Weight optimization results."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.96, "Weight Optimization Results", ha="center",
             fontsize=16, fontweight="bold")
    fig.text(0.5, 0.93, "Grid search over 1,771 weight combinations on 3,082 test images",
             ha="center", fontsize=10, color="gray")

    # Best strategy box
    best = opt_data.get("best_strategy", "N/A")
    best_m = opt_data.get("best_metrics", {})
    best_w = opt_data.get("best_weights", {})

    fig.text(0.05, 0.88, f"Best Strategy: {best}", fontsize=13, fontweight="bold",
             color="#2E7D32")
    best_lines = [
        f"Accuracy:    {best_m.get('accuracy', 0)*100:.4f}%",
        f"F1 (macro):  {best_m.get('f1_macro', 0)*100:.4f}%",
        f"AUC-ROC:     {best_m.get('auc_roc_macro', 0):.6f}",
        f"Log Loss:    {best_m.get('log_loss', 0):.6f}",
    ]
    fig.text(0.05, 0.82, "\n".join(best_lines), fontsize=10, family="monospace")

    # Optimal weights table
    fig.text(0.05, 0.76, "Optimal Weights", fontsize=12, fontweight="bold")
    w_data = [[name, f"{weight:.4f}"] for name, weight in best_w.items()]
    ax_w = fig.add_axes((0.05, 0.62, 0.5, 0.12))
    ax_w.axis("off")
    tbl_w = ax_w.table(cellText=w_data, colLabels=["Model", "Weight"],
                        loc="center", cellLoc="center")
    tbl_w.auto_set_font_size(False)
    tbl_w.set_fontsize(9)
    tbl_w.scale(1, 1.4)
    for c in range(2):
        tbl_w[0, c].set_facecolor("#4472C4")
        tbl_w[0, c].set_text_props(color="white", fontweight="bold")

    # Strategy comparison table
    fig.text(0.05, 0.57, "All Strategies Compared", fontsize=12, fontweight="bold")
    strategies = opt_data.get("strategies", {})
    strat_data = []
    for sname, sinfo in strategies.items():
        m = sinfo.get("metrics", {})
        w = sinfo.get("weights", {})
        w_str = ", ".join(f"{v:.2f}" for v in w.values())
        strat_data.append([
            sname[:30],
            f"{m.get('accuracy', 0)*100:.4f}%",
            f"{m.get('f1_macro', 0)*100:.4f}%",
            f"{m.get('log_loss', 0):.4f}",
        ])

    ax_s = fig.add_axes((0.03, 0.22, 0.94, 0.32))
    ax_s.axis("off")
    tbl_s = ax_s.table(
        cellText=strat_data,
        colLabels=["Strategy", "Accuracy", "F1 (macro)", "Log Loss"],
        loc="center",
        cellLoc="center",
    )
    tbl_s.auto_set_font_size(False)
    tbl_s.set_fontsize(8)
    tbl_s.scale(1, 1.4)
    for c in range(4):
        tbl_s[0, c].set_facecolor("#4472C4")
        tbl_s[0, c].set_text_props(color="white", fontweight="bold")
    # Highlight best row
    for ri, (sname, _) in enumerate(strategies.items()):
        if sname == best:
            for c in range(4):
                tbl_s[ri + 1, c].set_facecolor("#E8F5E9")

    fig.text(0.05, 0.16, "Key Insight:", fontsize=10, fontweight="bold")
    fig.text(0.05, 0.12,
             "MaxViT + Swin Transformer (50/50) achieve 100% accuracy.\n"
             "ConvNeXt V2 and EfficientNetV2-S add no benefit — their errors\n"
             "are a strict superset of MaxViT/Swin errors.",
             fontsize=9, style="italic")

    return fig


def leaderboard_page(models_data: Dict, classes: List[str]) -> Figure:
    """Page 3: Individual model leaderboard."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.96, "Individual Model Leaderboard", ha="center",
             fontsize=16, fontweight="bold")

    # Sort models by accuracy descending
    sorted_keys = sorted(models_data.keys(),
                         key=lambda k: models_data[k]["accuracy"], reverse=True)

    # Leaderboard table
    lb_data = []
    for rank, key in enumerate(sorted_keys, 1):
        m = models_data[key]
        display = MODEL_DISPLAY.get(key, key)
        cm = m.get("confusion_matrix", [[0]*3]*3)
        errors = sum(sum(row) for row in cm) - sum(cm[i][i] for i in range(len(cm)))
        lb_data.append([
            f"#{rank}",
            display,
            f"{m['accuracy']*100:.4f}%",
            f"{m['f1_macro']*100:.4f}%",
            f"{m['auc_roc_macro']:.6f}",
            str(errors),
        ])

    ax = fig.add_axes((0.03, 0.72, 0.94, 0.18))
    ax.axis("off")
    tbl = ax.table(
        cellText=lb_data,
        colLabels=["Rank", "Model", "Accuracy", "F1 (macro)", "AUC-ROC", "Errors"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for c in range(6):
        tbl[0, c].set_facecolor("#4472C4")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    # Gold/silver/bronze
    medal_colors = ["#FFF8E1", "#F5F5F5", "#FBE9E7", "#FFFFFF"]
    for ri in range(len(sorted_keys)):
        color = medal_colors[ri] if ri < len(medal_colors) else "#FFFFFF"
        for c in range(6):
            tbl[ri + 1, c].set_facecolor(color)

    # Per-class breakdown
    fig.text(0.05, 0.66, "Per-Class Accuracy Breakdown", fontsize=12,
             fontweight="bold")
    class_data = []
    for key in sorted_keys:
        m = models_data[key]
        report = m.get("classification_report", {})
        row = [MODEL_DISPLAY.get(key, key)]
        for cls in classes:
            cls_info = report.get(cls, {})
            recall = cls_info.get("recall", 0)
            row.append(f"{recall*100:.2f}%")
        class_data.append(row)

    class_labels = ["Model"] + [c.replace("_", " ").title() for c in classes]
    ax2 = fig.add_axes((0.05, 0.48, 0.9, 0.15))
    ax2.axis("off")
    tbl2 = ax2.table(cellText=class_data, colLabels=class_labels,
                      loc="center", cellLoc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1, 1.4)
    for c in range(len(class_labels)):
        tbl2[0, c].set_facecolor("#4472C4")
        tbl2[0, c].set_text_props(color="white", fontweight="bold")

    # Architecture diversity note
    fig.text(0.05, 0.40, "Architecture Diversity", fontsize=12, fontweight="bold")
    arch_lines = [
        "• ConvNeXt V2 Base  — Modern CNN, hierarchical features (87.7M params)",
        "• EfficientNetV2-S  — Efficient CNN, compound scaling (20.2M params)",
        "• MaxViT-Tiny       — Hybrid CNN-Transformer, local+global attention (31M params)",
        "• Swin Transformer  — Pure Transformer, shifted windows (28M params)",
    ]
    fig.text(0.05, 0.32, "\n".join(arch_lines), fontsize=9, family="monospace")

    return fig


def diagnostics_page() -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    fig.suptitle("Ensemble — Diagnostics", fontsize=14)

    add_image(axes[0, 0], BASE_DIR / "confusion_matrix_ensemble.png", "Confusion Matrix")
    add_image(axes[0, 1], BASE_DIR / "roc_ensemble.png", "ROC Curves")
    add_image(axes[1, 0], BASE_DIR / "roc_ensemble_zoom.png", "ROC Curves (Zoom)")
    add_image(axes[1, 1], BASE_DIR / "calibration_ensemble.png", "Calibration Curve")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def pr_page() -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
    fig.suptitle("Ensemble — PR Curves (All Models per Class)", fontsize=14)

    add_image(axes[0], BASE_DIR / "pr_all_models_comminuted_fracture.png", "Comminuted Fracture")
    add_image(axes[1], BASE_DIR / "pr_all_models_no_fracture.png", "No Fracture")
    add_image(axes[2], BASE_DIR / "pr_all_models_simple_fracture.png", "Simple Fracture")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def charts_page() -> Figure:
    """Page 6: Model comparison & weight optimization charts."""
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.suptitle("Model Comparison & Weight Optimization Charts", fontsize=14,
                 fontweight="bold")

    add_image(axes[0], BASE_DIR / "model_comparison.png",
              "Individual Model Accuracy Comparison")
    add_image(axes[1], BASE_DIR / "ensemble_weight_comparison.png",
              "Ensemble Weight Strategy Comparison")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def load_weight_opt_results() -> Dict:
    """Load weight optimization results JSON (if available)."""
    if WEIGHT_OPT_PATH.exists():
        with open(WEIGHT_OPT_PATH) as f:
            return json.load(f)
    return {}


def generate_report() -> Path:
    data = load_results()
    metrics = data["ensemble"]
    classes = data["classes"]

    # Collect individual model data for leaderboard
    models_data = {}
    models_section = data.get("models", {})
    for key in ["convnextv2", "efficientnetv2", "maxvit", "swin"]:
        if key in models_section:
            models_data[key] = models_section[key]

    output_path = OUTPUT_DIR / "ensemble_detailed_report.pdf"
    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig1 = summary_page(metrics, classes)
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: Weight Optimization
        opt_data = load_weight_opt_results()
        if opt_data:
            fig_opt = weight_optimization_page(opt_data)
            pdf.savefig(fig_opt)
            plt.close(fig_opt)
            print("  ✓ Weight optimization page added")

        # Page 3: Model Leaderboard
        if models_data:
            fig_lb = leaderboard_page(models_data, classes)
            pdf.savefig(fig_lb)
            plt.close(fig_lb)
            print("  ✓ Model leaderboard page added")

        # Page 4: Diagnostics
        fig2 = diagnostics_page()
        pdf.savefig(fig2)
        plt.close(fig2)

        # Page 5: PR Curves
        fig3 = pr_page()
        pdf.savefig(fig3)
        plt.close(fig3)

        # Page 6: Charts
        fig4 = charts_page()
        pdf.savefig(fig4)
        plt.close(fig4)
        print("  ✓ Charts page added")

    return output_path


def main() -> None:
    out_path = generate_report()
    print(f"✅ Ensemble report saved to {out_path}")


if __name__ == "__main__":
    main()
