"""
Generate summary report artifacts from ensemble evaluation results.
Outputs:
- results/ensemble_eval/ensemble_summary.md
- results/ensemble_eval/ensemble_summary.pdf
- results/ensemble_eval/model_leaderboard.csv
- results/ensemble_eval/model_comparison.png
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_PATH = Path("results/ensemble_eval/ensemble_results.json")
OUTPUT_DIR = Path("results/ensemble_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    with RESULTS_PATH.open() as f:
        return json.load(f)


def build_leaderboard(data):
    rows = []
    for key, metrics in data["models"].items():
        rows.append(
            {
                "model_key": key,
                "model": metrics["model"],
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "auc_roc_macro": metrics["auc_roc_macro"],
            }
        )
    rows.append(
        {
            "model_key": "ensemble",
            "model": data["ensemble"]["model"],
            "accuracy": data["ensemble"]["accuracy"],
            "precision_macro": data["ensemble"]["precision_macro"],
            "recall_macro": data["ensemble"]["recall_macro"],
            "f1_macro": data["ensemble"]["f1_macro"],
            "auc_roc_macro": data["ensemble"]["auc_roc_macro"],
        }
    )
    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    return rows


def write_leaderboard_csv(rows):
    out_path = OUTPUT_DIR / "model_leaderboard.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_key",
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "auc_roc_macro",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def build_markdown(data, rows):
    lines = []
    lines.append("# Ensemble Evaluation Summary")
    lines.append("")
    lines.append(f"**Samples:** {data['num_samples']}")
    lines.append(f"**Classes:** {', '.join(data['classes'])}")
    lines.append("")
    lines.append("## Leaderboard (sorted by accuracy)")
    lines.append("")
    lines.append("| Rank | Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | AUC-ROC (Macro) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {} | {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |".format(
                idx,
                row["model"],
                row["accuracy"],
                row["precision_macro"],
                row["recall_macro"],
                row["f1_macro"],
                row["auc_roc_macro"],
            )
        )
    lines.append("")
    lines.append("## Best Model Summary")
    best = rows[0]
    lines.append(
        f"**Top performer:** {best['model']} with accuracy {best['accuracy']:.5f} and macro F1 {best['f1_macro']:.5f}."
    )
    lines.append("")
    lines.append("## Per-class metrics (Ensemble)")
    report = data["ensemble"]["classification_report"]
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls in data["classes"]:
        cls_row = report[cls]
        lines.append(
            "| {} | {:.5f} | {:.5f} | {:.5f} | {} |".format(
                cls,
                cls_row["precision"],
                cls_row["recall"],
                cls_row["f1-score"],
                int(cls_row["support"]),
            )
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Confusion matrices: `results/ensemble_eval/confusion_matrix_*.png`")
    lines.append("- ROC curves: `results/ensemble_eval/roc_*.png`")
    lines.append("- Leaderboard CSV: `results/ensemble_eval/model_leaderboard.csv`")
    lines.append("- Comparison plot: `results/ensemble_eval/model_comparison.png`")
    lines.append("- PDF summary: `results/ensemble_eval/ensemble_summary.pdf`")
    lines.append("")
    return "\n".join(lines)


def save_markdown(markdown_text):
    out_path = OUTPUT_DIR / "ensemble_summary.md"
    out_path.write_text(markdown_text)
    return out_path


def plot_comparison(rows):
    labels = [row["model"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]
    f1 = [row["f1_macro"] for row in rows]

    x = range(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, accuracy, width=0.4, label="Accuracy")
    plt.bar([i + 0.4 for i in x], f1, width=0.4, label="Macro F1")
    plt.xticks([i + 0.2 for i in x], labels, rotation=30, ha="right")
    plt.ylim(0.98, 1.0)
    plt.ylabel("Score")
    plt.title("Model Comparison (Accuracy vs Macro F1)")
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def generate_pdf(markdown_text, rows):
    out_path = OUTPUT_DIR / "ensemble_summary.pdf"
    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.5, 0.97, "Ensemble Evaluation Summary", ha="center", fontsize=16)
        fig.text(0.5, 0.94, "4-model ensemble for bone fracture classification", ha="center", fontsize=10)

        # Leaderboard table
        table_data = [[
            str(idx),
            row["model"],
            f"{row['accuracy']:.5f}",
            f"{row['precision_macro']:.5f}",
            f"{row['recall_macro']:.5f}",
            f"{row['f1_macro']:.5f}",
            f"{row['auc_roc_macro']:.5f}",
        ] for idx, row in enumerate(rows, start=1)]

        col_labels = [
            "Rank",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC",
        ]

        ax = fig.add_axes([0.05, 0.45, 0.9, 0.45])
        ax.axis("off")
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        # Notes section
        notes = (
            "Notes:\n"
            "- Metrics computed on full test set (3082 images).\n"
            "- Ensemble uses soft voting over model probabilities.\n"
            "- See results/ensemble_eval for confusion matrices and ROC curves."
        )
        fig.text(0.05, 0.25, notes, fontsize=9, va="top")

        pdf.savefig(fig)
        plt.close(fig)
    return out_path


def main():
    data = load_results()
    rows = build_leaderboard(data)
    write_leaderboard_csv(rows)
    markdown_text = build_markdown(data, rows)
    save_markdown(markdown_text)
    plot_comparison(rows)
    generate_pdf(markdown_text, rows)
    print("✅ Ensemble summary artifacts generated.")


if __name__ == "__main__":
    main()
