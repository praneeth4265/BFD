"""
Generate error analysis artifacts from saved ensemble probabilities.

Outputs (under reports/ensemble_eval/error_analysis/):
- misclassified_ensemble.csv
- model_disagreements.csv
- error_summary.json
- misclassified_grid_<class>.png

Requires:
- reports/ensemble_eval/data/ensemble_probabilities.npz
- test images at datasets/augmented/test/<class>/*.png|*.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]
MODEL_KEYS = ["convnextv2", "efficientnetv2", "maxvit", "swin", "ensemble"]


def list_test_images(data_dir: Path) -> List[Path]:
    images: List[Path] = []
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_path in class_dir.glob(ext):
                images.append(img_path)
    return images


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_misclassified_grid(
    images: List[Path],
    rows: List[Dict[str, object]],
    class_name: str,
    output_path: Path,
    max_items: int = 12,
) -> None:
    rows = rows[:max_items]
    if not rows:
        return

    cols = 4
    rows_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.2, rows_count * 3.2))
    if rows_count == 1:
        axes = np.array([axes])

    for ax in axes.flat:
        ax.axis("off")

    for idx, row in enumerate(rows):
        img_path = Path(row["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"pred: {row['ensemble_pred']}\nconf: {row['ensemble_conf']:.3f}",
            fontsize=9,
        )

    fig.suptitle(f"Misclassified Samples — True: {class_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate error analysis artifacts")
    parser.add_argument("--data-dir", default="datasets/augmented/test")
    parser.add_argument("--results-dir", default="reports/ensemble_eval")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    prob_path = results_dir / "ensemble_probabilities.npz"
    if not prob_path.exists():
        raise FileNotFoundError(f"Missing probabilities: {prob_path}")

    data = np.load(prob_path, allow_pickle=True)
    y_true = data["y_true"]

    probs_by_model: Dict[str, np.ndarray] = {
        "convnextv2": data["probs_convnextv2"],
        "efficientnetv2": data["probs_efficientnetv2"],
        "maxvit": data["probs_maxvit"],
        "swin": data["probs_swin"],
        "ensemble": data["probs_ensemble"],
    }

    data_dir = Path(args.data_dir)
    images = list_test_images(data_dir)
    if len(images) != len(y_true):
        raise ValueError(
            f"Image count mismatch: {len(images)} images vs {len(y_true)} labels. "
            "Check data_dir or regenerate probabilities."
        )

    preds_by_model = {
        k: np.argmax(v, axis=1) for k, v in probs_by_model.items()
    }
    conf_by_model = {
        k: np.max(v, axis=1) for k, v in probs_by_model.items()
    }

    rows: List[Dict[str, object]] = []
    misclassified: List[Dict[str, object]] = []
    disagreements: List[Dict[str, object]] = []

    for idx, img_path in enumerate(images):
        true_idx = int(y_true[idx])
        true_label = CLASSES[true_idx]
        ensemble_pred_idx = int(preds_by_model["ensemble"][idx])
        ensemble_pred = CLASSES[ensemble_pred_idx]
        ensemble_conf = float(conf_by_model["ensemble"][idx])
        ensemble_prob_true = float(probs_by_model["ensemble"][idx, true_idx])

        model_preds = {k: CLASSES[int(preds_by_model[k][idx])] for k in MODEL_KEYS}
        unique_preds = sorted(set(model_preds.values()))
        agreement_count = sum(1 for k in MODEL_KEYS if model_preds[k] == ensemble_pred)

        row = {
            "index": idx,
            "image_path": str(img_path),
            "true_label": true_label,
            "ensemble_pred": ensemble_pred,
            "ensemble_conf": ensemble_conf,
            "ensemble_prob_true": ensemble_prob_true,
            "agreement_count": agreement_count,
            "unique_preds": ",".join(unique_preds),
        }
        for key in MODEL_KEYS:
            row[f"{key}_pred"] = model_preds[key]
            row[f"{key}_conf"] = float(conf_by_model[key][idx])
        rows.append(row)

        if ensemble_pred_idx != true_idx:
            misclassified.append(row)

        if len(unique_preds) > 1:
            disagreements.append(row)

    out_dir = results_dir / "error_analysis"
    ensure_dir(out_dir)

    fieldnames = [
        "index",
        "image_path",
        "true_label",
        "ensemble_pred",
        "ensemble_conf",
        "ensemble_prob_true",
        "agreement_count",
        "unique_preds",
    ] + [f"{k}_pred" for k in MODEL_KEYS] + [f"{k}_conf" for k in MODEL_KEYS]

    save_csv(out_dir / "misclassified_ensemble.csv", misclassified, fieldnames)
    save_csv(out_dir / "model_disagreements.csv", disagreements, fieldnames)

    # Summary counts
    summary = {
        "total_samples": int(len(y_true)),
        "misclassified_count": int(len(misclassified)),
        "disagreement_count": int(len(disagreements)),
    }

    errors_by_true = {cls: 0 for cls in CLASSES}
    errors_by_pair = {cls: {c: 0 for c in CLASSES} for cls in CLASSES}
    for row in misclassified:
        errors_by_true[row["true_label"]] += 1
        errors_by_pair[row["true_label"]][row["ensemble_pred"]] += 1

    summary["errors_by_true"] = errors_by_true
    summary["errors_by_pair"] = errors_by_pair

    with open(out_dir / "error_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot grids per true class (top-k by confidence)
    misclassified_sorted = sorted(misclassified, key=lambda r: r["ensemble_conf"], reverse=True)
    for cls in CLASSES:
        subset = [r for r in misclassified_sorted if r["true_label"] == cls]
        if subset:
            plot_misclassified_grid(
                images,
                subset,
                cls,
                out_dir / f"misclassified_grid_{cls}.png",
                max_items=args.top_k,
            )

    print(f"✅ Error analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
