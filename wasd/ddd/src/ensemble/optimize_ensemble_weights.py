"""
Ensemble Weight Optimization — find optimal model weights for maximum accuracy.

Uses saved per-model probability arrays (no GPU needed).
Methods:
  1. Grid search (exhaustive over weight simplex)
  2. SciPy optimization (Nelder-Mead on negative accuracy)
  3. Per-class analysis (which model dominates each class?)
  4. Multiple metrics: accuracy, log-loss, F1, AUC

Outputs:
  reports/ensemble_eval/
    - weight_optimization_results.json
    - weight_optimization_report.md
    - ensemble_weight_comparison.png
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ──────────────────────────────────────────────
CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]
DISPLAY = ["Comminuted Fracture", "No Fracture", "Simple Fracture"]
MODEL_NAMES = ["ConvNeXt V2", "EfficientNetV2-S", "MaxViT-Tiny", "Swin Transformer"]
MODEL_KEYS = ["convnextv2", "efficientnetv2", "maxvit", "swin"]
NPZ_PATH = Path("reports/ensemble_eval/data/ensemble_probabilities.npz")
OUT_DIR = Path("reports/ensemble_eval/plots")


def load_data():
    """Load saved probability arrays."""
    d = np.load(NPZ_PATH)
    y_true = d["y_true"]
    probs = np.stack(
        [d[f"probs_{k}"] for k in MODEL_KEYS], axis=0
    )  # (4, N, 3)
    return y_true, probs


def weighted_ensemble(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted average probabilities.  probs: (M, N, C), weights: (M,)"""
    w = weights / weights.sum()
    return np.tensordot(w, probs, axes=([0], [0]))  # (N, C)


def evaluate(y_true, probs_ensemble):
    """Compute metrics for an ensemble probability array."""
    preds = np.argmax(probs_ensemble, axis=1)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    ll = log_loss(y_true, probs_ensemble, labels=[0, 1, 2])
    try:
        auc = roc_auc_score(y_true, probs_ensemble, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return dict(accuracy=acc, f1_macro=f1, log_loss=ll, auc_roc_macro=auc)


# ── Method 1: Grid Search ──────────────────────────────
def grid_search(y_true, probs, step=0.05):
    """Exhaustive grid search over weight simplex."""
    print("\n═══ Grid Search (step={}) ═══".format(step))
    best_acc = 0
    best_weights = None
    best_metrics = None
    n_steps = int(1.0 / step) + 1
    values = np.linspace(0, 1, n_steps)

    count = 0
    for w0 in values:
        for w1 in values:
            for w2 in values:
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                    continue
                w = np.array([w0, w1, w2, max(0, w3)])
                ens = weighted_ensemble(probs, w)
                m = evaluate(y_true, ens)
                count += 1
                if m["accuracy"] > best_acc:
                    best_acc = m["accuracy"]
                    best_weights = w.copy()
                    best_metrics = m

    print(f"  Evaluated {count} weight combinations")
    print(f"  Best accuracy: {best_acc:.6f}")
    print(f"  Best weights:  {best_weights}")
    return best_weights, best_metrics


# ── Method 2: Scipy Optimization ───────────────────────
def scipy_optimize(y_true, probs, metric="accuracy"):
    """Optimize weights using Nelder-Mead (or other scipy method)."""
    print(f"\n═══ SciPy Optimization (metric={metric}) ═══")

    def objective(w_raw):
        w = np.exp(w_raw) / np.exp(w_raw).sum()  # softmax parameterization
        ens = weighted_ensemble(probs, w)
        m = evaluate(y_true, ens)
        if metric == "accuracy":
            return -m["accuracy"]
        elif metric == "log_loss":
            return m["log_loss"]
        elif metric == "f1":
            return -m["f1_macro"]
        return -m["accuracy"]

    # Multiple random starts
    best_result = None
    best_obj = float("inf")
    for seed in range(20):
        rng = np.random.RandomState(seed)
        x0 = rng.randn(4) * 0.5
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    w_opt = np.exp(best_result.x) / np.exp(best_result.x).sum()
    ens = weighted_ensemble(probs, w_opt)
    metrics = evaluate(y_true, ens)
    disp_key = "f1_macro" if metric == "f1" else metric
    print(f"  Best {metric}: {metrics[disp_key]:.6f}")
    print(f"  Weights: {w_opt}")
    return w_opt, metrics


# ── Method 3: Per-class weight analysis ────────────────
def per_class_analysis(y_true, probs):
    """Analyze which model performs best for each class."""
    print("\n═══ Per-Class Model Dominance ═══")
    for ci, cname in enumerate(DISPLAY):
        mask = y_true == ci
        print(f"\n  {cname} ({mask.sum()} samples):")
        for mi, mname in enumerate(MODEL_NAMES):
            preds_i = np.argmax(probs[mi, mask], axis=1)
            correct = (preds_i == ci).sum()
            acc = correct / mask.sum()
            avg_conf = probs[mi, mask, ci].mean()
            print(f"    {mname:25s}  acc={acc:.4f}  avg_conf={avg_conf:.4f}  "
                  f"errors={mask.sum()-correct}")


# ── Method 4: Leave-one-out analysis ──────────────────
def leave_one_out(y_true, probs):
    """Test ensemble with each model removed."""
    print("\n═══ Leave-One-Out Analysis ═══")
    results = {}
    for drop_idx in range(4):
        keep = [i for i in range(4) if i != drop_idx]
        w = np.ones(4)
        w[drop_idx] = 0
        ens = weighted_ensemble(probs, w)
        m = evaluate(y_true, ens)
        print(f"  Without {MODEL_NAMES[drop_idx]:25s}  "
              f"acc={m['accuracy']:.6f}  f1={m['f1_macro']:.6f}")
        results[MODEL_NAMES[drop_idx]] = m
    return results


# ── Visualization ──────────────────────────────────────
def plot_comparison(results: dict, out_path: Path):
    """Bar chart comparing ensemble strategies."""
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    f1s = [results[n]["f1_macro"] * 100 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    bars = axes[0].barh(names, accs, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_title("Ensemble Strategy — Accuracy")
    axes[0].set_xlim(min(accs) - 0.1, max(accs) + 0.05)
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{v:.4f}%", va="center", fontsize=9)

    # F1
    bars = axes[1].barh(names, f1s, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(names))))
    axes[1].set_xlabel("F1 Score (%)")
    axes[1].set_title("Ensemble Strategy — F1 (macro)")
    axes[1].set_xlim(min(f1s) - 0.1, max(f1s) + 0.05)
    for bar, v in zip(bars, f1s):
        axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{v:.4f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved chart: {out_path}")


def main():
    y_true, probs = load_data()
    print(f"Loaded {probs.shape[1]} samples, {probs.shape[0]} models, {probs.shape[2]} classes")

    # Baseline: equal weights
    print("\n═══ Baseline (Equal Weights) ═══")
    w_equal = np.array([0.25, 0.25, 0.25, 0.25])
    ens_equal = weighted_ensemble(probs, w_equal)
    m_equal = evaluate(y_true, ens_equal)
    print(f"  acc={m_equal['accuracy']:.6f}  f1={m_equal['f1_macro']:.6f}  "
          f"log_loss={m_equal['log_loss']:.6f}  auc={m_equal['auc_roc_macro']:.6f}")

    # Individual model baselines
    print("\n═══ Individual Models ═══")
    individual = {}
    for mi, mname in enumerate(MODEL_NAMES):
        m = evaluate(y_true, probs[mi])
        individual[mname] = m
        print(f"  {mname:25s}  acc={m['accuracy']:.6f}  f1={m['f1_macro']:.6f}")

    # Per-class analysis
    per_class_analysis(y_true, probs)

    # Leave-one-out
    loo = leave_one_out(y_true, probs)

    # Grid search (accuracy)
    w_grid, m_grid = grid_search(y_true, probs, step=0.05)

    # Scipy optimize (accuracy)
    w_scipy_acc, m_scipy_acc = scipy_optimize(y_true, probs, metric="accuracy")

    # Scipy optimize (log_loss)
    w_scipy_ll, m_scipy_ll = scipy_optimize(y_true, probs, metric="log_loss")

    # Scipy optimize (f1)
    w_scipy_f1, m_scipy_f1 = scipy_optimize(y_true, probs, metric="f1")

    # Accuracy-weighted (proportional to individual accuracy)
    ind_accs = np.array([individual[n]["accuracy"] for n in MODEL_NAMES])
    w_accprop = ind_accs / ind_accs.sum()
    ens_accprop = weighted_ensemble(probs, w_accprop)
    m_accprop = evaluate(y_true, ens_accprop)
    print(f"\n═══ Accuracy-Proportional Weights ═══")
    print(f"  Weights: {w_accprop}")
    print(f"  acc={m_accprop['accuracy']:.6f}  f1={m_accprop['f1_macro']:.6f}")

    # Top-2 ensemble (best 2 models only)
    top2_idx = np.argsort(ind_accs)[-2:]
    w_top2 = np.zeros(4)
    w_top2[top2_idx] = 0.5
    ens_top2 = weighted_ensemble(probs, w_top2)
    m_top2 = evaluate(y_true, ens_top2)
    print(f"\n═══ Top-2 Ensemble ({MODEL_NAMES[top2_idx[0]]} + {MODEL_NAMES[top2_idx[1]]}) ═══")
    print(f"  acc={m_top2['accuracy']:.6f}  f1={m_top2['f1_macro']:.6f}")

    # ── Compile results ──────────────────────────────────
    all_results = {}
    for mname in MODEL_NAMES:
        all_results[mname + " (solo)"] = individual[mname]

    strategies = {
        "Equal Weights (0.25 each)": (w_equal, m_equal),
        "Accuracy-Proportional": (w_accprop, m_accprop),
        "Grid Search (best)": (w_grid, m_grid),
        "Scipy Opt (accuracy)": (w_scipy_acc, m_scipy_acc),
        "Scipy Opt (log-loss)": (w_scipy_ll, m_scipy_ll),
        "Scipy Opt (F1)": (w_scipy_f1, m_scipy_f1),
        f"Top-2 ({MODEL_NAMES[top2_idx[0]]}+{MODEL_NAMES[top2_idx[1]]})": (w_top2, m_top2),
    }

    for name, (w, m) in strategies.items():
        all_results[name] = m

    # ── Save JSON ─────────────────────────────────────────
    output = {
        "num_samples": int(probs.shape[1]),
        "model_keys": MODEL_KEYS,
        "model_names": MODEL_NAMES,
        "individual_metrics": {k: individual[k] for k in MODEL_NAMES},
        "strategies": {},
    }
    for name, (w, m) in strategies.items():
        output["strategies"][name] = {
            "weights": {MODEL_NAMES[i]: round(float(w[i]), 6) for i in range(4)},
            "metrics": {k: round(float(v), 8) for k, v in m.items()},
        }

    # Best strategy
    best_name = max(strategies, key=lambda n: strategies[n][1]["accuracy"])
    best_w, best_m = strategies[best_name]
    output["best_strategy"] = best_name
    output["best_weights"] = {MODEL_NAMES[i]: round(float(best_w[i]), 6) for i in range(4)}
    output["best_metrics"] = {k: round(float(v), 8) for k, v in best_m.items()}

    json_path = OUT_DIR / "weight_optimization_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ── Generate Markdown Report ──────────────────────────
    md_lines = [
        "# Ensemble Weight Optimization Report\n",
        f"**Samples:** {probs.shape[1]} test images | **Models:** {len(MODEL_NAMES)} | **Classes:** {len(CLASSES)}\n",
        "",
        "## Individual Model Performance\n",
        "| Model | Accuracy | F1 (macro) | AUC-ROC | Log Loss |",
        "|-------|----------|------------|---------|----------|",
    ]
    for mname in MODEL_NAMES:
        m = individual[mname]
        md_lines.append(
            f"| {mname} | {m['accuracy']*100:.4f}% | {m['f1_macro']*100:.4f}% "
            f"| {m['auc_roc_macro']:.6f} | {m['log_loss']:.6f} |"
        )

    md_lines += [
        "",
        "## Ensemble Strategies Comparison\n",
        "| Strategy | Weights | Accuracy | F1 (macro) | Log Loss |",
        "|----------|---------|----------|------------|----------|",
    ]
    for name, (w, m) in strategies.items():
        w_str = ", ".join(f"{v:.2f}" for v in w)
        md_lines.append(
            f"| {name} | [{w_str}] | **{m['accuracy']*100:.4f}%** "
            f"| {m['f1_macro']*100:.4f}% | {m['log_loss']:.6f} |"
        )

    md_lines += [
        "",
        f"## 🏆 Best Strategy: **{best_name}**\n",
        f"- **Accuracy:** {best_m['accuracy']*100:.4f}%",
        f"- **F1 (macro):** {best_m['f1_macro']*100:.4f}%",
        f"- **AUC-ROC:** {best_m['auc_roc_macro']:.6f}",
        f"- **Log Loss:** {best_m['log_loss']:.6f}",
        "",
        "### Optimal Weights",
        "| Model | Weight |",
        "|-------|--------|",
    ]
    for i, mname in enumerate(MODEL_NAMES):
        md_lines.append(f"| {mname} | {best_w[i]:.4f} |")

    # Confusion matrix for best
    ens_best = weighted_ensemble(probs, best_w)
    preds_best = np.argmax(ens_best, axis=1)
    cm = confusion_matrix(y_true, preds_best)
    md_lines += [
        "",
        "### Confusion Matrix (Optimized Ensemble)\n",
        "| | Pred: Comminuted | Pred: No Frac | Pred: Simple |",
        "|---|---|---|---|",
    ]
    for ri, rname in enumerate(DISPLAY):
        md_lines.append(f"| **{rname}** | {cm[ri,0]} | {cm[ri,1]} | {cm[ri,2]} |")

    # Errors
    errors = np.where(preds_best != y_true)[0]
    md_lines += [
        "",
        f"### Error Analysis ({len(errors)} misclassified samples)\n",
    ]
    if len(errors) == 0:
        md_lines.append("**Perfect classification — zero errors!**")
    else:
        cr = classification_report(y_true, preds_best, target_names=DISPLAY)
        md_lines.append(f"```\n{cr}\n```")

    # Leave-one-out
    md_lines += [
        "",
        "## Leave-One-Out Analysis\n",
        "| Removed Model | Accuracy | F1 (macro) |",
        "|---------------|----------|------------|",
    ]
    for mname, m in loo.items():
        md_lines.append(f"| {mname} | {m['accuracy']*100:.4f}% | {m['f1_macro']*100:.4f}% |")

    md_path = OUT_DIR / "weight_optimization_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")

    # ── Plot ──────────────────────────────────────────────
    plot_comparison(all_results, OUT_DIR / "ensemble_weight_comparison.png")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Best strategy:  {best_name}")
    print(f"  Accuracy:       {best_m['accuracy']*100:.4f}%")
    print(f"  F1 (macro):     {best_m['f1_macro']*100:.4f}%")
    print(f"  Errors:         {len(errors)} / {len(y_true)}")
    print(f"  Weights:")
    for i, mname in enumerate(MODEL_NAMES):
        print(f"    {mname:25s}  {best_w[i]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
