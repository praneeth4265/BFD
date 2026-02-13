"""
Ensemble evaluation for 4-model bone fracture classification.
Evaluates ConvNeXt V2, EfficientNetV2-S, MaxViT, and Swin on the test set,
and computes a soft-voting ensemble.

Outputs:
- Per-model metrics (accuracy, precision, recall, F1, AUC-ROC)
- Ensemble metrics
- Confusion matrices and ROC curves
- Results JSON in results/ensemble_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


CLASSES = ['comminuted_fracture', 'no_fracture', 'simple_fracture']

MODEL_CONFIGS = [
    {
        'key': 'convnextv2',
        'display': 'ConvNeXt V2 Base',
        'timm_name': 'convnextv2_base.fcmae_ft_in22k_in1k',
        'checkpoint': 'models/checkpoints/convnextv2_3class_augmented_best.pth',
    },
    {
        'key': 'efficientnetv2',
        'display': 'EfficientNetV2-S',
        'timm_name': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'checkpoint': 'models/checkpoints/efficientnetv2_3class_augmented_best.pth',
    },
    {
        'key': 'maxvit',
        'display': 'MaxViT-Tiny',
        'timm_name': 'maxvit_tiny_tf_224.in1k',
        'checkpoint': 'models/checkpoints/maxvit_3class_augmented_best.pth',
    },
    {
        'key': 'swin',
        'display': 'Swin Transformer (Tiny)',
        'timm_name': 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
        'checkpoint': 'models/checkpoints/swin_3class_augmented_best.pth',
    },
]


class BoneFractureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = CLASSES

        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, idx


def load_model(timm_name, checkpoint_path, device):
    model = timm.create_model(timm_name, pretrained=False, num_classes=len(CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_metrics(y_true, y_prob, model_name):
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    auc_macro = roc_auc_score(
        y_true, y_prob, multi_class='ovr', average='macro'
    )

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    roc_data = {}
    for class_idx, class_name in enumerate(CLASSES):
        y_true_bin = (y_true == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, class_idx])
        roc_data[class_name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
        }

    return {
        'model': model_name,
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1),
        'precision_weighted': float(weighted_precision),
        'recall_weighted': float(weighted_recall),
        'f1_weighted': float(weighted_f1),
        'auc_roc_macro': float(auc_macro),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_curves': roc_data,
    }


def plot_confusion_matrix(cm, title, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curves(roc_data, title, output_path):
    plt.figure(figsize=(7, 6))
    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (class_name, curve), color in zip(roc_data.items(), color_cycle):
        plt.plot(
            curve['fpr'],
            curve['tpr'],
            label=class_name,
            color=color,
            linewidth=2,
        )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate 4-model ensemble on test set')
    parser.add_argument('--data-dir', default='datasets/augmented/test')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--results-dir', default='results/ensemble_eval')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = BoneFractureDataset(args.data_dir, transform=transform)
    if args.max_samples:
        dataset.images = dataset.images[: args.max_samples]
        dataset.labels = dataset.labels[: args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    n_samples = len(dataset)
    y_true = np.zeros(n_samples, dtype=int)
    label_set = np.zeros(n_samples, dtype=bool)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_probs = {}
    model_metrics = {}

    for config in MODEL_CONFIGS:
        print(f"\n🔍 Evaluating {config['display']}...")
        model = load_model(config['timm_name'], config['checkpoint'], device)
        probs = np.zeros((n_samples, len(CLASSES)), dtype=np.float32)

        with torch.no_grad():
            for images, labels, indices in tqdm(loader, desc=config['key']):
                images = images.to(device)
                outputs = model(images)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()

                for i, idx in enumerate(indices.numpy()):
                    probs[idx] = batch_probs[i]
                    if not label_set[idx]:
                        y_true[idx] = labels[i].item()
                        label_set[idx] = True

        model_probs[config['key']] = probs
        metrics = compute_metrics(y_true, probs, config['display'])
        model_metrics[config['key']] = metrics

        cm_path = results_dir / f"confusion_matrix_{config['key']}.png"
        roc_path = results_dir / f"roc_{config['key']}.png"
        plot_confusion_matrix(np.array(metrics['confusion_matrix']),
                              f"{config['display']} Confusion Matrix",
                              cm_path)
        plot_roc_curves(metrics['roc_curves'],
                        f"{config['display']} ROC Curves",
                        roc_path)

    # Ensemble (soft voting)
    print("\n🔮 Evaluating ensemble (soft voting)...")
    stacked_probs = np.stack([model_probs[m['key']] for m in MODEL_CONFIGS], axis=0)
    ensemble_probs = stacked_probs.mean(axis=0)

    ensemble_metrics = compute_metrics(y_true, ensemble_probs, 'Ensemble (Soft Voting)')

    plot_confusion_matrix(
        np.array(ensemble_metrics['confusion_matrix']),
        "Ensemble Confusion Matrix",
        results_dir / "confusion_matrix_ensemble.png",
    )
    plot_roc_curves(
        ensemble_metrics['roc_curves'],
        "Ensemble ROC Curves",
        results_dir / "roc_ensemble.png",
    )

    results = {
        'num_samples': int(n_samples),
        'classes': CLASSES,
        'models': model_metrics,
        'ensemble': ensemble_metrics,
    }

    results_path = results_dir / 'ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save probabilities for PR / calibration plots
    np.savez_compressed(
        results_dir / 'ensemble_probabilities.npz',
        y_true=y_true,
        **{f"probs_{k}": v for k, v in model_probs.items()},
        probs_ensemble=ensemble_probs,
    )

    print(f"\n✅ Results saved to {results_path}")
    print(f"📊 Plots saved in {results_dir}")


if __name__ == '__main__':
    main()
