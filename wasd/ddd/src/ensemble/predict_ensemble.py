"""
Run ensemble inference on a folder of images and export predictions to CSV.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from ensemble_model import EnsembleModel, CLASSES


class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None, max_samples=None):
        self.folder = Path(folder)
        self.transform = transform
        self.images = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.images.extend(self.folder.rglob(ext))
        self.images = sorted(self.images)
        if max_samples:
            self.images = self.images[:max_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(path)


def main():
    parser = argparse.ArgumentParser(description="4-model ensemble inference")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", default="ensemble_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(args.input_dir, transform=transform, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ensemble = EnsembleModel()

    output_path = Path(args.output_csv)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "prediction",
            "confidence",
            "agreement_count",
            *[f"prob_{c}" for c in CLASSES],
        ])

        for batch, paths in loader:
            preds = ensemble.predict(batch.to(ensemble.device))
            for pred, path in zip(preds, paths):
                writer.writerow([
                    path,
                    pred.prediction,
                    f"{pred.confidence:.6f}",
                    pred.agreement_count,
                    *[f"{pred.probabilities[c]:.6f}" for c in CLASSES],
                ])

    print(f"✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
