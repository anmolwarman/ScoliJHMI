"""Image-based ODI classification inference script.

This module provides utilities to extract ResNet-50 embeddings from an image
and classify them with a pre-trained random forest.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# ---------------------------------------------------------------------------
# Model and transform initialization
# ---------------------------------------------------------------------------
try:  # Pillow 9 compatibility
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _load_feature_extractor() -> nn.Module:
    """Load an ImageNet-pretrained ResNet-50 without the final fully-connected layer."""
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


def _load_random_forest():
    model_path = Path(__file__).resolve().parent / "models" / "odi_rf.joblib"
    return joblib.load(model_path)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_EXTRACTOR = _load_feature_extractor().to(DEVICE)
RANDOM_FOREST = _load_random_forest()


def predict_image(path: str) -> Tuple[str, float]:
    """Predict the ODI class for an image.

    Args:
        path: Path to the input image file.

    Returns:
        A tuple containing the predicted class label ("<=40" or ">40") and the
        associated probability from the random forest classifier.
    """

    image = Image.open(path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = FEATURE_EXTRACTOR(tensor)

    embedding = embedding.view(embedding.size(0), -1).cpu().numpy()
    probabilities = RANDOM_FOREST.predict_proba(embedding)[0]
    best_idx = probabilities.argmax()
    predicted_label = RANDOM_FOREST.classes_[best_idx]
    predicted_probability = float(probabilities[best_idx])
    return predicted_label, predicted_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ODI class from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    label, probability = predict_image(args.image_path)
    print(f"Predicted ODI class: {label}")
    print(f"Probability: {probability:.4f}")


if __name__ == "__main__":
    main()
