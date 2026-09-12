"""
predict.py
----------
Load a trained model checkpoint and classify a single image.

Usage:
    python3 src/predict.py --image path/to/image.jpg
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import CIFAR10CNN
from utils import CLASS_NAMES


def load_model(checkpoint_path: str, device: torch.device) -> CIFAR10CNN:
    model = CIFAR10CNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor


def predict(image_path: str, checkpoint_path: str = "models/best_model.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")

    model = load_model(checkpoint_path, device)
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = probabilities.argmax().item()

    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100

    print(f"\nPrediction: {predicted_class} ({confidence:.1f}% confidence)\n")
    print("Top 3 predictions:")
    top3 = torch.topk(probabilities, 3)
    for idx, prob in zip(top3.indices.tolist(), top3.values.tolist()):
        print(f"  {CLASS_NAMES[idx]:<12} {prob * 100:.1f}%")

    return predicted_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image using the trained CIFAR-10 CNN")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth",
                         help="Path to the trained model checkpoint")
    args = parser.parse_args()

    predict(args.image, args.checkpoint)
