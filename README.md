# CIFAR-10 Image Classifier — CNN (PyTorch)

A from-scratch Convolutional Neural Network that classifies images from the **CIFAR-10** dataset into 10 categories (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

Built as a hands-on ML project: clean code structure, training + evaluation scripts, accuracy/loss plots, a confusion matrix, and a script to classify your own images.

---

## What it does

- Loads CIFAR-10 (60,000 32×32 color images, 10 classes) via `torchvision`.
- Trains a custom CNN (3 conv blocks + batch norm + dropout + fully connected head).
- Tracks training/validation loss & accuracy per epoch, saves plots.
- Generates a confusion matrix on the test set.
- Saves the best model checkpoint (`.pth`).
- Includes a `predict.py` script to classify any single image you give it.

## Results

With the default settings (15 epochs, Adam optimizer, data augmentation) you should see **~78–83% test accuracy** — solid for a from-scratch CNN with no pretrained backbone.

---

## Project structure
