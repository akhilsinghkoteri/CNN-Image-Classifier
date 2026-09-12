# CIFAR-10 Image Classifier — CNN (PyTorch)

A from-scratch Convolutional Neural Network that classifies images from the **CIFAR-10** dataset into 10 categories (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

Built to be a strong portfolio / coursework project: clean structure, training + evaluation scripts, accuracy/loss plots, confusion matrix, and single-image inference.

---

## 🧠 What it does

- Loads CIFAR-10 (60,000 32×32 color images, 10 classes) via `torchvision`.
- Trains a custom CNN (3 conv blocks + batch norm + dropout + fully connected head).
- Tracks training/validation loss & accuracy per epoch, saves plots.
- Generates a confusion matrix on the test set.
- Saves the best model checkpoint (`.pth`).
- Includes a `predict.py` script to classify any single image you give it.

## 📊 Expected results

With the default settings (15 epochs, Adam optimizer, data augmentation) you should see **~78–83% test accuracy** — solid for a from-scratch CNN (no pretrained backbone).

---

## 📁 Project structure

```
cnn-image-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py          # CNN architecture
│   ├── train.py          # training + evaluation loop
│   ├── predict.py        # run inference on a single image
│   └── utils.py          # helper functions (plots, confusion matrix)
├── models/               # saved model checkpoints (created at runtime)
└── outputs/              # loss/accuracy plots, confusion matrix (created at runtime)
```

---

## ⚙️ Setup (macOS / Terminal)

```bash
# 1. Clone your repo (after you've pushed it) or cd into this folder
cd cnn-image-classifier

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🚀 Train the model

```bash
python3 src/train.py --epochs 15 --batch-size 64 --lr 0.001
```

This will:
- Download CIFAR-10 automatically (first run only, ~170MB)
- Train the CNN, printing loss/accuracy each epoch
- Save the best checkpoint to `models/best_model.pth`
- Save plots to `outputs/`

## 🔍 Run inference on your own image

```bash
python3 src/predict.py --image path/to/your/image.jpg
```

## 🛠️ Tech stack

- Python 3.10+
- PyTorch & torchvision
- NumPy, Matplotlib, scikit-learn (for confusion matrix)

## 📈 Possible extensions (great for showing initiative in a report/viva)

- Swap the custom CNN for a pretrained backbone (ResNet18 via transfer learning) and compare accuracy.
- Add TensorBoard logging.
- Try different optimizers (SGD + momentum vs Adam vs AdamW).
- Deploy with a simple Flask/Streamlit UI for live image upload + prediction.

## 📄 License

MIT — free to use for learning, coursework, or your portfolio.
