# 🔢 Handwritten Digit Recognition

A beginner-friendly CNN-based handwritten digit recognizer trained on the **MNIST dataset** using TensorFlow/Keras. This is **Stage 1** of a future handwritten mathematical expression recognition project.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📁 Project Structure

```
Handwritten-Digit-Recognition/
├── notebooks/
│   └── digit_recognition_training.ipynb   # Training notebook (run in Google Colab)
├── saved_model/
│   └── digit_cnn.keras                    # Trained model (generated after training)
├── test_images/                           # Your test images go here
├── predict.py                             # Prediction script
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

A simple CNN built **from scratch** (no pretrained models):

```
Input (28×28×1 grayscale image)
  │
  ├─ Conv2D (32 filters, 3×3, ReLU)
  ├─ MaxPooling2D (2×2)
  │
  ├─ Conv2D (64 filters, 3×3, ReLU)
  ├─ MaxPooling2D (2×2)
  │
  ├─ Flatten
  ├─ Dense (128, ReLU)
  ├─ Dropout (0.5)
  └─ Dense (10, Softmax) → Predicted digit (0–9)
```

**Expected accuracy:** ~99% on the MNIST test set.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A Google account (for Google Colab training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

### Step 2: Install Dependencies (for local prediction)

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Training is designed to run in **Google Colab** (free GPU access).

### Option A: Google Colab (Recommended)

1. Open the training notebook in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

   - Go to [Google Colab](https://colab.research.google.com/)
   - Click **File → Upload notebook**
   - Upload `notebooks/digit_recognition_training.ipynb`

2. **Enable GPU** (faster training):
   - Go to **Runtime → Change runtime type**
   - Select **GPU** as the hardware accelerator

3. **Run all cells** (Runtime → Run all)

4. The notebook will:
   - Download and preprocess the MNIST dataset
   - Build and train the CNN model (~2 minutes with GPU)
   - Evaluate accuracy and show a confusion matrix
   - Save the model and prompt you to download it

5. **Download the trained model** (`saved_model/digit_cnn.keras`) when prompted.

6. Place the downloaded model in the `saved_model/` directory of this project.

### Option B: Local Training

If you have TensorFlow installed locally:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/digit_recognition_training.ipynb
```

---

## 🔮 Making Predictions

Once you have a trained model in `saved_model/digit_cnn.keras`:

### Basic Usage

```bash
python predict.py path/to/your/digit_image.png
```

### Example

```bash
python predict.py test_images/my_digit.png
```

### Sample Output

```
📦 Loading model from: saved_model/digit_cnn.keras
🖼️  Processing image: test_images/my_digit.png

=============================================
  🔢 Predicted Digit:  3
  📊 Confidence:       98.75%
=============================================

  Probabilities for each digit:
  ------------------------------
  0:   0.01%
  1:   0.05%
  2:   0.42%
  3:  98.75%  ██████████████████████████████ ◄
  4:   0.12%
  5:   0.03%
  6:   0.01%
  7:   0.28%
  8:   0.19%
  9:   0.14%
```

### Options

```bash
python predict.py --help
python predict.py digit.png --model path/to/custom_model.keras
```

### Tips for Best Results

- Use **clear, high-contrast** images (dark digit on white background, or vice versa)
- The script automatically handles:
  - ✅ Color → grayscale conversion
  - ✅ Resizing to 28×28
  - ✅ Normalization to [0, 1]
  - ✅ Auto-inversion (detects light/dark background)
- Supported formats: PNG, JPG, BMP, TIFF, etc.

---

## 🔧 Image Preprocessing Pipeline

Every input image goes through these steps before prediction:

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Grayscale | Convert to single-channel grayscale |
| 2 | Resize | Scale to 28×28 pixels (LANCZOS interpolation) |
| 3 | Normalize | Scale pixel values from [0, 255] → [0.0, 1.0] |
| 4 | Auto-Invert | If background is light, invert to match MNIST format |
| 5 | Reshape | Reshape to (1, 28, 28, 1) for CNN input |

---

## 📊 Dataset: MNIST

| Property | Value |
|----------|-------|
| Total images | 70,000 |
| Training set | 60,000 |
| Test set | 10,000 |
| Image size | 28×28 pixels |
| Channels | 1 (grayscale) |
| Classes | 10 (digits 0–9) |

---

## 🗺️ Roadmap

This project is **Stage 1** of a larger handwritten mathematical expression recognition system:

- [x] **Stage 1** — Digit recognition (0–9) ← *You are here*
- [ ] **Stage 2** — Extend to mathematical symbols (+, −, ×, ÷, =, etc.)
- [ ] **Stage 3** — Expression segmentation and parsing
- [ ] **Stage 4** — Full expression evaluation

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras** — Model building and training
- **NumPy** — Numerical operations
- **Pillow (PIL)** — Image loading and preprocessing
- **Matplotlib & Seaborn** — Visualization
- **scikit-learn** — Evaluation metrics

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
