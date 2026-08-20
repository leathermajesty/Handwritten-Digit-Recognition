# 🔢 Handwritten Digit Recognition

A CNN-based handwritten digit recognition project built with **TensorFlow/Keras** and trained on the **MNIST dataset**.

This is **Stage 1** of a future handwritten mathematical expression recognition system.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-red)

---

## 🧠 What I've Done

- Trained a CNN from scratch on MNIST.
- Built a model with **225,034 parameters**.
- Saved the trained model as a `.keras` file.
- Tested the model on custom handwritten images.
- Built an image preprocessing pipeline to handle different input image conditions.
- Converted different images into a common **28×28 MNIST-style format** before prediction.
- Tested handwritten digits from **0 to 9** successfully.

---

## 🏗️ Model

```text
Input (28×28×1)
      ↓
Conv2D (32) + ReLU
      ↓
MaxPooling
      ↓
Conv2D (64) + ReLU
      ↓
MaxPooling
      ↓
Flatten
      ↓
Dense (128) + ReLU
      ↓
Dropout
      ↓
Dense (10) + Softmax
      ↓
Digit 0–9
```

**Total parameters:** 225,034

---

## 🔧 Image Preprocessing

Custom images do not always look like MNIST images, so a preprocessing pipeline was added.

```text
Input Image
     ↓
Grayscale / Foreground Detection
     ↓
Thresholding
     ↓
Digit Detection & Cropping
     ↓
Resize while preserving shape
     ↓
Stroke Adjustment
     ↓
Center / Padding
     ↓
28×28 MNIST-style Image
     ↓
CNN
     ↓
Predicted Digit
```

The goal is to make different types of handwritten images look similar to the images used during model training.

---

## 🚀 Current Result

The model can now recognize custom handwritten digits such as:

```text
0  →  0
1  →  1
2  →  2
3  →  3
...
9  →  9
```

The preprocessing step significantly improves prediction on images that originally have different sizes, backgrounds, brightness, or stroke thickness.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Jupyter Notebook
- MNIST

---

## 🗺️ Roadmap

- [x] **Stage 1** — Handwritten digit recognition
- [x] **Stage 1.5** — Robust image preprocessing
- [ ] **Stage 2** — Mathematical symbols (`+`, `−`, `×`, `÷`, `=`, etc.)
- [ ] **Stage 3** — Expression segmentation and parsing
- [ ] **Stage 4** — Mathematical expression evaluation

### Final Goal

Recognize handwritten expressions such as:

```text
2 + 3 × 5
```

and convert them into a machine-readable expression for evaluation.
