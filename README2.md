# Handwritten Digit Recognition

A simple CNN-based handwritten digit recognition project using **TensorFlow/Keras and MNIST**.

## What I've Done

- Trained a CNN from scratch on the MNIST dataset.
- Built a model with **225K parameters** to classify digits `0–9`.
- Saved and loaded the trained model as a `.keras` file.
- Tested the model on custom handwritten images.
- Built an image preprocessing pipeline to convert different input images into a common **28×28 MNIST-style format**.
- Added handling for differences in:
  - brightness/background
  - digit polarity
  - size and position
  - stroke thickness
  - noise
- Successfully tested the preprocessing + model on handwritten digits from **0 to 9**.

## Current Pipeline

```text
Input Image
     ↓
Image Processing
     ↓
Crop + Resize + Center
     ↓
28×28 MNIST-style Image
     ↓
CNN Model
     ↓
Predicted Digit
```

## Tech Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- Jupyter Notebook
- MNIST

## Next Goal

Extend the project from recognizing individual digits to recognizing **mathematical symbols and handwritten expressions**.

Example:

```text
2 + 3 × 5
```

→ recognize the symbols → reconstruct the expression → evaluate it.
