"""
Handwritten Digit Prediction Script
====================================

Accepts a handwritten digit image and predicts the digit (0–9)
using the trained CNN model.

Usage:
    python predict.py <image_path>
    python predict.py <image_path> --model saved_model/digit_cnn.keras

Example:
    python predict.py test_images/digit_5.png
"""

import argparse
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = os.path.join("saved_model", "digit_cnn.keras")
IMG_HEIGHT = 28
IMG_WIDTH = 28


# ─── Image Preprocessing ────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a handwritten digit image for prediction.

    Steps:
        1. Load the image
        2. Convert to grayscale (single channel)
        3. Resize to 28×28 pixels
        4. Normalize pixel values to [0, 1]
        5. Invert colors if needed (MNIST has white digits on black background)
        6. Reshape for the CNN input: (1, 28, 28, 1)

    Args:
        image_path: Path to the input image file.

    Returns:
        Preprocessed image as a numpy array with shape (1, 28, 28, 1).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Cannot open image '{image_path}': {e}")

    # Step 1: Convert to grayscale
    img = img.convert("L")

    # Step 2: Resize to 28×28
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

    # Step 3: Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype="float32") / 255.0

    # Step 4: Invert if needed
    # MNIST digits are white-on-black. If the input image has a white/light
    # background (mean pixel value > 0.5), invert it.
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    # Step 5: Reshape for CNN input → (1, 28, 28, 1)
    img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

    return img_array


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_digit(model: keras.Model, image_path: str) -> dict:
    """
    Predict the digit in a handwritten image.

    Args:
        model: Trained Keras CNN model.
        image_path: Path to the handwritten digit image.

    Returns:
        Dictionary containing:
            - predicted_digit: The predicted digit (0–9)
            - confidence: Confidence percentage of the prediction
            - probabilities: Full probability distribution over all 10 digits
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    probabilities = predictions[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict a handwritten digit (0–9) from an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py test_images/digit_3.png
  python predict.py my_digit.jpg --model saved_model/digit_cnn.keras
        """,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to a handwritten digit image (PNG, JPG, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})",
    )

    args = parser.parse_args()

    # ── Load the model ──
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("   Please train the model first using the training notebook,")
        print("   then place the saved model in the 'saved_model/' directory.")
        sys.exit(1)

    print(f"📦 Loading model from: {args.model}")
    model = keras.models.load_model(args.model)

    # ── Run prediction ──
    print(f"🖼️  Processing image: {args.image}")
    result = predict_digit(model, args.image)

    # ── Display results ──
    print("\n" + "=" * 45)
    print(f"  🔢 Predicted Digit:  {result['predicted_digit']}")
    print(f"  📊 Confidence:       {result['confidence']:.2f}%")
    print("=" * 45)

    print("\n  Probabilities for each digit:")
    print("  " + "-" * 30)
    for digit, prob in enumerate(result["probabilities"]):
        bar = "█" * int(prob * 30)
        marker = " ◄" if digit == result["predicted_digit"] else ""
        print(f"  {digit}: {prob * 100:6.2f}%  {bar}{marker}")

    print()


if __name__ == "__main__":
    main()
