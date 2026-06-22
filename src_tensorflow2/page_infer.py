"""
Full-page handwritten text recognition with automatic line detection and extraction.

Usage:
    python page_infer.py path/to/page.png
    python page_infer.py path/to/folder
"""
import os
import sys
import cv2
import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from DataLoader import Batch, FilePaths
from SamplePreprocessor import preprocessor
from Model import Model, DecoderType


def detect_lines(img, min_line_height=10):
    """
    Detect text lines using horizontal projection.
    Works well for plain documents and screenshots.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Adaptive threshold for varied lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 10)

    # Invert if needed (text should be white)
    if np.mean(binary) > 128:
        binary = 255 - binary

    # Dilate horizontally to connect text on same line
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

    # Horizontal projection: sum white pixels per row
    horizontal = np.sum(binary, axis=1)
    text_rows = np.where(horizontal > 20)[0]

    if len(text_rows) == 0:
        return []

    # Find gaps to separate lines
    lines = []
    line_start = text_rows[0]

    for i in range(1, len(text_rows)):
        if text_rows[i] - text_rows[i-1] > 2:
            line_end = text_rows[i-1]
            height = line_end - line_start + 1

            if height >= min_line_height:
                # Find x bounds
                line_binary = binary[line_start:line_end+1, :]
                vertical = np.sum(line_binary, axis=0)
                text_cols = np.where(vertical > 0)[0]

                if len(text_cols) > 0:
                    lines.append((line_start, line_end, text_cols[0], text_cols[-1]))

            line_start = text_rows[i]

    # Last line
    if line_start <= text_rows[-1]:
        line_end = text_rows[-1]
        height = line_end - line_start + 1
        if height >= min_line_height:
            line_binary = binary[line_start:line_end+1, :]
            vertical = np.sum(line_binary, axis=0)
            text_cols = np.where(vertical > 0)[0]
            if len(text_cols) > 0:
                lines.append((line_start, line_end, text_cols[0], text_cols[-1]))

    return lines


def extract_lines(img, line_regions):
    """Extract individual line images from the full page."""
    lines = []
    for y_start, y_end, x_start, x_end in line_regions:
        # Add padding
        pad = 5
        y_start = max(0, y_start - pad)
        y_end = min(img.shape[0], y_end + pad)
        x_start = max(0, x_start - pad)
        x_end = min(img.shape[1], x_end + pad)

        line_img = img[y_start:y_end, x_start:x_end]
        lines.append(line_img)

    return lines


def recognize_lines(model, lines, fillers):
    """Recognize text in each extracted line."""
    results = []

    for line_img in lines:
        # Convert to grayscale if needed
        if len(line_img.shape) == 3:
            line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

        # Preprocess
        processed = preprocessor(line_img, Model.imgSize, enhance=False)

        # Create batch (pad with fillers)
        batch = Batch(None, [processed] + fillers)

        # Recognize
        recognized = model.inferBatch(batch)[0]
        results.append(recognized.strip())

    return results


def try_spellcheck(text):
    """Optional post-correction."""
    try:
        from SpellChecker import correct_sentence
        return correct_sentence(text).strip()
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python page_infer.py <image_or_folder>")
        sys.exit(1)

    path = sys.argv[1]

    # Handle single image or folder
    if os.path.isfile(path):
        images = [path]
    elif os.path.isdir(path):
        imgs = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            imgs.extend(sorted(__import__('glob').glob(os.path.join(path, '**', ext), recursive=True)))
        images = imgs
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

    # Load model
    charList = open("../model/charList.txt").read()
    model = Model(charList, DecoderType.BestPath, mustRestore=False)

    # Load fillers for batch padding
    fillers = []
    for i in range(1, Model.batchSize):
        f = "../data/check_image/a (%d).png" % i
        if os.path.exists(f):
            fillers.append(preprocessor(cv2.imread(f, cv2.IMREAD_GRAYSCALE),
                                       Model.imgSize, enhance=False))

    print("\n" + "="*70)
    print(f"Recognizing {len(images)} page(s) with full-page line detection")
    print("="*70)

    for img_path in images:
        print(f"\n{os.path.basename(img_path)}")
        print("-" * 70)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Could not read image")
            continue

        # Detect lines
        line_regions = detect_lines(img)
        print(f"  Detected {len(line_regions)} lines")

        if not line_regions:
            print(f"  No lines detected")
            continue

        # Extract lines
        lines = extract_lines(img, line_regions)

        # Recognize text
        results = recognize_lines(model, lines, fillers)

        # Display results
        print(f"\n  Recognized text:")
        for i, text in enumerate(results, 1):
            corrected = try_spellcheck(text)
            print(f"    Line {i}: '{text}'")
            if corrected and corrected != text:
                print(f"             > '{corrected}' (corrected)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
