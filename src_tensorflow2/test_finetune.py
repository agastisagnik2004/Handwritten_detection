"""
Test fine-tuned model on prescription images.
Uses model_finetune/ checkpoints (not base model).
"""
import os
import sys
import cv2

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MODEL_DIR", "model_finetune")  # Use fine-tuned checkpoints

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from DataLoader import Batch, FilePaths
from SamplePreprocessor import preprocessor
from Model import Model, DecoderType


def test_finetune_image(image_path):
    """Test fine-tuned model on a single image."""

    # Check if fine-tuned checkpoint exists
    if not os.path.exists("../model_finetune/charList.txt"):
        print("[ERROR] Fine-tuned model not found at ../model_finetune/")
        print("Run: python finetune_on_dataset.py --dataset Training --epochs 10 --train")
        return False

    print(f"\n{'='*70}")
    print(f"TESTING FINE-TUNED MODEL")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    print(f"Checkpoint: model_finetune/")
    print(f"{'='*70}\n")

    # Load image (handle paths with special characters)
    # Use absolute path
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return False

    # Read image using numpy to avoid OpenCV encoding issues with special chars
    try:
        import numpy as np
        with open(image_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"[ERROR] Could not read image: {e}")
        return False

    if img is None:
        print(f"[ERROR] Could not decode image")
        return False

    print(f"Image size: {img.shape}")

    # Load model
    try:
        charList = open("../model_finetune/charList.txt").read()
        model = Model(charList, DecoderType.BestPath, mustRestore=False)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

    # Preprocess
    processed = preprocessor(img, Model.imgSize, enhance=False)

    # Load fillers for batch padding
    fillers = []
    for i in range(1, Model.batchSize):
        f = "../data/check_image/a (%d).png" % i
        if os.path.exists(f):
            fillers.append(preprocessor(cv2.imread(f, cv2.IMREAD_GRAYSCALE),
                                       Model.imgSize, enhance=False))

    # Infer
    batch = Batch(None, [processed] + fillers)
    recognized = model.inferBatch(batch)[0]

    print(f"\n[RESULT]")
    print(f"  Recognized: '{recognized}'")

    # Try spell correction
    try:
        from SpellChecker import correct_sentence
        corrected = correct_sentence(recognized).strip()
        if corrected != recognized:
            print(f"  Corrected : '{corrected}'")
    except:
        pass

    print(f"\n{'='*70}\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_finetune.py <image_num_or_path>")
        print("\nExample:")
        print("  python test_finetune.py 44")
        print("  python test_finetune.py ../dataset/Doctor*/Testing/testing_words/44.png")
        sys.exit(1)

    arg = sys.argv[1]

    # If argument is just a number, build the path internally
    if arg.isdigit():
        import glob
        pattern = f"../dataset/*rescription*BD dataset/Testing/testing_words/{arg}.png"
        matches = glob.glob(pattern)
        if matches:
            image_path = matches[0]
        else:
            print(f"[ERROR] Image {arg}.png not found")
            sys.exit(1)
    else:
        image_path = arg

    success = test_finetune_image(image_path)
    sys.exit(0 if success else 1)
