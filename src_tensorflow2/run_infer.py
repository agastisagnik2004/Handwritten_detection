"""
Standalone inference runner for the Handwritten Line-Text Recognition model.

Why this exists:
  - main.py imports editdistance + autocorrect (validation / spell-check) which
    have install friction on Python 3.13. This runner drives the neural network
    directly (CNN -> BLSTM -> CTC), so it needs none of that.
  - TensorFlow 2.21 ships Keras 3, which removed the legacy rnn_cell.LSTMCell the
    model uses. We enable the Keras-2 compat shim BEFORE importing tensorflow.

Usage (from anywhere):
    python run_infer.py                      # recognize every image in ../dataset
    python run_infer.py path/to/image.png    # recognize a single image
    python run_infer.py path/to/folder       # recognize every image in a folder

NOTE: Output is only meaningful once a trained checkpoint (snapshot-*) is placed
in ../model/. Without one, the network runs with random weights and prints gibberish.
"""
import os
import sys
import glob

# Must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Model.py / DataLoader.py use paths relative to this src dir ('../model', './logs').
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import cv2  # noqa: E402
from DataLoader import Batch, FilePaths  # noqa: E402
from SamplePreprocessor import preprocessor  # noqa: E402
from Model import Model, DecoderType  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_images(arg):
    """Resolve the CLI arg into a list of image paths."""
    if arg:
        if os.path.isdir(arg):
            target_dir = arg
        elif os.path.isfile(arg):
            return [arg]
        else:
            sys.exit("Path not found: %s" % arg)
    else:
        target_dir = os.path.join("..", "dataset")
        if not os.path.isdir(target_dir):
            print("No ../dataset folder; falling back to %s" % FilePaths.fnInfer)
            return [FilePaths.fnInfer]

    imgs = sorted(
        p for p in glob.glob(os.path.join(target_dir, "*"))
        if p.lower().endswith(IMG_EXTS)
    )
    if not imgs:
        sys.exit("No images found in: %s" % target_dir)
    return imgs


def load_fillers():
    """The model decodes a fixed batch of Model.batchSize; pad with sample images."""
    fillers = []
    for i in range(1, Model.batchSize):
        f = "../data/check_image/a (%d).png" % i
        fillers.append(preprocessor(cv2.imread(f, cv2.IMREAD_GRAYSCALE),
                                    Model.imgSize, enhance=False))
    return fillers


def try_spellcheck(text):
    """Optional post-correction; silently skipped if autocorrect isn't available."""
    try:
        from SpellChecker import correct_sentence
        return correct_sentence(text).strip()
    except Exception:
        return None


def main():
    images = collect_images(sys.argv[1] if len(sys.argv) > 1 else None)

    charList = open("../model/charList.txt").read()
    model = Model(charList, DecoderType.BestPath, mustRestore=False)

    fillers = load_fillers()
    print("\n" + "=" * 60)
    print("Recognizing %d image(s)" % len(images))
    print("=" * 60)
    for path in images:
        img = preprocessor(cv2.imread(path, cv2.IMREAD_GRAYSCALE), imgSize=Model.imgSize)
        batch = Batch(None, [img] + fillers)
        recognized = model.inferBatch(batch)[0]
        corrected = try_spellcheck(recognized)
        print("\n%s" % os.path.basename(path))
        print("  recognized: '%s'" % recognized)
        if corrected is not None:
            print("  corrected : '%s'" % corrected)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
