#!/usr/bin/env python3
"""
FINETUNE ADAPTER (Top-level)

Architecture:
  - main.py = BASE MODEL TRAINING SCRIPT (trains on any dataset)
  - finetune_on_dataset.py = ADAPTER (prepares prescription data, calls main.py)

Flow:
  1. Prepare prescription dataset in IAM format
  2. Set MODEL_DIR environment to use separate checkpoints (model_finetune/)
  3. Call main.py --train with prescription data
  4. Save checkpoints to model_finetune/ (separate from base model)

Usage:
  python finetune_on_dataset.py --dataset Training --epochs 10 --train
  python finetune_on_dataset.py --dataset Validation --epochs 5 --train
  python finetune_on_dataset.py --dataset Training --epochs 20 --prepare-only
"""
import os
import csv
import shutil
import argparse
import subprocess
import sys
import glob


def prepare_prescription_data(dataset_split="Training", output_dir="data"):
    """
    Prepare prescription dataset in IAM format for main.py.

    Converts:
      dataset/Doctor's Handwritten Prescription BD dataset/Training/
        training_labels.csv
        training_words/0.png, 1.png, ...

    To:
      data/lines.txt (IAM format)
      data/lines/word/word-T/word-T-000000.png (directory structure)
    """
    os.makedirs(f"{output_dir}/lines", exist_ok=True)

    # Find prescription dataset (handle special characters in folder name)
    dataset_base = "dataset"
    if not os.path.exists(dataset_base):
        print(f"Error: {dataset_base}/ folder not found")
        return False

    # Find the prescription dataset folder
    dataset_path = None
    for item in os.listdir(dataset_base):
        if "Prescription" in item or "prescription" in item:
            dataset_path = os.path.join(dataset_base, item)
            break

    if not dataset_path or not os.path.isdir(dataset_path):
        print(f"Error: Prescription dataset not found")
        print(f"Expected: dataset/Doctor's Handwritten Prescription BD dataset/")
        return False
    print(f"[1/3] Found dataset: {dataset_path}")

    # Build paths for specified split
    split_dir = os.path.join(dataset_path, dataset_split)
    labels_file = os.path.join(split_dir, f"{dataset_split.lower()}_labels.csv")
    words_dir = os.path.join(split_dir, f"{dataset_split.lower()}_words")

    if not os.path.exists(labels_file) or not os.path.exists(words_dir):
        print(f"Error: {dataset_split} data not found")
        return False

    print(f"[2/3] Processing {dataset_split} data...")

    lines_txt = ""
    image_count = 0

    # Read CSV and copy images
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['IMAGE']
            medicine_name = row['MEDICINE_NAME']

            src_img = os.path.join(words_dir, img_name)
            if not os.path.exists(src_img):
                continue

            # Create IAM-format ID: word-T-000000 (T=Training, V=Validation)
            split_letter = dataset_split[0]
            line_id = f"word-{split_letter}-{image_count:06d}"

            # Directory structure: data/lines/word/word-T/word-T-000000.png
            dir_path = os.path.join(output_dir, "lines", "word", f"word-{split_letter}")
            os.makedirs(dir_path, exist_ok=True)

            # Copy image
            dst_img = os.path.join(dir_path, f"{line_id}.png")
            shutil.copy2(src_img, dst_img)

            # IAM format: FILE_ID ok X Y W H GREY TEXT
            text = medicine_name.replace(" ", "|")
            lines_txt += f"{line_id} ok 0 0 100 50 0 0 {text}\n"

            image_count += 1

    # Write lines.txt
    output_file = os.path.join(output_dir, "lines.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(lines_txt)

    print(f"[3/3] Created {output_file} with {image_count} samples")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="FINETUNE ADAPTER: Fine-tune base model on prescription dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Training",
        choices=["Training", "Validation"],
        help="Dataset to use: Training or Validation (default: Training)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Max epochs to train (default: 10)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Prepare data AND start training with base model"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare data, don't train"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"FINETUNE ADAPTER v1.0")
    print(f"Dataset: {args.dataset} | Epochs: {args.epochs}")
    print("=" * 70)

    # Step 1: Prepare prescription data
    print("\nPreparing prescription dataset...")
    if not prepare_prescription_data(dataset_split=args.dataset):
        print("[ERROR] Data preparation failed")
        return False

    if args.prepare_only:
        print("\n[OK] Data prepared. Ready to train.")
        print("\nTo start training:")
        print(f"  python finetune_on_dataset.py --dataset {args.dataset} --epochs {args.epochs} --train")
        return True

    # Step 2: Train with base model using separate checkpoints
    if args.train:
        print("\nStarting fine-tuning with base model...")
        print("-" * 70)

        # Set up environment for separate checkpoints
        env = os.environ.copy()
        env['MODEL_DIR'] = 'model_finetune'  # Separate checkpoint directory

        # Call main.py with custom checkpoint dir
        cmd = [sys.executable, "src_tensorflow2/main.py", "--train"]

        print(f"Command: {' '.join(cmd)}")
        print(f"Checkpoint dir: model_finetune/")
        print("-" * 70 + "\n")

        result = subprocess.run(cmd, env=env, cwd=".")

        if result.returncode == 0:
            print("\n[OK] Fine-tuning completed!")
            print(f"Checkpoints saved to: model_finetune/")
            return True
        else:
            print("\n[ERROR] Fine-tuning failed")
            return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
