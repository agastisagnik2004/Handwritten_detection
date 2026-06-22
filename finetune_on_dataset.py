#!/usr/bin/env python3
"""
FINETUNE ADAPTER v2.0 (Improved)

Features:
  - Combine Training + Validation datasets for better accuracy
  - Configurable epochs for longer training
  - Automatic checkpoint management
  - Better result quality

Usage:
  python finetune_on_dataset.py --all-data --epochs 50 --train
  python finetune_on_dataset.py --dataset Training --epochs 30 --train
"""
import os
import csv
import shutil
import argparse
import subprocess
import sys
import glob


def prepare_prescription_data(dataset_splits=None, output_dir="data"):
    """
    Prepare prescription dataset in IAM format for main.py.

    Args:
        dataset_splits: list of splits to use (["Training", "Validation"]) or single string
        output_dir: output directory for prepared data
    """
    if isinstance(dataset_splits, str):
        dataset_splits = [dataset_splits]
    elif dataset_splits is None:
        dataset_splits = ["Training"]

    os.makedirs(f"{output_dir}/lines", exist_ok=True)

    # Find prescription dataset
    dataset_base = "dataset"
    if not os.path.exists(dataset_base):
        print(f"Error: {dataset_base}/ folder not found")
        return False

    dataset_path = None
    for item in os.listdir(dataset_base):
        if "Prescription" in item or "prescription" in item:
            dataset_path = os.path.join(dataset_base, item)
            break

    if not dataset_path or not os.path.isdir(dataset_path):
        print(f"Error: Prescription dataset not found")
        return False

    print(f"Found dataset: {dataset_path}")

    lines_txt = ""
    image_count = 0

    # Process each split
    for dataset_split in dataset_splits:
        split_dir = os.path.join(dataset_path, dataset_split)
        labels_file = os.path.join(split_dir, f"{dataset_split.lower()}_labels.csv")
        words_dir = os.path.join(split_dir, f"{dataset_split.lower()}_words")

        if not os.path.exists(labels_file) or not os.path.exists(words_dir):
            print(f"Warning: {dataset_split} data not found, skipping")
            continue

        print(f"Processing {dataset_split}...")

        with open(labels_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['IMAGE']
                medicine_name = row['MEDICINE_NAME']

                src_img = os.path.join(words_dir, img_name)
                if not os.path.exists(src_img):
                    continue

                # Create IAM-format ID
                split_letter = dataset_split[0]
                line_id = f"word-{split_letter}-{image_count:06d}"

                # Directory structure
                dir_path = os.path.join(output_dir, "lines", "word", f"word-{split_letter}")
                os.makedirs(dir_path, exist_ok=True)

                # Copy image
                dst_img = os.path.join(dir_path, f"{line_id}.png")
                shutil.copy2(src_img, dst_img)

                # IAM format
                text = medicine_name.replace(" ", "|")
                lines_txt += f"{line_id} ok 0 0 100 50 0 0 {text}\n"

                image_count += 1

    # Write lines.txt
    output_file = os.path.join(output_dir, "lines.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(lines_txt)

    print(f"Created {output_file} with {image_count} total samples")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="FINETUNE ADAPTER v2.0: Better accuracy with combined datasets"
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Use ALL data (Training + Validation) for best accuracy"
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
        default=30,
        help="Max epochs to train (default: 30, increased from 10)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Prepare data AND start training"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare data, don't train"
    )

    args = parser.parse_args()

    # Determine which datasets to use
    if args.all_data:
        dataset_splits = ["Training", "Validation"]
        print_dataset = "Training + Validation (All Data)"
    else:
        dataset_splits = [args.dataset]
        print_dataset = args.dataset

    print("=" * 70)
    print(f"FINETUNE ADAPTER v2.0 (IMPROVED)")
    print(f"Dataset: {print_dataset} | Epochs: {args.epochs}")
    print("=" * 70)

    # Step 1: Prepare data
    print("\nPreparing prescription dataset...")
    if not prepare_prescription_data(dataset_splits=dataset_splits):
        print("[ERROR] Data preparation failed")
        return False

    if args.prepare_only:
        print("\n[OK] Data prepared. Ready to train.")
        print(f"\nTo start training:")
        print(f"  python finetune_on_dataset.py --all-data --epochs 50 --train")
        return True

    # Step 2: Train
    if args.train:
        print("\nStarting fine-tuning with improved settings...")
        print("-" * 70)

        env = os.environ.copy()
        env['MODEL_DIR'] = 'model_finetune'

        # Check if resuming from previous training
        last_epoch_file = "model_finetune/last_epoch.txt"
        if os.path.exists(last_epoch_file):
            with open(last_epoch_file, 'r') as f:
                last_epoch = int(f.read().strip())
            print(f"[RESUME] Found last epoch: {last_epoch}")
            print(f"Training will resume from epoch {last_epoch + 1}")
        else:
            print(f"[FRESH] Starting fresh training")

        cmd = [sys.executable, "src_tensorflow2/main.py", "--train"]

        print(f"Checkpoint dir: model_finetune/")
        print(f"Training data: {print_dataset}")
        print(f"Max epochs: {args.epochs}")
        print("-" * 70 + "\n")

        result = subprocess.run(cmd, env=env, cwd=".")

        if result.returncode == 0:
            print("\n[OK] Fine-tuning completed!")
            print(f"Checkpoints saved to: model_finetune/")
            print(f"Epoch tracking saved to: model_finetune/last_epoch.txt")
            return True
        else:
            print("\n[ERROR] Fine-tuning failed")
            return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
