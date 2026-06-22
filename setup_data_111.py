
"""Extract data_111 parquet files and organize into IAM format"""
import pandas as pd
import os
from pathlib import Path

print("Loading data_111 parquet files...")

# Load training data
train_df = pd.read_parquet("data_111/train.parquet")
print(f"Train samples: {len(train_df)}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"Sample row:")
print(train_df.iloc[0])

# Check structure
print("\nChecking data structure...")
print(train_df.head())

# Create lines.txt from training data
os.makedirs("data", exist_ok=True)
lines_txt = ""

for idx, row in train_df.iterrows():
    # Get file ID and text
    file_id = row.get('id', row.get('filename', f'line_{idx}'))
    text = str(row.get('text', row.get('label', 'unknown'))).strip()

    if not text or text == 'unknown':
        continue

    # Replace spaces with pipes
    text_formatted = text.replace(' ', '|')

    # IAM format line
    lines_txt += f"{file_id} ok 0 0 100 50 0 0 {text_formatted}\n"

# Write file
with open("data/lines.txt", "w", encoding="utf-8") as f:
    f.write(lines_txt)

print(f"\nSuccessfully created data/lines.txt with {len(lines_txt.splitlines())} entries from data_111/train.parquet")
print("\nFirst 5 entries:")
for line in lines_txt.split("\n")[:5]:
    if line:
        print(f"  {line[:80]}")
