# load_data.py
import os

# Locate the raw data folder relative to this file (Assign 1/data/raw)
DATA_FOLDER = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "raw"))

all_texts = []
if os.path.isdir(DATA_FOLDER):
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_FOLDER, filename), "r", encoding="utf-8") as f:
                all_texts.append(f.read())
else:
    raise FileNotFoundError(f"Data folder not found: {DATA_FOLDER}")

print(f"Loaded {len(all_texts)} files from {DATA_FOLDER}")
