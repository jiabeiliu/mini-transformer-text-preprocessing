# data_collection_preprocessing.py
"""
data_collection_preprocessing.py

This script will:
 - Ensure a `sample_text_dataset` folder contains ~200MB of raw text (download a public dataset subset if needed)
 - Run cleaning: deduplication, HTML/markdown removal, short-doc filter (<50 words), basic normalization
 - Tokenize with a Hugging Face tokenizer and chunk long sequences into 512-token blocks
 - Save tokenized blocks as a torch .pt file and provide a small DataLoader smoke test

Designed to run in the project's venv.
"""

import argparse
import os
import re
import hashlib
from pathlib import Path
from tqdm import tqdm
import torch

TARGET_RAW_BYTES = 200 * 1024 * 1024  # 200 MB
PROJECT_ROOT = Path(__file__).resolve().parent
# New, clearer data layout
RAW_DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_FOLDER = PROJECT_ROOT / "data" / "processed"
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)


def download_200mb_subset(target_bytes=TARGET_RAW_BYTES):
    """Stream a public dataset (OpenWebText) and save raw text files until target size reached."""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "datasets library required to download data: pip install datasets") from e

    cur_bytes = sum(f.stat().st_size for f in RAW_DATA_FOLDER.glob("*.txt"))
    if cur_bytes >= target_bytes:
        print(f"Already have {cur_bytes} bytes >= target {target_bytes}")
        return

    print("Downloading streaming subset from OpenWebText until ~200MB is reached...")
    ds = load_dataset("openwebtext", split="train", streaming=True)
    idx = 0
    for item in ds:
        text = item.get("text") or item.get("content") or ""
        if not text or len(text) < 200:  # skip extremely short items
            continue
        fname = RAW_DATA_FOLDER / f"download_{idx:06d}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(text)
        idx += 1
        cur_bytes += fname.stat().st_size
        if idx % 50 == 0:
            print(f"Downloaded {idx} files, {cur_bytes/1024/1024:.2f} MB")
        if cur_bytes >= target_bytes:
            print(f"Reached target size: {cur_bytes} bytes")
            break


def load_texts_from_folder(folder=RAW_DATA_FOLDER):
    texts = []
    for p in sorted(folder.glob("*.txt")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception:
            continue
    print(
        f"Loaded {len(texts)} files from {folder} (total {sum(len(t) for t in texts)} chars)")
    return texts


def clean_and_filter_texts(texts, min_words=50):
    seen = set()
    cleaned = []
    for t in texts:
        # Basic normalization
        s = t.lower()
        # Remove HTML tags
        s = re.sub(r"<[^>]+>", " ", s)
        # Remove markdown links and images
        s = re.sub(r"!\[.*?\]\(.*?\)", " ", s)
        s = re.sub(r"\[.*?\]\(.*?\)", " ", s)
        # Remove extra whitespace
        s = re.sub(r"\s+", " ", s).strip()
        # Remove non-printable/uncommon chars
        s = re.sub(r"[^\x00-\x7F]+", " ", s)

        # Deduplicate by content hash
        h = hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        # Filter short documents
        if len(s.split()) < min_words:
            continue

        cleaned.append(s)

    print(
        f"After cleaning/filtering: {len(cleaned)} documents (from {len(texts)})")
    return cleaned


def tokenize_and_chunk(texts, model_name="gpt2", block_size=512):
    from transformers import AutoTokenizer

    if not texts:
        raise ValueError("No documents remain after cleaning; provide more input text or lower --min-words.")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_blocks_input_ids = []
    all_blocks_attention = []

    for t in tqdm(texts, desc="Tokenizing texts"):
        enc = tokenizer(t, add_special_tokens=True, return_attention_mask=True)
        input_ids = enc["input_ids"]
        attention = enc["attention_mask"]
        # chunk into non-overlapping blocks of block_size
        for i in range(0, len(input_ids), block_size):
            chunk_ids = input_ids[i: i + block_size]
            chunk_att = attention[i: i + block_size]
            # pad if shorter than block size
            if len(chunk_ids) < block_size:
                pad_len = block_size - len(chunk_ids)
                chunk_ids = chunk_ids + [tokenizer.pad_token_id] * pad_len
                chunk_att = chunk_att + [0] * pad_len
            all_blocks_input_ids.append(
                torch.tensor(chunk_ids, dtype=torch.long))
            all_blocks_attention.append(
                torch.tensor(chunk_att, dtype=torch.long))

    input_ids_tensor = torch.stack(all_blocks_input_ids)
    attention_tensor = torch.stack(all_blocks_attention)
    print(f"Created {input_ids_tensor.size(0)} blocks of size {block_size}")
    return {"input_ids": input_ids_tensor, "attention_mask": attention_tensor}


def save_tokenized_blocks(tokenized, out_path=PROCESSED_DATA_FOLDER / "tokenized_200mb.pt"):
    os.makedirs(out_path.parent, exist_ok=True)
    torch.save(tokenized, str(out_path))
    print(
        f"Saved tokenized blocks to {out_path} (shapes: input_ids={tokenized['input_ids'].shape})")


def small_dataloader_smoke_test(tokenized, batch_size=8):
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(
        tokenized["input_ids"], tokenized["attention_mask"])
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dl:
        print("Batch shapes:", batch[0].shape, batch[1].shape)
        break


def main():
    parser = argparse.ArgumentParser(description="Clean and tokenize local .txt files for language-model coursework.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_FOLDER)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_FOLDER / "tokenized_200mb.pt")
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()

    if not args.input_dir.is_dir() or not any(args.input_dir.glob("*.txt")):
        parser.error(f"No .txt files found in {args.input_dir}. Add licensed/local text files or pass --input-dir.")
    texts = load_texts_from_folder(args.input_dir)

    # 3) Clean, dedupe, filter
    cleaned = clean_and_filter_texts(texts, min_words=args.min_words)

    # 4) Tokenize and chunk
    tokenized = tokenize_and_chunk(cleaned, model_name="gpt2", block_size=args.block_size)

    # 5) Save tokenized blocks
    save_tokenized_blocks(tokenized, args.output)

    # 6) Smoke test DataLoader
    small_dataloader_smoke_test(tokenized)


if __name__ == "__main__":
    main()
