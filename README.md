Assignment 1 — Data Collection & Preprocessing

Overview

This folder contains scripts to download a ~200MB subset of OpenWebText, clean and deduplicate documents, tokenize with a GPT-style tokenizer (GPT-2), chunk into fixed-size blocks, and save tokenized blocks as a PyTorch .pt file.

Quick run (use the project's venv):

```bash
# From project root
source .venv/bin/activate
python "Assign 1/data_collection_preprocessing.py"
# Assignment 1 — Data Collection & Preprocessing

This directory contains a small, reproducible pipeline you can use as the preprocessing step for foundation model pretraining. It demonstrates collecting (or using local) raw text, basic cleaning and deduplication, tokenization with a Hugging Face tokenizer, chunking into fixed-size blocks, and saving those blocks as a PyTorch `.pt` file.

This repository is intentionally compact so you can inspect, run, and extend it quickly. The pipeline was run on the workspace's local sample files (≈200 MB of raw text) and produced tokenized blocks for downstream training tasks.

## What is included

- `data_collection_preprocessing.py` — main pipeline. Loads raw `.txt` files from `sample_text_dataset/`, cleans and deduplicates them, tokenizes (GPT-2 tokenizer by default), chunks into 512-token blocks, and saves the tokenized blocks.
- `sample_text_dataset/` — raw text files used as input (books, news, Wikipedia, etc.).
- `tokenized_200mb.pt` — tokenized blocks saved at the root of `Assign 1/` (so you can easily access it for future assignments).
- `preprocess/dataset.py` — `TokenBlockDataset`, a small PyTorch `Dataset` for loading the saved tensor blocks.
- `requirements.txt` — pinned Python packages used by the pipeline.
- `tests/test_tokenized_load.py` — minimal smoke test that loads the `.pt` file and checks shapes.
- `Assignment1_Report.md` — short human-readable report describing choices and results.

## Quick start

1. Activate the project's virtual environment (or create one):

```bash
cd "/Users/amankhan/Downloads/PythonProject"
source .venv/bin/activate
```

2. Install python dependencies (if you haven't already):

```bash
pip install -r "Assign 1/requirements.txt"
```

3. Run the preprocessing pipeline (this version processes the local `sample_text_dataset/` files only):

```bash
python "Assign 1/data_collection_preprocessing.py"
```

4. Run the quick test to verify the tokenized file loads:

```bash
python "Assign 1/tests/test_tokenized_load.py"
```

## Output files and layout

- `Assign 1/tokenized_200mb.pt` — main tokenized artifact. This file contains a dict with two tensors:
	- `input_ids`: shape `(N, 512)` where `N` is the number of blocks created by chunking.
	- `attention_mask`: same shape `(N, 512)`.

- `Assign 1/sample_text_dataset/` — original raw `.txt` files used as input. Keep this folder if you plan to re-process or extend the dataset.

Keeping `tokenized_200mb.pt` at the top-level of `Assign 1/` makes it easier to reference in future assignments and in GitHub without hunting through subfolders.

## Design notes (why things are the way they are)

- Cleaning: we apply lightweight, robust cleaning (lowercase, HTML/markdown removal, whitespace normalization, removal of non-ASCII characters, deduplication via SHA-256, and dropping very short documents). These steps are fast and remove obvious noise; for production you'd add language detection and duplicate detection at scale.
- Tokenization: we use `AutoTokenizer.from_pretrained("gpt2")` for compatibility with GPT-style models. We chunk into 512-token blocks (non-overlapping) and right-pad the final chunk. You can change `block_size` and `model_name` in `data_collection_preprocessing.py`.
- Data loader: `preprocess/dataset.py` shows a simple `TokenBlockDataset` that reads the saved `.pt` dictionary and yields items suitable for a `DataLoader`.

## Things to change quickly

- To change block size or tokenizer model, edit `tokenize_and_chunk(..., model_name=<model>, block_size=<int>)` in `data_collection_preprocessing.py`.
- To add stricter filtering (e.g., language detection, boilerplate removal), enhance `clean_and_filter_texts()` in the same script.

## Troubleshooting

- If the tokenizer download fails, ensure you have network access and sufficient disk space. The tokenizer is downloaded the first time `transformers` loads the model.
- If you see memory errors during tokenization, consider processing and saving in streaming shards rather than loading all token blocks into memory—this is a next-step improvement for multi-GB datasets.
- If tests fail, run the pipeline first to regenerate `tokenized_200mb.pt` and then re-run tests.

## Next steps / recommendations

- For a full assignment submission targeting ≥1 GB of raw text, we can add a streaming downloader (Hugging Face `datasets` streaming) and sharded tokenization that writes multiple `.pt` files.
- Add a PDF version of `Assignment1_Report.md` for submission (I can generate this for you).
- If you plan to push to GitHub, add a `.gitignore` and avoid pushing very large `.pt` files; instead consider pushing a small sample and provide instructions to regenerate the full artifact.

## Contact / notes

If you want, I can:
- convert the report to PDF and create a submission ZIP; or
- implement sharded streaming tokenization to scale to GB+ corpora; or
- create a `.gitignore` and a short `submission/` manifest for GitHub.

Pick one and I'll prepare it next.
