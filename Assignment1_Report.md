Assignment 1 — Data Collection & Preprocessing for Foundation Model Pre-Training

> Portfolio note (2026-10-06): This is the original coursework narrative. Its ~200 MB corpus, generated artifact, and several `Assign 1/` paths are **not present in this GitHub checkout**, so the historical size/result claims are not independently reproducible here. Use the current README and CLI for a fresh run with your own licensed text; do not treat this report as a verified benchmark.

1. Dataset sources and size

- Source: existing `sample_text_dataset` folder that comes with the repo (mixed domain samples: books, news, openwebtext, stackexchange, wikipedia).
- Raw characters loaded: 207,838,292 (approx 198–208 MB of text content). This satisfies the user's request to work with ~200MB of data for the assignment demonstration (note: assignment asked for >=1GB; this workspace sample is ~200MB).

2. Cleaning strategies

- Lowercased all text to normalize casing.
- Removed HTML tags via regex and stripped Markdown links and images.
- Normalized whitespace and removed non-ASCII characters (keeps basic printable ASCII only).
- Deduplicated documents using SHA-256 content hash.
- Removed short documents (<50 words) to avoid low-quality samples.

Rationale: these operations are lightweight, fast, and remove obvious noise. For production pretraining, more advanced filtering such as language detection, heuristics for boilerplate removal, and aggressive deduplication across large corpora would be used.

3. Tokenization choices

- Tokenizer: Hugging Face `AutoTokenizer.from_pretrained("gpt2")` (GPT-2 BPE tokenizer).
- Block size: 512 tokens (non-overlapping chunking).
- Padding: right-pad shorter chunks with tokenizer.pad_token_id.

Rationale: GPT-style tokenizer is compatible with transformer models for next-token prediction. Block size 512 is a common choice balancing context and memory.

4. Data loader implementation

- `preprocess/dataset.py` contains `TokenBlockDataset`, a small `torch.utils.data.Dataset` that loads the saved tokenized blocks (`tokenized_200mb.pt`) and returns `input_ids` and `attention_mask` per item.
- A simple `DataLoader` smoke-test is included in `data_collection_preprocessing.py` which prints batch shapes.

5. Challenges and notes

- Tokenization required installing `transformers` and `torch`. Tokenizer download occurs the first time and needs network access.
- The minimal cleaning pipeline may remove some non-English characters; extend if multilingual datasets are used.
- For very large corpora (>GB), streaming tokenization and sharded storage (e.g., multiple .pt shards) would be recommended.

6. Files produced

- `sample_text_dataset/tokenized_200mb.pt` — contains a dict with `input_ids` (N,512) and `attention_mask` (N,512) tensors.
- `preprocess/dataset.py` — PyTorch dataset loader.
- `Assign 1/data_collection_preprocessing.py` — main pipeline script (processing local dataset only).
- `Assign 1/requirements.txt`, `README.md`, `tests/test_tokenized_load.py`.

7. How to run (reproducibility)

```bash
source .venv/bin/activate
pip install -r "Assign 1/requirements.txt"
python "Assign 1/data_collection_preprocessing.py"
# Run tests
python "Assign 1/tests/test_tokenized_load.py"
```

8. Reflections

This exercise implements the core preprocessing steps for foundation model pretraining: cleaning, deduplication, tokenization, and batching. To scale to multi-GB corpora, I'd add streaming tokenization (write shards incrementally), stronger deduplication (minhash/LSH), language filtering, and more robust artifact/versioning for reproducibility.


Prepared by: Automated preprocessing script
Date: 2025-10-31
