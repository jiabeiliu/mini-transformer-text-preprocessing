# Text preprocessing for a small Transformer (coursework, Part 1)

This repository contains the data-preparation half of a two-part learning project. It cleans local text, removes exact duplicate documents, tokenizes with the GPT-2 tokenizer, and writes fixed-length token blocks. The companion [small Transformer training notebook](https://github.com/jiabeiliu/mini-gpt-training-experiment) is Part 2.

## What runs today

```text
Local .txt files → normalize/filter → exact-document deduplication
                 → GPT-2 token IDs → padded blocks → PyTorch .pt file
```

The script does **not** download a dataset by default. No raw corpus or generated checkpoint is committed here. Use text you have permission to process; dataset quality, licensing, and provenance are your responsibility. The filename `tokenized_200mb.pt` is historical and does not guarantee that 200 MB of source text was processed.

## Reproduce

Use Python 3.11 and install the dependencies in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data/raw
# Put your own UTF-8 .txt files in data/raw/.
python data_collection_preprocessing.py --input-dir data/raw --output data/processed/tokenized_blocks.pt
```

For a tiny smoke input, use `--min-words 5 --block-size 32`; meaningful model training needs a much larger, legally usable corpus. The first run downloads the GPT-2 tokenizer. The script reports how many files and blocks it processed and errors clearly if there is no usable input.

The output is a PyTorch dictionary with `input_ids` and `attention_mask`, each shaped `(number_of_blocks, block_size)`. Padding positions have attention mask `0`. Generated data is ignored by Git.

Run the network-free unit checks with `python -m unittest discover -s tests -v`. A separate local smoke run using the repository's historical report as temporary input produced 19 blocks of 32 tokens; that verifies plumbing only, not the historical 200 MB claim or training quality.

## Design and limitations

- Cleaning lowercases text, strips simple HTML/Markdown markup and non-ASCII characters, and drops short documents. These choices are easy to inspect but discard multilingual text and some useful formatting.
- Deduplication is exact after normalization; near-duplicates remain.
- Token blocks are non-overlapping and the final block of each document is padded. This is a teaching pipeline, not an optimized large-scale data loader.
- The optional `download_200mb_subset` function remains in the source for the original assignment, but the documented run processes only local files and does not imply that OpenWebText is available or licensed for every use.

See [Assignment1_Report.md](Assignment1_Report.md) for the original coursework context. Do not present this preprocessing step alone as a trained foundation model.
