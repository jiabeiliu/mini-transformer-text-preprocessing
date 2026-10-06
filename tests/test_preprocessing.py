import tempfile
import unittest
from pathlib import Path

import torch

from data_collection_preprocessing import clean_and_filter_texts, save_tokenized_blocks


class PreprocessingTests(unittest.TestCase):
    def test_exact_duplicate_and_short_document_filter(self):
        documents = ["<p>Alpha beta gamma delta epsilon.</p>",
                     "alpha beta gamma delta epsilon.", "short"]
        cleaned = clean_and_filter_texts(documents, min_words=5)
        self.assertEqual(cleaned, ["alpha beta gamma delta epsilon."])

    def test_artifact_shape_and_keys(self):
        payload = {"input_ids": torch.tensor([[1, 2, 0]]),
                   "attention_mask": torch.tensor([[1, 1, 0]])}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tokens.pt"
            save_tokenized_blocks(payload, output)
            saved = torch.load(output, weights_only=True)
        self.assertEqual(set(saved), {"input_ids", "attention_mask"})
        self.assertEqual(tuple(saved["input_ids"].shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
