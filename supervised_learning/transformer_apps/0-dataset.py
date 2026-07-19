#!/usr/bin/env python3
"""Loads and prepares a dataset for machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Loads the Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize the datasets and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create pretrained tokenizers for the translation dataset.

        Args:
            data: Dataset containing Portuguese-English sentence pairs.

        Returns:
            The Portuguese tokenizer and the English tokenizer.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        return tokenizer_pt, tokenizer_en
