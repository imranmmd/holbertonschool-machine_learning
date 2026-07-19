#!/usr/bin/env python3
"""Loads and prepares a dataset for machine translation."""

from transformers import AutoTokenizer
from setup import load_pt2en


class Dataset:
    """Loads the Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize the dataset and pretrained tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create tokenizers for the Portuguese and English text.

        Args:
            data: A tf.data.Dataset containing Portuguese-English
                sentence pairs.

        Returns:
            tokenizer_pt: Pretrained Portuguese BERT tokenizer.
            tokenizer_en: Pretrained English BERT tokenizer.
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        return tokenizer_pt, tokenizer_en
