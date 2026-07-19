#!/usr/bin/env python3
"""Load and prepare a dataset for machine translation."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Prepare the Portuguese-to-English translation dataset."""

    def __init__(self):
        """Load, tokenize, and encode the translation datasets."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Create subword tokenizers from the training dataset.

        Args:
            data: Dataset containing Portuguese-English sentence pairs.

        Returns:
            A tuple containing the Portuguese and English tokenizers.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            (
                pt.numpy().decode('utf-8')
                for pt, _ in data
            ),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            (
                en.numpy().decode('utf-8')
                for _, en in data
            ),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a Portuguese-English translation pair.

        Args:
            pt: Tensor containing a Portuguese sentence.
            en: Tensor containing the corresponding English sentence.

        Returns:
            Lists containing the Portuguese and English token IDs.
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False
        )

        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False
        )

        pt_start = self.tokenizer_pt.vocab_size
        en_start = self.tokenizer_en.vocab_size

        pt_tokens = [pt_start] + pt_tokens + [pt_start + 1]
        en_tokens = [en_start] + en_tokens + [en_start + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Wrap the encode method for use in a TensorFlow data pipeline.

        Args:
            pt: Tensor containing a Portuguese sentence.
            en: Tensor containing the corresponding English sentence.

        Returns:
            Encoded Portuguese and English tensors.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
