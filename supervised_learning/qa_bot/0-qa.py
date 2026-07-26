#!/usr/bin/env python3
"""Question-answering using a BERT model from TensorFlow Hub."""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


TOKENIZER = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

MODEL = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """Find and return an answer to a question from a reference document.

    Args:
        question (str): Question to answer.
        reference (str): Document containing the possible answer.

    Returns:
        str: Extracted answer, or None if no valid answer is found.
    """
    question_tokens = TOKENIZER.tokenize(question)
    reference_tokens = TOKENIZER.tokenize(reference)

    max_reference_length = 512 - len(question_tokens) - 3
    reference_tokens = reference_tokens[:max_reference_length]

    tokens = (
        [TOKENIZER.cls_token]
        + question_tokens
        + [TOKENIZER.sep_token]
        + reference_tokens
        + [TOKENIZER.sep_token]
    )

    input_word_ids = TOKENIZER.convert_tokens_to_ids(tokens)

    question_segment = [0] * (len(question_tokens) + 2)
    reference_segment = [1] * (len(reference_tokens) + 1)
    input_type_ids = question_segment + reference_segment

    input_mask = [1] * len(input_word_ids)

    model_inputs = {
        "input_word_ids": tf.constant([input_word_ids]),
        "input_mask": tf.constant([input_mask]),
        "input_type_ids": tf.constant([input_type_ids]),
    }

    outputs = MODEL(model_inputs)

    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    reference_start = len(question_tokens) + 2

    start_logits[:reference_start] = -np.inf
    end_logits[:reference_start] = -np.inf

    start_index = int(np.argmax(start_logits))
    end_index = int(np.argmax(end_logits))

    if (
        start_index == 0
        or end_index == 0
        or start_index > end_index
        or start_index >= len(tokens)
        or end_index >= len(tokens)
    ):
        return None

    answer_tokens = tokens[start_index:end_index + 1]
    answer = TOKENIZER.convert_tokens_to_string(answer_tokens).strip()

    if not answer:
        return None

    return answer
