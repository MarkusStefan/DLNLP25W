#%% ############################################################
#%% # TINY SHAKESPEARE DATA PREPARATION
#%% ############################################################

"""
Preprocess the Tiny Shakespeare dataset (`tiny.txt`) for subsequent language
model experiments. Outputs word-level and character-level resources tailored
for n-gram modeling, NumPy RNN training, and PyTorch experiments.
"""

#%% # Imports and file locations
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

BASE_DIR = Path(__file__).parent
RAW_PATH = BASE_DIR / "tiny.txt"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {RAW_PATH}")
raw_text = RAW_PATH.read_text(encoding="utf-8")
print(f"Raw text length (characters): {len(raw_text)}")


#%% # Sentence splitting and tokenization utilities

def split_lines(text: str) -> list[str]:
    """Return non-empty lines stripped of surrounding whitespace."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def tokenize_words(line: str) -> list[str]:
    """
    Lowercase and split text into tokens. Keeps apostrophes and treats
    punctuation as separate tokens to maintain language structure.
    """
    lowered = line.lower()
    return re.findall(r"[a-z']+|[0-9]+|[^\s\w]", lowered)


lines = split_lines(raw_text)
tokenized_lines = [tokenize_words(line) for line in lines]

all_word_tokens = [token for sentence in tokenized_lines for token in sentence]
word_vocab = sorted(set(all_word_tokens))
word_token_to_idx = {token: idx for idx, token in enumerate(word_vocab)}

print(f"Number of non-empty lines: {len(lines)}")
print(f"Word vocabulary size: {len(word_vocab)}")


#%% # Train/test split for word-level work

split_ratio = 0.8
split_idx = math.floor(len(tokenized_lines) * split_ratio)

train_sentences = tokenized_lines[:split_idx]
test_sentences = tokenized_lines[split_idx:]

train_tokens = [token for sentence in train_sentences for token in sentence]
test_tokens = [token for sentence in test_sentences for token in sentence]


#%% # Sliding windows for word-level RNN tasks

sequence_length = 12  # 12 input tokens -> predict the 13th
word_token_ids = [word_token_to_idx[token] for token in all_word_tokens]

word_windows: list[list[int]] = []
for start in range(len(word_token_ids) - sequence_length):
    window = word_token_ids[start : start + sequence_length + 1]
    word_windows.append(window)

print(f"Word-level training windows: {len(word_windows)}")


#%% # Persist word-level artifacts

word_artifacts = {
    "vocab": word_vocab,
    "token_to_idx": word_token_to_idx,
    "train_sentences": train_sentences,
    "test_sentences": test_sentences,
    "word_sequence_length": sequence_length,
}

(OUTPUT_DIR / "tiny_shakespeare_words.json").write_text(
    json.dumps(word_artifacts, indent=2), encoding="utf-8"
)

np.save(OUTPUT_DIR / "tiny_shakespeare_word_windows.npy", np.array(word_windows, dtype=np.int32))


#%% # Character-level resources

chars = sorted(set(raw_text))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
encoded_text = np.array([char_to_idx[ch] for ch in raw_text], dtype=np.int32)

char_sequence_length = 60
char_step = 3

char_windows: list[list[int]] = []
for start in range(0, len(encoded_text) - char_sequence_length - 1, char_step):
    window = encoded_text[start : start + char_sequence_length + 1]
    char_windows.append(window.tolist())

print(f"Character vocabulary size: {len(chars)}")
print(f"Character-level sequences: {len(char_windows)}")

char_artifacts = {
    "chars": chars,
    "char_to_idx": char_to_idx,
    "sequence_length": char_sequence_length,
    "step": char_step,
}

(OUTPUT_DIR / "tiny_shakespeare_chars.json").write_text(
    json.dumps(char_artifacts, indent=2), encoding="utf-8"
)

np.save(OUTPUT_DIR / "tiny_shakespeare_char_windows.npy", np.array(char_windows, dtype=np.int32))


#%% # Summary

print("Saved artifacts:")
for path in sorted(OUTPUT_DIR.glob("tiny_shakespeare_*")):
    size_kb = path.stat().st_size / 1024
    print(f"- {path.name} ({size_kb:.1f} KB)")
