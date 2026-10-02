#!/usr/bin/env python3
# Vocabulary builder
# Build a 6000-token vocabulary using word-level and subword tokenization

import json
import re
from collections import Counter
from pathlib import Path

# Load training data
data_files = sorted(Path("data").rglob("*.txt"))
if not data_files:
    raise FileNotFoundError("No .txt training files were found under the data/ folder.")

text = "\n\n".join(
    data_file.read_text(encoding="utf-8")
    for data_file in data_files
)

# ============================================================
# STEP 1: Build character set (base tokens 0-127)
# ============================================================
base_chars = sorted(list(set(text)))
print(f"Total unique characters: {len(base_chars)}")

char_to_id = {ch: i for i, ch in enumerate(base_chars)}
id_to_char = {i: ch for i, ch in enumerate(base_chars)}

# ============================================================
# STEP 2: Extract words and build word-level vocabulary
# ============================================================
# Split text into words
words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
word_freq = Counter(words)

# Get top words
vocab_size_target = 6000
num_word_tokens = 5500  # Reserve space for subword tokens
top_words = word_freq.most_common(num_word_tokens)

print(f"Top words found: {len(top_words)}")

# ============================================================
# STEP 3: Build final vocabulary
# ============================================================
vocab = []
vocab_set = set()


def add_token(token):
    if token not in vocab_set and len(vocab) < vocab_size_target:
        vocab.append(token)
        vocab_set.add(token)

# Add characters (base tokens)
for character in base_chars:
    add_token(character)

# Add words as multi-character tokens
for word, freq in top_words:
    add_token(word)

# Pad with empty strings if needed
while len(vocab) < vocab_size_target:
    add_token(f"<UNK_{len(vocab)}>")

print(f"Final vocabulary size: {len(vocab)}")

# ============================================================
# STEP 4: Save vocabulary
# ============================================================
vocab_data = {
    "vocab_size": len(vocab),
    "tokens": vocab,
    "char_tokens": len(base_chars),
    "word_tokens": len(top_words)
}

for output_path in ("vocab_6000.json", "vocab.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

print("\n✓ Vocabulary saved to vocab_6000.json")
print(f"  - Characters: {len(base_chars)}")
print(f"  - Words: {len(top_words)}")
print(f"  - Total: {len(vocab)}")
print(f"\nFirst 10 tokens: {vocab[:10]}")
print(f"Sample word tokens: {vocab[len(base_chars):len(base_chars)+10]}")
