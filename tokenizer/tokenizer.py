import json
import os

VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "vocab_6000.json"
)
TARGET_VOCAB_SIZE = 6000

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

tokens = vocab_data["tokens"]
if len(tokens) != TARGET_VOCAB_SIZE:
    raise ValueError(
        f"Expected {TARGET_VOCAB_SIZE} tokens in {VOCAB_PATH}, "
        f"found {len(tokens)}. Run the vocabulary builder first."
    )

if len(set(tokens)) != TARGET_VOCAB_SIZE:
    raise ValueError("The vocabulary contains duplicate tokens.")

vocab_size = len(tokens)
stoi = {token: index for index, token in enumerate(tokens)}
itos = {index: token for index, token in enumerate(tokens)}
unknown_token = next(
    token
    for token in tokens
    if token.startswith("<UNK_")
)
single_character_tokens = {token for token in tokens if len(token) == 1}
word_tokens = sorted(
    (token for token in tokens if len(token) > 1),
    key=len,
    reverse=True
)
word_trie = {}
for token in word_tokens:
    node = word_trie
    for character in token:
        node = node.setdefault(character, {})
    node[None] = stoi[token]

def encode(text):
    encoded = []
    position = 0

    while position < len(text):
        node = word_trie
        next_position = position
        matched_position = position
        matched_token_id = None

        while next_position < len(text):
            character = text[next_position]
            if character not in node:
                break
            node = node[character]
            next_position += 1
            if None in node:
                matched_token_id = node[None]
                matched_position = next_position

        if matched_token_id is not None:
            encoded.append(matched_token_id)
            position = matched_position
        else:
            character = text[position]
            if character not in single_character_tokens:
                encoded.append(stoi[unknown_token])
            else:
                encoded.append(stoi[character])
            position += 1

    return encoded


def decode(numbers):
    return "".join(itos[index] for index in numbers)

with open(
    os.path.join(os.path.dirname(__file__), "tokenizer.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "version": "1.0",
            "type": "word_character",
            "vocab_size": vocab_size,
            "token_to_id": stoi,
            "id_to_token": {str(index): token for index, token in itos.items()}
        },
    f,
    ensure_ascii=False,
    indent=2
    )

if __name__ == "__main__":
    sample = "Hello"
    encoded = encode(sample)
    decoded = decode(encoded)
    print("Tokenizer Test")
    print("Original:", sample)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Vocabulary size:", vocab_size)
