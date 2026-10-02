# tokenizer.py

# Open the training data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a list of all unique characters
chars = sorted(list(set(text)))

# Vocabulary size
vocab_size = len(chars)

# Create character -> number mapping
stoi = {ch: i for i, ch in enumerate(chars)}

# Create number -> character mapping
itos = {i: ch for i, ch in enumerate(chars)}


# Encode: text -> numbers
def encode(text):
    return [stoi[ch] for ch in text]


# Decode: numbers -> text
def decode(numbers):
    return "".join(itos[i] for i in numbers)


# Test the tokenizer
sample = "Artificial intelligence"

encoded = encode(sample)
decoded = decode(encoded)

print("Original:")
print(sample)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(decoded)

print("\nVocabulary size:")
print(vocab_size)