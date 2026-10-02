# generate.py

import torch

from model import MiniGPT, BLOCK_SIZE

from tokenizer import (
    encode,
    decode,
    vocab_size
)


# -----------------------------
# Device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load Model
# -----------------------------

model = MiniGPT(vocab_size)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


print("Mini LLM loaded!")
print("Type 'exit' to stop.\n")


# -----------------------------
# Chat Loop
# -----------------------------

while True:

    prompt = input("You: ")

    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    # Convert prompt to tokens
    encoded = encode(prompt)

    if len(encoded) == 0:
        print("AI: Please enter some text.")
        continue

    # Convert to tensor
    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    # Generate
    with torch.no_grad():

        output = model.generate(
            input_tensor,
            max_new_tokens=200,
            temperature=0.8
        )

    # Convert back to text
    result = decode(
        output[0].tolist()
    )

    print("\nAI:", result)
    print()