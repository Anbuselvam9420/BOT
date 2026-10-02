# train.py

import torch
import torch.nn as nn
from model import MiniGPT, BLOCK_SIZE
from tokenizer import encode, vocab_size


# -----------------------------
# Device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# -----------------------------
# Read Data
# -----------------------------

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Characters in dataset:", len(text))


# Convert text to numbers
data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# -----------------------------
# Train / Validation Split
# -----------------------------

split = int(0.9 * len(data))

train_data = data[:split]

val_data = data[split:]


# -----------------------------
# Get Batch
# -----------------------------

BATCH_SIZE = 32


def get_batch(data):

    ix = torch.randint(
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack(
        [
            data[i:i + BLOCK_SIZE]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + BLOCK_SIZE + 1]
            for i in ix
        ]
    )

    return x.to(device), y.to(device)


# -----------------------------
# Create Model
# -----------------------------

model = MiniGPT(vocab_size)

model = model.to(device)


print("Model created.")

print(
    "Number of parameters:",
    sum(p.numel() for p in model.parameters())
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# -----------------------------
# Training
# -----------------------------

MAX_ITERS = 5000

PRINT_EVERY = 500


for iteration in range(MAX_ITERS):

    # Get batch
    xb, yb = get_batch(train_data)

    # Forward
    logits, loss = model(
        xb,
        yb
    )

    # Clear gradients
    optimizer.zero_grad(
        set_to_none=True
    )

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()


    # Print progress
    if iteration % PRINT_EVERY == 0:

        print(
            "Step:",
            iteration,
            "Loss:",
            loss.item()
        )


# -----------------------------
# Save Model
# -----------------------------

torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nTraining complete!")

print("Model saved as model.pth")