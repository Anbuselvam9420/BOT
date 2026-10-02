# training.py

from pathlib import Path

import torch

from chatbot.transformer import TransformerLM, Config
from tokenizer.tokenizer import encode, vocab_size


# ============================================================
# SETTINGS
# ============================================================

BLOCK_SIZE = Config.max_seq_len
BATCH_SIZE = 32

MAX_ITERS = 6500



PRINT_EVERY = 100
EVAL_EVERY = 500
CHECKPOINT_EVERY = 100

LEARNING_RATE = 3e-4


# ============================================================
# 1. DEVICE
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("=" * 50)
print("        MINI LLM TRAINING")
print("=" * 50)

print("Device:", device)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading all training data from data/...")

data_files = sorted(Path("data").rglob("*.txt"))

if not data_files:
    raise FileNotFoundError(
        "No .txt training files were found under the data/ folder."
    )

text_parts = []

for data_file in data_files:
    file_text = data_file.read_text(encoding="utf-8")
    text_parts.append(file_text)
    print(f"{data_file} characters: {len(file_text)}")

text = "\n\n".join(text_parts)
text = "".join(
    character
    for character in text
    if character in "\n\r\t" or ord(character) >= 32
)

print("Training files:", len(data_files))
print("Total characters:", len(text))


# ============================================================
# 3. TOKENIZE DATA
# ============================================================

print("\nConverting text into token IDs...")


encoded_text = encode(text)

data = torch.tensor(
    encoded_text,
    dtype=torch.long
)


print("Total tokens:", len(data))


# ============================================================
# 4. CHECK DATA SIZE
# ============================================================

if len(data) <= BLOCK_SIZE + 1:
    raise ValueError(
        f"Not enough training data!\n"
        f"Tokens: {len(data)}\n"
        f"Block size: {BLOCK_SIZE}\n"
        f"Add more text to data.txt and conversation.txt."
    )


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

split = int(0.90 * len(data))

train_data = data[:split]
validation_data = data[split:]


print("\nTraining tokens:", len(train_data))
print("Validation tokens:", len(validation_data))


# ============================================================
# 6. BATCH SETTINGS
# ============================================================

print("Batch size:", BATCH_SIZE)
print("Block size:", BLOCK_SIZE)


# ============================================================
# 7. GET BATCH FUNCTION
# ============================================================

def get_batch(data):

    # Make sure the dataset is large enough
    if len(data) <= BLOCK_SIZE:
        raise ValueError(
            "Dataset is too small for the selected BLOCK_SIZE."
        )

    # Random starting positions
    ix = torch.randint(
        low=0,
        high=len(data) - BLOCK_SIZE,
        size=(BATCH_SIZE,)
    )

    # Input sequences
    x = torch.stack(
        [
            data[i:i + BLOCK_SIZE]
            for i in ix
        ]
    )

    # Target sequences
    # Target is input shifted by one token
    y = torch.stack(
        [
            data[i + 1:i + BLOCK_SIZE + 1]
            for i in ix
        ]
    )

    # Move to CPU/GPU
    x = x.to(device)
    y = y.to(device)

    return x, y
# ============================================================
# 8. CREATE MODEL
# ============================================================

print("\nCreating model...")

# Create configuration
config = Config()

# Set vocabulary size
config.vocab_size = vocab_size

# Create Transformer model
model = TransformerLM(config)

# Move model to CPU/GPU
model = model.to(device)


# ============================================================
# 9. COUNT PARAMETERS
# ============================================================

number_of_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print("Vocabulary size:", vocab_size)

print(
    "Number of parameters:",
    number_of_parameters
)


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


print("Learning rate:", LEARNING_RATE)

# ============================================================
# 10.5 LOAD CHECKPOINT
# ============================================================

import os

start_step = 0

if os.path.exists("checkpoint.pth"):

    try:

        checkpoint = torch.load(
            "checkpoint.pth",
            map_location=device
        )

        checkpoint_vocab_size = checkpoint.get("vocab_size")

        if checkpoint_vocab_size != vocab_size:
            print(
                "Checkpoint vocabulary size does not match the current "
                f"vocabulary ({checkpoint_vocab_size} != {vocab_size})."
            )
            print("Starting training from scratch.")
        else:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            start_step = checkpoint["step"]

            print(
                f"Resuming from step {start_step}"
            )

    except Exception as e:

        print(
            "Could not load checkpoint."
        )

        print(e)

        start_step = 0


# ============================================================
# 11. LOSS EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, dataset in [
        ("train", train_data),
        ("validation", validation_data)
    ]:

        losses = torch.zeros(10)

        for k in range(10):

            xb, yb = get_batch(dataset)

            logits, loss = model(
                xb,
                yb
            )

            losses[k] = loss.item()

        results[name] = losses.mean().item()

    model.train()

    return results


# ============================================================
# 12. TRAINING
# ============================================================

print("\n")
print("=" * 50)
print("TRAINING STARTED")
print("=" * 50)


for step in range(start_step, MAX_ITERS):

    # Get training batch
    xb, yb = get_batch(train_data)


    # Forward pass
    logits, loss = model(
        xb,
        yb
    )


    # Clear old gradients
    optimizer.zero_grad(
        set_to_none=True
    )


    # Backpropagation
    loss.backward()


    # Update model
    optimizer.step()


    # Print progress
    if step % PRINT_EVERY == 0:

        print(
            f"Step {step:5d} | "
            f"Training Loss: {loss.item():.4f}"
        )


    # Evaluate model
    if step % EVAL_EVERY == 0:

        losses = estimate_loss()

        print(
            f"           "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['validation']:.4f}"
        )
    # Save checkpoint every 100 steps
    if step > 0 and step % CHECKPOINT_EVERY == 0:

        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "vocab_size":
                vocab_size,

            "block_size":
                BLOCK_SIZE,

            "step":
                step
        }

        torch.save(
            checkpoint,
            f"checkpoint_step_{step}.pth"
        )

        # Update latest checkpoint
        torch.save(
            checkpoint,
            "checkpoint.pth"
        )

        print(
            f"Checkpoint saved at step {step}"
        )

# ============================================================
# 13. SAVE MODEL
# ============================================================

print("\n")
print("=" * 50)
print("TRAINING FINISHED")
print("=" * 50)


torch.save(
    model.state_dict(),
    "model.pth"
)


print("\nModel saved successfully!")
print("File: model.pth")


# ============================================================
# 14. SAVE CHECKPOINT
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "vocab_size":
        vocab_size,

    "block_size":
        BLOCK_SIZE,

    "step":
        MAX_ITERS
}


torch.save(
    checkpoint,
    "checkpoint.pth"
)


print("\nCheckpoint saved!")
print("File: checkpoint.pth")       

                                                     
print("\nYour LLM training is complete! 🤖")         