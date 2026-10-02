# chatbot.py

import torch

from .transformer import TransformerLM, Config
from tokenizer.tokenizer import encode, decode, vocab_size

# ============================================================
# 1. DEVICE
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


print("=" * 50)
print("         LET WE talk")
print("=" * 50)

print("Device:", device)


# ============================================================
# 2. CREATE MODEL
# ============================================================
config = Config()
config.vocab_size = vocab_size

model = TransformerLM(config)
model = model.to(device)



# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

try:

    state_dict = torch.load(
        "model.pth",
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

except FileNotFoundError:

    print("\nERROR: model.pth was not found.")

    print("Please train the model first:")
    print("python train.py")

    exit()

except RuntimeError as error:

    print("\nERROR: model.pth does not match the current model configuration.")
    print("The current tokenizer uses 6000 tokens, but this model was trained")
    print("with a different vocabulary size. Retrain the model from scratch:")
    print("python -m training.training")
    print(f"\nDetails: {error}")

    exit()


# ============================================================
# 4. EVALUATION MODE
# ============================================================

model.eval()


print("\nModel loaded successfully!")
print("Your chatbot is ready.")
print("Type 'exit' to stop.")
print("Type 'clear' to clear conversation.\n")


# ============================================================
# 5. CONVERSATION HISTORY
# ============================================================

conversation_history = ""


# ============================================================
# 6. CHAT LOOP
# ============================================================

while True:

    try:

        user_message = input("You: ")

    except KeyboardInterrupt:

        print("\n\nGoodbye!")
        break


    # Remove extra spaces
    user_message = user_message.strip()


    # Ignore empty messages
    if user_message == "":
        continue


    # ========================================================
    # EXIT
    # ========================================================

    if user_message.lower() == "exit":

        print("BRO: Goodbye! Have a great day.")

        break


    # ========================================================
    # CLEAR HISTORY
    # ========================================================

    if user_message.lower() == "clear":

        conversation_history = ""

        print("AI: Conversation cleared.\n")

        continue


    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    conversation_history += (
        "\nUser: "
        + user_message
        + "\nAssistant:"
    )


    # ========================================================
    # ENCODE
    # ========================================================

    try:

        encoded = encode(
            conversation_history
        )

    except Exception as error:

        print("Encoding error:", error)

        continue


    # ========================================================
    # CHECK INPUT
    # ========================================================

    if len(encoded) == 0:

        print("BRO: I could not understand that.")

        continue


    # ========================================================
    # CONVERT TO TENSOR
    # ========================================================

    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )


    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    with torch.no_grad():

        output = model.generate(
            input_tensor,
            max_new_tokens=200,
            temperature=0.8
        )


    # ========================================================
    # DECODE
    # ========================================================

    generated_text = decode(
        output[0].tolist()
    )


    # ========================================================
    # EXTRACT ASSISTANT RESPONSE
    # ========================================================

    if "Assistant:" in generated_text:

        response = generated_text.split(
            "Assistant:"
        )[-1]

    else:

        response = generated_text


    # ========================================================
    # STOP AT NEXT USER MESSAGE
    # ========================================================

    if "User:" in response:

        response = response.split(
            "User:"
        )[0]


    # ========================================================
    # CLEAN RESPONSE
    # ========================================================

    response = response.strip()


    # ========================================================
    # DISPLAY RESPONSE
    # ========================================================

    print(
        "\nBRO:",
        response
    )

    print()


    # ========================================================
    # ADD RESPONSE TO HISTORY
    # ========================================================

    conversation_history += (
        "\n"
        + response
    )