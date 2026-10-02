# generate.py

import os
import json
import argparse

import torch

from transformer import Config, TransformerLM, load_model


# ============================================================
# Tokenizer
# ============================================================

class Tokenizer:
    """
    Simple tokenizer loader.

    Expected tokenizer.json format:

    {
        "token_to_id": {
            "<pad>": 0,
            "<unk>": 1,
            "hello": 2
        },
        "id_to_token": {
            "0": "<pad>",
            "1": "<unk>",
            "2": "hello"
        }
    }
    """

    def __init__(self, tokenizer_path):

        if not os.path.exists(tokenizer_path):

            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}"
            )

        with open(
            tokenizer_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.token_to_id = data["token_to_id"]

        self.id_to_token = {
            int(key): value
            for key, value in data["id_to_token"].items()
        }

        # Special tokens
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.unk_id = self.token_to_id.get(
            self.unk_token,
            0
        )

        self.bos_id = self.token_to_id.get(
            self.bos_token,
            None
        )

        self.eos_id = self.token_to_id.get(
            self.eos_token,
            None
        )

    # --------------------------------------------------------
    # Encode text
    # --------------------------------------------------------

    def encode(self, text):

        text = text.strip()

        if not text:

            return []

        words = text.split()

        token_ids = []

        for word in words:

            token_id = self.token_to_id.get(
                word,
                self.unk_id
            )

            token_ids.append(token_id)

        return token_ids

    # --------------------------------------------------------
    # Decode token IDs
    # --------------------------------------------------------

    def decode(self, token_ids):

        tokens = []

        for token_id in token_ids:

            token = self.id_to_token.get(
                int(token_id),
                self.unk_token
            )

            # Skip special tokens
            if token in [
                "<pad>",
                "<bos>",
                "<eos>"
            ]:

                continue

            tokens.append(token)

        return " ".join(tokens)


# ============================================================
# Load tokenizer
# ============================================================

def load_tokenizer(path):

    print(
        f"Loading tokenizer: {path}"
    )

    tokenizer = Tokenizer(path)

    print(
        f"Vocabulary size: {len(tokenizer.token_to_id):,}"
    )

    return tokenizer


# ============================================================
# Load Transformer model
# ============================================================

def load_transformer(
    model_path,
    config
):

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print(
        f"Loading model: {model_path}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TransformerLM(
        config
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=config.device
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model.to(
        config.device
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# Generate tokens
# ============================================================

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
):

    # --------------------------------------------------------
    # Validate temperature
    # --------------------------------------------------------

    if temperature <= 0:

        raise ValueError(
            "Temperature must be greater than 0."
        )

    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    token_ids = tokenizer.encode(
        prompt
    )

    # --------------------------------------------------------
    # Empty prompt
    # --------------------------------------------------------

    if len(token_ids) == 0:

        if tokenizer.bos_id is not None:

            token_ids = [
                tokenizer.bos_id
            ]

        else:

            token_ids = [
                tokenizer.unk_id
            ]

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=model.config.device
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Convert back to text
    # --------------------------------------------------------

    output_ids = generated_ids[
        0
    ].tolist()

    generated_text = tokenizer.decode(
        output_ids
    )

    return generated_text


# ============================================================
# Interactive chat
# ============================================================

def interactive_mode(
    model,
    tokenizer,
    args
):

    print()
    print("=" * 60)
    print("Transformer LLM")
    print("=" * 60)
    print()
    print("Type your prompt.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        try:

            prompt = input(
                "You: "
            )

        except KeyboardInterrupt:

            print()
            print("Exiting...")

            break

        except EOFError:

            print()
            print("Exiting...")

            break

        prompt = prompt.strip()

        if not prompt:

            continue

        if prompt.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "Goodbye!"
            )

            break

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )

            print()
            print(
                "Model:",
                response
            )
            print()

        except RuntimeError as error:

            print()
            print(
                "Generation error:"
            )

            print(error)

            print()


# ============================================================
# Single prompt mode
# ============================================================

def single_prompt_mode(
    model,
    tokenizer,
    args
):

    response = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print()
    print(response)
    print()


# ============================================================
# Argument parser
# ============================================================

def create_argument_parser():

    parser = argparse.ArgumentParser(
        description="Generate text using your Transformer LLM."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default="transformer_model.pt",
        help="Path to trained model."
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to tokenizer.json."
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for generation."
    )

    # --------------------------------------------------------
    # Number of tokens
    # --------------------------------------------------------

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens."
    )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature."
    )

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K sampling."
    )

    return parser


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("Loading your LLM")
    print("=" * 60)

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = create_argument_parser()

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.max_new_tokens <= 0:

        raise ValueError(
            "max_new_tokens must be greater than 0."
        )

    if args.top_k is not None:

        if args.top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )

    if args.temperature <= 0:

        raise ValueError(
            "temperature must be greater than 0."
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = Config()

    print()
    print(
        f"Device: {config.device}"
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer(
        args.tokenizer
    )

    # --------------------------------------------------------
    # Check vocabulary
    # --------------------------------------------------------

    tokenizer_vocab_size = len(
        tokenizer.token_to_id
    )

    if tokenizer_vocab_size != config.vocab_size:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Tokenizer vocabulary: "
            f"{tokenizer_vocab_size}"
        )

        print(
            f"Model vocabulary: "
            f"{config.vocab_size}"
        )

        print(
            "These should normally be the same."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_transformer(
        model_path=args.model,
        config=config
    )

    # --------------------------------------------------------
    # Generation mode
    # --------------------------------------------------------

    if args.prompt is not None:

        single_prompt_mode(
            model,
            tokenizer,
            args
        )

    else:

        interactive_mode(
            model,
            tokenizer,
            args
        )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()