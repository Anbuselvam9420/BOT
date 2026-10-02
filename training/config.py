# config.py

import os
import torch


# ============================================================
# LLM Configuration
# ============================================================

class Config:
    """
    Central configuration for the LLM.

    Keep the same configuration when:
    - Training the model
    - Loading the model
    - Generating text
    """

    # ========================================================
    # Project
    # ========================================================

    project_name = "MyLLM"

    # ========================================================
    # Device
    # ========================================================

    # CUDA is used if available.
    # Otherwise CPU is used.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========================================================
    # Random Seed
    # ========================================================

    seed = 42

    # ========================================================
    # Tokenizer
    # ========================================================

    # Total number of tokens in the vocabulary.
    #
    # IMPORTANT:
    # This MUST match the tokenizer vocabulary size.

    vocab_size = 6000

    # Special tokens

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    # Token IDs

    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3

    # ========================================================
    # Sequence / Context Length
    # ========================================================

    # Maximum number of tokens the model can see at once.

    max_seq_len = 256

    # ========================================================
    # Transformer Architecture
    # ========================================================

    # Embedding / hidden dimension

    d_model = 256

    # Number of Transformer blocks

    num_layers = 6

    # Number of attention heads

    num_heads = 8

    # Feed-forward network dimension

    d_ff = 1024

    # ========================================================
    # Dropout
    # ========================================================

    dropout = 0.1

    # ========================================================
    # Weight Initialization
    # ========================================================

    initializer_range = 0.02

    # ========================================================
    # Training Configuration
    # ========================================================

    # Batch size

    batch_size = 4

    # Number of training epochs

    epochs = 5

    # Learning rate

    learning_rate = 3e-4

    # Weight decay

    weight_decay = 0.01

    # Gradient clipping

    max_grad_norm = 1.0

    # Gradient accumulation.
    #
    # Example:
    # batch_size = 4
    # gradient_accumulation_steps = 4
    #
    # Effective batch size = 16

    gradient_accumulation_steps = 4

    # ========================================================
    # Learning Rate Scheduler
    # ========================================================

    use_scheduler = True

    # Warmup steps

    warmup_steps = 500

    # ========================================================
    # Mixed Precision
    # ========================================================

    # Mixed precision can reduce GPU memory usage.

    use_amp = True

    # ========================================================
    # DataLoader
    # ========================================================

    num_workers = 0

    pin_memory = True

    # ========================================================
    # Dataset
    # ========================================================

    train_file = "data/train.txt"

    validation_file = "data/validation.txt"

    # ========================================================
    # Checkpoints
    # ========================================================

    checkpoint_dir = "checkpoints"

    checkpoint_name = "transformer_model.pt"

    # ========================================================
    # Generated Output
    # ========================================================

    output_dir = "outputs"

    # ========================================================
    # Generation Configuration
    # ========================================================

    max_new_tokens = 100

    temperature = 0.8

    top_k = 50

    # ========================================================
    # Logging
    # ========================================================

    log_interval = 10

    save_interval = 500

    # ========================================================
    # Validation
    # ========================================================

    validation_interval = 500

    # ========================================================
    # Paths
    # ========================================================

    tokenizer_file = "tokenizer.json"

    model_file = os.path.join(
        checkpoint_dir,
        checkpoint_name
    )


# ============================================================
# Create Default Configuration
# ============================================================

config = Config()


# ============================================================
# Utility Functions
# ============================================================

def get_device():
    """
    Return the device used by the model.
    """

    return config.device


def get_model_config():
    """
    Return the global configuration object.
    """

    return config


def print_config():
    """
    Print important configuration values.
    """

    print()
    print("=" * 60)
    print("LLM CONFIGURATION")
    print("=" * 60)

    print(
        f"Project              : {config.project_name}"
    )

    print(
        f"Device               : {config.device}"
    )

    print(
        f"Vocabulary Size      : {config.vocab_size:,}"
    )

    print(
        f"Context Length       : {config.max_seq_len}"
    )

    print(
        f"Model Dimension      : {config.d_model}"
    )

    print(
        f"Transformer Layers   : {config.num_layers}"
    )

    print(
        f"Attention Heads      : {config.num_heads}"
    )

    print(
        f"Feed Forward Size    : {config.d_ff}"
    )

    print(
        f"Dropout              : {config.dropout}"
    )

    print(
        f"Batch Size           : {config.batch_size}"
    )

    print(
        f"Learning Rate        : {config.learning_rate}"
    )

    print(
        f"Weight Decay         : {config.weight_decay}"
    )

    print(
        f"Epochs               : {config.epochs}"
    )

    print(
        f"Gradient Accumulation: "
        f"{config.gradient_accumulation_steps}"
    )

    print(
        f"Mixed Precision      : {config.use_amp}"
    )

    print(
        f"Tokenizer             : "
        f"{config.tokenizer_file}"
    )

    print(
        f"Model File            : "
        f"{config.model_file}"
    )

    print("=" * 60)
    print()


# ============================================================
# Validate Configuration
# ============================================================

def validate_config():
    """
    Check whether the configuration is valid.
    """

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    if config.vocab_size <= 0:

        raise ValueError(
            "vocab_size must be greater than 0."
        )

    # --------------------------------------------------------
    # Context length
    # --------------------------------------------------------

    if config.max_seq_len <= 0:

        raise ValueError(
            "max_seq_len must be greater than 0."
        )

    # --------------------------------------------------------
    # Model dimension
    # --------------------------------------------------------

    if config.d_model <= 0:

        raise ValueError(
            "d_model must be greater than 0."
        )

    # --------------------------------------------------------
    # Attention heads
    # --------------------------------------------------------

    if config.num_heads <= 0:

        raise ValueError(
            "num_heads must be greater than 0."
        )

    # --------------------------------------------------------
    # Check dimension compatibility
    # --------------------------------------------------------

    if config.d_model % config.num_heads != 0:

        raise ValueError(
            "d_model must be divisible by num_heads."
        )

    # --------------------------------------------------------
    # Transformer layers
    # --------------------------------------------------------

    if config.num_layers <= 0:

        raise ValueError(
            "num_layers must be greater than 0."
        )

    # --------------------------------------------------------
    # Feed-forward dimension
    # --------------------------------------------------------

    if config.d_ff <= 0:

        raise ValueError(
            "d_ff must be greater than 0."
        )

    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    if not 0.0 <= config.dropout < 1.0:

        raise ValueError(
            "dropout must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    if config.learning_rate <= 0:

        raise ValueError(
            "learning_rate must be greater than 0."
        )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    if config.batch_size <= 0:

        raise ValueError(
            "batch_size must be greater than 0."
        )

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    if config.epochs <= 0:

        raise ValueError(
            "epochs must be greater than 0."
        )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if config.temperature <= 0:

        raise ValueError(
            "temperature must be greater than 0."
        )

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    if config.top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )

    # --------------------------------------------------------
    # Special token IDs
    # --------------------------------------------------------

    special_ids = [
        config.pad_token_id,
        config.unk_token_id,
        config.bos_token_id,
        config.eos_token_id
    ]

    if len(set(special_ids)) != len(special_ids):

        raise ValueError(
            "Special token IDs must be unique."
        )

    # --------------------------------------------------------
    # Special token IDs must fit vocabulary
    # --------------------------------------------------------

    for token_id in special_ids:

        if token_id >= config.vocab_size:

            raise ValueError(
                "Special token ID is larger than "
                "or equal to vocab_size."
            )

    return True


# ============================================================
# Set Random Seed
# ============================================================

def set_seed(seed=None):
    """
    Set random seeds for reproducibility.
    """

    if seed is None:

        seed = config.seed

    # Python / PyTorch seed

    torch.manual_seed(
        seed
    )

    # CUDA seed

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    # CuDNN settings

    if torch.cuda.is_available():

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================
# Create Required Directories
# ============================================================

def create_directories():
    """
    Create directories needed by the project.
    """

    os.makedirs(
        config.checkpoint_dir,
        exist_ok=True
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True
    )


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    print(
        "Checking configuration..."
    )

    # Validate

    validate_config()

    # Set seed

    set_seed()

    # Create directories

    create_directories()

    # Print configuration

    print_config()

    print(
        "Configuration is valid."
    )

    print(
        f"Using device: {config.device}"
    )