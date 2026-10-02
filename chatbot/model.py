import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. CONFIGURATION
# ============================================================

class ModelConfig:
    """
    Configuration for the small GPT-style language model.
    """

    def __init__(
        self,
        vocab_size=6000,
        block_size=128,
        n_embd=256,
        n_layer=6,
        n_head=8,
        dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout


# ============================================================
# 2. CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask.

    The causal mask prevents the model from seeing future tokens.
    """

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Creates Q, K and V together
        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        # Output projection
        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Size of each attention head
        self.head_dim = config.n_embd // config.n_head

        # Causal mask
        mask = torch.tril(
            torch.ones(
                config.block_size,
                config.block_size
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                config.block_size,
                config.block_size
            )
        )

    def forward(self, x):
        """
        x shape:

        (batch_size, sequence_length, embedding_size)
        """

        batch_size, seq_len, n_embd = x.size()

        # ----------------------------------------------------
        # Create Query, Key and Value
        # ----------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.split(
            self.n_embd,
            dim=2
        )

        # ----------------------------------------------------
        # Split into attention heads
        # ----------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # ----------------------------------------------------
        # Causal masking
        # ----------------------------------------------------

        attention_scores = attention_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Convert scores into probabilities
        # ----------------------------------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.attn_dropout(
            attention_weights
        )

        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        y = attention_weights @ v

        # ----------------------------------------------------
        # Combine heads
        # ----------------------------------------------------

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            batch_size,
            seq_len,
            self.n_embd
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        y = self.out_proj(y)

        y = self.resid_dropout(y)

        return y


# ============================================================
# 3. FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):
    """
    Feed-forward network used inside each Transformer block.
    """

    def __init__(self, config):
        super().__init__()

        self.network = nn.Sequential(

            # Expand embedding dimension
            nn.Linear(
                config.n_embd,
                4 * config.n_embd
            ),

            # GELU activation
            nn.GELU(),

            # Project back
            nn.Linear(
                4 * config.n_embd,
                config.n_embd
            ),

            # Dropout
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 4. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    One complete Transformer block.

    Contains:
        LayerNorm
        Self Attention
        Residual connection
        Feed Forward Network
        Residual connection
    """

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            config.n_embd
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.ln2 = nn.LayerNorm(
            config.n_embd
        )

        self.feed_forward = FeedForward(
            config
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Attention + residual connection
        # ----------------------------------------------------

        x = x + self.attention(
            self.ln1(x)
        )

        # ----------------------------------------------------
        # Feed forward + residual connection
        # ----------------------------------------------------

        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# 5. GPT-STYLE LANGUAGE MODEL
# ============================================================

class GPTLanguageModel(nn.Module):
    """
    Small GPT-style decoder-only Transformer.

    The model receives token IDs and predicts the next token.
    """

    def __init__(self, config=None):

        super().__init__()

        if config is None:
            config = ModelConfig()

        self.config = config

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            config.dropout
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        self.ln_f = nn.LayerNorm(
            config.n_embd
        )

        # ----------------------------------------------------
        # Language-model output head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight initialization
        self.apply(self._init_weights)

        # Tie token embedding weights with output weights
        self.lm_head.weight = self.token_embedding.weight

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(
        self,
        idx,
        targets=None
    ):
        """
        Parameters
        ----------
        idx:
            Token IDs.

            Shape:
            (batch_size, sequence_length)

        targets:
            Expected next token IDs.

        Returns
        -------
        logits:
            Prediction scores for every token.

        loss:
            Cross entropy loss if targets are provided.
        """

        batch_size, seq_len = idx.shape

        # Make sure sequence is not too long
        if seq_len > self.config.block_size:

            raise ValueError(
                f"Sequence length {seq_len} is greater than "
                f"block size {self.config.block_size}"
            )

        # ----------------------------------------------------
        # Position IDs
        # ----------------------------------------------------

        positions = torch.arange(
            0,
            seq_len,
            device=idx.device
        )

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_embeddings = self.token_embedding(
            idx
        )

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        position_embeddings = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Combine token + position embeddings
        # ----------------------------------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.dropout(x)

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.ln_f(x)

        # ----------------------------------------------------
        # Predict vocabulary probabilities
        # ----------------------------------------------------

        logits = self.lm_head(x)

        loss = None

        # ----------------------------------------------------
        # Calculate training loss
        # ----------------------------------------------------

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    # ========================================================
    # TEXT GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None
    ):
        """
        Generate new tokens from the model.

        Parameters
        ----------
        idx:
            Starting token IDs.

        max_new_tokens:
            Number of tokens to generate.

        temperature:
            Controls randomness.

        top_k:
            Limits sampling to the top K tokens.
        """

        for _ in range(max_new_tokens):

            # ------------------------------------------------
            # Crop context if it becomes too long
            # ------------------------------------------------

            idx_cond = idx[
                :,
                -self.config.block_size:
            ]

            # ------------------------------------------------
            # Get predictions
            # ------------------------------------------------

            logits, _ = self(
                idx_cond
            )

            # Only use prediction for final token
            logits = logits[:, -1, :]

            # ------------------------------------------------
            # Temperature
            # ------------------------------------------------

            logits = logits / temperature

            # ------------------------------------------------
            # Top-K sampling
            # ------------------------------------------------

            if top_k is not None:

                top_values, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1)
                    )
                )

                minimum_value = top_values[
                    :, -1
                ].unsqueeze(-1)

                logits = torch.where(
                    logits < minimum_value,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            # ------------------------------------------------
            # Convert logits to probabilities
            # ------------------------------------------------

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # ------------------------------------------------
            # Sample next token
            # ------------------------------------------------

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # ------------------------------------------------
            # Add token to sequence
            # ------------------------------------------------

            idx = torch.cat(
                (
                    idx,
                    next_token
                ),
                dim=1
            )

        return idx


# ============================================================
# 6. MODEL CREATION FUNCTION
# ============================================================

def create_model(
    vocab_size=6000,
    block_size=128,
    n_embd=256,
    n_layer=6,
    n_head=8,
    dropout=0.1
):
    """
    Easy function for creating the model.
    """

    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout
    )

    model = GPTLanguageModel(
        config
    )

    return model


# ============================================================
# 7. SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing GPT Language Model")
    print("=" * 60)

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        vocab_size=6000,
        block_size=128,
        n_embd=256,
        n_layer=6,
        n_head=8,
        dropout=0.1
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Create fake input
    # --------------------------------------------------------

    batch_size = 2
    sequence_length = 32

    x = torch.randint(
        0,
        6000,
        (
            batch_size,
            sequence_length
        ),
        device=device
    )

    y = torch.randint(
        0,
        6000,
        (
            batch_size,
            sequence_length
        ),
        device=device
    )

    # --------------------------------------------------------
    # Forward test
    # --------------------------------------------------------

    logits, loss = model(
        x,
        y
    )

    print(
        f"Input shape: "
        f"{x.shape}"
    )

    print(
        f"Output shape: "
        f"{logits.shape}"
    )

    print(
        f"Initial loss: "
        f"{loss.item():.4f}"
    )

    print("=" * 60)
    print("MODEL TEST SUCCESSFUL")
    print("=" * 60)