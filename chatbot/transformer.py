# transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

class Config:
    vocab_size = 6000

    max_seq_len = 256

    # Model size
    d_model = 256
    num_heads = 8
    num_layers = 6

    # Feed-forward hidden size
    d_ff = 4 * d_model

    # Dropout
    dropout = 0.1

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Multi-Head Causal Self Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.d_model % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

        # Query, Key and Value projection
        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model
        )

        # Output projection
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        mask = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                config.max_seq_len,
                config.max_seq_len
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # ----------------------------------------------------
        # Reshape into heads
        # ----------------------------------------------------

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # ----------------------------------------------------
        # Causal masking
        # Prevent looking at future tokens
        # ----------------------------------------------------

        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.attn_dropout(attention)

        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        y = attention @ v

        # ----------------------------------------------------
        # Combine heads
        # ----------------------------------------------------

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        y = self.out_proj(y)

        y = self.resid_dropout(y)

        return y


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                config.d_model,
                config.d_ff
            ),

            nn.GELU(),

            nn.Linear(
                config.d_ff,
                config.d_model
            ),

            nn.Dropout(
                config.dropout
            )
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.ln2 = nn.LayerNorm(
            config.d_model
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
        # Feed Forward + residual connection
        # ----------------------------------------------------

        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# GPT Transformer
# ============================================================

class TransformerLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # ----------------------------------------------------
        # Positional embeddings
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
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
                for _ in range(config.num_layers)
            ]
        )

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        self.ln_f = nn.LayerNorm(
            config.d_model
        )

        # ----------------------------------------------------
        # Language model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    # --------------------------------------------------------
    # Weight initialization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        input_ids,
        targets=None
    ):

        B, T = input_ids.shape

        if T > self.config.max_seq_len:

            raise ValueError(
                f"Sequence length {T} exceeds "
                f"maximum length "
                f"{self.config.max_seq_len}"
            )

        # ----------------------------------------------------
        # Positions
        # ----------------------------------------------------

        positions = torch.arange(
            0,
            T,
            device=input_ids.device
        )

        positions = positions.unsqueeze(0)

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = (
            token_embeddings
            +
            position_embeddings
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
        # Vocabulary logits
        # ----------------------------------------------------

        logits = self.lm_head(x)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    # --------------------------------------------------------
    # Text generation
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None
    ):

        self.eval()

        for _ in range(max_new_tokens):

            # ------------------------------------------------
            # Keep only maximum context
            # ------------------------------------------------

            input_context = input_ids[
                :, -self.config.max_seq_len:
            ]

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits, _ = self(
                input_context
            )

            # ------------------------------------------------
            # Last token
            # ------------------------------------------------

            logits = logits[:, -1, :]

            # ------------------------------------------------
            # Temperature
            # ------------------------------------------------

            logits = logits / temperature

            # ------------------------------------------------
            # Top-K sampling
            # ------------------------------------------------

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                minimum = values[:, -1].unsqueeze(-1)

                logits = torch.where(
                    logits < minimum,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            # ------------------------------------------------
            # Convert to probabilities
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
            # Add token
            # ------------------------------------------------

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

        return input_ids


# ============================================================
# Parameter Count
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Save Model
# ============================================================

def save_model(
    model,
    path="transformer_model.pt"
):

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                key: value
                for key, value
                in vars(model.config).items()
                if not key.startswith("__")
            }
        },
        path
    )

    print(
        f"Model saved to: {path}"
    )


# ============================================================
# Load Model
# ============================================================

def load_model(
    path,
    config
):

    model = TransformerLM(config)

    checkpoint = torch.load(
        path,
        map_location=config.device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(config.device)

    model.eval()

    return model


# ============================================================
# Example Training Step
# ============================================================

def training_step(
    model,
    optimizer,
    input_ids,
    targets
):

    model.train()

    optimizer.zero_grad()

    logits, loss = model(
        input_ids,
        targets
    )

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    return loss.item()


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Small Transformer Language Model")
    print("=" * 60)

    config = Config()

    print(
        f"Device: {config.device}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TransformerLM(
        config
    )

    model.to(
        config.device
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = count_parameters(
        model
    )

    print(
        f"Parameters: {parameters:,}"
    )

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    batch_size = 2
    sequence_length = 32

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (
            batch_size,
            sequence_length
        ),
        device=config.device
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (
            batch_size,
            sequence_length
        ),
        device=config.device
    )

    # --------------------------------------------------------
    # Forward test
    # --------------------------------------------------------

    logits, loss = model(
        input_ids,
        targets
    )

    print(
        f"Input shape:  {input_ids.shape}"
    )

    print(
        f"Logits shape: {logits.shape}"
    )

    print(
        f"Loss:         {loss.item():.4f}"
    )

    # --------------------------------------------------------
    # Generation test
    # --------------------------------------------------------

    generated = model.generate(
        input_ids[:1, :5],
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )

    print(
        f"Generated shape: {generated.shape}"
    )

    print("=" * 60)
    print("Transformer test completed successfully.")
    print("=" * 60)