# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Model Configuration
# -----------------------------

BLOCK_SIZE = 128
N_EMBED = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1


# -----------------------------
# Self Attention
# -----------------------------

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        # Prevent looking at future tokens
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Attention scores
        weights = q @ k.transpose(-2, -1)

        weights = weights * (k.shape[-1] ** -0.5)

        # Causal mask
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        weights = self.dropout(weights)

        v = self.value(x)

        output = weights @ v

        return output


# -----------------------------
# Multi Head Attention
# -----------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.projection = nn.Linear(
            num_heads * head_size,
            N_EMBED
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        output = self.projection(output)

        return self.dropout(output)


# -----------------------------
# Feed Forward Network
# -----------------------------

class FeedForward(nn.Module):

    def __init__(self, n_embed):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(n_embed, 4 * n_embed),

            nn.ReLU(),

            nn.Linear(4 * n_embed, n_embed),

            nn.Dropout(DROPOUT)
        )

    def forward(self, x):

        return self.network(x)


# -----------------------------
# Transformer Block
# -----------------------------

class Block(nn.Module):

    def __init__(self, n_embed, n_head):

        super().__init__()

        head_size = n_embed // n_head

        self.attention = MultiHeadAttention(
            n_head,
            head_size
        )

        self.feed_forward = FeedForward(
            n_embed
        )

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.ln1(x)
        )

        # Feed forward + residual connection
        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# -----------------------------
# Mini GPT
# -----------------------------

class MiniGPT(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            N_EMBED
        )

        # Position embedding
        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBED
        )

        # Transformer blocks
        self.blocks = nn.Sequential(

            *[
                Block(N_EMBED, N_HEAD)
                for _ in range(N_LAYER)
            ]
        )

        self.ln_final = nn.LayerNorm(N_EMBED)

        # Final output layer
        self.lm_head = nn.Linear(
            N_EMBED,
            vocab_size
        )

    def forward(self, index, targets=None):

        B, T = index.shape

        # Token embeddings
        token_embeddings = self.token_embedding(index)

        # Position embeddings
        positions = torch.arange(
            T,
            device=index.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Combine token + position
        x = token_embeddings + position_embeddings

        # Transformer
        x = self.blocks(x)

        x = self.ln_final(x)

        # Output
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    # -------------------------
    # Text Generation
    # -------------------------

    def generate(
        self,
        index,
        max_new_tokens,
        temperature=1.0
    ):

        for _ in range(max_new_tokens):

            # Keep only last BLOCK_SIZE tokens
            index_condition = index[:, -BLOCK_SIZE:]

            logits, loss = self(
                index_condition
            )

            # Get last token
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Convert to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Add token
            index = torch.cat(
                (index, next_token),
                dim=1
            )

        return index