import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, n_heads=8, context_length=256):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # Causal mask: prevents attending to future tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):

        B, T, C = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into multiple attention heads
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # Causal masking
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        attention = torch.softmax(scores, dim=-1)

        # Apply attention to values
        x = attention @ V

        # Combine attention heads
        x = x.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(x)


class FeedForward(nn.Module):

    def __init__(self, d_model=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, d_model=512, n_heads=8, context_length=256):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)

    def forward(self, x):

        # Pre-LN + residual connection
        x = x + self.attention(self.ln1(x))

        # Pre-LN + residual connection
        x = x + self.feed_forward(self.ln2(x))

        return x


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size=8000,
        d_model=512,
        context_length=256,
        n_layers=8,
        n_heads=8
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            context_length,
            d_model
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                context_length=context_length
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, tokens):

        B, T = tokens.shape

        # Token + positional embeddings
        positions = torch.arange(
            T,
            device=tokens.device
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization + language-model head
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits


# Test
if __name__ == "__main__":

    model = GPT()

    tokens = torch.tensor([
        [315, 325, 67, 284]
    ])

    logits = model(tokens)

    print("Logits shape:", logits.shape)