"""
Raw PyTorch Transformer for AttentionForge Studio.

No Hugging Face.
No torch.nn.Transformer.
No torch.nn.MultiheadAttention.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_terms = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_diagnostics: bool = True):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            masked_scores = raw_scores.masked_fill(mask == 0, float("-inf"))
        else:
            masked_scores = raw_scores

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        output = self.out_proj(self.combine_heads(context))

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "q": q,
                "k": k,
                "v": v,
                "raw_scores": raw_scores,
                "masked_scores": masked_scores,
                "attention_weights": attention_weights,
            }

        return output, diagnostics


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, ff_hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_diagnostics: bool = True):
        attention_output, diagnostics = self.attention(x, mask, return_diagnostics=return_diagnostics)
        x = self.norm1(x + self.dropout(attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x, diagnostics


class MiniTransformerLM(nn.Module):
    """
    A larger educational decoder-only Transformer preset.

    This is still a mini model compared with modern LLMs, but it is large enough
    to feel more serious than a baby beginner demo.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        num_heads: int = 6,
        num_layers: int = 6,
        ff_hidden_dim: int = 768,
        max_len: int = 96,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_hidden_dim = ff_hidden_dim
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, idx: torch.Tensor, causal_mask: bool = True, return_diagnostics: bool = True):
        _, seq_len = idx.shape

        if seq_len > self.max_len:
            idx = idx[:, -self.max_len :]
            seq_len = idx.shape[1]

        x = self.token_embedding(idx)
        x = self.positional_encoding(x)

        mask = self.make_causal_mask(seq_len, idx.device) if causal_mask else None

        diagnostics_by_layer = []
        for block in self.blocks:
            x, diagnostics = block(x, mask, return_diagnostics=return_diagnostics)
            if return_diagnostics:
                diagnostics_by_layer.append(diagnostics)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, diagnostics_by_layer

    @torch.no_grad()
    def generate_raw(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.75,
        top_k: int = 12,
    ) -> torch.Tensor:
        """
        Generate from the trained mini Transformer checkpoint.

        top_k sampling keeps output readable for a focused portfolio demo while
        still sampling from the raw PyTorch model's logits.
        """
        self.eval()

        for _ in range(max_new_tokens):
            context = idx[:, -self.max_len :]
            logits, _ = self(context, causal_mask=True, return_diagnostics=False)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                values, indices = torch.topk(logits, k=top_k, dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(dim=-1, index=indices, src=values)
                logits = filtered

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
