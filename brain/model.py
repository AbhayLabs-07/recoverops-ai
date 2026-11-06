"""Custom decoder-only transformer architecture built from scratch in PyTorch for RecoverBrain."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponential frequencies for Rotary Position Embeddings (RoPE)."""
    assert dim % 2 == 0, "Dimension must be divisible by 2 for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Return [max_seq_len, dim // 2, 2] as cos and sin
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply precomputed RoPE cos/sin rotations to queries or keys.

    x: [batch, heads, seq_len, head_dim]
    freqs: [seq_len, head_dim // 2, 2]
    """
    b, h, s, d = x.shape
    x_pairs = x.reshape(b, h, s, d // 2, 2)
    cos = freqs[:s, :, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, s, d // 2]
    sin = freqs[:s, :, 1].unsqueeze(0).unsqueeze(0)

    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos

    return torch.stack([rot_x1, rot_x2], dim=-1).reshape(b, h, s, d)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward layer."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalAttention(nn.Module):
    """Multi-head causal self-attention using scaled_dot_product_attention and RoPE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert (
            self.head_dim * self.n_heads == self.d_model
        ), "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # PyTorch scaled_dot_product_attention invokes FlashAttention / memory-efficient kernels
        drop_rate = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=drop_rate, is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-RMSNorm and residual connections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class RecoverBrainModel(nn.Module):
    """Full causal decoder transformer model for RecoverBrain."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE table
        head_dim = config.d_model // config.n_heads
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        # Initialize weights with depth scaling
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        scaled_std = std / math.sqrt(2 * self.config.n_layers)

        for name, p in self.named_parameters():
            if "tok_embeddings" in name or "gate_proj" in name or "up_proj" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "o_proj" in name or "down_proj" in name:
                nn.init.normal_(p, mean=0.0, std=scaled_std)
            elif "weight" in name and "norm" in name:
                nn.init.ones_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, s = input_ids.shape
        x = self.tok_embeddings(input_ids)
        rope = self.rope_freqs[:s]

        for layer in self.layers:
            x = layer(x, rope)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 32,
        eos_token_id: int = 2,
    ) -> list[int]:
        """Greedy deterministic autoregressive token generation."""
        self.eval()
        device = next(self.parameters()).device
        generated = list(prompt_tokens)

        for _ in range(max_new_tokens):
            seq_len = len(generated)
            if seq_len >= self.config.max_seq_len:
                break

            inp = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = self.forward(inp)
            next_token_logits = logits[0, -1, :]
            next_token = int(torch.argmax(next_token_logits).item())

            generated.append(next_token)
            if next_token == eos_token_id:
                break

        return generated

