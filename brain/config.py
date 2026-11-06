"""Configuration definitions, model presets, and parameter counting utilities for RecoverBrain."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

try:
    import torch.serialization
    # Register ModelConfig in PyTorch weights_only allowlist
    torch.serialization.add_safe_globals([ModelConfig])
except Exception:
    pass


PresetName = Literal["tiny", "small", "base"]


@dataclass(frozen=True)
class ModelConfig:
    """Architectural configuration parameters for RecoverBrain decoder transformer."""
    name: PresetName
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int
    d_ff: int
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0


# Target Parameter Scale:
#   tiny:  ~2M  (d_model=128, n_layers=4,  n_heads=4,  vocab=4096,  d_ff=768)  -> ~1.97M
#   small: ~25M (d_model=384, n_layers=8,  n_heads=6,  vocab=8192,  d_ff=1800) -> ~24.45M
#   base:  ~97M (d_model=768, n_layers=12, n_heads=12, vocab=16384, d_ff=2048) -> ~97.54M

PRESETS: dict[PresetName, ModelConfig] = {
    "tiny": ModelConfig(
        name="tiny",
        d_model=128,
        n_layers=4,
        n_heads=4,
        vocab_size=4096,
        d_ff=768,
        max_seq_len=256,
    ),
    "small": ModelConfig(
        name="small",
        d_model=384,
        n_layers=8,
        n_heads=6,
        vocab_size=8192,
        d_ff=1800,
        max_seq_len=256,
    ),
    "base": ModelConfig(
        name="base",
        d_model=768,
        n_layers=12,
        n_heads=12,
        vocab_size=16384,
        d_ff=2048,
        max_seq_len=256,
    ),
}


def get_preset(name: str) -> ModelConfig:
    """Retrieve model configuration preset by name."""
    clean = name.strip().lower()
    if clean not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {list(PRESETS.keys())}")
    return PRESETS[clean]  # type: ignore


def count_parameters(model) -> int:
    """Count total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
