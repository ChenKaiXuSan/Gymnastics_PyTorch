"""Masked pre-norm transformer encoder shared by every attention module.

The spatial, short-term and long-term encoders of the cycle-aware fusion
model all reduce to "self-attention over a token axis with some tokens
missing".  This module provides that core once so every encoder has
identical masking semantics.

Research Motivation:
    Monocular pose estimates contain missing joints (occlusion, truncation)
    and windows contain padding frames.  Attention must neither read from nor
    be corrupted by those tokens.  Instead of ``key_padding_mask`` (which
    yields NaN attention rows when every key of a query is masked) a full
    boolean attention mask is built per sequence in which every query keeps at
    least itself as an allowed key.  Masked outputs are then zeroed so that
    invalid tokens carry no information downstream.

Method:
    Pre-norm transformer block (Xiong et al., 2020):

        x = x + MHA(LN(x), LN(x), LN(x); allowed)
        x = x + MLP(LN(x))

    with ``allowed[n, q, k] = token_valid[n, k] & band[q, k]`` where ``band``
    is an optional ``[L, L]`` structural mask (used by the short-term branch
    to restrict attention to a local temporal window).

Shapes:
    tokens        [N, L, D]   N independent sequences of L tokens
    token_valid   [N, L]      bool
    band          [L, L]      bool, true = attention allowed (optional)
    output        [N, L, D]   zero at invalid tokens
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """One pre-norm multi-head self-attention block.

    Attributes:
        attention: Multi-head attention with ``batch_first=True``.
        mlp: Position-wise feed-forward network.
    """

    def __init__(self, dim: int, heads: int, *, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = max(1, int(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, blocked: torch.Tensor | None) -> torch.Tensor:
        """Apply the block.

        Args:
            tokens: ``[N, L, D]`` input tokens.
            blocked: ``[N * heads, L, L]`` bool mask, true where attention is
                *not* allowed, or ``None`` for full attention.

        Returns:
            ``[N, L, D]`` output tokens.
        """
        normalized = self.norm_attention(tokens)
        context, _ = self.attention(normalized, normalized, normalized, attn_mask=blocked, need_weights=False)
        tokens = tokens + context
        return tokens + self.mlp(self.norm_mlp(tokens))


def build_blocked_mask(
    token_valid: torch.Tensor,
    heads: int,
    band: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the ``[N * heads, L, L]`` blocked-attention mask.

    Args:
        token_valid: ``[N, L]`` bool validity of keys.
        heads: Number of attention heads (mask is repeated per head).
        band: Optional ``[L, L]`` bool structural mask, true = allowed.

    Returns:
        Bool mask, true where attention must be blocked.  Every query row
        keeps at least its own position allowed, so no row is fully blocked.
    """
    if token_valid.ndim != 2:
        raise ValueError("token_valid must have shape [N, L]")
    sequences, length = token_valid.shape
    allowed = token_valid.bool()[:, None, :].expand(sequences, length, length)
    if band is not None:
        if band.shape != (length, length):
            raise ValueError("band must have shape [L, L]")
        allowed = allowed & band.bool()[None]
    # Queries whose every key is masked would produce NaN softmax rows; let
    # them attend to themselves instead (their output is zeroed later anyway).
    eye = torch.eye(length, dtype=torch.bool, device=token_valid.device)[None]
    allowed = allowed | (eye & ~allowed.any(dim=-1, keepdim=True))
    blocked = ~allowed
    # Repeat per head: [N, L, L] -> [N * heads, L, L] with head as the fast axis,
    # matching nn.MultiheadAttention's expected layout.
    return blocked.repeat_interleave(heads, dim=0)


class MaskedTransformerEncoder(nn.Module):
    """Stack of :class:`TransformerBlock` with validity-aware masking.

    Attributes:
        blocks: The transformer blocks.
        norm: Final layer normalisation.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        layers: int,
        *,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.dim = int(dim)
        self.heads = int(heads)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: torch.Tensor,
        band: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token sequences.

        Args:
            tokens: ``[N, L, D]`` inputs.
            token_valid: ``[N, L]`` bool; invalid tokens are never attended to
                and their outputs are zero.
            band: Optional ``[L, L]`` structural mask (true = allowed).

        Returns:
            ``[N, L, D]`` encoded tokens.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("tokens must have shape [N, L, D] with D = dim")
        if token_valid.shape != tokens.shape[:2]:
            raise ValueError("token_valid must have shape [N, L]")
        blocked = build_blocked_mask(token_valid, self.heads, band)
        # Zero invalid inputs so their (masked) values cannot leak through
        # residual connections of the query positions.
        tokens = torch.where(token_valid[..., None], tokens, torch.zeros_like(tokens))
        for block in self.blocks:
            tokens = block(tokens, blocked)
        tokens = self.norm(tokens)
        return torch.where(token_valid[..., None], tokens, torch.zeros_like(tokens))


def sinusoidal_positions(length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding ``[L, D]`` (Vaswani et al., 2017)."""
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    half = dim // 2
    frequency = torch.exp(-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) * torch.arange(half, device=device, dtype=dtype) / max(half, 1))
    angle = position * frequency[None]
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0 : 2 * half : 2] = torch.sin(angle)
    encoding[:, 1 : 2 * half : 2] = torch.cos(angle)
    return encoding
