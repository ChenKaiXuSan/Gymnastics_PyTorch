"""Masked bidirectional feature exchange between aligned pose views."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BidirectionalCrossViewAttention(nn.Module):
    """Exchange same-frame joint features with one shared attention module."""

    def __init__(self, hidden_channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if num_heads <= 0 or hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.hidden_channels = int(hidden_channels)
        self.num_heads = int(num_heads)
        self.norm = nn.LayerNorm(hidden_channels)
        self.attention = nn.MultiheadAttention(
            hidden_channels,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def _validate(
        self,
        face: Tensor,
        side: Tensor,
        valid_face: Tensor,
        valid_side: Tensor,
    ) -> None:
        if face.ndim != 4:
            raise ValueError("view features must have shape [B, T, J, C]")
        if side.shape != face.shape:
            raise ValueError("face and side features must have equal shape")
        if face.shape[-1] != self.hidden_channels:
            raise ValueError("view feature channels must match hidden_channels")
        if valid_face.shape != face.shape[:-1] or valid_side.shape != face.shape[:-1]:
            raise ValueError("view validity masks must have shape [B, T, J]")
        if side.device != face.device or side.dtype != face.dtype:
            raise ValueError("face and side features must share device and dtype")

    def _exchange(
        self,
        query: Tensor,
        source: Tensor,
        query_valid: Tensor,
        source_valid: Tensor,
    ) -> Tensor:
        batch, frames, joints, channels = query.shape
        query_valid = query_valid.bool()
        source_valid = source_valid.bool()
        normalized_query = self.norm(query)
        normalized_source = self.norm(source)
        normalized_source = torch.where(
            source_valid[..., None], normalized_source, torch.zeros_like(normalized_source)
        )
        flat_query = normalized_query.reshape(batch * frames, joints, channels)
        flat_source = normalized_source.reshape(batch * frames, joints, channels)
        flat_source_valid = source_valid.reshape(batch * frames, joints)
        has_source = flat_source_valid.any(dim=-1)
        key_padding_mask = ~flat_source_valid
        if (~has_source).any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[~has_source, 0] = False
        context, _ = self.attention(
            flat_query,
            flat_source,
            flat_source,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        context = torch.where(
            has_source[:, None, None], context, torch.zeros_like(context)
        ).reshape(batch, frames, joints, channels)
        enhanced = query + context
        return torch.where(
            query_valid[..., None], enhanced, torch.zeros_like(enhanced)
        )

    def forward(
        self,
        face: Tensor,
        side: Tensor,
        valid_face: Tensor,
        valid_side: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return face-from-side and side-from-face enhanced features."""
        self._validate(face, side, valid_face, valid_side)
        return (
            self._exchange(face, side, valid_face, valid_side),
            self._exchange(side, face, valid_side, valid_face),
        )
