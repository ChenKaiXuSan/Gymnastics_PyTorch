"""Bidirectional Cross-View Attention between the two motion-guided views.

Research Motivation:
    After FiLM each view carries a motion-guided description of every joint,
    but it has been computed *without looking at the other view*.  Deciding
    how reliable View A is for joint ``j`` at time ``t`` is only meaningful
    relative to View B: a joint that disagrees strongly with the other view
    while the other view is anatomically and temporally consistent is the
    signature of a single-view failure.  Cross-view attention lets

        Motion-Guided View A  <->  Motion-Guided View B

    exchange information before the reliability of each joint is estimated,
    so the reliability head sees *contextualised* view features rather than
    two isolated descriptions.

Method:
    One shared attention block is applied in both directions from the
    pre-update features (so the operation is symmetric under swapping the
    views):

        C_A = H_A + Attn(q = LN(H_A), k = v = LN(H_B); keys masked by valid_B)
        C_B = H_B + Attn(q = LN(H_B), k = v = LN(H_A); keys masked by valid_A)

    followed by a position-wise MLP with residual connection.  Attention
    runs over the ``J`` joints of the *same frame* in the other view, i.e.
    each joint of A can read from every joint of B at time ``t``.

Attention dimensions:
    queries  [B * T, J, D]   (joints of the receiving view at one frame)
    keys     [B * T, J, D]   (joints of the other view at the same frame)
    values   [B * T, J, D]
    output   [B * T, J, D]  -> reshaped to [B, T, J, D]

    Temporal context is not mixed here on purpose: temporal reasoning has
    already been folded into ``H`` by the motion branches, and per-frame
    exchange keeps the attention cost at ``O(T * J^2)``.

Masking:
    Keys that are invalid in the source view are excluded.  If a frame has
    no valid source joint at all, the context is zero and the query features
    pass through unchanged.  Query positions that are invalid in their own
    view produce zero output.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    """One pre-norm cross-attention block with a feed-forward sub-layer."""

    def __init__(self, dim: int, heads: int, *, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.norm_query = nn.LayerNorm(dim)
        self.norm_source = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = max(1, int(dim * mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, query: torch.Tensor, source: torch.Tensor, source_valid: torch.Tensor) -> torch.Tensor:
        """Attend from ``query`` tokens to ``source`` tokens.

        Args:
            query: ``[N, L, D]`` receiving tokens.
            source: ``[N, L, D]`` tokens of the other view.
            source_valid: ``[N, L]`` bool validity of ``source``.

        Returns:
            ``[N, L, D]`` updated query tokens.
        """
        has_source = source_valid.any(dim=-1)
        key_padding = ~source_valid
        if (~has_source).any():
            # Fully masked rows would yield NaN; unmask one key for those
            # sequences and discard their context below.
            key_padding = key_padding.clone()
            key_padding[~has_source, 0] = False
        context, _ = self.attention(
            self.norm_query(query),
            self.norm_source(source),
            self.norm_source(source),
            key_padding_mask=key_padding,
            need_weights=False,
        )
        context = torch.where(has_source[:, None, None], context, torch.zeros_like(context))
        query = query + context
        return query + self.mlp(self.norm_mlp(query))


class BidirectionalCrossViewAttention(nn.Module):
    """Symmetric feature exchange ``A <- B`` and ``B <- A`` over joints per frame.

    Attributes:
        blocks: Shared cross-attention blocks applied in both directions.
    """

    def __init__(self, hidden_dim: int, *, heads: int = 4, layers: int = 1, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.hidden_dim = int(hidden_dim)
        self.blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(layers))

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        valid_a: torch.Tensor,
        valid_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange information between the views.

        Args:
            feature_a: ``H_A`` with shape ``[B, T, J, D]``.
            feature_b: ``H_B`` with shape ``[B, T, J, D]``.
            valid_a: ``[B, T, J]`` bool validity of View A joints.
            valid_b: ``[B, T, J]`` bool validity of View B joints.

        Returns:
            Tuple ``(C_A, C_B)`` each ``[B, T, J, D]``; zero at invalid
            query joints.
        """
        if feature_a.shape != feature_b.shape or feature_a.ndim != 4 or feature_a.shape[-1] != self.hidden_dim:
            raise ValueError("view features must both have shape [B, T, J, D]")
        if valid_a.shape != feature_a.shape[:-1] or valid_b.shape != feature_a.shape[:-1]:
            raise ValueError("validity masks must have shape [B, T, J]")
        batch, frames, joints, dim = feature_a.shape
        # Fold batch and time: attention exchanges the J joints of one frame.
        tokens_a = feature_a.reshape(batch * frames, joints, dim)
        tokens_b = feature_b.reshape(batch * frames, joints, dim)
        mask_a = valid_a.bool().reshape(batch * frames, joints)
        mask_b = valid_b.bool().reshape(batch * frames, joints)
        for block in self.blocks:
            # Both directions read the pre-update features so the exchange is
            # symmetric under swapping the views.
            updated_a = block(tokens_a, tokens_b, mask_b)
            updated_b = block(tokens_b, tokens_a, mask_a)
            tokens_a, tokens_b = updated_a, updated_b
        tokens_a = torch.where(mask_a[..., None], tokens_a, torch.zeros_like(tokens_a))
        tokens_b = torch.where(mask_b[..., None], tokens_b, torch.zeros_like(tokens_b))
        return tokens_a.reshape(batch, frames, joints, dim), tokens_b.reshape(batch, frames, joints, dim)
