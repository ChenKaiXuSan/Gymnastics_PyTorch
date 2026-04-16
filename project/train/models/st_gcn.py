#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/train/models/st_gcn.py
Project: /workspace/code/project/train/models
Created Date: Thursday April 16th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday April 16th 2026 11:15:40 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn

try:
    from project.train.map_config import ID_TO_INDEX, SKELETON_CONNECTIONS
except Exception:
    ID_TO_INDEX = {}
    SKELETON_CONNECTIONS = []


def _build_edge_index(
    num_nodes: int,
    edges: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Build undirected edge list with self-links."""
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")

    edge_list: List[Tuple[int, int]] = [(i, i) for i in range(num_nodes)]
    if edges is not None:
        for src, dst in edges:
            if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
                continue
            edge_list.append((src, dst))
            edge_list.append((dst, src))
    return edge_list


def _edge_from_map_config() -> List[Tuple[int, int]]:
    """Convert global joint ids to contiguous indices when map config is available."""
    if not ID_TO_INDEX or not SKELETON_CONNECTIONS:
        return []

    converted: List[Tuple[int, int]] = []
    for src_id, dst_id in SKELETON_CONNECTIONS:
        if src_id in ID_TO_INDEX and dst_id in ID_TO_INDEX:
            converted.append((ID_TO_INDEX[src_id], ID_TO_INDEX[dst_id]))
    return converted


def _normalize_adjacency(
    num_nodes: int,
    edge_index: Iterable[Tuple[int, int]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return normalized adjacency D^{-1/2} A D^{-1/2} as (V, V)."""
    a = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for src, dst in edge_index:
        a[src, dst] = 1.0

    degree = a.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree.clamp(min=1e-6), -0.5)
    d_inv_sqrt = torch.diag(degree_inv_sqrt)
    return d_inv_sqrt @ a @ d_inv_sqrt


class GraphConv(nn.Module):
    """Spatial graph conv with fixed normalized adjacency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square matrix with shape (V,V)")

        self.register_buffer("adjacency", adjacency)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        x = self.proj(x)
        return torch.einsum("nctv,vw->nctw", x, self.adjacency)


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv + temporal conv + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd to keep center alignment")

        padding = (temporal_kernel_size - 1) // 2
        self.gcn = GraphConv(in_channels, out_channels, adjacency)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    """ST-GCN classifier for skeleton sequences.

    Supported input formats:
    - x: (N, T, V, C) or (N, C, T, V)
    - p_left/p_right: each (N, T, V, C), fused by average
    """

    def __init__(
        self,
        num_class: int,
        num_point: int,
        in_channels: int = 3,
        graph_edges: Optional[Sequence[Tuple[int, int]]] = None,
        temporal_kernel_size: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_class <= 0:
            raise ValueError(f"num_class must be positive, got {num_class}")
        if num_point <= 0:
            raise ValueError(f"num_point must be positive, got {num_point}")

        if graph_edges is None:
            graph_edges = _edge_from_map_config()

        edge_index = _build_edge_index(num_nodes=num_point, edges=graph_edges)
        adjacency = _normalize_adjacency(num_nodes=num_point, edge_index=edge_index)

        self.num_point = num_point
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)

        self.backbone = nn.ModuleList(
            [
                STGCNBlock(
                    in_channels,
                    64,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                STGCNBlock(
                    64,
                    64,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                STGCNBlock(
                    64,
                    64,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                STGCNBlock(
                    64,
                    128,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=2,
                    dropout=dropout,
                ),
                STGCNBlock(
                    128,
                    128,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                STGCNBlock(
                    128,
                    256,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=2,
                    dropout=dropout,
                ),
                STGCNBlock(
                    256,
                    256,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
            ]
        )

        self.cls_head = nn.Conv2d(256, num_class, kernel_size=1)

    def _to_nctv(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {tuple(x.shape)}")

        # Prefer (N,T,V,C) from dataloader outputs.
        if x.shape[-1] == self.in_channels and x.shape[-2] == self.num_point:
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.shape[1] == self.in_channels and x.shape[-1] == self.num_point:
            # Already (N,C,T,V)
            pass
        else:
            raise ValueError(
                "Input shape is not supported. Expected (N,T,V,C) or (N,C,T,V) "
                f"with V={self.num_point}, C={self.in_channels}, got {tuple(x.shape)}"
            )

        n, c, t, v = x.shape
        x = x.permute(0, 2, 3, 1).reshape(n, t, v * c)
        x = x.transpose(1, 2)  # (N, V*C, T)
        x = self.data_bn(x)
        x = x.transpose(1, 2).reshape(n, t, v, c).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        p_left: Optional[torch.Tensor] = None,
        p_right: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x is None:
            if p_left is None or p_right is None:
                raise ValueError("Provide either x, or both p_left and p_right")
            if p_left.shape != p_right.shape:
                raise ValueError(
                    f"p_left and p_right shape mismatch: {tuple(p_left.shape)} vs {tuple(p_right.shape)}"
                )
            x = 0.5 * (p_left + p_right)

        feat = self._to_nctv(cast(torch.Tensor, x))

        for block in self.backbone:
            feat = block(feat)

        # Global average pooling over temporal and joint dims.
        feat = feat.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        feat = self.cls_head(feat)
        return feat.flatten(1)


class STGCNFusion(STGCN):
    """Alias for dual-view usage, keeping naming compatibility in configs."""

    pass


def build_stgcn_from_hparams(hparams) -> STGCN:
    """Build ST-GCN from hydra-style hparams."""
    model_cfg = getattr(hparams, "model", hparams)
    num_class = int(getattr(model_cfg, "model_class_num", 3))
    num_point = int(getattr(model_cfg, "num_point", len(ID_TO_INDEX) or 17))
    in_channels = int(getattr(model_cfg, "in_channels", 3))
    temporal_kernel_size = int(getattr(model_cfg, "temporal_kernel_size", 9))
    dropout = float(getattr(model_cfg, "dropout", 0.2))

    return STGCN(
        num_class=num_class,
        num_point=num_point,
        in_channels=in_channels,
        graph_edges=_edge_from_map_config(),
        temporal_kernel_size=temporal_kernel_size,
        dropout=dropout,
    )
