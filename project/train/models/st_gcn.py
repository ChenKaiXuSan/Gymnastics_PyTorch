#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/train/models/st_gcn.py
Project: /workspace/code/project/train/models
Created Date: Thursday April 16th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday April 16th 2026 11:18:55 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from project.train.map_config import INDICES, FILTERED_SKELETON_CONNECTIONS


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
    if not FILTERED_SKELETON_CONNECTIONS or not INDICES:
        return []

    converted: List[Tuple[int, int]] = []
    for src_id, dst_id in FILTERED_SKELETON_CONNECTIONS:
        if src_id in INDICES and dst_id in INDICES:
            converted.append((INDICES.index(src_id), INDICES.index(dst_id)))
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


def _build_stgcn_adjacency(
    num_nodes: int,
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build ST-GCN adjacency tensor A with shape (K, V, V)."""
    edge_index = _build_edge_index(num_nodes=num_nodes, edges=edges)
    a = _normalize_adjacency(num_nodes=num_nodes, edge_index=edge_index, device=device)
    # Use a single partition (K=1) while keeping the standard ST-GCN tensor shape.
    return a.unsqueeze(0)


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=1)

    def forward(
        self, x: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.ndim != 3:
            raise ValueError(f"adjacency must be (K,V,V), got shape {tuple(a.shape)}")
        if a.shape[0] != self.kernel_size:
            raise ValueError(
                f"adjacency K mismatch, expected {self.kernel_size}, got {a.shape[0]}"
            )

        x = self.conv(x)  # (N, out_channels*K, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # Aggregate graph partitions.
        x = torch.einsum("nkctv,kvw->nctw", x, a)
        return x.contiguous(), a


class STGCNUnit(nn.Module):
    """One ST-GCN unit: graph conv + temporal conv + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if len(kernel_size) != 2:
            raise ValueError(
                "kernel_size must be (temporal_kernel_size, spatial_kernel_size)"
            )
        if kernel_size[0] % 2 == 0:
            raise ValueError("temporal kernel size must be odd")

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = 0 if self.residual is None else self.residual(x)
        x, a = self.gcn(x, a)
        x = self.tcn(x) + res
        return self.relu(x), a


class STGCN(nn.Module):
    """ST-GCN classifier for skeleton sequences.

    Supported input formats:
    - x: (N, T, V, C), (N, C, T, V), or (N, C, T, V, M)
    """

    def __init__(
        self,
        num_class: int,
        num_point: int,
        in_channels: int = 3,
        graph_edges: Optional[Sequence[Tuple[int, int]]] = None,
        temporal_kernel_size: int = 9,
        dropout: float = 0.2,
        edge_importance_weighting: bool = True,
    ) -> None:
        super().__init__()
        if num_class <= 0:
            raise ValueError(f"num_class must be positive, got {num_class}")
        if num_point <= 0:
            raise ValueError(f"num_point must be positive, got {num_point}")

        if graph_edges is None:
            graph_edges = _edge_from_map_config()

        adjacency = _build_stgcn_adjacency(num_nodes=num_point, edges=graph_edges)
        self.register_buffer("_adjacency_tensor", adjacency)

        self.num_point = num_point
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)

        adjacency_tensor = self._get_adjacency()
        spatial_kernel_size = int(adjacency_tensor.shape[0])
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {"dropout": 0.0}

        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNUnit(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
                STGCNUnit(64, 64, kernel_size, 1, dropout=dropout),
                STGCNUnit(64, 64, kernel_size, 1, dropout=dropout),
                STGCNUnit(64, 64, kernel_size, 1, dropout=dropout),
                STGCNUnit(64, 128, kernel_size, 2, dropout=dropout),
                STGCNUnit(128, 128, kernel_size, 1, dropout=dropout),
                STGCNUnit(128, 128, kernel_size, 1, dropout=dropout),
                STGCNUnit(128, 256, kernel_size, 2, dropout=dropout),
                STGCNUnit(256, 256, kernel_size, 1, dropout=dropout),
                STGCNUnit(256, 256, kernel_size, 1, dropout=dropout),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [
                    nn.Parameter(torch.ones_like(adjacency_tensor))
                    for _ in self.st_gcn_networks
                ]
            )
        else:
            self.edge_importance = [1.0] * len(self.st_gcn_networks)

        self.cls_head_twist = nn.Conv2d(256, num_class, kernel_size=1)
        self.cls_head_posture = nn.Conv2d(256, num_class, kernel_size=1)
        self.cls_head_relax = nn.Conv2d(256, num_class, kernel_size=1)
        self.cls_head_total = nn.Conv2d(256, num_class, kernel_size=1)

    def _to_nctvm(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input into ST-GCN standard shape (N,C,T,V,M)."""
        if x.ndim == 4:
            # (N,T,V,C)
            if x.shape[-1] == self.in_channels and x.shape[-2] == self.num_point:
                x = x.permute(0, 3, 1, 2).contiguous()
            # (N,C,T,V)
            elif x.shape[1] == self.in_channels and x.shape[-1] == self.num_point:
                pass
            else:
                raise ValueError(
                    "Input shape is not supported. Expected (N,T,V,C) or (N,C,T,V) "
                    f"with V={self.num_point}, C={self.in_channels}, got {tuple(x.shape)}"
                )
            x = x.unsqueeze(-1)  # M=1
        elif x.ndim == 5:
            # Already (N,C,T,V,M)
            if x.shape[1] != self.in_channels or x.shape[3] != self.num_point:
                raise ValueError(
                    "5D input must be (N,C,T,V,M) "
                    f"with V={self.num_point}, C={self.in_channels}, got {tuple(x.shape)}"
                )
        else:
            raise ValueError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")
        return x

    def _get_adjacency(self) -> torch.Tensor:
        adjacency = getattr(self, "_adjacency_tensor")
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("_adjacency_tensor is not a torch.Tensor")
        return adjacency

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            x: (N, T, V, C), (N, C, T, V), or (N, C, T, V, M)
        Returns:
            dict of logits for each task, each of shape (N, num_class)
        """

        x_tensor = self._to_nctvm(x)
        n, c, t, v, m = x_tensor.size()

        # data normalization (same pattern as original ST-GCN)
        x_tensor = x_tensor.permute(0, 4, 3, 1, 2).contiguous()  # (N,M,V,C,T)
        x_tensor = x_tensor.view(n * m, v * c, t)
        x_tensor = self.data_bn(x_tensor)
        x_tensor = x_tensor.view(n, m, v, c, t)
        x_tensor = x_tensor.permute(0, 1, 3, 4, 2).contiguous()  # (N,M,C,T,V)
        x_tensor = x_tensor.view(n * m, c, t, v)

        adjacency_tensor = self._get_adjacency()

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x_tensor, _ = gcn(x_tensor, adjacency_tensor * importance)

        # Global pooling + instance average
        x_tensor = x_tensor.mean(dim=(2, 3), keepdim=True)
        x_tensor = x_tensor.view(n, m, -1, 1, 1).mean(dim=1)

        # Four classification heads for multi-task learning
        logits_twist = self.cls_head_twist(x_tensor).flatten(1)
        logits_posture = self.cls_head_posture(x_tensor).flatten(1)
        logits_relax = self.cls_head_relax(x_tensor).flatten(1)
        logits_total = self.cls_head_total(x_tensor).flatten(1)

        return {
            "twist": logits_twist,
            "posture": logits_posture,
            "relax": logits_relax,
            "total": logits_total,
        }


def build_stgcn_from_hparams(hparams) -> STGCN:
    """Build ST-GCN from hydra-style hparams."""
    model_cfg = getattr(hparams, "model", hparams)
    num_class = int(getattr(model_cfg, "model_class_num", 3))
    num_point = int(len(INDICES))  # Override with actual indices length if available
    in_channels = int(getattr(model_cfg, "in_channels", 3))
    temporal_kernel_size = int(getattr(model_cfg, "temporal_kernel_size", 9))
    dropout = float(getattr(model_cfg, "dropout", 0.2))
    edge_importance_weighting = bool(
        getattr(model_cfg, "edge_importance_weighting", True)
    )

    return STGCN(
        num_class=num_class,
        num_point=num_point,
        in_channels=in_channels,
        graph_edges=_edge_from_map_config(),
        temporal_kernel_size=temporal_kernel_size,
        dropout=dropout,
        edge_importance_weighting=edge_importance_weighting,
    )
