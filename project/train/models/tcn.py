from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn


class TCN(nn.Module):
    def __init__(
        self,
        num_class: int,
        num_point: int,
        in_channels: int = 3,
        tcn_channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_class <= 0:
            raise ValueError(f"num_class must be positive, got {num_class}")
        if num_point <= 0:
            raise ValueError(f"num_point must be positive, got {num_point}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive odd, got {kernel_size}")
        if len(tcn_channels) == 0:
            raise ValueError("tcn_channels must not be empty")

        self.num_point = int(num_point)
        self.in_channels = int(in_channels)
        self.num_class = int(num_class)

        layers = []
        input_dim = self.num_point * self.in_channels
        for out_ch in tcn_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            input_dim = out_ch

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.heads = nn.ModuleDict(
            {
                "twist": nn.Linear(input_dim, self.num_class),
                "posture": nn.Linear(input_dim, self.num_class),
                "relax": nn.Linear(input_dim, self.num_class),
                "total": nn.Linear(input_dim, self.num_class),
            }
        )

    def _forward_4d(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (N, T, J, C)
        n, t, j, c = x.shape
        if j != self.num_point or c != self.in_channels:
            raise ValueError(
                f"Expected input shape (N,T,{self.num_point},{self.in_channels}), got {tuple(x.shape)}"
            )

        x = x.reshape(n, t, j * c).transpose(1, 2).contiguous()  # (N, J*C, T)
        x = self.tcn(x)  # (N, C_out, T)
        x = self.pool(x).squeeze(-1)  # (N, C_out)
        return {task: head(x) for task, head in self.heads.items()}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 4D: (N,T,J,C), 5D: (N,S,T,J,C)
        if x.ndim == 4:
            return self._forward_4d(x)

        if x.ndim == 5:
            n, s, t, j, c = x.shape
            flat = x.reshape(n * s, t, j, c)
            flat_logits = self._forward_4d(flat)
            return {
                task: logit.reshape(n, s, -1).mean(dim=1)
                for task, logit in flat_logits.items()
            }

        raise ValueError(f"Expected input ndim 4 or 5, got shape {tuple(x.shape)}")
