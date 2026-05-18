from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn


class TCN(nn.Module):
    def __init__(
        self,
        num_class: int,
        num_total_class: int,
        num_point: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.num_point = int(num_point)
        self.in_channels = int(in_channels)
        self.num_class = int(num_class)
        self.num_total_class = int(num_total_class)

        input_dim = self.num_point * self.in_channels

        self.net = nn.ModuleDict(
            {
                "twist": nn.Sequential(
                    TemporalBlock(input_dim, 64, 3, dilation=1),
                    TemporalBlock(64, 128, 3, dilation=2),
                    TemporalBlock(128, 256, 3, dilation=4),
                    TemporalBlock(256, 256, 3, dilation=8),
                ),
                "posture": nn.Sequential(
                    TemporalBlock(input_dim, 64, 3, dilation=1),
                    TemporalBlock(64, 128, 3, dilation=2),
                    TemporalBlock(128, 256, 3, dilation=4),
                    TemporalBlock(256, 256, 3, dilation=8),
                ),
                "relax": nn.Sequential(
                    TemporalBlock(input_dim, 64, 3, dilation=1),
                    TemporalBlock(64, 128, 3, dilation=2),
                    TemporalBlock(128, 256, 3, dilation=4),
                    TemporalBlock(256, 256, 3, dilation=8),
                ),
            }
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.heads = nn.ModuleDict(
            {
                "twist": nn.Linear(256, self.num_class),
                "posture": nn.Linear(256, self.num_class),
                "relax": nn.Linear(256, self.num_class),
                "total": nn.Linear(256 * 3, self.num_total_class),
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 4D: (N,T,J,C)
        # x: (N, T, J, C)
        n, t, j, c = x.shape

        x = x.reshape(n, t, j * c).transpose(1, 2).contiguous()  # (N, J*C, T)

        x = {task: net(x) for task, net in self.net.items()}  # (N, C_out, T)
        x = {
            task: self.pool(layer).squeeze(-1) for task, layer in x.items()
        }  # (N, C_out)

        x["total"] = torch.cat(
            [x[task] for task in ["twist", "posture", "relax"]], dim=1
        )

        twist_logits = self.heads["twist"](x["twist"])
        posture_logits = self.heads["posture"](x["posture"])
        relax_logits = self.heads["relax"](x["relax"])
        total_logits = self.heads["total"](x["total"])

        return {
            "twist": twist_logits,
            "posture": posture_logits,
            "relax": relax_logits,
            "total": total_logits,
        }


class TemporalBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=padding, dilation=dilation
        )

        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, padding=padding, dilation=dilation
        )

        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):

        res = self.residual(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        return x + res
