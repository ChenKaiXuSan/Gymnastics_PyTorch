#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: pose_fusion_res_3dcnn.py
Author: Kaixu Chen <chenkaixusan@gmail.com>

A 3D-CNN backbone that fuses RGB clips and keypoint-derived attention maps
via a lightweight Pose-Attn Fusion module (channel-wise gated mixing).
It supports:
- Selecting fusion stages (fusion_layers)
- Side heads for per-joint heatmap supervision (optional)
- Saving gate weights and side feature maps for interpretability

Inputs
------
RGB  : (N, 3, T, H, W)
Attn : (N, C_ctx, T, H, W)  # C_ctx=1 or num_joints

Output
------
- logits: (N, num_classes)
- (optional) aux: {"side_preds": List[Tensor], "gate_scales": List[Tensor]}
"""
from __future__ import annotations

import os
import math
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from project.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# -------------------------- Pose-Attention Fusion Block --------------------------
class PoseAttnFusion(nn.Module):
    """
    Channel-wise gated fusion of RGB feature and pose-attention feature.

    out = Norm( RGB_feat * g + Pose_feat * (1 - g) [+ residual x] )
    where g = sigmoid(Conv(ReLU(Norm(Conv([RGB_feat, Pose_feat])))) / temp)

    Notes:
    - Use GroupNorm as default for stability in small-batch settings.
    - gate_init_bias > 0 biases early training toward RGB branch.
    - Automatically interpolates attn spatial-temporal size to x.
    """
    def __init__(
        self,
        in_channels: int,
        context_channels: int = 1,
        norm: str = "gn",           # "bn" | "gn" | "ln" | "none"
        use_residual: bool = True,
        gate_init_bias: float = 2.0,
        gate_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.gate_temp = gate_temp

        self.rgb_conv  = nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False)
        self.attn_conv = nn.Conv3d(context_channels, in_channels, 1, bias=False)

        self.norm_rgb  = self._make_norm(norm, in_channels)
        self.norm_attn = self._make_norm(norm, in_channels)
        self.norm_out  = self._make_norm(norm, in_channels) if norm != "none" else nn.Identity()

        self.gate_conv1 = nn.Conv3d(in_channels * 2, in_channels, 1, bias=False)
        self.gate_norm1 = self._make_norm(norm, in_channels)
        self.gate_conv2 = nn.Conv3d(in_channels, in_channels, 1, bias=True)
        nn.init.constant_(self.gate_conv2.bias, gate_init_bias)

        self.act = nn.ReLU(inplace=True)
        self.last_scale: Optional[torch.Tensor] = None  # (N,C,T,H,W)

    @staticmethod
    def _make_norm(kind: str, c: int) -> nn.Module:
        if kind == "bn":
            return nn.BatchNorm3d(c)
        if kind == "gn":
            return nn.GroupNorm(32 if c >= 32 else max(1, c // 2), c)
        if kind == "ln":
            return nn.GroupNorm(1, c)  # LN-like
        return nn.Identity()

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        # Align dtype/device & THW
        attn = attn.to(dtype=x.dtype, device=x.device)
        if x.shape[-3:] != attn.shape[-3:]:
            attn = F.interpolate(attn, size=x.shape[-3:], mode="trilinear", align_corners=False)

        # Two-stream encode
        rgb_feat  = self.norm_rgb(self.rgb_conv(x))
        attn_feat = self.norm_attn(self.attn_conv(attn))

        # Gate (channel-wise)
        g = self.gate_conv1(torch.cat([rgb_feat, attn_feat], dim=1))
        g = self.act(self.gate_norm1(g))
        g = self.gate_conv2(g)
        if self.gate_temp != 1.0:
            g = g / self.gate_temp
        g = torch.sigmoid(g)

        self.last_scale = g.detach()

        # Fuse (+ optional residual)
        fused = rgb_feat * g + attn_feat * (1.0 - g)
        out = fused + x if self.use_residual else fused
        out = self.norm_out(out)
        return out


# -------------------------- Fusion Config Mapping ----------------------------
FUSE_LAYERS_MAPPING = {
    "single": {i: [i] for i in range(5)},
    "multi":  {
        0: [0],
        1: [0, 1],
        2: [0, 1, 2],
        3: [0, 1, 2, 3],
        4: [0, 1, 2, 3, 4],
    },
}

# blocks[0]: stem          →  64ch
# blocks[1]: layer1 (x3)   → 256ch
# blocks[2]: layer2 (x4)   → 512ch
# blocks[3]: layer3 (x6)   → 1024ch
# blocks[4]: layer4 (x3)   → 2048ch
# blocks[5]: head (GAP+FC) → logits
DIM_LIST = [64, 256, 512, 1024, 2048]


# ---------------------------- Helpers: image saving --------------------------
def _to_uint8(x: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    x = x.detach().float().cpu()
    x_min, x_max = x.min(), x.max()
    if (x_max - x_min) < eps:
        x = torch.zeros_like(x)
    else:
        x = (x - x_min) / (x_max - x_min + eps)
    x = (x * 255.0).clamp(0, 255).byte().numpy()
    return x

def _save_grid(images: List[np.ndarray], save_path: str, ncols: int = 4, pad: int = 2) -> None:
    if len(images) == 0:
        return
    h, w = images[0].shape[:2]
    n = len(images)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    grid_h = nrows * h + (nrows - 1) * pad
    grid_w = ncols * w + (ncols - 1) * pad
    canvas = np.full((grid_h, grid_w), 0, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, ncols)
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = img
    Image.fromarray(canvas, mode="L").save(save_path)


# ---------------------------- Main Model Class -------------------------------
class PoseFusionRes3DCNN(BaseModel):
    def __init__(self, hparams: OmegaConf) -> None:
        super().__init__(hparams)

        m = hparams.model
        ablation = m.get("ablation_study", "multi")
        fusion_layers = m.fusion_layers
        if isinstance(fusion_layers, int):
            fusion_layers = FUSE_LAYERS_MAPPING[ablation].get(fusion_layers, [])
        self.fusion_layers: List[int] = list(fusion_layers)
        logger.info(f"Fusion at blocks (0=stem..4=layer4): {self.fusion_layers}")

        self.num_classes = int(m.model_class_num)
        self.ckpt = m.get("ckpt_path", "None")
        self.attn_channels = int(m.get("attn_channels", 1))
        self.use_side = bool(m.get("use_side_heads", False))

        # Expect your init_resnet() to build a backbone with .blocks[0..5]
        # (0..4 = stages, 5 = head). Otherwise adapt this indexing.
        self.model = self.init_resnet(self.num_classes, self.ckpt)
        self.blocks = nn.ModuleList([self.model.blocks[i] for i in range(6)])

        # Fusion modules per stage
        self.attn_fusions = nn.ModuleList([
            PoseAttnFusion(
                in_channels=dim,
                context_channels=self.attn_channels,
                norm=m.get("fusion_norm", "gn"),
                use_residual=bool(m.get("fusion_residual", True)),
                gate_init_bias=float(m.get("gate_init_bias", 2.0)),
                gate_temp=float(m.get("gate_temp", 1.0)),
            ) if i in self.fusion_layers else nn.Identity()
            for i, dim in enumerate(DIM_LIST)
        ])

        # Side heads for per-joint (or 1ch) maps at chosen stages
        self.side_heads = nn.ModuleList([
            (nn.Conv3d(dim, self.attn_channels, kernel_size=1) if self.use_side and i in {1,2,3,4}
             else nn.Identity())
            for i, dim in enumerate(DIM_LIST)
        ])

    # ---------------------------- Forward ------------------------------------
    def forward(
        self,
        video: torch.Tensor,
        attn_map: torch.Tensor,
        return_aux: bool = False
    ):
        """
        video   : (N,3,T,H,W)
        attn_map: (N,C_ctx,T,H,W)
        """
        aux: Optional[Dict[str, List[torch.Tensor]]] = {"side_preds": [], "gate_scales": []} if (return_aux or self.use_side) else None

        x = video
        for idx in range(5):  # 0..4 stages
            x = self.blocks[idx](x)

            # side prediction logits
            if (return_aux or self.use_side) and not isinstance(self.side_heads[idx], nn.Identity):
                side_pred = self.side_heads[idx](x)  # (N, C_ctx, Ti, Hi, Wi)
                aux["side_preds"].append(side_pred)

            # fusion
            if not isinstance(self.attn_fusions[idx], nn.Identity):
                # ensure THW alignment is done inside fusion
                x = self.attn_fusions[idx](x, attn_map)
                if (return_aux or self.use_side) and hasattr(self.attn_fusions[idx], "last_scale") and self.attn_fusions[idx].last_scale is not None:
                    # (N, C_i) channel-mean gate weights for logging
                    g = self.attn_fusions[idx].last_scale.mean(dim=(2,3,4)).detach()
                    aux["gate_scales"].append(g)

        # head -> logits
        logits = self.blocks[5](x)  # expected to return (N, num_classes)

        return (logits, aux) if (return_aux or self.use_side) else logits

    # ---------------------------- Visualization ------------------------------
    def get_gate_scales(self) -> List[Optional[torch.Tensor]]:
        """Return per-stage channel-mean gate scales (CPU) if available."""
        out: List[Optional[torch.Tensor]] = []
        for fusion in self.attn_fusions:
            if isinstance(fusion, PoseAttnFusion) and fusion.last_scale is not None:
                out.append(fusion.last_scale.mean(dim=(0,2,3,4)).detach().cpu())  # (C,)
            else:
                out.append(None)
        return out

    def save_attention_maps(self, save_dir: str = "fusion_vis") -> None:
        """
        Save bar charts of channel-mean gate weights per fused stage.
        """
        os.makedirs(save_dir, exist_ok=True)
        for idx, scale in enumerate(self.get_gate_scales()):
            if scale is None:
                continue
            arr = scale.numpy()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 3))
            plt.bar(range(len(arr)), arr)
            plt.title(f"Gate Weights – Block {idx}")
            plt.xlabel("Channel")
            plt.ylabel("Weight")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"block{idx}_gate.png"))
            plt.close()
        logger.info(f"Gate weight figures saved to: {save_dir}")

    def save_side_feature_maps(
        self,
        side_preds: List[torch.Tensor],
        save_dir: str = "fusion_vis/side_maps",
        aggregate: str = "mean",     # "mean" | "max" | "t=<int>"
        max_channels: int = 16,
        ncols: int = 4
    ) -> None:
        """
        Convert side head 3D logits (B,C,T,H,W) to 2D grids and save as PNG.
        """
        os.makedirs(save_dir, exist_ok=True)

        # parse aggregate option
        t_idx = None
        if aggregate.startswith("t="):
            try:
                t_idx = int(aggregate.split("=", 1)[1])
            except Exception as e:
                raise ValueError(f"Invalid aggregate '{aggregate}', expect 't=<int>'") from e

        for li, P in enumerate(side_preds):
            P = torch.sigmoid(P)       # visualize probabilities
            B, C, T, H, W = P.shape
            layer_dir = os.path.join(save_dir, f"layer{li}")
            os.makedirs(layer_dir, exist_ok=True)

            for b in range(B):
                if t_idx is not None:
                    t_sel = t_idx if t_idx >= 0 else (T + t_idx)
                    t_sel = max(0, min(T - 1, t_sel))
                    M = P[b, :, t_sel]           # (C,H,W)
                elif aggregate == "max":
                    M = P[b].amax(dim=1)         # (C,H,W)
                else:
                    M = P[b].mean(dim=1)         # (C,H,W)

                C_use = min(C, max_channels)
                imgs = [_to_uint8(M[ch]) for ch in range(C_use)]
                save_path = os.path.join(layer_dir, f"b{b}_grid.png")
                _save_grid(imgs, save_path, ncols=ncols, pad=2)

        logger.info(f"Side feature maps saved to: {save_dir}")


# ---------------------------- Quick Test Entry -------------------------------
if __name__ == "__main__":
    cfg = OmegaConf.create({
        "model": {
            "model_class_num": 3,
            "fusion_layers": [2, 3, 4],     # fuse at layer2..4
            "ckpt_path": "",
            "ablation_study": "multi",
            "attn_channels": 1,             # or J
            "use_side_heads": True,
            "fusion_norm": "gn",
            "fusion_residual": True,
            "gate_init_bias": 2.0,
            "gate_temp": 1.0,
        }
    })
    model = PoseFusionRes3DCNN(cfg)
    rgb  = torch.randn(2, 3, 8, 224, 224)
    attn = torch.randn(2, 1, 8, 224, 224)

    logits, aux = model(rgb, attn, return_aux=True)
    print("logits:", logits.shape)               # (2, 3)
    print("side heads:", len(aux["side_preds"])) # == #stages with side heads
    model.save_attention_maps("test_fusion_vis/gates")
    model.save_side_feature_maps(aux["side_preds"], save_dir="test_fusion_vis/side", aggregate="mean")
