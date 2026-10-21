#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lawfft.py - Standalone LAWFFT model architecture for PyTorch & TensorRT export.
Architecture designed by umzi2 (https://github.com/umzi2) in resselt (https://github.com/rewaifu/resselt).
Local Adaptive Weighted Fourier Feature Transformer.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Global normalization mode for FFT custom operators in LAWFFT:
# 0 = "backward" (PyTorch default used in LAWFFT)
# 1 = "ortho"
DEFAULT_LAWFFT_NORMALIZED: int = 0


class LawfftRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, normalized: int = DEFAULT_LAWFFT_NORMALIZED):
        ctx.normalized = normalized
        norm_str = "ortho" if normalized == 1 else "backward"
        y = torch.fft.rfft2(x, dim=(-2, -1), norm=norm_str)
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g, x: torch.Value, normalized: int = DEFAULT_LAWFFT_NORMALIZED):
        return g.op(
            "ai.onnx.contrib::Rfft",
            x,
            normalized_i=normalized,
            onesided_i=1,
            signal_ndim_i=2,
        )


class LawfftIRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_ri: torch.Tensor, normalized: int = DEFAULT_LAWFFT_NORMALIZED):
        ctx.normalized = normalized
        norm_str = "ortho" if normalized == 1 else "backward"
        x_c = torch.view_as_complex(x_ri)
        return torch.fft.irfft2(x_c, dim=(-2, -1), norm=norm_str)

    @staticmethod
    def symbolic(g, x: torch.Value, normalized: int = DEFAULT_LAWFFT_NORMALIZED):
        return g.op(
            "ai.onnx.contrib::Irfft",
            x,
            normalized_i=normalized,
            onesided_i=1,
            signal_ndim_i=2,
        )


SampleMods = Literal[
    "conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv", "dysample"
]


class DySample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= groups or in_channels % groups != 0:
            raise ValueError("Incorrect in_channels and groups values.")

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)
        return output


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods = "pixelshuffle",
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,
        group: int = 4,
    ) -> None:
        m: List[nn.Module] = []
        if upsample == "pixelshuffledirect":
            m.append(nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1))
            m.append(nn.PixelShuffle(scale))
        elif upsample == "pixelshuffle":
            m.append(nn.Conv2d(in_dim, mid_dim, 3, 1, 1))
            m.append(nn.LeakyReLU(inplace=True))
            if scale in [2, 3]:
                m.append(nn.Conv2d(mid_dim, mid_dim * scale**2, 3, 1, 1))
                m.append(nn.PixelShuffle(scale))
            elif scale == 4:
                m.append(nn.Conv2d(mid_dim, mid_dim * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.Conv2d(mid_dim, mid_dim * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            m.append(DySample(in_dim, out_dim, scale=scale, groups=group))
        elif upsample == "nearest+conv":
            m.append(nn.UpsamplingNearest2d(scale_factor=scale))
            m.append(nn.Conv2d(in_dim, mid_dim, 3, 1, 1))
            m.append(nn.LeakyReLU(inplace=True))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(f"upsampler '{upsample}' is not supported")

        super().__init__(*m)
        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    3,
                    [
                        "conv",
                        "pixelshuffledirect",
                        "pixelshuffle",
                        "nearest+conv",
                        "dysample",
                    ].index(upsample),
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class FSAS(nn.Module):
    """Fourier Spatial Attention Sub-block with windowed / full FFT."""
    def __init__(
        self,
        dim: int = 64,
        mid_factor: float = 1.0,
        window_size: int = 8,
        windowed: bool = True,
        normalized: int = DEFAULT_LAWFFT_NORMALIZED,
    ) -> None:
        super().__init__()
        self.to_hidden = nn.Conv2d(dim, int(dim * mid_factor * 3), 1)
        self.to_hidden_dw = nn.Conv2d(
            int(dim * mid_factor * 3),
            int(dim * mid_factor * 3),
            3,
            1,
            1,
            groups=int(dim * mid_factor * 3),
        )
        self.project_out = nn.Conv2d(int(dim * mid_factor), dim, 1)
        self.patch = windowed
        self.norm = LayerNorm(int(dim * mid_factor))
        self.patch_size = window_size
        self.normalized = normalized

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        norm_str = "ortho" if self.normalized == 1 else "backward"

        if self.patch:
            B, C, H, W = q.shape
            q_patch = q.view(
                B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size
            ).permute(0, 1, 2, 4, 3, 5)
            k_patch = k.view(
                B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size
            ).permute(0, 1, 2, 4, 3, 5)

            q_fft = torch.fft.rfft2(q_patch.float(), norm=norm_str)
            k_fft = torch.fft.rfft2(k_patch.float(), norm=norm_str)

            out = q_fft * k_fft
            out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size), norm=norm_str)
            out = out.to(v.dtype)
            out = out.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        else:
            q_fft = torch.fft.rfft2(q.float(), norm=norm_str)
            k_fft = torch.fft.rfft2(k.float(), norm=norm_str)
            out = q_fft * k_fft
            out = torch.fft.irfft2(out, norm=norm_str)
            out = out.to(v.dtype)

        out = self.norm(out)
        output = v * out
        output = self.project_out(output)
        return output


class DynamicLocal(nn.Module):
    def __init__(self, dim: int = 64, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel, 1, kernel // 2, groups=dim)
        self.dw = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.dw(self.conv(x) * x)


class SFSAS(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        split: float = 0.25,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        windowed: bool = True,
        normalized: int = DEFAULT_LAWFFT_NORMALIZED,
    ) -> None:
        super().__init__()
        local = int(split * dim)
        global_dim = dim - local
        self.local = nn.Sequential(DynamicLocal(local), DynamicLocal(local, 5))
        self.att = FSAS(global_dim, t_mid_factor, window_size, windowed, normalized=normalized)

    def forward(self, x: Tensor) -> Tensor:
        local, glob = torch.split(x, [self.local[0].conv.in_channels, self.att.to_hidden.in_channels], dim=1)
        return torch.cat([self.local(local), self.att(glob)], dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim: int = 64, mlp_factor: float = 2.66) -> None:
        super().__init__()
        hidden = int(dim * mlp_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.silu(x1) * x2)


class MetaBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        split: float = 0.25,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        windowed: bool = True,
        mlp_factor: float = 2.66,
        normalized: int = DEFAULT_LAWFFT_NORMALIZED,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        self.att = SFSAS(dim, split, t_mid_factor, window_size, windowed, normalized=normalized)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_factor)

    def forward(self, x: Tensor) -> Tensor:
        x = self.att(self.norm(x)) + x
        return self.ffn(self.norm2(x)) + x


class ResidualMeta(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        meta_b: int = 6,
        split: float = 0.25,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        mlp: float = 2.66,
        normalized: int = DEFAULT_LAWFFT_NORMALIZED,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            *[
                MetaBlock(dim, split, t_mid_factor, window_size, bool(i % 2), mlp, normalized=normalized)
                for i in range(meta_b)
            ]
            + [DynamicLocal(dim)]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + x


class LAWFFT(nn.Module):
    """Local Adaptive Weighted Fourier Feature Transformer"""

    def __init__(
        self,
        *,
        in_ch: int = 3,
        dim: int = 60,
        split: float = 0.25,
        scale: int = 4,
        n_rblock: int = 4,
        n_mblock: int = 6,
        t_mid_factor: float = 1.0,
        window_size: int = 8,
        mlp_factor: float = 2.66,
        unshuffle_mod: bool = False,
        upsampler: SampleMods = "pixelshuffledirect",
        mid_dim: int = 48,
        normalized: int = DEFAULT_LAWFFT_NORMALIZED,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.pad = 1
        current_scale = scale
        if unshuffle_mod and scale < 3:
            unshuffle = 4 // scale
            self.in_to_dim = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(in_ch * unshuffle**2, dim, 3, 1, 1),
            )
            self.pad = unshuffle * window_size
            current_scale = 4
        else:
            self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)
            self.pad = window_size

        self.register_buffer(
            "window_size", torch.tensor(window_size, dtype=torch.uint8)
        )
        self.body = nn.Sequential(
            *[
                ResidualMeta(
                    dim,
                    n_mblock,
                    split,
                    t_mid_factor,
                    window_size,
                    mlp_factor,
                    normalized=normalized,
                )
                for _ in range(n_rblock)
            ]
        )
        self.upscale = UniUpsample(upsampler, current_scale, dim, in_ch, mid_dim)

    def check_img_size(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        scaled_size = self.pad
        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, inp: Tensor) -> Tensor:
        _, _, h, w = inp.shape
        x = self.check_img_size(inp)
        x = self.in_to_dim(x)
        x = self.body(x) + x
        x = self.upscale(x)
        if x.shape[2] > h * self.scale or x.shape[3] > w * self.scale:
            x = x[:, :, : h * self.scale, : w * self.scale]
        return x


def detect_lawfft_params(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Inspects LAWFFT state_dict to deduce all hyperparameters."""
    params = {
        "in_ch": 3,
        "dim": 60,
        "split": 0.25,
        "scale": 4,
        "n_rblock": 4,
        "n_mblock": 6,
        "t_mid_factor": 1.0,
        "window_size": 8,
        "mlp_factor": 2.66,
        "unshuffle_mod": False,
        "upsampler": "pixelshuffledirect",
        "mid_dim": 48,
    }

    # in_ch and dim
    for key, tensor in state_dict.items():
        if key == "in_to_dim.weight":
            params["dim"] = tensor.shape[0]
            params["in_ch"] = tensor.shape[1]
            params["unshuffle_mod"] = False
            break
        elif key == "in_to_dim.1.weight":
            params["dim"] = tensor.shape[0]
            params["unshuffle_mod"] = True
            break

    # window_size
    if "window_size" in state_dict:
        params["window_size"] = int(state_dict["window_size"].item())

    # n_rblock and n_mblock
    r_blocks = [
        int(k.split(".")[1])
        for k in state_dict.keys()
        if k.startswith("body.") and k.split(".")[1].isdigit()
    ]
    if r_blocks:
        params["n_rblock"] = max(r_blocks) + 1

    m_blocks = [
        int(k.split(".")[3])
        for k in state_dict.keys()
        if k.startswith("body.0.residual.") and k.split(".")[3].isdigit()
    ]
    if m_blocks:
        params["n_mblock"] = max(m_blocks) + 1

    # MetaUpsample
    if "upscale.MetaUpsample" in state_dict:
        meta = state_dict["upscale.MetaUpsample"].cpu().numpy()
        if len(meta) >= 7:
            sample_mods = ["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv", "dysample"]
            idx = int(meta[1])
            if idx < len(sample_mods):
                params["upsampler"] = sample_mods[idx]
            params["scale"] = int(meta[2])
            params["mid_dim"] = int(meta[5])

    return params
