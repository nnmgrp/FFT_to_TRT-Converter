#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfisrv2.py - Standalone GFISRV2 model architecture for PyTorch & TensorRT export.
Architecture designed by umzi2 (https://github.com/umzi2) in resselt (https://github.com/rewaifu/resselt).
Zero external Spandrel dependencies; exports clean ai.onnx.contrib::Rfft / Irfft nodes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Global normalization mode for FFT custom operators:
# 1 = "ortho" (normalized by 1/sqrt(N) in both directions; native in updated plugin)
# 0 = "backward" (unnormalized forward, 1/N inverse)
DEFAULT_FFT_NORMALIZED: int = 1


class CustomRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, normalized: int = DEFAULT_FFT_NORMALIZED):
        ctx.normalized = normalized
        norm_str = "ortho" if normalized == 1 else "backward"
        y = torch.fft.rfft2(x, dim=(2, 3), norm=norm_str)
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g, x: torch.Value, normalized: int = DEFAULT_FFT_NORMALIZED):
        return g.op(
            "ai.onnx.contrib::Rfft",
            x,
            normalized_i=normalized,
            onesided_i=1,
            signal_ndim_i=2,
        )


class CustomIRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_ri: torch.Tensor, normalized: int = DEFAULT_FFT_NORMALIZED):
        ctx.normalized = normalized
        norm_str = "ortho" if normalized == 1 else "backward"
        x_c = torch.view_as_complex(x_ri)
        return torch.fft.irfft2(x_c, dim=(2, 3), norm=norm_str)

    @staticmethod
    def symbolic(g, x: torch.Value, normalized: int = DEFAULT_FFT_NORMALIZED):
        return g.op(
            "ai.onnx.contrib::Irfft",
            x,
            normalized_i=normalized,
            onesided_i=1,
            signal_ndim_i=2,
        )


class CustomRfft2Wrap(nn.Module):
    def __init__(self, normalized: int = DEFAULT_FFT_NORMALIZED) -> None:
        super().__init__()
        self.normalized = normalized

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            norm_str = "ortho" if self.normalized == 1 else "backward"
            y = torch.fft.rfft2(x, dim=(2, 3), norm=norm_str)
            return torch.view_as_real(y)
        return CustomRFFT2.apply(x, self.normalized)


class CustomIrfft2Wrap(nn.Module):
    def __init__(self, normalized: int = DEFAULT_FFT_NORMALIZED) -> None:
        super().__init__()
        self.normalized = normalized

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            norm_str = "ortho" if self.normalized == 1 else "backward"
            x_c = torch.view_as_complex(x)
            return torch.fft.irfft2(x_c, dim=(2, 3), norm=norm_str)
        return CustomIRFFT2.apply(x, self.normalized)


def pad_to_multiple(x: Tensor, multiple: int, mode: str = "reflect") -> Tensor:
    """Pad tensor height and width to a multiple of specified size."""
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)


SAMPLE_MODS = (
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "transpose+conv",
    "lda",
    "pa_up",
)


class DySample(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= groups or in_channels % groups != 0:
            raise ValueError("Incorrect in_channels and groups values.")

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )
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
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device) + 0.5
        coords_w = torch.arange(w, device=x.device) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
        )
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)
        if self.end_convolution:
            output = self.end_conv(output)
        return output


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


class RMSNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones([dim, 1, 1]))
        self.offset = nn.Parameter(torch.zeros([dim, 1, 1]))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed + self.offset


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalized: int = DEFAULT_FFT_NORMALIZED) -> None:
        super().__init__()
        self.rn = RMSNorm(out_channels * 2)
        self.post_norm = RMSNorm(out_channels)
        self.fdc = nn.Conv2d(in_channels * 2, out_channels * 2, 1, bias=True)
        self.fpe = nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            3,
            padding=1,
            groups=in_channels * 2,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.irfft2 = CustomIrfft2Wrap(normalized=normalized)
        self.rfft2 = CustomRfft2Wrap(normalized=normalized)

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        b, c, h, _ = x.shape
        ffted = self.rfft2(x)
        ffted = ffted.permute(0, 4, 1, 2, 3).contiguous()
        ffted = ffted.view(b, c * 2, h, -1).to(orig_dtype)
        ffted = self.rn(ffted)
        ffted = self.fpe(ffted) + ffted
        ffted = self.fdc(ffted)
        ffted = self.gelu(ffted)
        ffted = ffted.view(b, c, 2, h, -1).permute(0, 1, 3, 4, 2).contiguous().float()
        out = self.irfft2(ffted)
        out = self.post_norm(out.to(orig_dtype))
        return out


class InceptionDWConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
        shift: int = 0,
        normalized: int = DEFAULT_FFT_NORMALIZED,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.convs = [
            FourierUnit(in_channels - 3 * gc, in_channels - 3 * gc, normalized=normalized),
            nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
        ]
        self.pconv = self.convs[shift % 4]
        self.dwconv_hw = self.convs[(shift + 1) % 4]
        self.dwconv_w = self.convs[(shift + 2) % 4]
        self.dwconv_h = self.convs[(shift + 3) % 4]
        indexs = [in_channels - 3 * gc, gc, gc, gc]
        self.split_indexes = (
            indexs[shift % 4],
            indexs[(shift + 1) % 4],
            indexs[(shift + 2) % 4],
            indexs[(shift + 3) % 4],
        )

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (
                self.pconv(x_id),
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
            ),
            dim=1,
        )


class GatedCNNBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        shift: int = 0,
        normalized: int = DEFAULT_FFT_NORMALIZED,
    ) -> None:
        super().__init__()
        hidden = int(expansion_ratio * dim)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.act = nn.SiLU()
        conv_channels = dim
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = InceptionDWConv2d(dim, shift=shift, normalized=normalized)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        g, i, c = torch.split(x, self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x * self.gamma + shortcut


class UniUpsampleV3(nn.Sequential):
    def __init__(
        self,
        upsample: str = "pixelshuffle",
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,
        group: int = 4,
        dysample_end_kernel: int = 1,
    ) -> None:
        m: List[nn.Module] = []
        if upsample == "pixelshuffledirect":
            m.append(nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1))
            m.append(nn.PixelShuffle(scale))
        elif upsample == "pixelshuffle":
            m.append(nn.Conv2d(in_dim, mid_dim, 3, 1, 1))
            m.append(nn.SiLU(True))
            if scale in [2, 3]:
                m.append(nn.Conv2d(mid_dim, mid_dim * scale**2, 3, 1, 1))
                m.append(nn.PixelShuffle(scale))
            elif scale == 4:
                m.append(nn.Conv2d(mid_dim, mid_dim * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.Conv2d(mid_dim, mid_dim * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            elif scale == 8:
                m.append(nn.Conv2d(mid_dim, mid_dim * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.Conv2d(mid_dim, mid_dim * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.Conv2d(mid_dim, mid_dim * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            else:
                raise ValueError(f"scale {scale} not supported for pixelshuffle")
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            m.append(
                DySample(
                    in_dim,
                    out_dim,
                    scale=scale,
                    groups=group,
                    end_kernel=dysample_end_kernel,
                )
            )
        elif upsample == "nearest+conv":
            m.append(nn.UpsamplingNearest2d(scale_factor=scale))
            m.append(nn.Conv2d(in_dim, mid_dim, 3, 1, 1))
            m.append(nn.SiLU(True))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(f"upsampler '{upsample}' is not supported")

        super().__init__(*m)
        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [3, SAMPLE_MODS.index(upsample), scale, in_dim, out_dim, mid_dim, group],
                dtype=torch.uint8,
            ),
        )


class GFISRV2(nn.Module):
    def __init__(
        self,
        *,
        in_nc: int = 3,
        dim: int = 48,
        expansion_ratio: float = 8 / 3,
        scale: int = 4,
        out_nc: int = 3,
        upsampler: str = "pixelshuffledirect",
        mid_dim: int = 32,
        pixel_unshuffle: bool = False,
        n_blocks: int = 24,
        normalized: int = DEFAULT_FFT_NORMALIZED,
    ) -> None:
        super().__init__()
        self.true_scale = scale
        if pixel_unshuffle and scale in [1, 2]:
            down = 4 // scale
            self.in_to_dim = nn.Sequential(
                nn.PixelUnshuffle(down),
                nn.Conv2d(in_nc * down * down, dim, 3, 1, 1),
            )
            scale = 4
            self.pad = down * 2
        else:
            self.in_to_dim = nn.Conv2d(in_nc, dim, 3, 1, 1)
            self.pad = 2

        self.gfisr_body = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim,
                    expansion_ratio=expansion_ratio,
                    shift=i,
                    normalized=normalized,
                )
                for i in range(n_blocks)
            ]
            + [
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.SiLU(True),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
            ]
        )
        self.upscale = UniUpsampleV3(
            upsampler, scale, dim, out_nc, mid_dim, dysample_end_kernel=3
        )
        self.scale = scale

    def load_state_dict(self, state_dict: Dict[str, Any], *args, **kwargs):
        if "upscale.MetaUpsample" not in state_dict:
            state_dict["upscale.MetaUpsample"] = self.upscale.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        x = pad_to_multiple(x, self.pad, mode="reflect")
        x = self.in_to_dim(x)
        x = self.gfisr_body(x) + x
        out = self.upscale(x)
        return out[:, :, : h * self.true_scale, : w * self.true_scale]


def detect_gfisrv2_params(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Inspects GFISRV2 state_dict to deduce all hyperparameters."""
    params = {
        "in_nc": 3,
        "out_nc": 3,
        "dim": 48,
        "scale": 4,
        "n_blocks": 24,
        "upsampler": "pixelshuffledirect",
        "mid_dim": 32,
        "pixel_unshuffle": False,
        "expansion_ratio": 8.0 / 3.0,
    }

    # in_nc and dim
    for key, tensor in state_dict.items():
        if key in ("in_to_dim.weight", "in_to_dim.1.weight"):
            out_ch, in_ch = tensor.shape[:2]
            params["dim"] = out_ch
            params["in_nc"] = in_ch
            break

    # n_blocks (only count GatedCNNBlock which contains .gamma)
    block_indices = [
        int(k.split(".")[1])
        for k in state_dict.keys()
        if k.startswith("gfisr_body.") and k.endswith(".gamma")
    ]
    if block_indices:
        params["n_blocks"] = max(block_indices) + 1

    # scale and upsampler from MetaUpsample
    for key, tensor in state_dict.items():
        if "MetaUpsample" in key:
            meta = tensor.cpu().numpy()
            if len(meta) >= 7:
                upsampler_idx = int(meta[1])
                if upsampler_idx < len(SAMPLE_MODS):
                    params["upsampler"] = SAMPLE_MODS[upsampler_idx]
                params["scale"] = int(meta[2])
                params["mid_dim"] = int(meta[5])
            break

    # fallback scale detection from upscale weights if MetaUpsample is absent
    if "MetaUpsample" not in str(state_dict.keys()):
        for key, tensor in state_dict.items():
            if "upscale.0.weight" in key:
                out_ch, in_ch = tensor.shape[:2]
                out_nc = params.get("out_nc", 3)
                if out_ch % out_nc == 0:
                    scale_sq = out_ch // out_nc
                    detected_scale = int(math.isqrt(scale_sq))
                    if detected_scale * detected_scale == scale_sq:
                        params["scale"] = detected_scale

    # expansion_ratio
    for key, tensor in state_dict.items():
        if "fc1.weight" in key and "gfisr_body" in key:
            out_ch = tensor.shape[0]
            hidden = out_ch // 2
            if hidden > 0 and params["dim"] > 0:
                params["expansion_ratio"] = hidden / params["dim"]
            break

    return params
