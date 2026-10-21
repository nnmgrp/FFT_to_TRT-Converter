#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch2onnx.py - Universal PyTorch to ONNX converter for FFT architectures.
Exports models with native TensorRT FFT plugin nodes (ai.onnx.contrib::Rfft/Irfft).
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import math
import argparse
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

# Add parent directory to sys.path so converter package can be imported directly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))

from converter.models.gfisrv2 import GFISRV2, detect_gfisrv2_params
from converter.models.lawfft import LAWFFT, detect_lawfft_params


def load_state_dict_from_file(model_path: str) -> Dict[str, torch.Tensor]:
    """Load weights from .pth, .pt, .ckpt, or .safetensors."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
            print(f"[LOAD] Loading safetensors: {model_path}")
            return load_file(model_path, device="cpu")
        except ImportError:
            raise ImportError("Package 'safetensors' is required to load .safetensors files. Run: pip install safetensors")

    print(f"[LOAD] Loading checkpoint: {model_path}")
    raw = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
        if "params" in raw and isinstance(raw["params"], dict):
            return raw["params"]
        if "params_ema" in raw and isinstance(raw["params_ema"], dict):
            return raw["params_ema"]
        return raw

    raise ValueError(f"Unrecognized checkpoint contents in {model_path}")


def identify_and_instantiate_model(
    state_dict: Dict[str, torch.Tensor],
    norm_mode: str = "auto",
) -> Tuple[nn.Module, Dict[str, Any], str]:
    """Identifies model architecture (GFISRV2 vs LAWFFT) and instantiates it with weights."""
    keys_str = " ".join(state_dict.keys())

    if "gfisr_body" in keys_str or "rn.scale" in keys_str or "post_norm.scale" in keys_str:
        arch_name = "GFISRV2"
        params = detect_gfisrv2_params(state_dict)
        norm_val = 1 if (norm_mode == "ortho" or norm_mode == "auto") else 0
        params["normalized"] = norm_val
        print(f"[ARCH] Detected: {arch_name}")
        print(f"[ARCH] Parameters: {params}")
        model = GFISRV2(**params)

    elif "FSAS" in keys_str or "body.0.residual" in keys_str or "t_mid_factor" in keys_str:
        arch_name = "LAWFFT"
        params = detect_lawfft_params(state_dict)
        norm_val = 0 if (norm_mode == "backward" or norm_mode == "auto") else 1
        params["normalized"] = norm_val
        print(f"[ARCH] Detected: {arch_name}")
        print(f"[ARCH] Parameters: {params}")
        model = LAWFFT(**params)

    else:
        raise ValueError(
            "Could not automatically identify model architecture from state_dict keys. "
            "Supported: GFISRV2, LAWFFT."
        )

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Ignore buffers that are generated dynamically
        critical_missing = [k for k in missing if not k.endswith("MetaUpsample") and not k.endswith("window_size")]
        if critical_missing:
            print(f"[WARN] Missing keys ({len(critical_missing)}): {critical_missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.eval()
    return model, params, arch_name


def register_fft_symbolics(opset: int = 17, normalized: int = 1):
    """Registers aten::fft_rfft2 and aten::fft_irfft2 custom symbolics for ONNX export."""
    @parse_args('v')
    def symbolic_rfft2(g, x):
        return g.op(
            "ai.onnx.contrib::Rfft",
            x,
            normalized_i=normalized,
            onesided_i=1,
            signal_ndim_i=2,
        )

    @parse_args('v')
    def symbolic_irfft2(g, x):
        return g.op(
            "ai.onnx.contrib::Irfft",
            x,
            normalized_i=normalized,
            onesided_i=1,
            signal_ndim_i=2,
        )

    register_custom_op_symbolic('aten::fft_rfft2', symbolic_rfft2, opset)
    register_custom_op_symbolic('aten::fft_irfft2', symbolic_irfft2, opset)
    print(f"[ONNX] Registered custom FFT symbolics (normalized={normalized}) for opset {opset}")


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int],
    dynamic: bool = False,
    opset: int = 17,
) -> bool:
    """Exports PyTorch model to ONNX."""
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    with torch.no_grad():
        test_out = model(dummy_input)
    print(f"[ONNX] Input Shape:  {tuple(dummy_input.shape)}")
    print(f"[ONNX] Output Shape: {tuple(test_out.shape)}")

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
        print("[ONNX] Dynamic axes enabled: (batch, height, width)")
    else:
        print(f"[ONNX] Static shape export: {input_shape}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[ONNX] Exporting to: {output_path}...")
    kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamo": False,
    }
    if dynamic_axes is not None:
        kwargs["dynamic_axes"] = dynamic_axes

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        **kwargs,
    )

    if not os.path.exists(output_path):
        print(f"[ERROR] Export failed; file not written: {output_path}")
        return False

    size_mb = os.path.getsize(output_path) / (1 << 20)
    print(f"[ONNX] [OK] Export complete! Size: {size_mb:.2f} MB")

    # Validate ONNX model
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("[ONNX] [OK] Model validation passed!")

        fft_nodes = [n for n in model_onnx.graph.node if n.op_type in ("Rfft", "Irfft")]
        print(f"[ONNX] [OK] Found {len(fft_nodes)} TensorRT DFT plugin nodes in graph:")
        if fft_nodes:
            sample_node = fft_nodes[0]
            attrs = {a.name: a.i for a in sample_node.attribute if a.type == onnx.AttributeProto.INT}
            print(f"       Sample ({sample_node.op_type}): domain='{sample_node.domain}', attributes={attrs}")
    except Exception as e:
        print(f"[WARN] ONNX check note: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Universal PyTorch to ONNX converter for FFT architectures")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to input .pth, .pt, or .safetensors file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output .onnx file (defaults to <model>.onnx)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes for variable spatial resolutions")
    parser.add_argument("--height", "-H", type=int, default=None, help="Input spatial height (default: auto based on scale)")
    parser.add_argument("--width", "-W", type=int, default=None, help="Input spatial width (default: auto based on scale)")
    parser.add_argument("--batch", "-b", type=int, default=1, help="Input batch size (default: 1)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--norm", choices=["auto", "ortho", "backward"], default="auto",
                        help="FFT normalization mode: 'ortho' (1), 'backward' (0), or 'auto' (ortho for GFISRV2, backward for LAWFFT)")
    parser.add_argument("--detect-only", action="store_true", help="Inspect and display model parameters without exporting")

    args = parser.parse_args()

    state_dict = load_state_dict_from_file(args.model)
    model, params, arch_name = identify_and_instantiate_model(state_dict, norm_mode=args.norm)

    if args.detect_only:
        print("\n=== Model Inspection ===")
        print(f"Architecture: {arch_name}")
        for k, v in params.items():
            print(f"  {k}: {v}")
        return

    # Determine input shape
    scale = params.get("scale", 4)
    in_channels = params.get("in_nc", params.get("in_ch", 3))

    h = args.height
    w = args.width
    if h is None or w is None:
        # Default spatial dimensions: 256 / scale or at least 64
        base_size = max(64, 256 // scale)
        # Pad to multiple of model requirement (e.g. 8 for LAWFFT or 2 for GFISRV2)
        pad_req = getattr(model, "pad", 8)
        base_size = ((base_size + pad_req - 1) // pad_req) * pad_req
        h = h or base_size
        w = w or base_size

    input_shape = (args.batch, in_channels, h, w)

    # Determine output path
    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(args.model)[0]
        mode_suffix = "dynamic" if args.dynamic else f"static_{h}x{w}"
        output_path = f"{base_name}_{mode_suffix}.onnx"

    norm_val = params.get("normalized", 1)
    register_fft_symbolics(opset=args.opset, normalized=norm_val)

    export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=input_shape,
        dynamic=args.dynamic,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
