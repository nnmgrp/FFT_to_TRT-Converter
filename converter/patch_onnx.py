#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_onnx.py - Post-export inspector and patch utility for ONNX models with FFT nodes.
Converts arbitrary or legacy ONNX FFT nodes to TensorRT DFT Plugin format (ai.onnx.contrib).
"""

import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import onnx
except ImportError:
    print("[ERROR] 'onnx' library is required. Install via: pip install onnx")
    sys.exit(1)


def inspect_onnx_fft_nodes(model: onnx.ModelProto):
    """Prints all FFT nodes and their attributes in the ONNX model."""
    fft_nodes = [n for n in model.graph.node if n.op_type in ("Rfft", "Irfft")]
    print(f"[INSPECT] Found {len(fft_nodes)} FFT nodes in graph:")
    for idx, node in enumerate(fft_nodes):
        attrs = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.INT:
                attrs[a.name] = a.i
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode("utf-8", errors="ignore")
            elif a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = list(a.ints)
            else:
                attrs[a.name] = f"type_{a.type}"
        if idx < 5 or idx == len(fft_nodes) - 1:
            print(f"  [{idx:02d}] {node.op_type} (domain='{node.domain}'): {attrs}")
        elif idx == 5:
            print(f"  ... ({len(fft_nodes) - 6} more nodes) ...")


def patch_onnx_fft_nodes(
    input_path: str,
    output_path: str,
    normalized: int = 1,
    onesided: int = 1,
    signal_ndim: int = 2,
    target_domain: str = "ai.onnx.contrib",
) -> bool:
    """Patches all Rfft and Irfft nodes to have exact attributes expected by TensorRT plugin."""
    print(f"[PATCH] Loading: {input_path}")
    model = onnx.load(input_path)
    inspect_onnx_fft_nodes(model)

    patched_count = 0
    for node in model.graph.node:
        if node.op_type not in ("Rfft", "Irfft"):
            continue

        # Set domain
        node.domain = target_domain

        # Clear existing attributes
        del node.attribute[:]

        # normalized: 1 for ortho, 0 for backward
        attr_norm = onnx.AttributeProto()
        attr_norm.name = "normalized"
        attr_norm.type = onnx.AttributeProto.INT
        attr_norm.i = normalized
        node.attribute.append(attr_norm)

        # onesided: 1
        attr_onesided = onnx.AttributeProto()
        attr_onesided.name = "onesided"
        attr_onesided.type = onnx.AttributeProto.INT
        attr_onesided.i = onesided
        node.attribute.append(attr_onesided)

        # signal_ndim: 2
        attr_ndim = onnx.AttributeProto()
        attr_ndim.name = "signal_ndim"
        attr_ndim.type = onnx.AttributeProto.INT
        attr_ndim.i = signal_ndim
        node.attribute.append(attr_ndim)

        patched_count += 1

    print(f"\n[PATCH] Successfully patched {patched_count} nodes to domain='{target_domain}', normalized={normalized}, onesided={onesided}, signal_ndim={signal_ndim}.")

    try:
        onnx.checker.check_model(model)
        print("[PATCH] [OK] ONNX model checker validation: PASSED")
    except Exception as e:
        print(f"[PATCH] [WARN] ONNX checker note: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    print(f"[PATCH] [OK] Saved patched model to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch and validate ONNX models for TensorRT DFT Plugin")
    parser.add_argument("input", type=str, help="Input ONNX file path")
    parser.add_argument("output", type=str, nargs="?", default=None, help="Output ONNX file path (optional; inspect only if omitted)")
    parser.add_argument("--norm", type=int, choices=[0, 1], default=1, help="FFT normalization: 1=ortho (default), 0=backward")
    parser.add_argument("--domain", type=str, default="ai.onnx.contrib", help="Custom op domain (default: ai.onnx.contrib)")

    args = parser.parse_args()

    if args.output is None:
        model = onnx.load(args.input)
        inspect_onnx_fft_nodes(model)
    else:
        patch_onnx_fft_nodes(
            input_path=args.input,
            output_path=args.output,
            normalized=args.norm,
            target_domain=args.domain,
        )


if __name__ == "__main__":
    main()
