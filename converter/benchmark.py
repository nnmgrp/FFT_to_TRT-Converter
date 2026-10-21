#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py - Validation & Performance Benchmarking Tool.
Compares PyTorch FP16/FP32 vs TensorRT Engine:
- Measures inference speed (latency ms, FPS, MegaPixels/sec).
- Measures numerical accuracy (MAE, Max Error, PSNR, Cosine Similarity).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add parent directory to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))

from converter.torch2onnx import load_state_dict_from_file, identify_and_instantiate_model
from converter.build_engine import DLLResolver, locate_default_plugin_dll


def calculate_metrics(out_torch: np.ndarray, out_trt: np.ndarray) -> Dict[str, float]:
    """Calculates numerical fidelity metrics between PyTorch and TensorRT outputs."""
    diff = np.abs(out_torch - out_trt)
    mae = float(np.mean(diff))
    max_err = float(np.max(diff))

    mse = float(np.mean((out_torch - out_trt) ** 2))
    if mse == 0:
        psnr = 999.0
    else:
        max_val = max(float(np.max(out_torch)), 1.0)
        psnr = float(20 * np.log10(max_val / np.sqrt(mse)))

    # Cosine similarity
    vec1 = out_torch.flatten().astype(np.float64)
    vec2 = out_trt.flatten().astype(np.float64)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = float(np.dot(vec1, vec2) / (norm1 * norm2 + 1e-12))

    return {
        "MAE": mae,
        "Max_Error": max_err,
        "PSNR_dB": psnr,
        "Cosine_Similarity": cosine_sim,
    }


class TRTInferenceRunner:
    """Manages TensorRT 10.x execution context and GPU memory buffers."""

    def __init__(self, engine_path: str, plugin_path: Optional[str] = None):
        import tensorrt as trt
        self.trt = trt

        DLLResolver.setup_cuda_paths()
        plugin = plugin_path or locate_default_plugin_dll()
        if plugin:
            DLLResolver.preload_cuda_and_plugin(plugin)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.output_name = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Runs inference synchronously on GPU."""
        shape = tuple(input_tensor.shape)
        self.context.set_input_shape(self.input_name, shape)

        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_dtype = torch.float16 if self.engine.get_tensor_dtype(self.output_name) == self.trt.DataType.HALF else torch.float32

        output_tensor = torch.empty(out_shape, dtype=out_dtype, device=input_tensor.device)

        # Feed device pointers
        self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output_tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.synchronize()

        return output_tensor


def benchmark_model(
    model_path: str,
    engine_path: str,
    height: int = 128,
    width: int = 128,
    runs: int = 20,
    warmup: int = 5,
    fp16: bool = True,
):
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. A GPU is required for benchmarking.")
        return

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"\n=======================================================")
    print(f"BENCHMARK: PyTorch vs TensorRT Engine on {gpu_name}")
    print(f"=======================================================")

    # 1. Load PyTorch model
    print(f"[PYTORCH] Loading model weights from: {model_path}...")
    state_dict = load_state_dict_from_file(model_path)
    model, params, arch = identify_and_instantiate_model(state_dict)
    model = model.to(device)
    if fp16:
        model = model.half()
    model.eval()

    # 2. Load TensorRT Engine
    print(f"[TENSORRT] Loading engine from: {engine_path}...")
    trt_runner = TRTInferenceRunner(engine_path)

    # 3. Prepare Input Tensor
    in_channels = params.get("in_nc", params.get("in_ch", 3))
    torch_dtype = torch.float16 if fp16 else torch.float32
    input_tensor = torch.randn(1, in_channels, height, width, dtype=torch_dtype, device=device)

    # 4. Numerical Accuracy Verification
    print("\n[VALIDATION] Verifying numerical output parity...")
    with torch.no_grad():
        out_torch = model(input_tensor).float().cpu().numpy()

    out_trt = trt_runner.run(input_tensor).float().cpu().numpy()

    metrics = calculate_metrics(out_torch, out_trt)
    print(f"  - Mean Absolute Error (MAE): {metrics['MAE']:.6f}")
    print(f"  - Max Absolute Error:        {metrics['Max_Error']:.6f}")
    print(f"  - PSNR:                      {metrics['PSNR_dB']:.2f} dB")
    print(f"  - Cosine Similarity:         {metrics['Cosine_Similarity']:.6f}")

    if metrics["PSNR_dB"] > 35.0:
        print("  [OK] Accuracy Check: EXCELLENT (High Fidelity Match)")
    elif metrics["PSNR_dB"] > 25.0:
        print("  [OK] Accuracy Check: ACCEPTABLE (Minor FP16 Divergence)")
    else:
        print("  [WARN] Accuracy Check: LOW PSNR - verify FFT normalization mode!")

    # 5. Speed Benchmarks
    print(f"\n[SPEED] Running {warmup} warmup iterations + {runs} timed iterations (Spatial: {width}x{height})...")

    # PyTorch Warmup & Timing
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch_times = []
        for _ in range(runs):
            start_event.record()
            _ = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            torch_times.append(start_event.elapsed_time(end_event))

    # TensorRT Warmup & Timing
    for _ in range(warmup):
        _ = trt_runner.run(input_tensor)
    torch.cuda.synchronize()

    trt_times = []
    for _ in range(runs):
        start_event.record()
        _ = trt_runner.run(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        trt_times.append(start_event.elapsed_time(end_event))

    med_torch = float(np.median(torch_times))
    med_trt = float(np.median(trt_times))
    speedup = med_torch / med_trt if med_trt > 0 else 0

    mpixels = (height * width) / 1e6
    torch_mpps = mpixels / (med_torch / 1000.0)
    trt_mpps = mpixels / (med_trt / 1000.0)

    print("\n=======================================================")
    print("BENCHMARK RESULTS")
    print("=======================================================")
    print(f"Resolution:          {width}x{height} pixels ({mpixels:.3f} MP)")
    print(f"PyTorch Median:      {med_torch:.2f} ms ({1000.0/med_torch:.1f} FPS, {torch_mpps:.2f} MP/s)")
    print(f"TensorRT Median:     {med_trt:.2f} ms ({1000.0/med_trt:.1f} FPS, {trt_mpps:.2f} MP/s)")
    print(f"Speedup Factor:      {speedup:.2f}x faster with TensorRT")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT Engine")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to PyTorch model (.pth / .safetensors)")
    parser.add_argument("--engine", "-e", type=str, required=True, help="Path to TensorRT .engine file")
    parser.add_argument("--height", "-H", type=int, default=128, help="Benchmark spatial height (default: 128)")
    parser.add_argument("--width", "-W", type=int, default=128, help="Benchmark spatial width (default: 128)")
    parser.add_argument("--runs", "-r", type=int, default=20, help="Number of benchmark iterations (default: 20)")
    parser.add_argument("--fp32", action="store_true", help="Run benchmark in FP32 instead of FP16")

    args = parser.parse_args()

    benchmark_model(
        model_path=args.model,
        engine_path=args.engine,
        height=args.height,
        width=args.width,
        runs=args.runs,
        fp16=not args.fp32,
    )


if __name__ == "__main__":
    main()
