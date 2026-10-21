#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_engine.py - Robust TensorRT Engine builder for models with custom FFT plugins.
Features automatic DLL dependency resolution, memory safety, dynamic profile control,
and engine metadata generation.
"""

import os
import sys
import json
import time
import ctypes
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TRTBuilder")


class DLLResolver:
    """Manages Windows DLL search directories and dependency preloading."""

    @staticmethod
    def setup_cuda_paths(explicit_cuda_dir: Optional[str] = None) -> List[str]:
        added = []
        candidates = []
        if explicit_cuda_dir and os.path.isdir(explicit_cuda_dir):
            candidates.append(explicit_cuda_dir)

        # Environment variables
        for env_var in ("CUDA_PATH", "CUDA_PATH_V13_1", "CUDA_PATH_V12_8", "CUDA_HOME"):
            val = os.environ.get(env_var)
            if val and os.path.isdir(val):
                candidates.append(os.path.join(val, "bin"))
                candidates.append(os.path.join(val, "bin", "x64"))

        # Common disk paths
        for drive in ("C:", "G:", "D:", "F:"):
            candidates.extend([
                f"{drive}\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64",
                f"{drive}\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin",
                f"{drive}\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin\\x64",
                f"{drive}\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin",
            ])

        if hasattr(os, "add_dll_directory"):
            for p in candidates:
                if os.path.isdir(p):
                    try:
                        os.add_dll_directory(p)
                        added.append(p)
                        logger.info(f"[DLL] Added DLL directory: {p}")
                    except Exception as e:
                        logger.debug(f"[DLL] Could not add {p}: {e}")

        # Also prepend to PATH for subprocesses or legacy loaders
        if added:
            os.environ["PATH"] = ";".join(added) + ";" + os.environ.get("PATH", "")

        return added

    @staticmethod
    def preload_cuda_and_plugin(plugin_path: Optional[str] = None) -> bool:
        """Preloads nvinfer, cufft, and the DFT plugin."""
        # 1. cuFFT loading (support CUDA 13 alias and CUDA 12)
        cufft_loaded = False
        for cufft_name in ("cufft64_13.dll", "cufft64_12.dll", "cufft64_11.dll"):
            try:
                ctypes.CDLL(cufft_name)
                logger.info(f"[DLL] [OK] Loaded cuFFT library: {cufft_name}")
                cufft_loaded = True
                break
            except OSError:
                continue

        if not cufft_loaded:
            logger.warning("[DLL] [WARN] Could not find cuFFT DLL (cufft64_13.dll / cufft64_12.dll).")

        # 2. nvinfer loading
        for nvinfer_name in ("nvinfer_10.dll", "nvinfer.dll"):
            try:
                ctypes.CDLL(nvinfer_name)
                logger.info(f"[DLL] [OK] Loaded TensorRT base library: {nvinfer_name}")
                break
            except OSError:
                continue

        # 3. Plugin loading
        if plugin_path:
            abs_plugin = os.path.abspath(plugin_path)
            if not os.path.isfile(abs_plugin):
                logger.error(f"[DLL] [FAIL] Plugin DLL not found at: {abs_plugin}")
                return False

            try:
                # On Windows, winmode=0x1000 uses LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                ctypes.CDLL(abs_plugin, winmode=0x1000)
                logger.info(f"[DLL] [OK] Successfully loaded DFT plugin from: {abs_plugin}")
                return True
            except OSError as e:
                # Fallback to winmode=0
                try:
                    ctypes.CDLL(abs_plugin, winmode=0)
                    logger.info(f"[DLL] [OK] Successfully loaded DFT plugin (winmode=0) from: {abs_plugin}")
                    return True
                except Exception as e2:
                    logger.error(f"[DLL] [FAIL] Failed to load plugin DLL: {e2}")
                    return False

        return True


def locate_default_plugin_dll() -> Optional[str]:
    """Locate prebuilt plugin DLL relative to repo."""
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "prebuilt" / "win64_cuda13_trt10" / "trt_dft_plugins.dll"
    if candidate.is_file():
        return str(candidate)
    local_cand = Path(__file__).resolve().parent / "trt_dft_plugins.dll"
    if local_cand.is_file():
        return str(local_cand)
    return None


class EngineBuilder:
    def __init__(self, trt_logger_level: str = "INFO"):
        import tensorrt as trt
        self.trt = trt

        level_map = {
            "INTERNAL_ERROR": trt.Logger.INTERNAL_ERROR,
            "ERROR": trt.Logger.ERROR,
            "WARNING": trt.Logger.WARNING,
            "INFO": trt.Logger.INFO,
            "VERBOSE": trt.Logger.VERBOSE,
        }
        self.logger = trt.Logger(level_map.get(trt_logger_level.upper(), trt.Logger.INFO))

    def verify_registered_plugins(self) -> List[str]:
        registry = self.trt.get_plugin_registry()
        found = []
        for name in ("Rfft", "Irfft"):
            creator = registry.get_creator(name, "1", "")
            if creator:
                found.append(name)
        return found

    def build(
        self,
        onnx_path: str,
        engine_path: str,
        fp16: bool = True,
        tf32: bool = True,
        workspace_mb: int = 2048,
        min_shape: Tuple[int, int, int, int] = (1, 3, 64, 64),
        opt_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
        max_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    ) -> bool:
        trt = self.trt

        logger.info("=" * 65)
        logger.info(f"BUILDING TENSORRT ENGINE: {os.path.basename(onnx_path)}")
        logger.info("=" * 65)

        # 1. Builder and Network creation
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)

        # 2. Parse ONNX
        parser = trt.OnnxParser(network, self.logger)
        logger.info(f"[PARSER] Parsing ONNX model: {onnx_path}...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                logger.error("[PARSER] [FAIL] Failed to parse ONNX:")
                for i in range(parser.num_errors):
                    logger.error(f"  Error {i}: {parser.get_error(i)}")
                return False

        logger.info(f"[PARSER] [OK] Successfully parsed {network.num_layers} layers.")

        # 3. Builder configuration
        config = builder.create_builder_config()
        workspace_bytes = workspace_mb * (1 << 20)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        logger.info(f"[CONFIG] Workspace limit: {workspace_mb} MB")

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("[CONFIG] Enabled FP16 mode")

        if tf32 and hasattr(trt.BuilderFlag, "TF32"):
            config.set_flag(trt.BuilderFlag.TF32)
            logger.info("[CONFIG] Enabled TF32 mode")

        # 4. Check for dynamic dimensions and configure optimization profile
        has_dynamic = False
        profile = builder.create_optimization_profile()
        input_info = {}

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = list(inp.shape)
            input_info[inp.name] = shape
            if any(dim == -1 for dim in shape):
                has_dynamic = True
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                logger.info(
                    f"[PROFILE] Dynamic input '{inp.name}' profile set:\n"
                    f"          MIN={min_shape}, OPT={opt_shape}, MAX={max_shape}"
                )

        if has_dynamic:
            config.add_optimization_profile(profile)
            logger.info("[PROFILE] [OK] Optimization profile added to builder config")
        else:
            logger.info(f"[PROFILE] Static inputs detected: {input_info}")

        # 5. Build serialized engine
        logger.info("[BUILD] Starting TensorRT engine compilation (autotuning kernels)...")
        start_time = time.time()
        try:
            engine_bytes = builder.build_serialized_network(network, config)
        except Exception as e:
            logger.error(f"[BUILD] [FAIL] build_serialized_network raised an exception: {e}")
            return False

        duration = time.time() - start_time
        if engine_bytes is None:
            logger.error("[BUILD] [FAIL] build_serialized_network returned None (engine build failed).")
            return False

        logger.info(f"[BUILD] [OK] Compilation finished in {duration:.1f} seconds.")

        # 6. Save engine using memoryview buffer protocol
        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(memoryview(engine_bytes))

        size_mb = os.path.getsize(engine_path) / (1 << 20)
        logger.info(f"[SAVE] [OK] Engine written to: {engine_path} ({size_mb:.2f} MB)")

        # 7. Generate metadata JSON alongside engine
        meta_path = f"{engine_path}.json"
        meta = {
            "onnx_source": os.path.basename(onnx_path),
            "engine_file": os.path.basename(engine_path),
            "size_mb": round(size_mb, 2),
            "build_duration_sec": round(duration, 1),
            "fp16": fp16,
            "tf32": tf32,
            "workspace_mb": workspace_mb,
            "is_dynamic": has_dynamic,
            "min_shape": list(min_shape) if has_dynamic else None,
            "opt_shape": list(opt_shape) if has_dynamic else None,
            "max_shape": list(max_shape) if has_dynamic else None,
            "inputs": input_info,
            "tensorrt_version": trt.__version__,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[SAVE] [OK] Metadata written to: {meta_path}")

        return True


def main():
    parser = argparse.ArgumentParser(description="TensorRT Engine Builder for FFT Architectures")
    parser.add_argument("--onnx", "-i", type=str, required=True, help="Path to input .onnx model")
    parser.add_argument("--engine", "-o", type=str, default=None, help="Path to output .engine file (defaults to <onnx>.engine)")
    parser.add_argument("--plugin", "-p", type=str, default=None, help="Path to trt_dft_plugins.dll (default: auto-detected from prebuilt/)")
    parser.add_argument("--cuda-dir", type=str, default=None, help="Explicit CUDA bin directory containing cufft and nvinfer")
    parser.add_argument("--workspace", "-w", type=int, default=2048, help="Max workspace memory in MB (default: 2048)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision (use FP32)")
    parser.add_argument("--log-level", choices=["INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE"], default="INFO", help="TRT logger severity")

    # Dynamic shape options
    parser.add_argument("--min-size", nargs=2, type=int, default=[64, 64], help="Min spatial H W for dynamic profile (default: 64 64)")
    parser.add_argument("--opt-size", nargs=2, type=int, default=[256, 256], help="Opt spatial H W for dynamic profile (default: 256 256)")
    parser.add_argument("--max-size", nargs=2, type=int, default=[512, 512], help="Max spatial H W for dynamic profile (default: 512 512)")
    parser.add_argument("--min-batch", type=int, default=1, help="Min batch size (default: 1)")
    parser.add_argument("--opt-batch", type=int, default=1, help="Opt batch size (default: 1)")
    parser.add_argument("--max-batch", type=int, default=1, help="Max batch size (default: 1)")

    args = parser.parse_args()

    # 1. Setup paths and load plugin
    DLLResolver.setup_cuda_paths(args.cuda_dir)
    plugin_path = args.plugin or locate_default_plugin_dll()
    if not plugin_path or not os.path.isfile(plugin_path):
        logger.error("[INIT] [FAIL] Could not locate trt_dft_plugins.dll. Please provide --plugin path.")
        sys.exit(1)

    if not DLLResolver.preload_cuda_and_plugin(plugin_path):
        logger.error("[INIT] [FAIL] Failed to preload dependencies or plugin.")
        sys.exit(1)

    # 2. Check plugin registry
    builder = EngineBuilder(trt_logger_level=args.log_level)
    registered = builder.verify_registered_plugins()
    if not registered:
        logger.error("[INIT] [FAIL] FFT plugins (Rfft / Irfft) not found in TensorRT plugin registry.")
        sys.exit(1)
    logger.info(f"[INIT] [OK] Confirmed FFT plugins in registry: {registered}")

    # 3. Output path
    engine_path = args.engine
    if engine_path is None:
        engine_path = os.path.splitext(args.onnx)[0] + ".engine"

    min_shape = (args.min_batch, 3, args.min_size[0], args.min_size[1])
    opt_shape = (args.opt_batch, 3, args.opt_size[0], args.opt_size[1])
    max_shape = (args.max_batch, 3, args.max_size[0], args.max_size[1])

    success = builder.build(
        onnx_path=args.onnx,
        engine_path=engine_path,
        fp16=not args.no_fp16,
        workspace_mb=args.workspace,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
