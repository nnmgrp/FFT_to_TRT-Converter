# FFT_to_TRT-Converter

<p align="center">
  <b><a href="README.md">English</a></b> |
  <b><a href="README_RU.md">Русский</a></b> |
  <b><a href="README_JA.md">日本語</a></b>
</p>

---

**High-Performance TensorRT FFT Plugin & Universal Model Converter for Windows**

Convert frequency-domain super-resolution and image restoration architectures (such as **GFISRV2**, **LAWFFT**, and custom neural FFT networks) from PyTorch checkpoints into ultra-fast NVIDIA TensorRT engines.

---

## Acknowledgements & Credits

- **[umzi2](https://github.com/umzi2)**: Creator and author of the **GFISRV2** (Gated Fourier Inception Super-Resolution v2) and **LAWFFT** (Local Adaptive Weighted Fourier Feature Transformer) neural architectures in the **[resselt](https://github.com/rewaifu/resselt)** repository.
- **[Alexey-Kamenev](https://github.com/Alexey-Kamenev/tensorrt-dft-plugins)**: Original Linux implementation of the TensorRT DFT plugin.
- **[chaiNNer](https://github.com/chaiNNer-org/chaiNNer)**: The visual node-based image processing GUI where these models are deployed.

---

## Key Breakthroughs & Features

- **Native Ortho & Backward FFT Normalization**:
  Fixed C++ plugin implementing cuBLAS-accelerated scaling ($\frac{1}{\sqrt{N}}$) for `norm="ortho"` (`normalized=1`) alongside standard `norm="backward"` (`normalized=0`). Eliminates the $\sqrt{N}$ amplitude distortion that previously degraded frequency-domain weights.
- **Exported `getCreators` Symbol**:
  `trt_dft_plugins.dll` now exports the `getCreators` entry point required by TensorRT 10.x. Resolves `SymbolAddress for getCreators could not be loaded` errors and enables native engine deserialization without external hacks.
- **Universal Model Auto-Detection**:
  `torch2onnx.py` automatically parses `.safetensors`, `.pth`, and `.pt` checkpoints, deducing architecture type (GFISRV2 vs LAWFFT), scale factors ($2\times, 4\times$), channel dimensions, and block counts with zero manual configuration.
- **Direct Plugin Emission**:
  Exports clean `ai.onnx.contrib::Rfft` and `ai.onnx.contrib::Irfft` nodes directly into the ONNX graph. No manual graph patching or attribute renaming scripts required.
- **Memory-Safe Engine Builder**:
  `build_engine.py` provides memory pool controls and sensible dynamic shape profiling (`min=64x64`, `opt=256x256`, `max=512x512`), preventing Out-Of-Memory (OOM) autotuner crashes on 8 GB GPUs.
- **Full Verification & Parity Benchmarking**:
  `benchmark.py` validates numerical parity between PyTorch FP16 and TensorRT engine outputs (calculating MAE, Max Error, PSNR in dB, and Cosine Similarity) and benchmarks latency and throughput.
- **Prebuilt Binaries Included**:
  Ready-to-use `trt_dft_plugins.dll` compiled for Windows 64-bit (CUDA 13 / TensorRT 10) in `prebuilt/win64_cuda13_trt10/`.

---

## Repository Structure

```
FFT_to_TRT-Converter/
├── converter/
│   ├── models/
│   │   ├── gfisrv2.py        # Standalone GFISRV2 implementation (by umzi2)
│   │   └── lawfft.py         # Standalone LAWFFT implementation (by umzi2)
│   ├── torch2onnx.py         # Universal PyTorch (.pth / .safetensors) -> ONNX exporter
│   ├── build_engine.py       # Robust TensorRT builder with dynamic profiles & memory safety
│   ├── benchmark.py          # Parity validation (PSNR/MAE) & latency benchmark
│   └── patch_onnx.py         # Utility to inspect / patch legacy ONNX graphs
├── plugin/
│   ├── CMakeLists.txt        # Root CMake build configuration
│   ├── build_windows.bat     # One-click Windows build script (MSVC 2022)
│   └── src/
│       └── dft_plugins/
│           ├── CMakeLists.txt
│           └── dft_plugins.cpp # C++ plugin with native ortho scaling & getCreators export
├── prebuilt/
│   └── win64_cuda13_trt10/
│       ├── trt_dft_plugins.dll # Compiled plugin DLL
│       └── trt_dft_plugins.lib # Import library
├── docs/
│   ├── ARCHITECTURE.md       # Math deep-dive: Ortho scaling, cuBLAS, & TRT plugin API
│   ├── CHAINNER_INTEGRATION.md # Guide for chaiNNer, tiling trade-offs, and autoloading
│   └── TROUBLESHOOTING.md    # Solutions for CUDA 13 cuFFT naming, OOM, and compute tiers
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Installation

Clone this repository and install Python dependencies:

```powershell
git clone https://github.com/nnmgrp/FFT_to_TRT-Converter.git
cd FFT_to_TRT-Converter
pip install -r requirements.txt
```

### 2. Export PyTorch to ONNX

Convert any GFISRV2 or LAWFFT checkpoint (`.pth` or `.safetensors`) directly to ONNX:

```powershell
# Fixed tile export (e.g. 256x256):
python converter\torch2onnx.py --model path\to\model.safetensors --height 256 --width 256

# Dynamic resolution export:
python converter\torch2onnx.py --model path\to\model.safetensors --dynamic
```

Inspect checkpoint structure without exporting:
```powershell
python converter\torch2onnx.py --model path\to\model.safetensors --detect-only
```

### 3. Build TensorRT Engine

Compile the ONNX graph into a high-performance TensorRT engine:

```powershell
python converter\build_engine.py --onnx path\to\model.onnx --workspace 2048
```

For dynamic resolution models, customize the optimization profile bounds:
```powershell
python converter\build_engine.py --onnx model_dynamic.onnx `
    --min-size 64 64 `
    --opt-size 256 256 `
    --max-size 512 512 `
    --workspace 2048
```

### 4. Benchmark & Parity Verification

Verify numerical correctness (ensuring PSNR > 35-40 dB) and compare PyTorch FP16 speed vs TensorRT:

```powershell
python converter\benchmark.py --model path\to\model.safetensors --engine path\to\model.engine --height 256 --width 256
```

---

## Compiling the C++ Plugin from Source

If you wish to recompile `trt_dft_plugins.dll` for custom CUDA or TensorRT versions:

1. Ensure **Visual Studio 2022** (with C++ Desktop Development), **CMake >= 3.20**, **CUDA Toolkit**, and **TensorRT** are installed.
2. Run the build script:
   ```powershell
   cd plugin
   .\build_windows.bat
   ```
   Or invoke CMake directly:
   ```powershell
   cmake -B build -S plugin -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
   cmake --build build --config Release
   ```

---

## Documentation

- [Architecture & Mathematical Foundations](docs/ARCHITECTURE.md)
- [chaiNNer Integration & Tiling Guide](docs/CHAINNER_INTEGRATION.md)
- [Troubleshooting & Known Issues](docs/TROUBLESHOOTING.md)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) or individual source headers for details.
