# chaiNNer Integration & Tiling Guide

<p align="center">
  <b><a href="CHAINNER_INTEGRATION.md">English</a></b> |
  <b><a href="CHAINNER_INTEGRATION_RU.md">Русский</a></b> |
  <b><a href="CHAINNER_INTEGRATION_JA.md">日本語</a></b>
</p>

---

This guide explains how to integrate converted TensorRT FFT engines into **chaiNNer**, optimize performance, and avoid common throughput bottlenecks.

---

## 1. Automatic Plugin Discovery in chaiNNer

Because `trt_dft_plugins.dll` exports `getCreators`, manual monkeypatching of chaiNNer's `load_engine.py` is **no longer required**!

To make chaiNNer load the plugin automatically:

### Option A: Standard Library Directory (Recommended)
Copy `trt_dft_plugins.dll` into chaiNNer's Python environment directory containing the TensorRT binaries:
```
<chaiNNer-dir>\python\python\Lib\site-packages\tensorrt_libs\
```
Or into the CUDA bin directory:
```
<CUDA_PATH>\bin\x64\
```
When chaiNNer initializes TensorRT, all DLLs in these standard directories are automatically resolved by Windows.

### Option B: Local chaiNNer Directory
Copy `trt_dft_plugins.dll` into the root directory of your chaiNNer installation:
```
<chaiNNer-dir>\trt_dft_plugins.dll
```

---

## 2. Understanding the Tiling Bottleneck

When processing high-resolution images in chaiNNer, the execution approach fundamentally determines performance.

### The 64x64 Static Trap

If an engine is compiled with a fixed static shape of `64x64`:
- Processing a $2500 \times 2500$ image requires dividing the image into **1,521 tiles**.
- Each tile triggers:
  1. CPU $\to$ GPU memory copy.
  2. TensorRT kernel launch (`execute_async_v3`).
  3. GPU $\to$ CPU memory copy.
  4. Boundary seam stitching in host RAM.
- Even if GPU execution per tile takes only $0.5\text{ ms}$, $1,521 \times 0.5\text{ ms} = 760\text{ ms}$, plus several seconds of PCIe transfer overhead.

### The Solution: Larger Tiles & Dynamic Engines

Compiling engines with larger static tile sizes or flexible dynamic shapes dramatically reduces tile count:

| Input Tile Size | Tiles for 2500x2500 | Relative Tile Count | Typical Latency Impact |
| :---: | :---: | :---: | :---: |
| **64 x 64** | 1,521 | $100\%$ (baseline) | Slowest (PCIe bound) |
| **128 x 128** | 380 | $25\%$ ($4\times$ fewer) | $3\times$ faster |
| **256 x 256** | 95 | $6.25\%$ ($16\times$ fewer) | $7\times$ faster |
| **512 x 512** | 24 | $1.56\%$ ($64\times$ fewer) | Maximum throughput |

### Recommended Engine Profiles

For GPUs with **8 GB VRAM** (e.g. RTX 3070, RTX 4060):
- **Static Option:** Build with `--height 256 --width 256`.
- **Dynamic Option:**
  ```powershell
  python converter\build_engine.py --onnx model_dynamic.onnx --min-size 64 64 --opt-size 256 256 --max-size 512 512 --workspace 2048
  ```

For GPUs with **16+ GB VRAM**:
- Dynamic `--max-size 1024 1024` allows processing entire typical anime / manga illustrations in 1 to 4 tiles.

---

## 3. Rift73 / chaiNNer-C (v0.3.8) Fork Specifics (TensorRT 11.4)

The **[Rift73/chaiNNer-C](https://github.com/Rift73/chaiNNer-C)** fork targets **TensorRT 11.4.0.106** and **CUDA 13.4**. This branch (`chaiNNer-C`) addresses key changes in NVIDIA's plugin mechanism:

1. **TensorRT 11 Library Linking (`nvinfer_11.dll`):**
   - Engines built for older TRT 10.x runtimes fail deserialization in TRT 11.
   - The plugin DLL on the `chaiNNer-C` branch links directly against `nvinfer_11.dll`.

2. **Mandatory `setLoggerFinder` Export:**
   - In TensorRT 11, `nvinfer1::PluginRegistry::loadLibraryCommon` rejects libraries that do not export `setLoggerFinder`. Our plugin provides valid exports for both `setLoggerFinder` and `getCreators`.

3. **Prevention of Windows Loader Lock (WinError 1114):**
   - Eliminated static plugin registry calls inside `DllMain`, preventing initialization failure during module loading.

4. **Installation in chaiNNer-C:**
   - The prebuilt DLL is available at `prebuilt/win64_cuda13_trt11/trt_dft_plugins.dll`.
   - Copy it into:
     ```
     <chaiNNer-C_dir>\python\python\Lib\site-packages\tensorrt_libs\
     ```
     or:
     ```
     <chaiNNer-C_dir>\resources\src\nodes\impl\tensorrt\
     ```
   - When TensorRT nodes run inside chaiNNer-C, the `Rfft` and `Irfft` operations are automatically discovered and loaded.
