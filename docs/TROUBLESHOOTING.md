# Troubleshooting & Common Issues

This guide addresses known edge cases, environment idiosyncrasies, and troubleshooting steps.

---

## 1. CUDA 13 cuFFT DLL Naming Mismatch

### Symptom
```
OSError: [WinError 126] The specified module could not be found: cufft64_13.dll
```

### Cause
In official NVIDIA CUDA 13.x distributions for Windows, NVIDIA retained the cuFFT library filename as `cufft64_12.dll` to preserve ABI compatibility. Some linkers and plugins compiled against CUDA 13 look explicitly for `cufft64_13.dll`.

### Solution
Create a file copy or symbolic link in your CUDA `bin` directory:
```powershell
# In PowerShell (Run as Administrator):
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
Copy-Item cufft64_12.dll cufft64_13.dll
```
The ABI is 100% identical and safe.

---

## 2. Out of Memory (OOM) During Engine Compilation

### Symptom
- System RAM usage surges to 95-100%, followed by python crashing.
- TensorRT log output:
  ```
  [TRT] [E] 9: [shapeShuffle.cpp::computeOffsets::422] Error Code 9: Out of Memory
  ```

### Cause
When dynamic axes are exported with `--max-size 2048 2048` or higher:
- An input of $2048 \times 2048$ with scale $4\times$ yields an internal feature map output of **$8192 \times 8192$**.
- Intermediate convolution layers across 24 blocks allocate dozens of gigabytes of activation memory.
- TensorRT's autotuner profiles kernels against the maximum shape in the profile, exceeding 8 GB or 16 GB VRAM.

### Solution
Limit the dynamic profile's maximum dimensions:
```powershell
python converter\build_engine.py --onnx model.onnx --min-size 64 64 --opt-size 256 256 --max-size 512 512 --workspace 2048
```
A maximum of $512 \times 512$ is memory-safe on 8 GB VRAM while still providing massive speedups over 64x64 tiling.

---

## 3. Device Architecture Incompatibility

### Symptom
```
[TRT] [E] IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error
(The engine plan file is generated on an incompatible device, expecting compute 12.0 got compute 8.6, please rebuild)
```

### Cause
TensorRT engines (`.engine` / `.plan`) contain compiled machine code optimized specifically for the target GPU's streaming multiprocessor (SM) architecture:
- Compute 8.6 = NVIDIA Ampere (RTX 3070, RTX 3080)
- Compute 8.9 = NVIDIA Ada Lovelace (RTX 4070, RTX 4080)
- Compute 12.0 = NVIDIA Blackwell (RTX 5060 Ti, RTX 5080)

An engine built on an Ampere GPU cannot run on Blackwell without rebuilding.

### Solution
Always compile `.engine` files on the target machine using `converter\build_engine.py`. ONNX models (`.onnx`) are fully portable across all GPUs and platforms.

---

## 4. Saving `IHostMemory` in TensorRT 10.x

### Symptom
```
AttributeError: 'tensorrt.tensorrt.IHostMemory' object has no attribute 'size'
```
or
```
AttributeError: 'tensorrt.tensorrt.IHostMemory' object has no attribute 'data'
```

### Cause
In TensorRT 10.x Python bindings, `IHostMemory` no longer exposes `.size()` or `.data()`. Instead, it implements Python's native Buffer Protocol (`__buffer__`).

### Solution
Wrap `engine_bytes` with `memoryview()`:
```python
with open(engine_path, "wb") as f:
    f.write(memoryview(engine_bytes))
```
This writes the serialized engine directly to disk without unnecessary memory copies.
