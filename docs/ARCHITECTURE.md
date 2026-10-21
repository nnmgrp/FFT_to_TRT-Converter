# Architecture & Technical Deep-Dive

<p align="center">
  <b><a href="ARCHITECTURE.md">English</a></b> |
  <b><a href="ARCHITECTURE_RU.md">Русский</a></b> |
  <b><a href="ARCHITECTURE_JA.md">日本語</a></b>
</p>

---

This document details the internal mathematical foundations, CUDA/cuFFT execution pipeline, and TensorRT plugin design of the **FFT_to_TRT-Converter**.

---

## 1. Mathematical Foundations of FFT Normalization

The Discrete Fourier Transform (DFT) for a discrete 2D spatial signal $x[m, n]$ of dimensions $H \times W$ ($N = H \times W$) is defined as:

$$\hat{X}[u, v] = \sum_{m=0}^{H-1} \sum_{n=0}^{W-1} x[m, n] \, e^{-j 2\pi \left(\frac{u m}{H} + \frac{v n}{W}\right)}$$

The Inverse Discrete Fourier Transform (IDFT) reconstructs the spatial signal:

$$x[m, n] = \frac{1}{N} \sum_{u=0}^{H-1} \sum_{v=0}^{W-1} \hat{X}[u, v] \, e^{j 2\pi \left(\frac{u m}{H} + \frac{v n}{W}\right)}$$

### Normalization Modes

Different deep learning frameworks and papers parameterize the scaling factor differently:

| Mode | PyTorch Flag | Forward Scaling Factor | Inverse Scaling Factor | End-to-End Cycle Scaling |
| :--- | :--- | :--- | :--- | :--- |
| **Backward** | `norm="backward"` (default) | $1.0$ | $\frac{1}{N} = \frac{1}{H \times W}$ | $1.0$ |
| **Ortho** | `norm="ortho"` | $\frac{1}{\sqrt{N}} = \frac{1}{\sqrt{H \times W}}$ | $\frac{1}{\sqrt{N}} = \frac{1}{\sqrt{H \times W}}$ | $1.0$ |
| **Forward** | `norm="forward"` | $\frac{1}{N}$ | $1.0$ | $1.0$ |

### Why Ortho Mode Matters in Neural Networks

Models such as **GFISRV2** apply convolutions, activations, and layer normalizations directly in the frequency domain:

$$\text{Features}_{\text{freq}} = \text{Conv}_{\text{freq}}(\text{RMSNorm}(\text{RFFT2}(x, \text{norm}="ortho")))$$

If a runtime executes with `norm="backward"` ($normalized=0$) when the weights were trained with `norm="ortho"` ($normalized=1$):
- Forward spectral magnitudes are magnified by a factor of $\sqrt{N} = \sqrt{H \times W}$.
- For a $64 \times 64$ patch, $\sqrt{N} = 64$.
- Convolutions and GELU activations receive activations $64\times$ larger than intended, completely destroying the neural activations.

---

## 2. cuFFT & cuBLAS Scaling Implementation

NVIDIA `cuFFT` does **not** normalize transforms by default:
- `cufftExecR2C` performs an unnormalized forward transform (scale $1.0$).
- `cufftExecC2R` performs an unnormalized inverse transform (scale $1.0$).

In the updated plugin (`dft_plugins.cpp`), we implement native support for `normalized == 1` (`norm="ortho"`) by chaining a fast cuBLAS scaling call directly after transform execution:

```cpp
// In RfftPlugin::enqueue:
if (normalized_ == 1) {
    float scale = 1.0f / std::sqrt(static_cast<float>(total_dft_size));
    // Complex output has 2 * total_complex_elements floats
    int32_t total_floats = 2 * (batch_size * signal_channels * out_h * out_w);
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
}

// In IrfftPlugin::enqueue:
if (normalized_ == 1) {
    float scale = 1.0f / std::sqrt(static_cast<float>(total_dft_size));
    int32_t total_floats = batch_size * signal_channels * out_h * out_w;
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
} else if (normalized_ == 0) {
    // Backward mode: inverse is divided by N
    float scale = 1.0f / static_cast<float>(total_dft_size);
    int32_t total_floats = batch_size * signal_channels * out_h * out_w;
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
}
```

This ensures identical numerical equivalence with PyTorch `torch.fft.rfft2(..., norm="ortho")` down to floating-point machine epsilon.

---

## 3. TensorRT Plugin API & Symbol Export

TensorRT 10.x uses a modular plugin architecture based on `IPluginCreator` and `IPluginV2DynamicExt`.

### Exporting `getCreators`

On Windows, dynamic libraries (DLLs) hide symbols unless explicitly exported via `__declspec(dllexport)` or a DEF file.
TensorRT's runtime deserialization (`deserializeCudaEngine`) and `init_libnvinfer_plugins()` query the entry point:

```cpp
extern "C" TRT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);
```

Without this exported symbol:
1. `init_libnvinfer_plugins()` ignores the library.
2. Engine deserialization fails with:
   ```
   IRuntime::deserializeCudaEngine: Error Code 3: API Usage Error
   (SymbolAddress for getCreators could not be loaded)
   ```
3. Consumers were previously forced to perform fragile DLL injection.

Our compiled `trt_dft_plugins.dll` exports `getCreators` directly, making it automatically discoverable by both TensorRT and chaiNNer.
