# アーキテクチャと技術的詳細

<p align="center">
  <b><a href="ARCHITECTURE.md">English</a></b> |
  <b><a href="ARCHITECTURE_RU.md">Русский</a></b> |
  <b><a href="ARCHITECTURE_JA.md">日本語</a></b>
</p>

---

本書では、**FFT_to_TRT-Converter** の内部における離散フーリエ変換の数学的基礎、CUDA/cuFFT 実行パイプライン、および TensorRT C++ プラグインの設計について詳細に解説します。

---

## 1. FFT 正規化の数学的基礎

2次元空間信号 $x[m, n]$ (解像度 $H \times W$、$N = H \times W$) に対する離散フーリエ変換 (DFT) は次のように定義されます:

$$\hat{X}[u, v] = \sum_{m=0}^{H-1} \sum_{n=0}^{W-1} x[m, n] \, e^{-j 2\pi \left(\frac{u m}{H} + \frac{v n}{W}\right)}$$

逆離散フーリエ変換 (IDFT) により、周波数空間から元の空間信号が再構成されます:

$$x[m, n] = \frac{1}{N} \sum_{u=0}^{H-1} \sum_{v=0}^{W-1} \hat{X}[u, v] \, e^{j 2\pi \left(\frac{u m}{H} + \frac{v n}{W}\right)}$$

### 正規化モードの定義

深層学習フレームワークや論文によって、正規化スケーリング係数の割り当て方式が異なります:

| モード | PyTorch の引数 | 前方向 (Forward) 係数 | 逆方向 (Inverse) 係数 | 往復サイクル (Forward $\to$ Inverse) |
| :--- | :--- | :--- | :--- | :--- |
| **Backward** | `norm="backward"` (デフォルト) | $1.0$ | $\frac{1}{N} = \frac{1}{H \times W}$ | $1.0$ |
| **Ortho** | `norm="ortho"` | $\frac{1}{\sqrt{N}} = \frac{1}{\sqrt{H \times W}}$ | $\frac{1}{\sqrt{N}} = \frac{1}{\sqrt{H \times W}}$ | $1.0$ |
| **Forward** | `norm="forward"` | $\frac{1}{N}$ | $1.0$ | $1.0$ |

### ニューラルネットワークにおける Ortho モードの重要性

**GFISRV2** などのモデルでは、畳み込み、活性化関数、レイヤー正規化が周波数領域で直接実行されます:

$$\text{Features}_{\text{freq}} = \text{Conv}_{\text{freq}}(\text{RMSNorm}(\text{RFFT2}(x, \text{norm}="ortho")))$$

重みが `norm="ortho"` ($normalized=1$) で学習されているにもかかわらず、ランタイムが `norm="backward"` ($normalized=0$) で実行された場合:
- 前方向のスペクトル振幅が $\sqrt{N} = \sqrt{H \times W}$ 倍に増大します。
- $64 \times 64$ のパッチの場合、$\sqrt{4096} = 64$ 倍の増幅となります。
- 後続の畳み込み層や GELU 活性化関数に 64 倍の過大入力が送られ、特徴量マップが飽和し、出力画像がノイズに破壊されます。

---

## 2. cuFFT および cuBLAS スケーリングの実装

NVIDIA `cuFFT` はデフォルトでは**正規化処理を行いません**:
- `cufftExecR2C` はスケーリングなしの前方向変換を実行します (係数 $1.0$)。
- `cufftExecC2R` はスケーリングなしの逆方向変換を実行します (係数 $1.0$)。

本プラグイン (`dft_plugins.cpp`) では、cuBLAS による高速スケーリングを変換直後に実行することで、`normalized == 1` (`norm="ortho"`) をネイティブサポートしています:

```cpp
// RfftPlugin::enqueue 内:
if (normalized_ == 1) {
    float scale = 1.0f / std::sqrt(static_cast<float>(total_dft_size));
    // 複素数出力は実部・虚部からなる 2 * total_complex_elements 個の float
    int32_t total_floats = 2 * (batch_size * signal_channels * out_h * out_w);
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
}

// IrfftPlugin::enqueue 内:
if (normalized_ == 1) {
    float scale = 1.0f / std::sqrt(static_cast<float>(total_dft_size));
    int32_t total_floats = batch_size * signal_channels * out_h * out_w;
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
} else if (normalized_ == 0) {
    // Backward モードでは逆変換を N で除算
    float scale = 1.0f / static_cast<float>(total_dft_size);
    int32_t total_floats = batch_size * signal_channels * out_h * out_w;
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
}
```

これにより、PyTorch の `torch.fft.rfft2(..., norm="ortho")` と浮動小数点の丸め誤差レベルで完全に一致する数値結果が得られます。

---

## 3. TensorRT プラグイン API とシンボルエクスポート

TensorRT 10.x は `IPluginCreator` および `IPluginV2DynamicExt` に基づくプラグイン機構を採用しています。

### `getCreators` のエクスポート

Windows における動的リンクライブラリ (DLL) は、明示的に `__declspec(dllexport)` を付与しない限り内部シンボルが非公開となります。
TensorRT のランタイムデシリアライズ (`deserializeCudaEngine`) や `init_libnvinfer_plugins()` は、以下のエントリポイントを照会します:

```cpp
extern "C" TRT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);
```

このシンボルがエクスポートされていない場合:
1. `init_libnvinfer_plugins()` はプラグインを認識しません。
2. エンジンのデシリアライズ時に以下のエラーでクラッシュします:
   ```
   IRuntime::deserializeCudaEngine: Error Code 3: API Usage Error
   (SymbolAddress for getCreators could not be loaded)
   ```
3. ユーザーは Python の `ctypes` 等を用いた危険な DLL 差し込みを強いられていました。

コンパイル済みの `trt_dft_plugins.dll` は `getCreators` を直接公開しているため、TensorRT 公式バインディングや chaiNNer から自動的にプラグインが読み込まれます。
