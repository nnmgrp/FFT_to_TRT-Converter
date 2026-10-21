# FFT_to_TRT-Converter

<p align="center">
  <b><a href="README.md">English</a></b> |
  <b><a href="README_RU.md">Русский</a></b> |
  <b><a href="README_JA.md">日本語</a></b>
</p>

---

**Windows 向け 高性能 TensorRT FFT プラグイン ＆ 汎用モデルコンバーター**

**GFISRV2**、**LAWFFT**、独自のニューラル FFT ネットワークなど、周波数領域を利用する超解像・画像復元モデルを PyTorch のチェックポイントから極めて高速な NVIDIA TensorRT エンジンへ変換します。

---

## 謝辞とアーキテクチャのクレジット

- **[umzi2](https://github.com/umzi2)**: **[resselt](https://github.com/rewaifu/resselt)** プロジェクトにおいて **GFISRV2** (Gated Fourier Inception Super-Resolution v2) および **LAWFFT** (Local Adaptive Weighted Fourier Feature Transformer) のニューラルネットワーク設計・開発者。
- **[Alexey-Kamenev](https://github.com/Alexey-Kamenev/tensorrt-dft-plugins)**: TensorRT 向け DFT プラグインのオリジナル（Linux）実装。
- **[chaiNNer](https://github.com/chaiNNer-org/chaiNNer)**: これらのモデルが活用されるノードベースの画像処理 GUI 環境。

---

## 主な特長と改善点

- **Ortho および Backward モードのネイティブ正規化対応**:
  修正された C++ プラグインは、cuBLAS によるハードウェアスケーリング ($\frac{1}{\sqrt{N}}$) を実装し、従来の `norm="backward"` (`normalized=0`) に加えて `norm="ortho"` (`normalized=1`) をネイティブサポートします。従来の変換で発生していた $\sqrt{N}$ 倍の振幅のズレによる画質破壊を完全に解消しました。
- **`getCreators` シンボルのエクスポート**:
  TensorRT 10.x で必須となるエントリポイント `getCreators` を `trt_dft_plugins.dll` に追加しました。これにより `SymbolAddress for getCreators could not be loaded` エラーが解消され、外部ハックなしにエンジンの直接デシリアライズが可能になりました。
- **モデルパラメータの完全自動検出**:
  `torch2onnx.py` は `.safetensors`、`.pth`、`.pt` チェックポイントを解析し、アーキテクチャ種別 (GFISRV2 / LAWFFT)、拡大率 ($2\times, 4\times$)、チャネル数、ブロック数、アップサンプラーを自動判定します。手動設定は一切不要です。
- **ONNX へのプラグインノード直接出力**:
  ONNX グラフの出力時に `ai.onnx.contrib::Rfft` および `ai.onnx.contrib::Irfft` を正しい属性値（`normalized=1`, `onesided=1`, `signal_ndim=2`）で直接生成します。手動での属性書き換えスクリプトは不要です。
- **メモリ安全なエンジンビルダー**:
  `build_engine.py` は適切なワークスペース制限と動的プロファイル範囲 (`min=64x64`, `opt=256x256`, `max=512x512`) を設定し、8 GB VRAM の GPU における autotuner の Out-Of-Memory (OOM) クラッシュを防止します。
- **精度検証とベンチマークスイート**:
  `benchmark.py` を使用して、PyTorch FP16 と TensorRT エンジン間の数値誤差（MAE、最大誤差、PSNR (dB)、コサイン類似度）および推論速度（遅延時間、FPS、MPixels/秒）を測定・検証できます。
- **ビルド済みバイナリを同梱**:
  Windows 64-bit (CUDA 13 / TensorRT 10) 向けのコンパイル済み `trt_dft_plugins.dll` が `prebuilt/win64_cuda13_trt10/` に収録されています。

---

## リポジトリ構成

```
FFT_to_TRT-Converter/
├── converter/
│   ├── models/
│   │   ├── gfisrv2.py        # GFISRV2 スタンドアロン実装 (umzi2 氏作)
│   │   └── lawfft.py         # LAWFFT スタンドアロン実装 (umzi2 氏作)
│   ├── torch2onnx.py         # 汎用 PyTorch (.pth / .safetensors) -> ONNX 変換スクリプト
│   ├── build_engine.py       # 動的プロファイル対応の堅牢な TensorRT ビルダー
│   ├── benchmark.py          # 精度検証 (PSNR/MAE) および速度計測ツール
│   └── patch_onnx.py         # 既存・旧形式の ONNX グラフ検証・パッチユーティリティ
├── plugin/
│   ├── CMakeLists.txt        # プラグインの CMake ビルド定義
│   ├── build_windows.bat     # Windows 向けワンクリックビルドスクリプト (MSVC 2022)
│   └── src/
│       └── dft_plugins/
│           ├── CMakeLists.txt
│           └── dft_plugins.cpp # C++ プラグインソース (ortho スケーリング & getCreators 実装)
├── prebuilt/
│   └── win64_cuda13_trt10/
│       ├── trt_dft_plugins.dll # コンパイル済み DLL
│       └── trt_dft_plugins.lib # インポートライブラリ (.lib)
├── docs/
│   ├── ARCHITECTURE.md       # 数学的背景: Ortho スケーリング、cuBLAS、TRT プラグイン設計
│   ├── CHAINNER_INTEGRATION.md # chaiNNer 連携ガイド、タイリング最適化、自動読み込み
│   └── TROUBLESHOOTING.md    # トラブルシューティング: CUDA 13 cuFFT 名、OOM、Compute Tier
├── requirements.txt
└── README_JA.md
```

---

## クイックスタート

### 1. インストール

リポジトリをクローンし、Python 依存関係をインストールします:

```powershell
git clone https://github.com/nnmgrp/FFT_to_TRT-Converter.git
cd FFT_to_TRT-Converter
pip install -r requirements.txt
```

### 2. PyTorch モデルから ONNX へのエクスポート

GFISRV2 または LAWFFT のチェックポイント (`.pth` または `.safetensors`) を直接 ONNX へ変換します:

```powershell
# 固定タイルサイズでエクスポート (例: 256x256):
python converter\torch2onnx.py --model path\to\model.safetensors --height 256 --width 256

# 可変解像度 (ダイナミック軸) でエクスポート:
python converter\torch2onnx.py --model path\to\model.safetensors --dynamic
```

エクスポートせずにモデルパラメータのみを確認する場合:
```powershell
python converter\torch2onnx.py --model path\to\model.safetensors --detect-only
```

### 3. TensorRT エンジンのビルド

生成した ONNX グラフから TensorRT エンジンをコンパイルします:

```powershell
python converter\build_engine.py --onnx path\to\model.onnx --workspace 2048
```

動的解像度モデルの場合、最適化プロファイル範囲を指定してビルドします:
```powershell
python converter\build_engine.py --onnx model_dynamic.onnx `
    --min-size 64 64 `
    --opt-size 256 256 `
    --max-size 512 512 `
    --workspace 2048
```

### 4. ベンチマークと精度検証

出力の数値的同一性を検証し (PSNR > 35〜40 dB)、PyTorch FP16 と TensorRT の推論速度を比較します:

```powershell
python converter\benchmark.py --model path\to\model.safetensors --engine path\to\model.engine --height 256 --width 256
```

---

## C++ プラグインのソースコードビルド

特定の CUDA または TensorRT バージョン向けに `trt_dft_plugins.dll` を再コンパイルする場合:

1. **Visual Studio 2022** (C++ デスクトップ開発)、**CMake >= 3.20**、**CUDA Toolkit**、**TensorRT** がインストールされていることを確認します。
2. ビルドスクリプトを実行します:
   ```powershell
   cd plugin
   .\build_windows.bat
   ```
   または CMake を直接実行します:
   ```powershell
   cmake -B build -S plugin -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
   cmake --build build --config Release
   ```

---

## ドキュメント

- [アーキテクチャと数学的基礎 (英語)](docs/ARCHITECTURE.md)
- [chaiNNer 連携とタイリング最適化ガイド (英語)](docs/CHAINNER_INTEGRATION.md)
- [トラブルシューティングと既知の問題 (英語)](docs/TROUBLESHOOTING.md)

---

## ライセンス

本プロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) または各ソースコードのヘッダーをご参照ください。
