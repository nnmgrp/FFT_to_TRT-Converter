# chaiNNer 連携とタイリング最適化ガイド

<p align="center">
  <b><a href="CHAINNER_INTEGRATION.md">English</a></b> |
  <b><a href="CHAINNER_INTEGRATION_RU.md">Русский</a></b> |
  <b><a href="CHAINNER_INTEGRATION_JA.md">日本語</a></b>
</p>

---

本書では、変換された TensorRT FFT エンジンを **chaiNNer** に統合し、タイリング処理に伴うボトルネックを回避して最大の推論スループットを得る方法を解説します。

---

## 1. chaiNNer におけるプラグインの自動認識

本プラグイン `trt_dft_plugins.dll` はエントリポイント `getCreators` を正しくエクスポートしているため、chaiNNer の `load_engine.py` を手動で書き換える必要は**一切ありません**。

chaiNNer にプラグインを自動認識させる手順:

### 方法 A: 標準ライブラリフォルダへの配置 (推奨)
`trt_dft_plugins.dll` を chaiNNer 内蔵 Python 環境の TensorRT ライブラリディレクトリにコピーします:
```
<chaiNNerインストール先>\python\python\Lib\site-packages\tensorrt_libs\
```
またはシステム全体の CUDA bin フォルダへ配置します:
```
<CUDA_PATH>\bin\x64\
```
chaiNNer が TensorRT を初期化する際、Windows はこれらの標準パスからプラグイン DLL を自動的に読み込みます。

### 方法 B: chaiNNer のルートフォルダへの配置
`chaiNNer.exe` が存在するルートディレクトリに `trt_dft_plugins.dll` をコピーします:
```
<chaiNNerインストール先>\trt_dft_plugins.dll
```

---

## 2. タイリングによる速度低下のメカニズム

chaiNNer で高解像度画像を処理する場合、入力タイルサイズの設定が処理速度を根本的に左右します。

### 64x64 固定タイルの落とし穴

静的な入力解像度 `64x64` でエンジンをビルドした場合:
- $2500 \times 2500$ のイラストを処理するために、chaiNNer は画像を **1,521 個のタイル**に細分化します。
- タイルごとに以下のオーバーヘッドが発生します:
  1. CPU $\to$ GPU へのメモリ転送 (PCIe 経由)。
  2. TensorRT カーネル起動 (`execute_async_v3`)。
  3. GPU $\to$ CPU への結果メモリ転送。
  4. ホスト RAM 上でのタイルの縫い合わせ（スティッチング）処理。
- 仮に GPU 側の単一タイル推論がわずか $0.5\text{ ms}$ で完了したとしても、$1,521 \times 0.5\text{ ms} = 760\text{ ms}$ に加え、数千回の PCIe 往復レイテンシによって数秒の遅延が生じます。一方、PyTorch は画像全体を 1 回のバッチで一括処理できるため、結果として PyTorch の方が速く見えてしまいます。

### 解決策: タイルサイズの拡大とダイナミックエンジンの活用

入力タイルサイズを大きくしてビルドすることで、タイル分割数を劇的に削減できます:

| 入力タイルサイズ | 2500x2500 処理時のタイル数 | 相対呼び出し回数 | 全体速度への影響 |
| :---: | :---: | :---: | :---: |
| **64 x 64** | 1,521 | $100\%$ (基準値) | 最も低速 (PCIe 転送律速) |
| **128 x 128** | 380 | $25\%$ ($4\times$ 削減) | 約 $3\times$ 倍高速化 |
| **256 x 256** | 95 | $6.25\%$ ($16\times$ 削減) | 約 $7\times$ 倍高速化 |
| **512 x 512** | 24 | $1.56\%$ ($64\times$ 削減) | 最大スループット |

### 推奨エンジンビルド設定

**8 GB VRAM** の GPU (例: RTX 3070, RTX 4060):
- **固定解像度の場合:** `--height 256 --width 256` でビルド。
- **可変解像度の場合:**
  ```powershell
  python converter\build_engine.py --onnx model_dynamic.onnx --min-size 64 64 --opt-size 256 256 --max-size 512 512 --workspace 2048
  ```

**16+ GB VRAM** の GPU:
- 可変解像度で `--max-size 1024 1024` を指定することで、一般的なイラストの大半をわずか 1〜4 タイルで一瞬に処理可能です。
