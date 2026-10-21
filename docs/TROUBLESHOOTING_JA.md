# トラブルシューティングと既知の問題

<p align="center">
  <b><a href="TROUBLESHOOTING.md">English</a></b> |
  <b><a href="TROUBLESHOOTING_RU.md">Русский</a></b> |
  <b><a href="TROUBLESHOOTING_JA.md">日本語</a></b>
</p>

---

本書では、Windows 環境における既知のエッジケース、環境固有の挙動、および各種エラーの対処方法をまとめます。

---

## 1. CUDA 13 cuFFT DLL 名の不一致問題

### 発生するエラー
```
OSError: [WinError 126] The specified module could not be found: cufft64_13.dll
```

### 原因
NVIDIA 公式の Windows 向け CUDA 13.x 配布物では、ABI の後方互換性を維持するために cuFFT の DLL ファイル名が `cufft64_12.dll` のまま維持されています。CUDA 13 向けにコンパイルされたリンカーやプラグインの一部は、`cufft64_13.dll` を明示的に要求します。

### 対処方法
インストール先の CUDA `bin` ディレクトリ内でファイルのコピーを作成します:
```powershell
# 管理者権限の PowerShell で実行:
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
Copy-Item cufft64_12.dll cufft64_13.dll
```
ABI は完全に同一であり、安全に使用できます。

---

## 2. エンジンコンパイル時のメモリ不足 (OOM)

### 発生するエラー
- システムの RAM 使用率が 95〜100% に急増し、Python プロセスが異常終了する。
- TensorRT のログに以下のエラーが出力される:
  ```
  [TRT] [E] 9: [shapeShuffle.cpp::computeOffsets::422] Error Code 9: Out of Memory
  ```

### 原因
可変解像度モデルで `--max-size 2048 2048` 以上の過大なプロファイルを指定した場合:
- 入力 $2048 \times 2048$、拡大率 $4\times$ の場合、内部特徴マップの解像度は **$8192 \times 8192$** に達します。
- 24 ブロックにわたる中間畳み込み層で数十ギガバイトのアクティベーションメモリが確保されます。
- TensorRT の autotuner が最大サイズに基づいてカーネルのベンチマークを実行するため、8 GB または 16 GB の VRAM を一瞬で枯渇させます。

### 対処方法
動的プロファイルの最大サイズを現実的な値に制限します:
```powershell
python converter\build_engine.py --onnx model.onnx --min-size 64 64 --opt-size 256 256 --max-size 512 512 --workspace 2048
```
最大 $512 \times 512$ であれば、8 GB VRAM でも OOM を起こさず安全にビルドでき、かつ 64x64 タイリングに対して圧倒的な高速化を実現できます。

---

## 3. GPU アーキテクチャの互換性エラー

### 発生するエラー
```
[TRT] [E] IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error
(The engine plan file is generated on an incompatible device, expecting compute 12.0 got compute 8.6, please rebuild)
```

### 原因
TensorRT エンジン (`.engine` / `.plan`) には、ターゲット GPU のストリーミングマルチプロセッサ (SM) アーキテクチャ専用に最適化されたバイナリ機械語が含まれています:
- Compute 8.6 = NVIDIA Ampere (RTX 3070, RTX 3080)
- Compute 8.9 = NVIDIA Ada Lovelace (RTX 4070, RTX 4080)
- Compute 12.0 = NVIDIA Blackwell (RTX 5060 Ti, RTX 5080)

Ampere 世代でビルドされたエンジンは、再ビルドしない限り Blackwell 世代では動作しません。

### 対処方法
エンジンファイル (`.engine`) は常に実際に実行するターゲットマシン上で `converter\build_engine.py` を実行してコンパイルしてください。中間ファイルの ONNX (`.onnx`) は完全にプラットフォーム非依存であり、どの GPU にも移植可能です。

---

## 4. TensorRT 10.x における `IHostMemory` の保存エラー

### 発生するエラー
```
AttributeError: 'tensorrt.tensorrt.IHostMemory' object has no attribute 'size'
```
または
```
AttributeError: 'tensorrt.tensorrt.IHostMemory' object has no attribute 'data'
```

### 原因
TensorRT 10.x の Python バインディングにおいて、`IHostMemory` オブジェクトから `.size()` および `.data()` メソッドが削除されました。代わりに、Python 標準のバッファプロトコル (`__buffer__`) を実装しています。

### 対処方法
`engine_bytes` を `memoryview()` でラップしてファイルに書き込みます:
```python
with open(engine_path, "wb") as f:
    f.write(memoryview(engine_bytes))
```
これにより、無駄なメモリコピーを行わずにシリアライズされたエンジンを直接ディスクへ保存できます。
