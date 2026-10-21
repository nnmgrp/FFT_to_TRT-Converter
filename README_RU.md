# FFT_to_TRT-Converter

<p align="center">
  <b><a href="README.md">English</a></b> |
  <b><a href="README_RU.md">Русский</a></b> |
  <b><a href="README_JA.md">日本語</a></b>
</p>

---

**Высокопроизводительный TensorRT FFT-плагин и универсальный конвертер моделей для Windows**

Конвертация частотных нейросетевых архитектур супер-разрешения и восстановления изображений (таких как **GFISRV2**, **LAWFFT** и кастомных FFT-моделей) из чекпоинтов PyTorch в оптимизированные движки NVIDIA TensorRT.

---

## Благодарности и авторство архитектур

- **[umzi2](https://github.com/umzi2)**: Создатель и автор нейросетевых архитектур **GFISRV2** (Gated Fourier Inception Super-Resolution v2) и **LAWFFT** (Local Adaptive Weighted Fourier Feature Transformer) в проекте **[resselt](https://github.com/rewaifu/resselt)**.
- **[Alexey-Kamenev](https://github.com/Alexey-Kamenev/tensorrt-dft-plugins)**: Базовая Linux-реализация плагина преобразования Фурье для TensorRT.
- **[chaiNNer](https://github.com/chaiNNer-org/chaiNNer)**: Модульная графическая среда обработки изображений на узлах, в которой применяются эти модели.

---

## Ключевые возможности и исправления

- **Нативная поддержка Ortho и Backward нормализации Фурье**:
  Обновлён C++ плагин с аппаратным cuBLAS-масштабированием ($\frac{1}{\sqrt{N}}$) для режима `norm="ortho"` (`normalized=1`) наряду со стандартным `norm="backward"` (`normalized=0`). Устранено расхождение амплитуды в $\sqrt{N}$ раз, которое ранее разрушало веса частотных моделей.
- **Экспорт символа `getCreators`**:
  В библиотеку `trt_dft_plugins.dll` добавлен экспорт `getCreators`, обязательный для TensorRT 10.x. Это решает ошибку `SymbolAddress for getCreators could not be loaded` и позволяет автоматически десериализовать движок без ручных внедрений DLL.
- **Универсальное автоопределение параметров**:
  Скрипт `torch2onnx.py` напрямую считывает чекпоинты `.safetensors`, `.pth` и `.pt`, автоматически определяя тип архитектуры (GFISRV2 или LAWFFT), масштаб ($2\times, 4\times$), число каналов, блоков и тип апсемплера без ручной настройки.
- **Прямой экспорт узлов плагина в ONNX**:
  Генерирует чистые узлы `ai.onnx.contrib::Rfft` и `ai.onnx.contrib::Irfft` сразу в ONNX-граф. Никаких сторонних скриптов патчинга или переименования атрибутов больше не требуется.
- **Безопасная сборка движка по памяти**:
  Скрипт `build_engine.py` управляет пулом памяти и задаёт разумные границы динамических профилей (`min=64x64`, `opt=256x256`, `max=512x512`), предотвращая вылеты autotuner по Out-Of-Memory (OOM) на видеокартах с 8 ГБ VRAM.
- **Верификация точности и бенчмарк**:
  Утилита `benchmark.py` сравнивает численную точность между PyTorch FP16 и TensorRT Engine (MAE, максимальная ошибка, PSNR в дБ, косинусное сходство) и замеряет задержку и пропускную способность (FPS и MPixels/сек).
- **Готовые бинарные файлы**:
  Скомпилированная библиотека `trt_dft_plugins.dll` под Windows 64-bit (CUDA 13 / TensorRT 10) уже находится в папке `prebuilt/win64_cuda13_trt10/`.

---

## Структура репозитория

```
FFT_to_TRT-Converter/
├── converter/
│   ├── models/
│   │   ├── gfisrv2.py        # Автономная реализация GFISRV2 (автор: umzi2)
│   │   └── lawfft.py         # Автономная реализация LAWFFT (автор: umzi2)
│   ├── torch2onnx.py         # Универсальный экспортёр PyTorch (.pth / .safetensors) -> ONNX
│   ├── build_engine.py       # Сборщик TensorRT с динамическими профилями и защитой памяти
│   ├── benchmark.py          # Проверка точности (PSNR/MAE) и замер скорости
│   └── patch_onnx.py         # Утилита для проверки / патчинга сторонних ONNX-графов
├── plugin/
│   ├── CMakeLists.txt        # Корневой CMake для плагина
│   ├── build_windows.bat     # Скрипт сборки под Windows (MSVC 2022) в один клик
│   └── src/
│       └── dft_plugins/
│           ├── CMakeLists.txt
│           └── dft_plugins.cpp # C++ плагин с ortho-масштабированием и экспортом getCreators
├── prebuilt/
│   └── win64_cuda13_trt10/
│       ├── trt_dft_plugins.dll # Скомпилированная DLL плагина
│       └── trt_dft_plugins.lib # Файл импорта .lib
├── docs/
│   ├── ARCHITECTURE.md       # Математика: Ortho-масштабирование, cuBLAS и API плагинов TRT
│   ├── CHAINNER_INTEGRATION.md # Инструкция по интеграции в chaiNNer, тайлинг и автозагрузка
│   └── TROUBLESHOOTING.md    # Решение проблем: имена cuFFT в CUDA 13, OOM и архитектуры GPU
├── requirements.txt
└── README_RU.md
```

---

## Быстрый старт

### 1. Установка

Склонируйте репозиторий и установите зависимости Python:

```powershell
git clone https://github.com/nnmgrp/FFT_to_TRT-Converter.git
cd FFT_to_TRT-Converter
pip install -r requirements.txt
```

### 2. Экспорт PyTorch в ONNX

Конвертируйте модель GFISRV2 или LAWFFT (`.pth` или `.safetensors`) напрямую в формат ONNX:

```powershell
# Экспорт со статическим тайлом (например, 256x256):
python converter\torch2onnx.py --model путь\к\модели.safetensors --height 256 --width 256

# Экспорт с динамическим разрешением:
python converter\torch2onnx.py --model путь\к\модели.safetensors --dynamic
```

Проверить параметры модели без экспорта:
```powershell
python converter\torch2onnx.py --model путь\к\модели.safetensors --detect-only
```

### 3. Сборка TensorRT Engine

Скомпилируйте граф ONNX в высокопроизводительный движок TensorRT:

```powershell
python converter\build_engine.py --onnx путь\к\модели.onnx --workspace 2048
```

Для моделей с динамическим разрешением укажите границы диапазона оптимизации:
```powershell
python converter\build_engine.py --onnx model_dynamic.onnx `
    --min-size 64 64 `
    --opt-size 256 256 `
    --max-size 512 512 `
    --workspace 2048
```

### 4. Бенчмарк и проверка точности

Убедитесь в математической точности движка (PSNR > 35–40 дБ) и сравните скорость PyTorch FP16 с TensorRT:

```powershell
python converter\benchmark.py --model путь\к\модели.safetensors --engine путь\к\модели.engine --height 256 --width 256
```

---

## Сборка C++ плагина из исходников

Если вам необходимо пересобрать `trt_dft_plugins.dll` под другую версию CUDA или TensorRT:

1. Убедитесь, что установлены **Visual Studio 2022** (с компонентами C++), **CMake >= 3.20**, **CUDA Toolkit** и **TensorRT**.
2. Запустите скрипт сборки:
   ```powershell
   cd plugin
   .\build_windows.bat
   ```
   Либо вызовите CMake напрямую:
   ```powershell
   cmake -B build -S plugin -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
   cmake --build build --config Release
   ```

---

## Документация

- [Архитектура и математические основы](docs/ARCHITECTURE.md)
- [Интеграция в chaiNNer и оптимизация тайлинга](docs/CHAINNER_INTEGRATION.md)
- [Устранение неполадок и частые ошибки](docs/TROUBLESHOOTING.md)

---

## Лицензия

Проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE) или заголовках исходных файлов.
