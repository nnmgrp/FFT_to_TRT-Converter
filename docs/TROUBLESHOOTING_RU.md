# Устранение неполадок и частые ошибки

<p align="center">
  <b><a href="TROUBLESHOOTING.md">English</a></b> |
  <b><a href="TROUBLESHOOTING_RU.md">Русский</a></b> |
  <b><a href="TROUBLESHOOTING_JA.md">日本語</a></b>
</p>

---

В данном руководстве собраны известные краевые случаи, особенности окружения Windows и методы устранения ошибок.

---

## 1. Несовпадение имён DLL cuFFT в CUDA 13.x

### Симптом
```
OSError: [WinError 126] Не найден указанный модуль: cufft64_13.dll
```

### Причина
В официальном дистрибутиве NVIDIA CUDA 13.x для Windows компания NVIDIA сохранила имя файла библиотеки cuFFT как `cufft64_12.dll` ради обратной совместимости ABI. Некоторые компиляторы и плагины, собранные под CUDA 13, ищут файл с именем `cufft64_13.dll`.

### Решение
Создайте копию файла в каталоге `bin` установленного CUDA Toolkit:
```powershell
# В PowerShell от имени Администратора:
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
Copy-Item cufft64_12.dll cufft64_13.dll
```
Бинарный интерфейс (ABI) идентичен на 100%, переименование полностью безопасно.

---

## 2. Нехватка памяти (OOM) при компиляции динамического движка

### Симптом
- Потребление оперативной памяти (RAM) подскакивает до 95–100%, процесс Python аварийно завершается.
- В логах TensorRT появляется сообщение:
  ```
  [TRT] [E] 9: [shapeShuffle.cpp::computeOffsets::422] Error Code 9: Out of Memory
  ```

### Причина
Если модель экспортирована с динамическими осями и параметром `--max-size 2048 2048` или выше:
- При разрешении входа $2048 \times 2048$ и апскейле $4\times$ размер выходной карты признаков достигает **$8192 \times 8192$**.
- Промежуточные 24 сверточных блока выделяют десятки гигабайт видеопамяти под активации.
- Алгоритм autotuner в TensorRT тестирует тактики ядер под максимальный указанный размер профиля, моментально исчерпывая 8 ГБ или 16 ГБ VRAM.

### Решение
Ограничьте максимальный размер профиля динамических форм:
```powershell
python converter\build_engine.py --onnx model.onnx --min-size 64 64 --opt-size 256 256 --max-size 512 512 --workspace 2048
```
Максимальный размер $512 \times 512$ полностью безопасен по памяти на видеокартах с 8 ГБ VRAM и при этом даёт колоссальный прирост скорости относительно тайлов 64x64.

---

## 3. Несовместимость поколений микроархитектуры GPU

### Симптом
```
[TRT] [E] IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error
(The engine plan file is generated on an incompatible device, expecting compute 12.0 got compute 8.6, please rebuild)
```

### Причина
Файлы движков TensorRT (`.engine` / `.plan`) содержат низкоуровневый машинный код (SASS), скомпилированный под конкретное поколение потоковых мультипроцессоров (SM) видеокарты:
- Compute 8.6 = NVIDIA Ampere (RTX 3070, RTX 3080)
- Compute 8.9 = NVIDIA Ada Lovelace (RTX 4070, RTX 4080)
- Compute 12.0 = NVIDIA Blackwell (RTX 5060 Ti, RTX 5080)

Движок, скомпилированный на видеокарте Ampere, не сможет запуститься на Blackwell без повторной компиляции.

### Решение
Всегда собирайте файл `.engine` непосредственно на целевом компьютере с помощью `converter\build_engine.py`. Промежуточные ONNX-модели (`.onnx`) полностью универсальны и переносимы между любыми видеокартами.

---

## 4. Ошибка сохранения `IHostMemory` в TensorRT 10.x

### Симптом
```
AttributeError: 'tensorrt.tensorrt.IHostMemory' object has no attribute 'size'
```
или
```
AttributeError: 'tensorrt.tensorrt.IHostMemory' object has no attribute 'data'
```

### Причина
В Python-биндингах TensorRT версии 10.x объект `IHostMemory` больше не имеет методов `.size()` или `.data()`. Вместо этого он реализует стандартный протокол буфера Python (`__buffer__`).

### Решение
Оборачивайте `engine_bytes` в `memoryview()`:
```python
with open(engine_path, "wb") as f:
    f.write(memoryview(engine_bytes))
```
Это позволяет напрямую записать сериализованный движок на диск без избыточного копирования памяти.
