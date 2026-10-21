# Архитектура и техническое описание

<p align="center">
  <b><a href="ARCHITECTURE.md">English</a></b> |
  <b><a href="ARCHITECTURE_RU.md">Русский</a></b> |
  <b><a href="ARCHITECTURE_JA.md">日本語</a></b>
</p>

---

В данном документе подробно рассматриваются математические основы преобразования Фурье, реализация вычислений в CUDA/cuFFT и архитектура C++ плагина TensorRT в проекте **FFT_to_TRT-Converter**.

---

## 1. Математические основы нормализации ДПФ (FFT)

Дискретное преобразование Фурье (ДПФ / DFT) для двумерного дискретного пространственного сигнала $x[m, n]$ размерностью $H \times W$ (где $N = H \times W$) определяется формулой:

$$\hat{X}[u, v] = \sum_{m=0}^{H-1} \sum_{n=0}^{W-1} x[m, n] \, e^{-j 2\pi \left(\frac{u m}{H} + \frac{v n}{W}\right)}$$

Обратное дискретное преобразование Фурье (ОДПФ / IDFT) восстанавливает исходный пространственный сигнал:

$$x[m, n] = \frac{1}{N} \sum_{u=0}^{H-1} \sum_{v=0}^{W-1} \hat{X}[u, v] \, e^{j 2\pi \left(\frac{u m}{H} + \frac{v n}{W}\right)}$$

### Режимы нормализации

В различных фреймворках машинного обучения и научных публикациях масштабирующий коэффициент распределяется по-разному:

| Режим | Параметр в PyTorch | Масштабирование прямого ДПФ | Масштабирование обратного ДПФ | Итоговый цикл (Forward $\to$ Inverse) |
| :--- | :--- | :--- | :--- | :--- |
| **Backward** | `norm="backward"` (по умолчанию) | $1.0$ | $\frac{1}{N} = \frac{1}{H \times W}$ | $1.0$ |
| **Ortho** | `norm="ortho"` | $\frac{1}{\sqrt{N}} = \frac{1}{\sqrt{H \times W}}$ | $\frac{1}{\sqrt{N}} = \frac{1}{\sqrt{H \times W}}$ | $1.0$ |
| **Forward** | `norm="forward"` | $\frac{1}{N}$ | $1.0$ | $1.0$ |

### Почему режим Ortho критически важен для нейросетей

Модели класса **GFISRV2** производят свертки, нелинейные активации и послойную нормализацию непосредственно в частотной области:

$$\text{Features}_{\text{freq}} = \text{Conv}_{\text{freq}}(\text{RMSNorm}(\text{RFFT2}(x, \text{norm}="ortho")))$$

Если среда инференса выполняет ДПФ в режиме `norm="backward"` ($normalized=0$), в то время как веса нейросети обучались в режиме `norm="ortho"` ($normalized=1$):
- Амплитуда прямого частотного спектра возрастает в $\sqrt{N} = \sqrt{H \times W}$ раз.
- Для фрагмента изображения $64 \times 64$ коэффициент искажения составляет $\sqrt{4096} = 64$ раза.
- Входные тензоры слоев Conv2d и GELU перенасыщаются значениями, в 64 раза превышающими ожидаемые, что полностью ломает карты признаков и генерирует визуальный шум.

---

## 2. Реализация cuFFT и cuBLAS масштабирования

Библиотека NVIDIA `cuFFT` по умолчанию **не нормализует** вычисления:
- `cufftExecR2C` выполняет немасштабированное прямое преобразование (множитель $1.0$).
- `cufftExecC2R` выполняет немасштабированное обратное преобразование (множитель $1.0$).

В обновлённом коде плагина (`dft_plugins.cpp`) реализована нативная поддержка `normalized == 1` (`norm="ortho"`) путём цепочечного вызова быстрой операции масштабирования cuBLAS непосредственно после выполнения преобразования:

```cpp
// В методе RfftPlugin::enqueue:
if (normalized_ == 1) {
    float scale = 1.0f / std::sqrt(static_cast<float>(total_dft_size));
    // Комплексный выход содержит 2 * total_complex_elements вещественных float
    int32_t total_floats = 2 * (batch_size * signal_channels * out_h * out_w);
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
}

// В методе IrfftPlugin::enqueue:
if (normalized_ == 1) {
    float scale = 1.0f / std::sqrt(static_cast<float>(total_dft_size));
    int32_t total_floats = batch_size * signal_channels * out_h * out_w;
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
} else if (normalized_ == 0) {
    // В режиме Backward обратное преобразование делится на N
    float scale = 1.0f / static_cast<float>(total_dft_size);
    int32_t total_floats = batch_size * signal_channels * out_h * out_w;
    cublasScalEx(cublas_handle, total_floats, &scale, CUDA_R_32F,
                 d_output, CUDA_R_32F, 1, CUDA_R_32F);
}
```

Благодаря этому достигается точное совпадение с поведением `torch.fft.rfft2(..., norm="ortho")` в PyTorch вплоть до машинного эпсилон чисел с плавающей запятой.

---

## 3. Архитектура плагинов TensorRT и экспорт символов

В TensorRT 10.x используется расширяемая система плагинов на базе интерфейсов `IPluginCreator` и `IPluginV2DynamicExt`.

### Экспорт функции `getCreators`

В ОС Windows динамические библиотеки (DLL) по умолчанию скрывают все внутренние функции, если они не помечены директивой `__declspec(dllexport)` или не объявлены в DEF-файле.
Функция десериализации TensorRT (`deserializeCudaEngine`) и метод `init_libnvinfer_plugins()` обращаются к стандартной точке входа:

```cpp
extern "C" TRT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);
```

Если данный символ не экспортирован:
1. Вызов `init_libnvinfer_plugins()` не регистрирует плагин.
2. Десериализация движка падает с фатальной ошибкой:
   ```
   IRuntime::deserializeCudaEngine: Error Code 3: API Usage Error
   (SymbolAddress for getCreators could not be loaded)
   ```
3. Пользователи были вынуждены прибегать к нестабильным хакам с принудительной подгрузкой DLL через Python `ctypes`.

Скомпилированная библиотека `trt_dft_plugins.dll` напрямую экспортирует функцию `getCreators`, благодаря чему плагин автоматически обнаруживается как официальными биндингами TensorRT, так и средой chaiNNer.
