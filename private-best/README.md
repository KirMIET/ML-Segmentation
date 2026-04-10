# Ensemble Segmentation Training Pipeline

Пайплайн для обучения ансамбля моделей семантической сегментации с поддержкой нескольких архитектур, фолдов и расширенных аугментаций.

## 📋 Содержание

- [Обзор](#обзор)
- [Структура проекта](#структура-проекта)
- [Архитектура пайплайна](#архитектура-пайплайна)
  - [1. Создание фолдов (fold_create.py)](#1-создание-фолдов-fold_createpy)
  - [2. Обучение ансамбля (train_ensemble.py)](#2-обучение-ансамбля-train_ensemblepy)
  - [3. Вычисление весов (ensemble_weights.py)](#3-вычисление-весов-ensemble_weightspy)
  - [4. Инференс (inference_ensemble.py)](#4-инференс-inference_ensemblepy)
- [Поддерживаемые модели](#поддерживаемые-модели)
- [Установка зависимостей](#установка-зависимостей)
- [Подготовка данных](#подготовка-данных)
- [Обучение](#обучение)
  - [Конфигурация](#конфигурация)
  - [Запуск обучения](#запуск-обучения)
- [Инференс (предсказание)](#инференс-предсказание)
  - [Конфигурация инференса](#конфигурация-инференса)
  - [Запуск инференса](#запуск-инференса)
- [Ключевые особенности](#ключевые-особенности)
  - [SAM оптимизатор](#sam-оптимизатор)
  - [Динамическая функция потерь](#динамическая-функция-потерь)
  - [Аугментации](#аугментации)
  - [Multi-scale обучение](#multi-scale-обучение)
  - [Test-Time Augmentation (TTA)](#test-time-augmentation-tta)
- [Структура конфигурации моделей](#структура-конфигурации-моделей)
- [Выходные данные](#выходные-данные)

---

## Обзор

Данный пайплайн реализует обучение ансамбля моделей для бинарной сегментации с использованием библиотеки `segmentation-models-pytorch`. Основные возможности:

- **Обучение нескольких архитектур** одновременно (UNet++, FPN, SegFormer, UPerNet)
- **K-fold кросс-валидация** с умным разделением по камерам и последовательностям
- **Комбинированная динамическая функция потерь** (BCE + Focal Tversky + Boundary)
- **Расширенные аугментации данных** (CutMix, CopyPaste, геометрические и оптические трансформации)
- **SAM (Sharpness-Aware Minimization)** оптимизатор
- **Multi-scale обучение**
- **Mixed precision training** (AMP)
- **Early stopping** с сохранением лучших чекпоинтов
- **Визуализация результатов** обучения
- **Ensemble inference** с настраиваемыми весами моделей

---

## Структура проекта

```
private-best/
├── train_ensemble.py          # Главный скрипт обучения
├── inference_ensemble.py      # Скрипт инференса (предсказания)
└── src/
    ├── config.py              # Централизованная конфигурация
    ├── models.py              # Фабрика моделей
    ├── losses.py              # Функции потерь (Combined, FocalTversky, Boundary, OHEM)
    ├── dataset.py             # Dataset для бинарной сегментации
    ├── augmentations.py       # Albumentations pipeline + coordinate maps
    ├── mixup_augmentations.py # CutMix и CopyPaste аугментации
    ├── training_utils.py      # SAM optimizer, train/val loops, визуализация
    ├── fold_create.py         # Утилита для создания фолдов
    └── ensemble_weights.py    # Вычисление оптимальных весов ансамбля
```

---

## Архитектура пайплайна

Пайплайн состоит из 4 основных этапов:

### 1. Создание фолдов (`fold_create.py`)

Скрипт для разделения датасета на фолды с учётом структуры данных:

- **Группировка по** `(camera_ip + sequence_code + product_id)`
- **Стратифицированное разделение** — внутри каждой камеры группы делятся на фолды
- **Сохранение распределения** камер в train/val выборках

**Использование:**
```bash
python src/fold_create.py
```

**Параметры (внутри скрипта):**
- `TRAIN_IMAGES_DIR` — путь к тренировочным изображениям
- `TRAIN_MASKS_DIR` — путь к маскам
- `OUTPUT_DIR` — директория для сохранения фолдов
- `N_SPLITS` — количество фолдов (по умолчанию 4)
- `SEED` — сид для воспроизводимости

**Структура выходных данных:**
```
dataset/dataset_fold/
├── fold_1/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── fold_2/
│   └── ...
└── fold_N/
    └── ...
```

### 2. Обучение ансамбля (`train_ensemble.py`)

Главный скрипт обучения, который:

1. Проходит по всем фолдам
2. Для каждой модели из `MODEL_CONFIGS`:
   - Создаёт модель с заданной архитектурой
   - Инициализирует оптимизатор (SAM или AdamW)
   - Запускает training loop с динамическим лоссом
   - Сохраняет лучший и последний чекпоинты
   - Визуализирует предсказания каждые N эпох
   - Применяет early stopping

**Использование:**
```bash
python train_ensemble.py
```

**Что происходит:**
- Для каждого фолда и модели создаётся отдельный эксперимент
- Все чекпоинты сохраняются в `model_checkpoints/{model_name}_fold_{fold}/`
- Истории обучения сохраняются в CSV
- Визуализации сохраняются в `view_train_img/{model_name}_fold_{fold}/`

### 3. Вычисление весов (`ensemble_weights.py`)

Утилита для вычисления оптимальных весов моделей в ансамбле:

- Загружает чекпоинты и находит соответствующие CSV файлы истории
- Находит лучший `val_dice` для каждой модели
- Нормализует веса методами `linear` или `softmax`

**Использование:**
```bash
python src/ensemble_weights.py
```

**Параметры:**
- `CHECKPOINT_PATHS` — список путей до чекпоинтов
- `WEIGHT_METHOD` — метод нормализации (`linear` или `softmax`)
- `SOFTMAX_TEMPERATURE` — температура для softmax
- `OUTPUT_WEIGHTS_PATH` — путь для сохранения JSON с весами

### 4. Инференс (`inference_ensemble.py`)

Скрипт для предсказания на тестовых данных:

- Загружает несколько моделей из чекпоинтов
- Делает предсказания с поддержкой:
  - **Ensemble** — усреднение предсказаний нескольких моделей
  - **TTA (Test-Time Augmentation)** — флипы, повороты
  - **Multi-scale** — предсказание на разных масштабах
  - **Morphology** — очистка маски от шума и дырок
- Сохраняет результат в CSV формате

**Использование:**
```bash
python inference_ensemble.py
```

---

## Поддерживаемые модели

| Модель | Описание | Энкодер | Веса энкодера |
|--------|----------|---------|---------------|
| **UnetPlusPlus** | U-Net++ с SCSE attention | timm-efficientnet-b4 | noisy-student |
| **FPN** | Feature Pyramid Network | tu-convnext_small | imagenet |
| **SegFormer** | Transformer-based сегментация | mit_b3 | imagenet |
| **UPerNet** | Unified Perceptual Parsing | resnext50_32x4d | swsl |

Каждая модель может иметь уникальные:
- learning rate, batch size, количество эпох
- optimizer (SAM или AdamW)
- входные каналы (3-channel RGB или 5-channel RGB+XY)
- multi-scale конфигурацию

---

## Установка зависимостей

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install matplotlib pandas numpy
pip install scikit-image
pip install tqdm
```

---

## Подготовка данных

1. **Организуйте данные** в следующем формате:
```
dataset/best_dataset/
├── train/
│   ├── images/  # Изображения (.jpg, .png, и т.д.)
│   └── masks/   # Маски (.png с 0/255 значениями)
└── test_images/ # Тестовые изображения
```

2. **Создайте фолды:**
```bash
python src/fold_create.py
```

3. **Проверьте структуру:**
```
dataset/dataset_fold/
├── fold_1/
│   ├── train/images, masks
│   └── val/images, masks
└── fold_2/...
```

---

## Обучение

### Конфигурация

Все параметры находятся в `src/config.py`:

**Основные:**
- `NUM_FOLDS` — количество фолдов
- `IMG_SIZE` — размер входного изображения (по умолчанию 352)
- `SEED` — сид для воспроизводимости
- `NUM_WORKERS` — количество workers для DataLoader

**Оптимизация:**
- `SAM_RHO` — параметр SAM оптимизатора (0.05)
- `GRAD_CLIP_MAX_NORM` — gradient clipping (1.0)

**Аугментации:**
- `CUTMIX_ALPHA` — параметр Beta распределения для CutMix
- `AUGMENTATION_PROB` — вероятность применения CutMix/CopyPaste (0.5)
- `COPYPASTE_MAX_OBJECTS` — максимальное количество копируемых объектов

**Лосс функция:**
- `BCE_WEIGHT`, `TVERSKY_WEIGHT`, `BOUNDARY_WEIGHT` — начальные веса
- `USE_OHEM`, `OHEM_RATIO` — Online Hard Example Mining
- `FOCAL_TVERSKY_ALPHA`, `FOCAL_TVERSKY_BETA`, `FOCAL_TVERSKY_GAMMA`

### Запуск обучения

```bash
python train_ensemble.py
```

**Процесс обучения:**
1. Для каждого фолда (1..NUM_FOLDS):
   - Загружаются train/val сэмплы
   - Для каждой модели из `MODEL_CONFIGS`:
     - Инициализируется модель, оптимизатор, scheduler
     - Запускается training loop:
       - **Warmup phase**: LinearLR с малым LR
       - **Main phase**: CosineAnnealingLR
       - Dynamic loss weight updates
       - AMP (mixed precision) для ускорения
       - Gradient clipping
     - Early stopping проверяет улучшение метрики
     - Сохраняются best/last чекпоинты

**Выходные данные:**
```
model_checkpoints/
├── UnetPlusPlus_fold_1/
│   ├── best_UnetPlusPlus_fold_1.pth
│   ├── last_UnetPlusPlus_fold_1.pth
│   └── history_UnetPlusPlus_fold_1.csv
├── SegFormer_fold_1/
│   └── ...
└── ...

view_train_img/
├── UnetPlusPlus_fold_1/
│   ├── epoch_005.png
│   ├── epoch_010.png
│   └── ...
└── ...
```

---

## Инференс (предсказание)

### Конфигурация инференса

Параметры в начале `inference_ensemble.py`:

```python
TEST_IMAGES_DIR = Path(r"dataset/best_dataset/test_images")
OUTPUT_CSV = "submissions/submission_best_of_the_world.csv"

CHECKPOINT_PATHS = [
    r"./model_checkpoints/SegFormer_fold_1/best_SegFormer_fold_1.pth",
    r"./model_checkpoints/SegFormer_fold_2/best_SegFormer_fold_2.pth",
]

ENSEMBLE_WEIGHTS_FILE = r"submissions/ensemble_weights_best.json"

THRESHOLD = 0.35

USE_TTA = False          # Test-Time Augmentation
USE_MULTI_SCALE = False  # Multi-scale prediction
USE_MORPHOLOGY = False   # Morphology cleanup

SCALE_FACTORS = [0.75, 1.0, 1.25]  # Для multi-scale
MORPHOLOGY_MIN_SIZE = 100           # Мин. размер объекта
MORPHOLOGY_MAX_HOLE_SIZE = 500      # Макс. размер дырки
```

### Запуск инференса

```bash
python inference_ensemble.py
```

**Процесс:**
1. Загружаются все модели из `CHECKPOINT_PATHS`
2. Загружаются веса ансамбля (если указаны)
3. Для каждого тестового изображения:
   - Делается предсказание каждой моделью
   - Опционально: TTA (fliplr, flipud, rotate 90/180/270)
   - Опционально: multi-scale предсказание
   - Предсказания усредняются с заданными весами
   - Бинаризация по порогу `THRESHOLD`
   - Опционально: морфологическая очистка
4. Результат сохраняется в CSV

**Формат выходного CSV:**
```csv
ImageId,mask
image1.png,"[[0,0,1,1,...],[0,1,1,0,...],...]"
image2.png,"..."
```

---

## Ключевые особенности

### SAM оптимизатор

**Sharpness-Aware Minimization** — оптимизатор, который минимизирует не только loss, но и его чувствительность к возмущениям весов. Это приводит к нахождению более "плоских" минимумов и лучшей обобщающей способности.

**Как работает:**
1. Первый forward/backward pass — вычисление градиентов
2. Perturb — возмущение весов в направлении градиентов
3. Второй forward/backward pass — вычисление градиентов на возмущённых весах
4. Unperturb — откат возмущения
5. Step базового оптимизатора (AdamW)

**Конфигурация:**
- `use_sam: True/False` — в конфиге модели
- `SAM_RHO = 0.05` — радиус возмущения

### Динамическая функция потерь

Комбинированный лосс состоит из трёх компонентов:

1. **BCE (с OHEM)** — Binary Cross Entropy с Online Hard Example Mining
   - Фокусируется на сложных примерах
   - `ohem_ratio = 0.2` — учит только 20% самых сложных пикселей

2. **Focal Tversky** — обобщение Dice loss с фокусировкой
   - `alpha = 0.3` — штраф за False Positives
   - `beta = 0.7` — штраф за False Negatives (более важный)
   - `gamma = 0.75` — параметр фокусировки

3. **Boundary Loss** — штрафует ошибки на границах объектов
   - Использует Sobel-фильтр для выделения границ
   - `sigma = 2.0`

**Динамическое расписание:**
- **Warmup phase**: больше BCE, симметричный Tversky
- **Main phase**: плавный переход к итоговым весам

### Аугментации

**Albumentations pipeline:**
- Геометрические: Horizontal/Vertical Flip, Rotate90, Rotate, Affine
- Оптические: Brightness/Contrast, Hue/Saturation, RGB Shift
- Шум: GaussianBlur, GaussNoise
- Морфологические: Sharpen, CLAHE

**CutMix:**
- Вырезает прямоугольную область из одного изображения и вставляет в другое
- `alpha = 1.0` — параметр Beta распределения для площади

**CopyPaste:**
- Извлекает отдельные объекты из маски (connected components)
- Копирует их в случайное место целевого изображения
- Случайный скейлинг (0.6x — 1.2x)
- `max_objects = 3` — максимальное количество копируемых объектов

### Multi-scale обучение

Во время обучения размер изображения случайно выбирается из заданного списка:

```python
"multi_scales": [256, 352]  # Каждый батч может быть 256x256 или 352x352
```

Это помогает модели быть инвариантной к масштабу объектов.

### Test-Time Augmentation (TTA)

Во время инференса:
- Предсказание на оригинальном изображении
- Предсказание на трансформированных версиях (fliplr, flipud, rotate)
- Обратная трансформация и усреднение

TTA повышает качество ценой увеличения времени предсказания.

---

## Структура конфигурации моделей

Каждая модель в `MODEL_CONFIGS` имеет следующие параметры:

```python
"ModelName": {
    "model_name": "ModelName",          # Отображаемое имя
    "encoder_name": "mit_b3",           # Энкодер из SMP
    "encoder_weights": "imagenet",      # Предобученные веса
    "batch_size": 12,                   # Размер батча
    "lr": 1.5e-4,                       # Learning rate
    "num_epochs": 50,                   # Максимум эпох
    "warmup_epochs": 6,                 # Эпохи warmup
    "early_stopping_patience": 10,      # Patience для early stopping
    "weight_decay": 1e-2,               # Weight decay
    "use_coordinates": False,           # Добавлять ли XY каналы (5-ch вход)
    "use_sam": False,                   # Использовать SAM оптимизатор
    "multi_scales": None,               # Список размеров [256, 352] или None
}
```

---

## Выходные данные

После полного цикла обучения:

**Чекпоинты:**
```
model_checkpoints/
└── {model_name}_fold_{fold}/
    ├── best_{model_name}_fold_{fold}.pth   # Лучшая модель
    ├── last_{model_name}_fold_{fold}.pth   # Последняя модель
    └── history_{model_name}_fold_{fold}.csv # История обучения
```

**Визуализации:**
```
view_train_img/
└── {model_name}_fold_{fold}/
    ├── epoch_005.png
    ├── epoch_010.png
    └── ...
```

**CSV история содержит:**
- `epoch`, `lr`, `train_loss`, `train_dice`, `train_iou`
- `val_loss`, `val_dice`, `val_iou`

---

## Примеры использования

### 1. Базовое обучение одной модели

```python
# В src/config.py
MODEL_CONFIGS = {
    "SegFormer": {
        # ... конфигурация ...
    },
}
NUM_FOLDS = 4
```

```bash
python train_ensemble.py
```

### 2. Вычисление весов для ансамбля

```python
# В src/ensemble_weights.py
CHECKPOINT_PATHS = [
    r"./model_checkpoints/SegFormer_fold_1/best_SegFormer_fold_1.pth",
    r"./model_checkpoints/SegFormer_fold_2/best_SegFormer_fold_2.pth",
    r"./model_checkpoints/UnetPlusPlus_fold_1/best_UnetPlusPlus_fold_1.pth",
]
WEIGHT_METHOD = "softmax"
```

```bash
python src/ensemble_weights.py
```

### 3. Инференс с TTA и Multi-scale

```python
# В inference_ensemble.py
USE_TTA = True
USE_MULTI_SCALE = True
SCALE_FACTORS = [0.75, 1.0, 1.25]
```

```bash
python inference_ensemble.py
```

---

## Примечание

Запускать, не расскомментируя.
