"""
Ensemble Weights Calculator
=============================
Вычисляет оптимальные веса для моделей ансамбля на основе val_dice из файлов истории.
Для каждого чекпоинта находит соответствующий CSV файл истории,
берёт лучший val_dice и нормализует веса.
"""

import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

CHECKPOINT_PATHS = [
    r"./model_checkpoints/SegFormer_fold_1/ema_best_SegFormer_fold_1.pth",
    r"./model_checkpoints/SegFormer_fold_2/ema_best_SegFormer_fold_2.pth",
    r"./model_checkpoints/SegFormer_fold_3/ema_best_SegFormer_fold_3.pth",
    r"./model_checkpoints/SegFormer_fold_4/ema_best_SegFormer_fold_4.pth",

    r"./model_checkpoints/Unet_fold_2/ema_best_Unet_fold_2.pth",
    r"./model_checkpoints/Unet_fold_3/ema_best_Unet_fold_3.pth",
    r"./model_checkpoints/Unet_fold_4/ema_best_Unet_fold_4.pth",
]

# Метод нормализации весов: "linear" или "softmax"
# linear: weight_i = val_dice_i / sum(val_dice_all)
# softmax: weight_i = exp(val_dice_i * temperature) / sum(exp(...))
WEIGHT_METHOD = "softmax"
SOFTMAX_TEMPERATURE = 20.0  # чем выше, тем сильнее разница между весами

# Путь для сохранения весов
OUTPUT_WEIGHTS_PATH = "submissions/ensemble_weights_king.json"


def load_ensemble_weights(weights_path: str) -> list[float]:
    """Загружает веса ансамбля из JSON файла.
    
    Returns:
        Список весов в порядке, соответствующем CHECKPOINT_PATHS
    """
    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)
    
    weights = []
    for ckpt_name, weight in weights_dict.items():
        weights.append(weight)
    
    return weights


def find_history_file(checkpoint_path: Path) -> Path | None:
    """Ищет CSV файл истории в той же директории, что и чекпоинт."""
    ckpt_dir = checkpoint_path.parent
    
    # Пытаемся найти по префиксу имени чекпоинта
    # best_UnetPlusPlus_fold_1.pth -> history_UnetPlusPlus_fold_1.csv
    ckpt_stem = checkpoint_path.stem  # best_UnetPlusPlus_fold_1
    parts = ckpt_stem.split("_", 1)  # ['best', 'UnetPlusPlus_fold_1']
    
    if len(parts) == 2:
        model_fold_part = parts[1]  # UnetPlusPlus_fold_1
        history_pattern = f"history_{model_fold_part}.csv"
        history_path = ckpt_dir / history_pattern
        if history_path.exists():
            return history_path
    
    # Если не нашли по имени, ищем все history_*.csv в директории
    history_files = list(ckpt_dir.glob("history_*.csv"))
    if len(history_files) == 1:
        return history_files[0]
    elif len(history_files) > 1:
        # Пытаемся найти по совпадению модели и фолда
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = checkpoint["config"]
        model_name = cfg["MODEL_NAME"]
        fold = cfg.get("fold", None)
        
        for hf in history_files:
            if model_name in hf.stem:
                if fold is None or f"fold_{fold}" in hf.stem:
                    return hf
        # Если не нашли точного совпадения, возвращаем первый
        return history_files[0]
    
    return None


def compute_ensemble_weights(
    checkpoint_paths: list[str],
    method: str = "linear",
    softmax_temperature: float = 10.0,
) -> dict[str, float]:
    """Вычисляет веса для ансамбля на основе val_dice из истории обучения.
    
    Args:
        checkpoint_paths: Список путей до чекпоинтов.
        method: Метод нормализации ("linear" или "softmax").
        softmax_temperature: Температура для softmax.
        
    Returns:
        Словарь {checkpoint_path: weight}.
    """
    model_scores = {}  # {checkpoint_path: best_val_dice}
    
    for ckpt_path_str in checkpoint_paths:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue
        
        # Загружаем чекпоинт для информации
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        cfg = checkpoint["config"]
        model_name = cfg["MODEL_NAME"]
        fold = cfg.get("fold", "?")
        
        # Ищем файл истории
        history_path = find_history_file(ckpt_path)
        
        if history_path is None:
            print(f"WARNING: No history file found for {ckpt_path.name}")
            # Используем val_dice из чекпоинта если есть
            val_dice = checkpoint.get("val_dice_ema", None)
            if val_dice is not None:
                print(f"  Using val_dice from checkpoint: {val_dice:.4f}")
                model_scores[ckpt_path_str] = val_dice
            else:
                print(f"  No val_dice in checkpoint either, skipping")
            continue
        
        # Загружаем историю и находим лучший val_dice
        history_df = pd.read_csv(history_path)
        
        if "val_dice" not in history_df.columns:
            print(f"WARNING: No 'val_dice' column in {history_path}")
            continue
        
        if "ema" in ckpt_path_str:
            best_val_dice = history_df["val_dice_ema"].max()
            best_epoch = history_df["val_dice_ema"].idxmax() + 1  # +1 т.к. индексация с 0
        else:
            best_val_dice = history_df["val_dice"].max()
            best_epoch = history_df["val_dice"].idxmax() + 1  # +1 т.к. индексация с 0
        
        model_scores[ckpt_path_str] = best_val_dice
        
        print(f"Model: {model_name} (fold {fold})")
        print(f"  History: {history_path.name}")
        print(f"  Best val_dice: {best_val_dice:.4f} (epoch {best_epoch})")
    
    if not model_scores:
        raise ValueError("No valid model scores found. Check checkpoint paths and history files.")
    
    # Нормализация весов
    scores = np.array(list(model_scores.values()))
    
    if method == "linear":
        weights = scores / scores.sum()
    elif method == "softmax":
        exp_scores = np.exp(scores * softmax_temperature)
        weights = exp_scores / exp_scores.sum()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'softmax'.")
    
    # Создаём итоговый словарь
    ensemble_weights = {path: float(w) for path, w in zip(model_scores.keys(), weights)}
    
    # Выводим результаты
    print(f"\n{'='*60}")
    print(f"Ensemble Weights ({method} normalization)")
    print(f"{'='*60}")
    for path, weight in ensemble_weights.items():
        ckpt_name = Path(path).name
        score = model_scores[path]
        print(f"  {ckpt_name:40s} val_dice={score:.4f}  weight={weight:.4f}")
    print(f"{'='*60}")
    
    return ensemble_weights


def main():
    """Главная функция."""
    if not CHECKPOINT_PATHS:
        raise ValueError("CHECKPOINT_PATHS is empty! Please add checkpoint paths.")
    
    print(f"Computing ensemble weights for {len(CHECKPOINT_PATHS)} checkpoints")
    print(f"Method: {WEIGHT_METHOD}")
    if WEIGHT_METHOD == "softmax":
        print(f"Softmax temperature: {SOFTMAX_TEMPERATURE}")
    print()
    
    ensemble_weights = compute_ensemble_weights(
        CHECKPOINT_PATHS,
        method=WEIGHT_METHOD,
        softmax_temperature=SOFTMAX_TEMPERATURE,
    )
    
    # Сохраняем в JSON
    output_path = Path(OUTPUT_WEIGHTS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ensemble_weights, f, indent=2)
    
    print(f"\nWeights saved to: {output_path}")
    print("To use in inference_ensemble.py, set:")
    print(f'  ENSEMBLE_WEIGHTS_FILE = r"{output_path}"')


if __name__ == "__main__":
    main()
