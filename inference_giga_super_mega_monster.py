"""
Inference Giga Super Mega Monster
===================================
Универсальный скрипт инференса ансамбля:
  - 3-канальный (RGB) и 5-канальный (RGB+XY) вход автоматически из конфига чекпоинта
  - Взвешивание моделей из ensemble_weights_best.json
  - TTA: flips (H, V) + rotations (90, 180, 270)
  - Multi-scale: [256, 288, 352]
  - Морфологическая постобработка (remove_small_objects + remove_small_holes)
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from skimage.morphology import remove_small_objects, remove_small_holes

import segmentation_models_pytorch as smp

# =========================
# CONFIG
# =========================
TEST_IMAGES_DIR = Path(r"dataset/best_dataset/test_images")
OUTPUT_CSV = "submissions/submission_king.csv"

CHECKPOINT_PATHS = [
    r"./changes/change13/UnetPlusPlus_fold_2/best_UnetPlusPlus_fold_2.pth",
    r"./changes/change13/UnetPlusPlus_fold_3/best_UnetPlusPlus_fold_3.pth",
    r"./changes/change13/UPerNet_fold_2/best_UPerNet_fold_2.pth",

    r"./changes/change14/SegFormer_fold_1/best_SegFormer_fold_1.pth",
    r"./changes/change14/UPerNet_fold_2/best_UPerNet_fold_2.pth",

    r"./model_checkpoints/SegFormer_fold_1/ema_best_SegFormer_fold_1.pth",
    r"./model_checkpoints/SegFormer_fold_2/ema_best_SegFormer_fold_2.pth",
    r"./model_checkpoints/SegFormer_fold_3/ema_best_SegFormer_fold_3.pth",
    r"./model_checkpoints/SegFormer_fold_4/ema_best_SegFormer_fold_4.pth",

    r"./model_checkpoints/Unet_fold_2/ema_best_Unet_fold_2.pth",
    r"./model_checkpoints/Unet_fold_3/ema_best_Unet_fold_3.pth",
    r"./model_checkpoints/Unet_fold_4/ema_best_Unet_fold_4.pth",
]
ENSEMBLE_WEIGHTS_FILE = r"submissions/ensemble_weights_king.json"
THRESHOLD = 0.46
MULTI_SCALES = [256, 288, 352]
MORPHOLOGY_MIN_SIZE = 100
MORPHOLOGY_MAX_HOLE_SIZE = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# =========================
# COORDINATE MAPS
# =========================
def create_coordinate_maps(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    x_map, y_map = np.meshgrid(x, y)
    return x_map, y_map


# =========================
# MODEL LOADING
# =========================
def build_model(model_name: str, encoder_name: str, in_channels: int) -> torch.nn.Module:
    model_cls = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "FPN": smp.FPN,
        "SegFormer": smp.Segformer,
        "UPerNet": smp.UPerNet,
    }
    cls = model_cls.get(model_name)
    if cls is None:
        raise ValueError(f"Unsupported model: {model_name}")
    return cls(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)


def load_checkpoints(paths: list[str]) -> list[dict]:
    models = []
    for ckpt_path in paths:
        ckpt_path = Path(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model_name = cfg["MODEL_NAME"]
        encoder_name = cfg["ENCODER_NAME"]
        use_coords = cfg.get("use_coordinates", False)
        in_channels = 5 if use_coords else 3

        model = build_model(model_name, encoder_name, in_channels)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        models.append({
            "model": model,
            "model_name": model_name,
            "in_channels": in_channels,
            "use_coordinates": use_coords,
        })
        print(f"  Loaded: {ckpt_path.name} | {model_name} | ch={in_channels} (coords={use_coords})")
    return models


def load_weights(ckpt_paths: list[str], weights_file: str) -> list[float]:
    with open(weights_file, "r") as f:
        weights_dict = json.load(f)

    weights = []
    for p in ckpt_paths:
        key = str(p)
        if key in weights_dict:
            weights.append(weights_dict[key])
        else:
            name = Path(p).name
            found = next((v for k, v in weights_dict.items() if Path(k).name == name), None)
            weights.append(found if found is not None else 1.0 / len(ckpt_paths))

    print(f"\nEnsemble weights: {[f'{w:.4f}' for w in weights]} (sum={sum(weights):.4f})")
    return weights


# =========================
# INFERENCE HELPERS
# =========================
def prepare_input(image_rgb: np.ndarray, target_size: int, in_channels: int) -> torch.Tensor:
    """
    image_rgb: всегда только RGB (H, W, 3)
    in_channels: 3 или 5
    """
    # Ресайзим только RGB
    img_resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Нормализуем RGB
    transformed = val_transform(image=img_resized)
    norm_rgb = transformed["image"]  # (target_size, target_size, 3)

    if in_channels == 5:
        # Генерируем правильные XY-координаты для целевого размера (target_size)
        x_map, y_map = create_coordinate_maps(target_size, target_size)
        coords = np.stack([x_map, y_map], axis=-1)  # (target_size, target_size, 2)
        
        # Склеиваем нормализованный RGB и XY-координаты
        combined = np.concatenate([norm_rgb, coords], axis=-1)  # (H, W, 5)
        tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    else:
        tensor = torch.from_numpy(norm_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    return tensor


def predict_single_scale(model_info: dict, image: np.ndarray, target_size: int) -> np.ndarray:
    """Один форвард на заданном размере."""
    inp = prepare_input(image, target_size, model_info["in_channels"])
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        logits = model_info["model"](inp)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    return prob


# =========================
# TTA
# =========================
def rotate_img(img: np.ndarray, k: int) -> np.ndarray:
    """Поворот изображения на k*90 градусов."""
    return np.rot90(img, k, axes=(0, 1))


def rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """Обратный поворот маски."""
    return np.rot90(mask, -k, axes=(0, 1))


def predict_multi_scale(model_info: dict, image: np.ndarray, scales: list[int]) -> np.ndarray:
    """Multi-scale inference: предсказание на каждом масштабе, ресайз к размеру входа и усреднение."""
    h, w = image.shape[:2]
    probs = []
    for scale_size in scales:
        p = predict_single_scale(model_info, image, scale_size)
        # Обязательно возвращаем маску к размеру текущего TTA-изображения (h, w)
        p_resized = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        probs.append(p_resized)
    return np.mean(probs, axis=0)


def predict_with_tta_multi_scale(model_info: dict, image_rgb: np.ndarray, scales: list[int]) -> np.ndarray:
    """TTA + multi-scale: 6 аугментаций × multi-scale → усреднение."""
    all_probs = []

    # --- 1. Оригинал ---
    p = predict_multi_scale(model_info, image_rgb, scales)
    all_probs.append(p)

    # --- 2. Horizontal flip ---
    flipped_h = cv2.flip(image_rgb, 1)
    p = predict_multi_scale(model_info, flipped_h, scales)
    all_probs.append(cv2.flip(p, 1))

    # --- 3. Vertical flip ---
    flipped_v = cv2.flip(image_rgb, 0)
    p = predict_multi_scale(model_info, flipped_v, scales)
    all_probs.append(cv2.flip(p, 0))

    # --- 4, 5, 6. Rotations 90, 180, 270 ---
    for k in [1, 2, 3]:
        rotated = rotate_img(image_rgb, k)
        p = predict_multi_scale(model_info, rotated, scales)
        # Возвращаем маску в исходный поворот
        all_probs.append(rotate_mask(p, k))

    # Теперь все 6 масок имеют строго одинаковый оригинальный размер исходной картинки
    return np.mean(all_probs, axis=0)


# =========================
# MORPHOLOGY
# =========================
def apply_morphology(mask: np.ndarray, min_size: int, max_hole: int) -> np.ndarray:
    """Очищает маску от мелкого мусора и заделывает дырки с помощью skimage."""
    cleaned = remove_small_objects(mask.astype(bool), max_size=MORPHOLOGY_MIN_SIZE)
    cleaned = remove_small_holes(cleaned, max_size=MORPHOLOGY_MAX_HOLE_SIZE)
    return cleaned.astype(np.uint8)


# =========================
# SERIALIZATION
# =========================
def serialize_mask(mask: np.ndarray) -> str:
    return json.dumps(mask.astype(np.uint8).tolist(), separators=(",", ":"))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Multi-scales: {MULTI_SCALES}")
    print(f"Morphology: min_size={MORPHOLOGY_MIN_SIZE}, max_hole={MORPHOLOGY_MAX_HOLE_SIZE}")

    print("\nLoading checkpoints...")
    models = load_checkpoints(CHECKPOINT_PATHS)
    weights = load_weights(CHECKPOINT_PATHS, ENSEMBLE_WEIGHTS_FILE)

    image_paths = sorted(p for p in TEST_IMAGES_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {TEST_IMAGES_DIR}")

    print(f"\nFound {len(image_paths)} images\n")

    rows = []
    for i, img_path in enumerate(image_paths, 1):
        img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Ensemble prediction with TTA + multi-scale
        model_probs = []
        for m in models:
            prob = predict_with_tta_multi_scale(m, img_rgb, MULTI_SCALES)
            model_probs.append(prob)

        ensemble_pred = np.average(model_probs, weights=weights, axis=0)

        # Thresholding
        mask = (ensemble_pred > THRESHOLD).astype(np.uint8)

        # Morphology cleanup
        mask = apply_morphology(mask, MORPHOLOGY_MIN_SIZE, MORPHOLOGY_MAX_HOLE_SIZE)

        rows.append({"ImageId": img_path.name, "mask": serialize_mask(mask)})

        if i % 100 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nDone. Saved to {OUTPUT_CSV}")
