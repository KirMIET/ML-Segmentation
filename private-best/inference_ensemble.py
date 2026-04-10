import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import remove_small_objects, remove_small_holes

import segmentation_models_pytorch as smp


TEST_IMAGES_DIR = Path(r"dataset/best_dataset/test_images")
OUTPUT_CSV = "submissions/submission_best_of_the_world.csv"

CHECKPOINT_PATHS = [
    r"./changes/change15/SegFormer_fold_1/best_SegFormer_fold_1.pth",
    r"./changes/change15/SegFormer_fold_2/best_SegFormer_fold_2.pth",
]

ENSEMBLE_WEIGHTS_FILE = r"submissions/ensemble_weights_best.json" #r"submissions/ensemble_weights_segformer.json"  

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THRESHOLD = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 3  # 3-channel RGB input only

USE_TTA = False
USE_MULTI_SCALE = False
USE_MORPHOLOGY = False

# Multi-scale parameters
SCALE_FACTORS = [0.75, 1.0, 1.25]

# Morphology parameters
MORPHOLOGY_MIN_SIZE = 100  # минимальный размер объекта (пиксели)
MORPHOLOGY_MAX_HOLE_SIZE = 500  # максимальный размер дырки (пиксели)


# =========================
# HELPERS (Morphology)
# =========================
def apply_morphology_skimage(mask: np.ndarray) -> np.ndarray:
    """Очищает маску от мелкого мусора и заделывает дырки с помощью skimage."""
    cleaned = remove_small_objects(mask.astype(bool), max_size=MORPHOLOGY_MIN_SIZE)
    cleaned = remove_small_holes(cleaned, max_size=MORPHOLOGY_MAX_HOLE_SIZE)
    return cleaned.astype(np.uint8)


def cv2_imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def build_model(model_name: str, encoder_name: str, in_channels: int):
    if model_name == "Unet":
        return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    elif model_name == "UnetPlusPlus":
        return smp.UnetPlusPlus(
                encoder_name=encoder_name, 
                encoder_weights=None, 
                in_channels=in_channels, 
                classes=1,
                decoder_attention_type='scse'  
            )
    elif model_name == "FPN":
        return smp.FPN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    elif model_name == "SegFormer":
        return smp.Segformer(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    elif model_name == "UPerNet":
        return smp.UPerNet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")


def serialize_mask(mask2d: np.ndarray) -> str:
    return json.dumps(mask2d.astype(np.uint8).tolist(), separators=(",", ":"))


# =========================
# LOAD ALL CHECKPOINTS
# =========================
if not CHECKPOINT_PATHS:
    raise ValueError("CHECKPOINT_PATHS is empty! Please add checkpoint paths to the config section.")

ensemble_models = []

for ckpt_path in CHECKPOINT_PATHS:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    model_name = cfg["MODEL_NAME"]
    encoder_name = cfg["ENCODER_NAME"]
    img_size = int(cfg["IMG_SIZE"])

    print(f"Loading checkpoint: {ckpt_path.name}")
    print(f"  MODEL_NAME     = {model_name}")
    print(f"  ENCODER_NAME   = {encoder_name}")
    print(f"  IMG_SIZE       = {img_size}")
    print(f"  INPUT_CHANNELS = {IN_CHANNELS}")

    model = build_model(model_name, encoder_name, IN_CHANNELS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    ensemble_models.append({
        "model": model,
        "model_name": model_name,
        "encoder_name": encoder_name,
        "img_size": img_size,
    })

print(f"\nLoaded {len(ensemble_models)} models for ensemble inference")

# Загрузка весов моделей (если указаны)
ensemble_weights_list = None
if ENSEMBLE_WEIGHTS_FILE is not None:
    weights_path = Path(ENSEMBLE_WEIGHTS_FILE)
    if not weights_path.exists():
        raise FileNotFoundError(f"Ensemble weights file not found: {weights_path}")
    
    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)
    
    # Преобразуем словарь в список в порядке CHECKPOINT_PATHS
    ensemble_weights_list = []
    for ckpt_path in CHECKPOINT_PATHS:
        ckpt_key = str(ckpt_path)
        if ckpt_key in weights_dict:
            ensemble_weights_list.append(weights_dict[ckpt_key])
        else:
            # Пытаемся найти по имени файла
            ckpt_name = Path(ckpt_path).name
            found = False
            for key, val in weights_dict.items():
                if Path(key).name == ckpt_name:
                    ensemble_weights_list.append(val)
                    found = True
                    break
            if not found:
                print(f"WARNING: No weight found for {ckpt_name}, using uniform")
                ensemble_weights_list.append(1.0 / len(CHECKPOINT_PATHS))
    
    print(f"\nLoaded ensemble weights: {ensemble_weights_list}")
    print(f"Sum of weights: {sum(ensemble_weights_list):.4f}")

# Используем Albumentations для нормализации
val_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# =========================
# INFERENCE LOGIC
# =========================
def prepare_input(image_rgb: np.ndarray, target_size: int) -> torch.Tensor:
    """Подготавливает входной тензор для модели (3-channel RGB)."""
    img_resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    transformed = val_transforms(image=img_resized)
    img_norm = transformed["image"]
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
    inp = img_tensor.unsqueeze(0).to(DEVICE)

    return inp


def predict_single_image_with_model(
    model_info: dict,
    image_rgb: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Делает предсказание одной модели на изображении."""
    model = model_info["model"]

    inp = prepare_input(image_rgb, target_size)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(inp)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    return probs


def predict_multi_scale_with_model(
    model_info: dict,
    image_rgb: np.ndarray,
    scales: list[float],
) -> np.ndarray:
    """Multi-scale testing для одной модели."""
    h, w = image_rgb.shape[:2]
    img_size = model_info["img_size"]
    scale_probs = []

    for scale in scales:
        target_size = (int(img_size * scale) // 32) * 32
        prob_scaled = predict_single_image_with_model(model_info, image_rgb, target_size)
        prob_orig_size = cv2.resize(prob_scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        scale_probs.append(prob_orig_size)

    return np.mean(scale_probs, axis=0)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Поворот изображения на угол, кратный 90 градусам."""
    if angle == 0:
        return image
    elif angle == 90:
        return np.rot90(image, k=1, axes=(0, 1))
    elif angle == 180:
        return np.rot90(image, k=2, axes=(0, 1))
    elif angle == 270:
        return np.rot90(image, k=3, axes=(0, 1))
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")


def rotate_mask(mask: np.ndarray, angle: int) -> np.ndarray:
    """Обратный поворот маски к исходной ориентации."""
    if angle == 0:
        return mask
    elif angle == 90:
        return np.rot90(mask, k=-1, axes=(0, 1))
    elif angle == 180:
        return np.rot90(mask, k=-2, axes=(0, 1))
    elif angle == 270:
        return np.rot90(mask, k=-3, axes=(0, 1))
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")


def predict_ensemble_single_image(image_rgb: np.ndarray) -> np.ndarray:
    """Делает предсказание ансамбля на одном изображении с поддержкой TTA и Multi-scale."""
    h, w = image_rgb.shape[:2]
    all_model_probs = []

    for model_info in ensemble_models:
        if USE_MULTI_SCALE:
            pred_probs = predict_multi_scale_with_model(model_info, image_rgb, SCALE_FACTORS)

            if USE_TTA:
                tta_probs = [pred_probs]
                for angle in [90, 180, 270]:
                    img_rot = rotate_image(image_rgb, angle)
                    prob_rot = predict_multi_scale_with_model(model_info, img_rot, SCALE_FACTORS)
                    prob_rot_restored = rotate_mask(prob_rot, angle)
                    tta_probs.append(prob_rot_restored)

                pred_probs = np.mean(tta_probs, axis=0)

        elif USE_TTA:
            def predict_and_restore(img, m_info):
                p = predict_single_image_with_model(m_info, img, target_size=m_info["img_size"])
                return cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

            tta_probs = [predict_and_restore(image_rgb, model_info)]

            img_hflip = cv2.flip(image_rgb, 1)
            prob_hflip = predict_and_restore(img_hflip, model_info)
            tta_probs.append(cv2.flip(prob_hflip, 1))

            img_vflip = cv2.flip(image_rgb, 0)
            prob_vflip = predict_and_restore(img_vflip, model_info)
            tta_probs.append(cv2.flip(prob_vflip, 0))

            for angle in [90, 180, 270]:
                img_rot = rotate_image(image_rgb, angle)
                prob_rot = predict_and_restore(img_rot, model_info)
                tta_probs.append(rotate_mask(prob_rot, angle))

            pred_probs = np.mean(tta_probs, axis=0)

        else:
            pred_probs = predict_single_image_with_model(model_info, image_rgb, target_size=model_info["img_size"])
            pred_probs = cv2.resize(pred_probs, (w, h), interpolation=cv2.INTER_LINEAR)

        all_model_probs.append(pred_probs)

    # Усреднение предсказаний всех моделей (взвешенное или равномерное)
    if ensemble_weights_list is not None:
        ensemble_pred = np.average(all_model_probs, weights=ensemble_weights_list, axis=0)
    else:
        ensemble_pred = np.mean(all_model_probs, axis=0)
    return ensemble_pred


# =========================
# MAIN INFERENCE LOOP
# =========================
image_paths = sorted([p for p in TEST_IMAGES_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS])
if not image_paths:
    raise FileNotFoundError(f"No images found in: {TEST_IMAGES_DIR}")

print(f"Found {len(image_paths)} test images")
rows = []

with torch.no_grad():
    for i, img_path in enumerate(image_paths, 1):
        img_bgr = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. ENSEMBLE PREDICTION (с поддержкой TTA и Multi-scale)
        pred_probs = predict_ensemble_single_image(img_rgb)

        # 2. Ресайз (на всякий случай)
        if pred_probs.shape != (H, W):
            pred_probs = cv2.resize(pred_probs, (W, H), interpolation=cv2.INTER_LINEAR)

        # Бинаризация
        mask = (pred_probs > THRESHOLD).astype(np.uint8)

        # 3. MORPHOLOGY CLEANUP (Опционально)
        if USE_MORPHOLOGY:
            mask = apply_morphology_skimage(mask)

        # Сохранение результата
        rows.append({
            "ImageId": img_path.name,
            "mask": serialize_mask(mask),
        })

        if i % 100 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)}")

submission_df = pd.DataFrame(rows)
submission_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("Done.")
print(f"Saved submission to: {OUTPUT_CSV}")
