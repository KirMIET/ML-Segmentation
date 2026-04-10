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


# =========================
# CONFIG
# =========================
TEST_IMAGES_DIR = Path(r"dataset/best_dataset/test_images")
OUTPUT_CSV = "submissions/submission_segformer.csv"

# путь к вашему чекпоинту после обучения
CHECKPOINT_PATH = Path(r".\changes\change14\SegFormer_fold_1\best_SegFormer_fold_1.pth")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_TTA = True
USE_MULTI_SCALE = True
USE_MORPHOLOGY = True

# Multi-scale parameters
SCALE_FACTORS = [0.75, 1.0, 1.25]

# Morphology parameters
MORPHOLOGY_MIN_SIZE = 100  # минимальный размер объекта (пиксели)
MORPHOLOGY_MAX_HOLE_SIZE = 500  # максимальный размер дырки (пиксели)


# =========================
# HELPERS (Coordinates, CRF, Morphology)
# =========================
def create_coordinate_maps(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Генерация нормализованных координат X и Y (как в обучении)."""
    x_coords = np.linspace(0, 1, width, dtype=np.float32)
    y_coords = np.linspace(0, 1, height, dtype=np.float32)
    x_map, y_map = np.meshgrid(x_coords, y_coords)
    return x_map.copy(), y_map.copy()


def apply_morphology_skimage(mask: np.ndarray) -> np.ndarray:
    """Очищает маску от мелкого мусора и заделывает дырки с помощью skimage."""
    # Удаляем мелкие объекты
    cleaned = remove_small_objects(mask.astype(bool), max_size=MORPHOLOGY_MIN_SIZE) 
    # Заделываем мелкие дырки
    cleaned = remove_small_holes(cleaned, max_size=MORPHOLOGY_MAX_HOLE_SIZE)
    return cleaned.astype(np.uint8)


def cv2_imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0: return None
    return cv2.imdecode(data, flags)


def build_model(model_name: str, encoder_name: str, in_channels: int):
    if model_name == "Unet":
        return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    elif model_name == "UnetPlusPlus":
        return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    elif model_name == "FPN":
        return smp.FPN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    elif model_name == "SegFormer":
        return smp.Segformer(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")


def serialize_mask(mask2d: np.ndarray) -> str:
    return json.dumps(mask2d.astype(np.uint8).tolist(), separators=(",", ":"))


# =========================
# LOAD CHECKPOINT
# =========================
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
cfg = checkpoint["config"]

MODEL_NAME = cfg["MODEL_NAME"]
ENCODER_NAME = cfg["ENCODER_NAME"]
IMG_SIZE = int(cfg["IMG_SIZE"])

INPUT_CHANNELS = 5

print("Loaded checkpoint config:")
print(f"MODEL_NAME     = {MODEL_NAME}")
print(f"ENCODER_NAME   = {ENCODER_NAME}")
print(f"IMG_SIZE       = {IMG_SIZE}")
print(f"INPUT_CHANNELS = {INPUT_CHANNELS}")

model = build_model(MODEL_NAME, ENCODER_NAME, INPUT_CHANNELS)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Используем Albumentations для нормализации 
val_transforms = A.Compose([
    # A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# =========================
# INFERENCE LOGIC
# =========================
def predict_single_image(image_rgb: np.ndarray, target_size: int) -> np.ndarray:
    """Делает ресайз к target_size, готовит 5-канальный тензор и делает прогон."""
    
    # 1. Ресайз к размеру, кратному 32 (в квадрат)
    img_resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 2. Нормализация
    transformed = val_transforms(image=img_resized)
    img_norm = transformed["image"] # [target_size, target_size, 3]

    # 3. Переводим в формат PyTorch [3, target_size, target_size]
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

    # 4. ИСПРАВЛЕНИЕ: Генерируем координаты динамически под текущий target_size!
    x_map, y_map = create_coordinate_maps(target_size, target_size)
    coords = np.stack([x_map, y_map], axis=0)
    coords_tensor = torch.from_numpy(coords).float()

    # Собираем 5 каналов
    inp = torch.cat([img_tensor, coords_tensor], dim=0).unsqueeze(0).to(DEVICE)

    # 5. Предикт
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(inp)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    return probs  # возвращает маску размера (target_size, target_size)


def predict_multi_scale(image_rgb: np.ndarray, scales: list[float]) -> np.ndarray:
    """Multi-scale testing: предсказание на нескольких масштабах с усреднением."""
    h, w = image_rgb.shape[:2]
    scale_probs = []
    
    for scale in scales:
        # ИСПРАВЛЕНИЕ: Базируем масштаб на IMG_SIZE (на котором учили модель)
        # И обязательно делаем размер кратным 32
        target_size = (int(IMG_SIZE * scale) // 32) * 32
        
        # Предикт на масштабированном изображении
        prob_scaled = predict_single_image(image_rgb, target_size)
        
        # Возврат маски к исходному размеру текущей картинки (до того как мы ее сжали в квадрат)
        prob_orig_size = cv2.resize(prob_scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        scale_probs.append(prob_orig_size)
    
    # Усреднение предсказаний со всех масштабов
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
    """Обратный поворот маски, к исходной ориентации."""
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


image_paths = sorted([p for p in TEST_IMAGES_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS])
if not image_paths:
    raise FileNotFoundError(f"No images found in: {TEST_IMAGES_DIR}")

print(f"Found {len(image_paths)} test images")
rows = []

with torch.no_grad():
    for i, img_path in enumerate(image_paths, 1):
        img_bgr = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # -------------------------------------
        # 1. MULTI-SCALE + TTA
        # -------------------------------------
        if USE_MULTI_SCALE:
            pred_probs = predict_multi_scale(img_rgb, SCALE_FACTORS)
            
            if USE_TTA:
                tta_probs = [pred_probs]
                for angle in [90, 180, 270]:
                    img_rot = rotate_image(img_rgb, angle)
                    prob_rot = predict_multi_scale(img_rot, SCALE_FACTORS)
                    prob_rot_restored = rotate_mask(prob_rot, angle)
                    tta_probs.append(prob_rot_restored)
                
                pred_probs = np.mean(tta_probs, axis=0)
                
        elif USE_TTA: # Если Multi-scale выключен, используем базовый IMG_SIZE
            tta_probs = []
            
            # Вспомогательная функция для TTA без мульти-скейла, возвращающая к оригинальному размеру
            def predict_and_restore(img):
                p = predict_single_image(img, target_size=IMG_SIZE)
                return cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

            tta_probs.append(predict_and_restore(img_rgb))
            
            img_hflip = cv2.flip(img_rgb, 1)
            prob_hflip = predict_and_restore(img_hflip)
            tta_probs.append(cv2.flip(prob_hflip, 1))
            
            img_vflip = cv2.flip(img_rgb, 0)
            prob_vflip = predict_and_restore(img_vflip)
            tta_probs.append(cv2.flip(prob_vflip, 0))
            
            for angle in [90, 180, 270]:
                img_rot = rotate_image(img_rgb, angle)
                prob_rot = predict_and_restore(img_rot)
                tta_probs.append(rotate_mask(prob_rot, angle))
            
            pred_probs = np.mean(tta_probs, axis=0)
            
        else:
            # Без TTA и multi-scale
            pred_probs = predict_single_image(img_rgb, target_size=IMG_SIZE)
            pred_probs = cv2.resize(pred_probs, (W, H), interpolation=cv2.INTER_LINEAR)

        # 2. Ресайз (на всякий случай, хотя код выше уже гарантирует размер H, W)
        if pred_probs.shape != (H, W):
            pred_probs = cv2.resize(pred_probs, (W, H), interpolation=cv2.INTER_LINEAR)

        # Бинаризация
        mask = (pred_probs > THRESHOLD).astype(np.uint8)

        # -------------------------------------
        # 3. MORPHOLOGY CLEANUP (Опционально)
        # -------------------------------------
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