import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A

import segmentation_models_pytorch as smp

# =========================
# CONFIG
# =========================
TEST_IMAGES_DIR = Path(r"dataset/best_dataset/test_images")
OUTPUT_CSV = "submissions/submission_simple.csv"
CHECKPOINT_PATHS = [

    r"./model_checkpoints/SegFormer_fold_1/ema_best_SegFormer_fold_1.pth",
    r"./model_checkpoints/SegFormer_fold_2/ema_best_SegFormer_fold_2.pth",
    r"./model_checkpoints/SegFormer_fold_3/ema_best_SegFormer_fold_3.pth",
    r"./model_checkpoints/SegFormer_fold_4/ema_best_SegFormer_fold_4.pth",

    r"./model_checkpoints/Unet_fold_2/ema_best_Unet_fold_2.pth",
    r"./model_checkpoints/Unet_fold_3/ema_best_Unet_fold_3.pth",
    r"./model_checkpoints/Unet_fold_4/ema_best_Unet_fold_4.pth",
]
ENSEMBLE_WEIGHTS_FILE = r"submissions/ensemble_weights_best.json"
THRESHOLD = 0.47
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 3
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# =========================
# HELPERS
# =========================
def build_model(model_name, encoder_name, in_channels):
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


def load_checkpoints(paths):
    models = []
    for ckpt_path in paths:
        ckpt_path = Path(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model_name = cfg["MODEL_NAME"]
        encoder_name = cfg["ENCODER_NAME"]
        img_size = int(cfg["IMG_SIZE"])

        model = build_model(model_name, encoder_name, IN_CHANNELS)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        models.append({"model": model, "img_size": img_size})
        print(f"Loaded: {ckpt_path.name} | {model_name}/{encoder_name} | size={img_size}")
    return models


def load_weights(ckpt_paths, weights_file):
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
            if found is None:
                print(f"WARNING: No weight for {name}, using uniform")
            weights.append(found if found else 1.0 / len(ckpt_paths))
    print(f"Ensemble weights: {weights} (sum={sum(weights):.4f})")
    return weights


def prepare_input(image_rgb, target_size):
    img = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    transformed = val_transform(image=img)
    tensor = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    return tensor


def predict_one(model_info, image_rgb):
    inp = prepare_input(image_rgb, model_info["img_size"])
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        logits = model_info["model"](inp)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    return prob


def rotate_img(img, k):
    return np.rot90(img, k, axes=(0, 1))

def rotate_mask(mask, k):
    # Исправлен баг с k
    return np.rot90(mask, k=-k, axes=(0, 1))

def predict_with_tta(model_info, image_rgb):
    h, w = image_rgb.shape[:2]
    target_size = model_info["img_size"]
    probs = []

    # 1. Original
    p = predict_one(model_info, image_rgb)
    probs.append(p)

    # 2. Flips
    for flip_code in [1, 0]:  # 1: horizontal, 0: vertical
        flipped = cv2.flip(image_rgb, flip_code)
        p = predict_one(model_info, flipped)
        # Разворачиваем флип на квадратной маске
        probs.append(cv2.flip(p, flip_code))

    # 3. Rotations
    for k in [1, 2, 3]:  # 90, 180, 270
        rotated = rotate_img(image_rgb, k)
        p = predict_one(model_info, rotated)
        # Разворачиваем поворот на квадратной маске
        probs.append(rotate_mask(p, k))

    # Усредняем 6 квадратных масок (target_size x target_size)
    avg_prob = np.mean(probs, axis=0)

    # Делаем финальный ресайз в оригинальное разрешение ОДИН РАЗ
    final_prob = cv2.resize(avg_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return final_prob


def serialize_mask(mask):
    return json.dumps(mask.astype(np.uint8).tolist(), separators=(",", ":"))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
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

        # Ensemble prediction with TTA
        model_probs = []
        for m in models:
            prob = predict_with_tta(m, img_rgb)
            model_probs.append(prob)

        ensemble_pred = np.average(model_probs, weights=weights, axis=0)
        mask = (ensemble_pred > THRESHOLD).astype(np.uint8)

        rows.append({"ImageId": img_path.name, "mask": serialize_mask(mask)})

        if i % 100 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nDone. Saved to {OUTPUT_CSV}")
