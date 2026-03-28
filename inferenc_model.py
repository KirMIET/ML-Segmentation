import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


# =========================
# CONFIG
# =========================
TEST_IMAGES_DIR = Path(
    r"dataset/best_dataset/test_images"
)
OUTPUT_CSV = "submission_change1.csv"

# путь к вашему чекпоинту после обучения
CHECKPOINT_PATH = Path(r"./model_checkpoints/best.pth")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# =========================
# HELPERS
# =========================
def cv2_imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def build_model(model_name: str, encoder_name: str, encoder_weights=None):
    if model_name == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif model_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif model_name == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")
    return model




def serialize_mask(mask2d: np.ndarray) -> str:
    return json.dumps(mask2d.astype(np.uint8).tolist(), separators=(",", ":"))


# =========================
# LOAD CHECKPOINT
# =========================
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

if "config" not in checkpoint:
    raise ValueError("Checkpoint does not contain 'config'.")

cfg = checkpoint["config"]

MODEL_NAME = cfg["MODEL_NAME"]
ENCODER_NAME = cfg["ENCODER_NAME"]
ENCODER_WEIGHTS = cfg.get("ENCODER_WEIGHTS", None)
IMG_SIZE = int(cfg["IMG_SIZE"])

print("Loaded checkpoint config:")
print("MODEL_NAME     =", MODEL_NAME)
print("ENCODER_NAME   =", ENCODER_NAME)
print("ENCODER_WEIGHTS=", ENCODER_WEIGHTS)
print("IMG_SIZE       =", IMG_SIZE)

model = build_model(
    model_name=MODEL_NAME,
    encoder_name=ENCODER_NAME,
    encoder_weights=None,  # веса энкодера уже придут из checkpoint
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

preprocess_input = None
if ENCODER_WEIGHTS is not None:
    preprocess_input = get_preprocessing_fn(ENCODER_NAME, pretrained=ENCODER_WEIGHTS)


# =========================
# INFERENCE
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
            print(f"[skip] cannot read: {img_path}")
            continue

        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # resize под вход модели
        inp = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        inp = inp.astype(np.float32)

        if preprocess_input is not None:
            inp = preprocess_input(inp)
        else:
            inp = inp / 255.0

        inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

        logits = model(inp)                      # [1, 1, H, W]
        probs = torch.sigmoid(logits)[0, 0]      # [H, W]
        pred = probs.detach().cpu().numpy()

        # возвращаем в исходный размер
        if pred.shape != (H, W):
            pred = cv2.resize(pred.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

        mask = (pred > THRESHOLD).astype(np.uint8)

        rows.append(
            {
                "ImageId": img_path.name,
                "mask": serialize_mask(mask),
            }
        )

        if i % 100 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)}")

submission_df = pd.DataFrame(rows)
submission_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("Done.")
print(f"Saved submission to: {OUTPUT_CSV}")
print(submission_df.head())