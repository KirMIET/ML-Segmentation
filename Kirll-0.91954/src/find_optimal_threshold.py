import json
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A

import segmentation_models_pytorch as smp

# =========================
# CONFIG
# =========================
VAL_DIR = Path(r"dataset/dataset_fold/fold_2/val")
CHECKPOINT_PATHS = [
    r"./model_checkpoints/SegFormer_fold_1/ema_best_SegFormer_fold_1.pth",
    r"./model_checkpoints/SegFormer_fold_2/ema_best_SegFormer_fold_2.pth",
    r"./model_checkpoints/SegFormer_fold_3/ema_best_SegFormer_fold_3.pth",
    r"./model_checkpoints/SegFormer_fold_4/ema_best_SegFormer_fold_4.pth",

    r"./model_checkpoints/Unet_fold_2/ema_best_Unet_fold_2.pth",
    r"./model_checkpoints/Unet_fold_3/ema_best_Unet_fold_3.pth",
    r"./model_checkpoints/Unet_fold_4/ema_best_Unet_fold_4.pth",
]

ENSEMBLE_WEIGHTS_FILE = r"submissions/ensemble_weights_king.json"
OUTPUT_FILE = r"submissions/optimal_threshold.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD_SEARCH_RANGE = (0.10, 0.70)
THRESHOLD_STEP = 0.01

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def create_coordinate_maps(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    x_map, y_map = np.meshgrid(x, y)
    return x_map, y_map


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
        img_size = int(cfg["IMG_SIZE"])

        model = build_model(model_name, encoder_name, in_channels)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        models.append({
            "model": model,
            "model_name": model_name,
            "in_channels": in_channels,
            "use_coordinates": use_coords,
            "img_size": img_size,
        })
        coords_str = "RGB+XY" if use_coords else "RGB"
        print(f"  Loaded: {ckpt_path.name} | {model_name} | {coords_str} (ch={in_channels}) | size={img_size}")
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


def prepare_input(image_rgb: np.ndarray, target_size: int, in_channels: int) -> torch.Tensor:
    """
    image_rgb: всегда только RGB (H, W, 3).
    Если in_channels == 5, координаты генерируются прямо под target_size.
    """
    img_resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    transformed = val_transform(image=img_resized)
    norm_rgb = transformed["image"]

    if in_channels == 5:
        x_map, y_map = create_coordinate_maps(target_size, target_size)
        coords = np.stack([x_map, y_map], axis=-1)
        combined = np.concatenate([norm_rgb, coords], axis=-1)
        tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    else:
        tensor = torch.from_numpy(norm_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    return tensor


def predict_one(model_info: dict, image_rgb: np.ndarray) -> np.ndarray:
    inp = prepare_input(image_rgb, model_info["img_size"], model_info["in_channels"])
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        logits = model_info["model"](inp)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    return prob


def load_val_pairs(val_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = val_dir / "images"
    mask_dir = val_dir / "masks"

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Val images or masks not found in {val_dir}")

    mask_files = sorted(mask_dir.glob("*.png"))
    pairs = []
    for mf in mask_files:
        stem = mf.stem
        img_path = image_dir / f"{stem}.jpg"
        if img_path.exists():
            pairs.append((img_path, mf))
        else:
            print(f"WARNING: No image for mask {mf.name}")

    print(f"Found {len(pairs)} validation pairs")
    return pairs


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"\nLoading checkpoints...")
    models = load_checkpoints(CHECKPOINT_PATHS)
    weights = load_weights(CHECKPOINT_PATHS, ENSEMBLE_WEIGHTS_FILE)
    val_pairs = load_val_pairs(VAL_DIR)

    if not val_pairs:
        raise ValueError("No validation pairs found!")

    # Подготавливаем массив порогов
    thresholds = np.arange(THRESHOLD_SEARCH_RANGE[0], THRESHOLD_SEARCH_RANGE[1] + THRESHOLD_STEP, THRESHOLD_STEP)
    thresholds = np.round(thresholds, 4)

    # Словарь для накопления Dice Score для каждого порога
    dice_sums_per_threshold = {thr: 0.0 for thr in thresholds}

    print(f"\nRunning ensemble inference & evaluating thresholds on {len(val_pairs)} samples...")
    
    for i, (img_path, mask_path) in enumerate(val_pairs, 1):
        # 1. Чтение изображения
        img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Чтение маски (переводим сразу в float32 для быстрого умножения)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 0).astype(np.float32)

        # 3. Инференс
        model_probs = []
        for m in models:
            # Теперь подаем только RGB, prepare_input разберется с XY сам
            prob = predict_one(m, img_rgb)
            prob_resized = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
            model_probs.append(prob_resized)

        ensemble_pred = np.average(model_probs, weights=weights, axis=0)

        # 4. Расчет Dice на лету для всех порогов
        for thr in thresholds:
            pred_mask = (ensemble_pred > thr).astype(np.float32)
            
            intersection = (pred_mask * gt_mask).sum()
            denom = pred_mask.sum() + gt_mask.sum()
            
            if denom == 0:
                dice = 1.0
            else:
                dice = (2.0 * intersection) / denom
                
            dice_sums_per_threshold[thr] += float(dice)

        if i % 50 == 0 or i == len(val_pairs):
            print(f"  Processed {i}/{len(val_pairs)}")


    best_dice = -1.0
    best_threshold = THRESHOLD_SEARCH_RANGE[0]
    results = []

    print(f"\nResults:")
    for thr in thresholds:
        mean_dice = dice_sums_per_threshold[thr] / len(val_pairs)
        results.append({"threshold": float(thr), "dice": float(mean_dice)})
        
        print(f"  thr={thr:.2f}  mean_dice={mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = thr

    print(f"\n{'='*50}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best Dice:      {best_dice:.4f}")
    print(f"{'='*50}")

    # Сохранение результатов
    output = {
        "best_threshold": float(best_threshold),
        "best_dice": float(best_dice),
        "val_dir": str(VAL_DIR),
        "checkpoints": [str(p) for p in CHECKPOINT_PATHS],
        "weights_file": ENSEMBLE_WEIGHTS_FILE,
        "all_results": results,
    }

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")