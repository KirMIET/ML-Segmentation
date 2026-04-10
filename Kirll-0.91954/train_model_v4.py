"""
Train Model V4 - Ensemble Segmentation Training Pipeline
=========================================================
Cross-validation training with 2-model ensemble:
- Transformer: SegFormer + MiT-B4
- CNN: UnetPlusPlus + ConvNeXt-small

Features:
- K-fold cross-validation on dataset/dataset_fold
- EMA (Exponential Moving Average) with decay=0.999
- Combined loss: 0.5*Lovász-Softmax + 0.5*BCE
- Augmentations: rotations, flips, scaling, color/noise (ImageNet normalization)
- AMP + gradient clipping + memory cleanup
- Early stopping by val_dice
- AdamW + CosineAnnealingWarmRestarts
"""

import os
import gc
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import LovaszLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import pandas as pd


# Paths
DATA_ROOT = Path("dataset/dataset_fold")
SAVE_DIR = Path("./model_checkpoints")
VISUALIZATION_DIR = Path("./view_train_img")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# General settings
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FOLDS = 4
THRESHOLD = 0.5
VISUALIZE_EVERY_N_EPOCHS = 5

# Dataset settings
IMG_SIZE = 352
NUM_WORKERS = 4

# Training settings 
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# AMP settings
USE_AMP = True
GRAD_CLIP_MAX_NORM = 1.0

# Loss weights (0.5*Lovász + 0.5*BCE)
LOVAZ_WEIGHT = 0.5
BCE_WEIGHT = 0.5

# EMA settings
EMA_DECAY = 0.999

# Augmentation parameters
HORIZONTAL_FLIP_P = 0.5
VERTICAL_FLIP_P = 0.3
RANDOM_ROTATE_90_P = 0.5
ROTATE_LIMIT = 30
AFFINE_SCALE = (0.85, 1.15)
AFFINE_ROTATE = (-15, 15)
BRIGHTNESS_CONTRAST_LIMIT = 0.1
GAUSSIAN_NOISE_VAR_LIMIT = (5, 20)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_CONFIGS = {
    "SegFormer": {
        "model_name": "SegFormer",
        "encoder_name": "mit_b4",
        "encoder_weights": "imagenet",
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 5e-2,
        "num_epochs": 50,
        "early_stopping_patience": 15,
    },

    "Unet": {
        "model_name": "Unet",
        "encoder_name": "tu-convnext_base",
        "encoder_weights": "imagenet",
        "batch_size": 8,
        "lr": 5e-4,
        "weight_decay": 3e-2,
        "num_epochs": 50,
        "early_stopping_patience": 10,
    },
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def dice_score_from_tp_fp_fn(total_intersection: float, total_pred: float, total_target: float, eps: float = 1e-7) -> float:
    """Compute global Dice score from accumulated TP/FP/FN statistics."""
    return (2.0 * total_intersection + eps) / (total_pred + total_target + eps)


def iou_score_from_tp_fp_fn(total_intersection: float, total_pred: float, total_target: float, eps: float = 1e-7) -> float:
    """Compute global IoU from accumulated TP/FP/FN statistics."""
    union = total_pred + total_target - total_intersection
    return (total_intersection + eps) / (union + eps)


def compute_tp_fp_fn(logits: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Compute intersection, prediction sum, target sum for dataset-level metrics."""
    probs = torch.sigmoid(logits.float())
    preds = (probs > THRESHOLD).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    return intersection.sum().item(), pred_sum.sum().item(), target_sum.sum().item()


class EMA:
    """Exponential Moving Average for model weights and buffers."""
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        # Сохраняем ВСЕ состояния (включая BatchNorm buffers)
        for name, param in model.state_dict().items():
            self.shadow[name] = param.clone().detach()

    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            # Обновляем по EMA только float-тензоры (веса и running_mean/var)
            if param.is_floating_point():
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
            # Целочисленные тензоры (например num_batches_tracked) просто копируем
            else:
                self.shadow[name] = param.data.clone()

    def apply_to_model(self, model: nn.Module) -> None:
        # Безопасно загружаем все состояние обратно в модель
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = state_dict.copy()


def get_train_transforms(img_size: int = IMG_SIZE, encoder_name: str = None, encoder_weights: str = None) -> A.Compose:
    """Get training augmentations with ImageNet normalization."""
    # Get preprocessing stats from SMP for the specific encoder
    if encoder_weights is not None:
        try:
            from segmentation_models_pytorch.encoders import get_preprocessing_params
            params = get_preprocessing_params(encoder_name, pretrained=encoder_weights)
            mean = params.get("mean", IMAGENET_MEAN)
            std = params.get("std", IMAGENET_STD)
        except Exception:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
    else:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=HORIZONTAL_FLIP_P),
        A.VerticalFlip(p=VERTICAL_FLIP_P),
        A.RandomRotate90(p=RANDOM_ROTATE_90_P),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=AFFINE_SCALE,
            rotate=AFFINE_ROTATE,
            mode=cv2.BORDER_REFLECT_101,
            p=0.6,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS_CONTRAST_LIMIT,
                contrast_limit=BRIGHTNESS_CONTRAST_LIMIT,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=1.0,
            ),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(var_limit=GAUSSIAN_NOISE_VAR_LIMIT, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = IMG_SIZE, encoder_name: str = None, encoder_weights: str = None) -> A.Compose:
    """Get validation augmentations (only resize + ImageNet normalization)."""
    # Get preprocessing stats from SMP for the specific encoder
    if encoder_weights is not None:
        try:
            from segmentation_models_pytorch.encoders import get_preprocessing_params
            params = get_preprocessing_params(encoder_name, pretrained=encoder_weights)
            mean = params.get("mean", IMAGENET_MEAN)
            std = params.get("std", IMAGENET_STD)
        except Exception:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
    else:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class BinarySegDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        img_size: int = IMG_SIZE,
        encoder_name: str = None,
        encoder_weights: str | None = "imagenet",
        augmentations: A.Compose | None = None,
        samples: list | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augmentations = augmentations

        self.samples = samples or []
        if not self.samples:
            # Auto-collect samples
            for mask_path in sorted(self.masks_dir.glob("*.png")):
                stem = mask_path.stem
                image_path = find_image_for_stem(self.images_dir, stem)
                if image_path is not None:
                    self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No paired image/mask samples found in {self.images_dir} and {self.masks_dir}")

        print(f"  Dataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image_rgb, mask=mask)
            image_rgb = augmented["image"]
            mask = augmented["mask"]
        else:
            image_rgb = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            image_rgb = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

        if isinstance(image_rgb, torch.Tensor):
            image_rgb = image_rgb.float()
        else:
            image_rgb = torch.from_numpy(image_rgb).float()

        if isinstance(mask, np.ndarray):
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask[None, ...]).float()
        elif isinstance(mask, torch.Tensor):
            mask = (mask > 0).float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

        return image_rgb, mask


class LovaszBCELoss(nn.Module):
    """Combined loss: 0.5*Lovász-Softmax + 0.5*BCE"""
    def __init__(self, lovasz_weight: float = LOVAZ_WEIGHT, bce_weight: float = BCE_WEIGHT):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.bce_weight = bce_weight
        self.lovasz = LovaszLoss(mode='binary', from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lovasz_loss = self.lovasz(logits, targets)
        bce_loss = self.bce(logits, targets)
        return self.lovasz_weight * lovasz_loss + self.bce_weight * bce_loss


def create_model(model_name: str, encoder_name: str, encoder_weights: str, in_channels: int = 3) -> nn.Module:
    """Create segmentation model."""
    if model_name == "Unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
    elif model_name == "SegFormer":
        return smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    ema: EMA | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    grad_clip_max_norm: float = GRAD_CLIP_MAX_NORM,
) -> dict:
    model.train()
    running_loss = 0.0
    total_intersection = 0.0
    total_pred = 0.0
    total_target = 0.0
    n = len(loader)

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP forward pass
        if grad_scaler is not None and USE_AMP:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, masks)

            # AMP backward pass
            grad_scaler.scale(loss).backward()

            # Gradient clipping with AMP
            if grad_clip_max_norm > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            # Standard forward/backward pass
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()

            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        batch_loss = loss.item()
        inter, p_sum, t_sum = compute_tp_fp_fn(logits.detach(), masks)
        total_intersection += inter
        total_pred += p_sum
        total_target += t_sum

        running_loss += batch_loss

        # Compute running dice for display
        running_dice = dice_score_from_tp_fp_fn(total_intersection, total_pred, total_target)
        pbar.set_postfix(loss=f"{batch_loss:.4f}", dice=f"{running_dice:.4f}")

    epoch_dice = dice_score_from_tp_fp_fn(total_intersection, total_pred, total_target)
    epoch_iou = iou_score_from_tp_fp_fn(total_intersection, total_pred, total_target)

    return {
        "loss": running_loss / n,
        "dice": epoch_dice,
        "iou": epoch_iou,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    epoch: int | None = None,
    ema: EMA | None = None,
    use_ema: bool = False,
) -> dict:
    """Validate model. If use_ema=True and ema is provided, uses EMA weights."""
    if use_ema and ema is not None:
        # Сохраняем ПОЛНЫЙ original state
        original_state = {name: param.clone().detach() for name, param in model.state_dict().items()}
        ema.apply_to_model(model)

    model.eval()
    running_loss = 0.0
    total_intersection = 0.0
    total_pred = 0.0
    total_target = 0.0
    n = len(loader)

    desc = f"  Val (EMA)" if use_ema else f"  Val"
    if epoch is not None:
        desc = f"  Val Epoch {epoch} (EMA)" if use_ema else f"  Val Epoch {epoch}"
    pbar = tqdm(loader, desc=desc, leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # AMP inference
        if USE_AMP and device == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, masks)
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)

        batch_loss = loss.item()
        inter, p_sum, t_sum = compute_tp_fp_fn(logits, masks)
        total_intersection += inter
        total_pred += p_sum
        total_target += t_sum

        running_loss += batch_loss

        # Compute running dice for display
        running_dice = dice_score_from_tp_fp_fn(total_intersection, total_pred, total_target)
        pbar.set_postfix(loss=f"{batch_loss:.4f}", dice=f"{running_dice:.4f}")

    # Restore original weights if EMA was used
    if use_ema and ema is not None:
        model.load_state_dict(original_state) # Загружаем состояние обратно правильно

    epoch_dice = dice_score_from_tp_fp_fn(total_intersection, total_pred, total_target)
    epoch_iou = iou_score_from_tp_fp_fn(total_intersection, total_pred, total_target)

    return {
        "loss": running_loss / n,
        "dice": epoch_dice,
        "iou": epoch_iou,
    }


@torch.no_grad()
def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    save_dir: Path,
    epoch: int,
    model_name: str,
    fold: int,
    num_samples: int = 4,
    threshold: float = THRESHOLD,
    ema: EMA | None = None,
):
    """Visualize predictions for a batch."""
    if ema is not None:
        # Use EMA weights for visualization
        original_state = {name: param.data.clone() for name, param in model.named_parameters()}
        ema.apply_to_model(model)

    model.eval()
    vis_dir = save_dir / f"{model_name}_fold_{fold}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        images, masks = next(iter(loader))
    except StopIteration:
        return

    images = images.to(device)
    masks = masks.to(device)

    if USE_AMP and device == 'cuda':
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
    else:
        logits = model(images)

    logits = logits.float()
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        mask = masks[i].cpu().numpy()[0]
        pred = preds[i].cpu().numpy()[0]

        viz_img = (img * 255).astype(np.uint8)
        viz_mask = (mask * 255).astype(np.uint8)
        viz_pred = (pred * 255).astype(np.uint8)

        if viz_img.shape[2] == 3:
            overlay = viz_img.copy()
            overlay[mask > 0] = [0, 255, 0]
            overlay_pred = viz_img.copy()
            overlay_pred[pred > 0] = [255, 0, 0]
        else:
            overlay = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2RGB)
            overlay_pred = overlay.copy()

        combined = np.hstack([
            cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR) if viz_img.shape[2] == 3 else cv2.cvtColor(viz_img, cv2.COLOR_GRAY2BGR),
            overlay if viz_img.shape[2] == 3 else cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            overlay_pred if viz_img.shape[2] == 3 else cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR),
        ])

        save_path = vis_dir / f"epoch_{epoch:03d}_sample_{i:02d}.png"
        cv2.imwrite(str(save_path), combined)

    # Restore original weights if EMA was used
    if ema is not None:
        for name, param in model.named_parameters():
            param.data.copy_(original_state[name])


def train_single_fold(
    fold: int,
    model_name: str,
    model_config: dict,
    train_samples: list,
    val_samples: list,
    images_dir: Path,
    masks_dir: Path,
) -> dict:
    """Train single model on single fold with EMA."""
    lr = model_config["lr"]
    batch_size = model_config["batch_size"]
    weight_decay = model_config["weight_decay"]
    encoder_name = model_config["encoder_name"]
    encoder_weights = model_config["encoder_weights"]

    # Per-model training settings
    num_epochs = model_config.get("num_epochs", NUM_EPOCHS)
    early_stopping_patience = model_config.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)

    print(f"\n{'='*60}")
    print(f"Training {model_name} on Fold {fold}")
    print(f"  LR={lr}, BS={batch_size}, Weight Decay={weight_decay}")
    print(f"  Encoder: {encoder_name} ({encoder_weights})")
    print(f"  Epochs={num_epochs}, ES Patience={early_stopping_patience}")
    print(f"  EMA Decay={EMA_DECAY}")
    print(f"{'='*60}")

    # Create transforms
    train_transforms = get_train_transforms(IMG_SIZE, encoder_name, encoder_weights)
    val_transforms = get_val_transforms(IMG_SIZE, encoder_name, encoder_weights)

    # Create datasets
    train_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        img_size=IMG_SIZE,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        augmentations=train_transforms,
        samples=train_samples,
    )

    val_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        img_size=IMG_SIZE,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        augmentations=val_transforms,
        samples=val_samples,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Create model
    model = create_model(model_name, encoder_name, encoder_weights, in_channels=3).to(DEVICE)

    # Create EMA
    ema = EMA(model, decay=EMA_DECAY)

    # Create loss, optimizer, scheduler
    loss_fn = LovaszBCELoss(lovasz_weight=LOVAZ_WEIGHT, bce_weight=BCE_WEIGHT).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # CosineAnnealingWarmRestarts scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    # AMP GradScaler
    grad_scaler = torch.amp.GradScaler() if USE_AMP and DEVICE == 'cuda' else None

    # Checkpoint directory
    checkpoint_subdir = SAVE_DIR / f"{model_name}_fold_{fold}"
    checkpoint_subdir.mkdir(parents=True, exist_ok=True)

    # Save prefixes
    model_save_prefix = f"best_{model_name}_fold_{fold}"
    ema_save_prefix = f"ema_best_{model_name}_fold_{fold}"
    last_save_prefix = f"last_{model_name}_fold_{fold}"
    ema_last_save_prefix = f"ema_last_{model_name}_fold_{fold}"

    best_val_dice = -1.0
    best_ema_val_dice = -1.0
    history = []
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\n  Epoch {epoch}/{num_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch,
            ema=ema,
            grad_scaler=grad_scaler,
            grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
        )

        # Validate (regular model)
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch,
            use_ema=False,
        )

        # Validate (EMA model)
        val_metrics_ema = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch,
            ema=ema,
            use_ema=True,
        )

        scheduler.step()

        # Log metrics
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_loss_ema": val_metrics_ema["loss"],
            "val_dice_ema": val_metrics_ema["dice"],
            "val_iou_ema": val_metrics_ema["iou"],
        }
        history.append(row)

        print(
            f"  Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={row['train_loss']:.4f} train_dice={row['train_dice']:.4f} train_iou={row['train_iou']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} val_iou={row['val_iou']:.4f} | "
            f"val_loss_ema={row['val_loss_ema']:.4f} val_dice_ema={row['val_dice_ema']:.4f} val_iou_ema={row['val_iou_ema']:.4f}"
        )

        # Visualization every N epochs (using EMA)
        # if epoch % VISUALIZE_EVERY_N_EPOCHS == 0:
        #     visualize_predictions(model, val_loader, DEVICE, VISUALIZATION_DIR, epoch, model_name, fold, ema=ema)

        # Save last model (both regular and EMA)
        torch.save(
            {
                "fold": fold,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": row["val_dice"],
                "val_dice_ema": row["val_dice_ema"],
                "config": {
                    "MODEL_NAME": model_name,
                    "ENCODER_NAME": encoder_name,
                    "ENCODER_WEIGHTS": encoder_weights,
                    "IMG_SIZE": IMG_SIZE,
                },
            },
            checkpoint_subdir / f"{last_save_prefix}.pth",
        )

        # Save best regular model (parallel, does not affect early stopping)
        if row["val_dice"] > best_val_dice:
            best_val_dice = row["val_dice"]

            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": row["val_dice"],
                    "val_iou": row["val_iou"],
                    "config": {
                        "MODEL_NAME": model_name,
                        "ENCODER_NAME": encoder_name,
                        "ENCODER_WEIGHTS": encoder_weights,
                        "IMG_SIZE": IMG_SIZE,
                    },
                },
                checkpoint_subdir / f"{model_save_prefix}.pth",
            )
            print(f"  >>> Saved new best model with val_dice={best_val_dice:.4f}")

        # Early stopping based on EMA val_dice
        if row["val_dice_ema"] > best_ema_val_dice:
            best_ema_val_dice = row["val_dice_ema"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": row["val_dice"],
                    "val_dice_ema": row["val_dice_ema"],
                    "val_iou_ema": row["val_iou_ema"],
                    "config": {
                        "MODEL_NAME": model_name,
                        "ENCODER_NAME": encoder_name,
                        "ENCODER_WEIGHTS": encoder_weights,
                        "IMG_SIZE": IMG_SIZE,
                    },
                },
                checkpoint_subdir / f"{ema_save_prefix}.pth",
            )
            print(f"  >>> Saved new best EMA model with val_dice_ema={best_ema_val_dice:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No EMA improvement for {epochs_without_improvement} epochs (best: regular={best_val_dice:.4f}, ema={best_ema_val_dice:.4f})")

        # Early stopping (based on EMA model val_dice)
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n  Early stopping triggered after {epoch} epochs.")
            print(f"  Best val_dice={best_val_dice:.4f}, Best val_dice_ema={best_ema_val_dice:.4f}")
            break

    # Save history
    history_df = pd.DataFrame(history)
    history_path = checkpoint_subdir / f"history_{model_name}_fold_{fold}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\n  Training finished. Best val_dice={best_val_dice:.4f}, Best val_dice_ema={best_ema_val_dice:.4f}")
    print(f"  History saved to {history_path}")

    # Memory cleanup
    del model, ema, optimizer, train_loader, val_loader, loss_fn
    if grad_scaler is not None:
        del grad_scaler
    torch.cuda.empty_cache()
    gc.collect()

    return history


def main():
    seed_everything(SEED)

    print(f"\n{'='*60}")
    print(f"Train Model V4 - Ensemble Training Pipeline")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"AMP: {USE_AMP}")
    print(f"Number of folds: {NUM_FOLDS}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Total experiments: {NUM_FOLDS} folds x {len(MODEL_CONFIGS)} models = {NUM_FOLDS * len(MODEL_CONFIGS)}")
    print(f"Loss: {LOVAZ_WEIGHT}*Lovász-Softmax + {BCE_WEIGHT}*BCE")
    print(f"EMA Decay: {EMA_DECAY}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Optimizer: AdamW + CosineAnnealing")
    print(f"{'='*60}")

    # Print model configs
    for mname, mcfg in MODEL_CONFIGS.items():
        print(f"  {mname}: encoder={mcfg['encoder_name']}, BS={mcfg['batch_size']}, LR={mcfg['lr']}")
    print(f"{'='*60}")

    # Cross-validation loop
    all_histories = {}

    for fold in range(2, NUM_FOLDS + 1):
        fold_dir = DATA_ROOT / f"fold_{fold}"
        train_dir = fold_dir / "train"
        val_dir = fold_dir / "val"

        train_images_dir = train_dir / "images"
        train_masks_dir = train_dir / "masks"
        val_images_dir = val_dir / "images"
        val_masks_dir = val_dir / "masks"

        # Check directories exist
        if not train_images_dir.exists() or not train_masks_dir.exists():
            print(f"WARNING: Train directory not found for fold {fold}: {train_dir}")
            continue
        if not val_images_dir.exists() or not val_masks_dir.exists():
            print(f"WARNING: Val directory not found for fold {fold}: {val_dir}")
            continue

        # Collect samples
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")

        train_samples = []
        for mask_path in sorted(train_masks_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_for_stem(train_images_dir, stem)
            if image_path is not None:
                train_samples.append((image_path, mask_path))

        val_samples = []
        for mask_path in sorted(val_masks_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_for_stem(val_images_dir, stem)
            if image_path is not None:
                val_samples.append((image_path, mask_path))

        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")

        # Train each model
        fold_histories = {}
        for model_name, model_config in MODEL_CONFIGS.items():
            # Clear cache before each model
            torch.cuda.empty_cache()
            gc.collect()

            history = train_single_fold(
                fold=fold,
                model_name=model_name,
                model_config=model_config,
                train_samples=train_samples,
                val_samples=val_samples,
                images_dir=train_images_dir,
                masks_dir=train_masks_dir,
            )
            fold_histories[model_name] = history

            # Clear cache after each model
            torch.cuda.empty_cache()
            gc.collect()

        all_histories[f"fold_{fold}"] = fold_histories

        # Clear cache after each fold
        torch.cuda.empty_cache()
        gc.collect()

    # Final summary
    print(f"\n{'='*60}")
    print(f"All training complete!")
    print(f"Checkpoints saved to: {SAVE_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")
    print(f"{'='*60}")

    # Save all histories
    for fold_name, fold_histories in all_histories.items():
        for model_name, history in fold_histories.items():
            history_df = pd.DataFrame(history)
            history_path = SAVE_DIR / f"history_{model_name}_{fold_name}.csv"
            history_df.to_csv(history_path, index=False)
            print(f"Saved {history_path}")

    print(f"\nAll histories saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
