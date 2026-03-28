import os
import random
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================
DATA_ROOT = Path(
    r"dataset/best_dataset/train"
)
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "masks"
SAVE_DIR = Path("./model_checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 352
BATCH_SIZE = 8
NUM_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 4
SEED = 42
THRESHOLD = 0.5

MODEL_NAME = "UnetPlusPlus"
ENCODER_NAME = "timm-efficientnet-b0"  
ENCODER_WEIGHTS = "noisy-student"    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# UTILS
# =========================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > THRESHOLD).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean().item()


def iou_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > THRESHOLD).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# =========================
# DATASET
# =========================
class BinarySegDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        img_size: int = 352,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size

        self.preprocess_input = None
        if encoder_weights is not None:
            self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)

        self.samples = []
        for mask_path in sorted(self.masks_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_for_stem(self.images_dir, stem)
            if image_path is not None:
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No paired image/mask samples found in {self.images_dir} and {self.masks_dir}")

        print(f"Found {len(self.samples)} paired samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image_rgb = image_rgb.astype(np.float32)

        if self.preprocess_input is not None:
            image_rgb = self.preprocess_input(image_rgb)
        else:
            image_rgb = image_rgb / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask[None, ...]).float()  # [1, H, W]

        return image, mask


# =========================
# MODEL
# =========================
def build_model() -> nn.Module:
    if MODEL_NAME == "Unet":
        model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif MODEL_NAME == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif MODEL_NAME == "FPN":
        model = smp.FPN(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
    return model


# =========================
# TRAIN / VAL LOOPS
# =========================
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, num_epochs):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(loader, desc=f"Train", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_dice = dice_score_from_logits(logits.detach(), masks)
        batch_iou = iou_score_from_logits(logits.detach(), masks)

        running_loss += batch_loss
        running_dice += batch_dice
        running_iou += batch_iou

        pbar.set_postfix(loss=f"{batch_loss:.4f}", dice=f"{batch_dice:.4f}")

    n = len(loader)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "iou": running_iou / n,
    }

@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device, epoch, num_epochs):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(loader, desc=f"Val", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        batch_loss = loss.item()
        batch_dice = dice_score_from_logits(logits, masks)
        batch_iou = iou_score_from_logits(logits, masks)

        running_loss += batch_loss
        running_dice += batch_dice
        running_iou += batch_iou

        pbar.set_postfix(loss=f"{batch_loss:.4f}", dice=f"{batch_dice:.4f}")

    n = len(loader)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "iou": running_iou / n,
    }


# =========================
# MAIN
# =========================
def main():
    seed_everything(SEED)

    full_dataset = BinarySegDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        img_size=IMG_SIZE,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
    )

    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model().to(DEVICE)

    loss_fn = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_dice = -1.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch, NUM_EPOCHS)
        val_metrics = validate_one_epoch(model, val_loader, loss_fn, DEVICE, epoch, NUM_EPOCHS)

        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train_loss={row['train_loss']:.4f} train_dice={row['train_dice']:.4f} train_iou={row['train_iou']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} val_iou={row['val_iou']:.4f}"
        )

        # last
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": row["val_dice"],
                "config": {
                    "MODEL_NAME": MODEL_NAME,
                    "ENCODER_NAME": ENCODER_NAME,
                    "ENCODER_WEIGHTS": ENCODER_WEIGHTS,
                    "IMG_SIZE": IMG_SIZE,
                },
            },
            SAVE_DIR / "last.pth",
        )

        # best
        if row["val_dice"] > best_val_dice:
            best_val_dice = row["val_dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": row["val_dice"],
                    "config": {
                        "MODEL_NAME": MODEL_NAME,
                        "ENCODER_NAME": ENCODER_NAME,
                        "ENCODER_WEIGHTS": ENCODER_WEIGHTS,
                        "IMG_SIZE": IMG_SIZE,
                    },
                },
                SAVE_DIR / "best.pth",
            )
            print(f"Saved new best model with val_dice={best_val_dice:.4f}")

    # сохранить историю
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(SAVE_DIR / "history.csv", index=False)
    print(f"Training finished. Best val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()