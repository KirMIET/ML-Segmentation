"""
Ensemble Segmentation Training Pipeline
========================================
Trains models (UNet++, FPN, SegFormer, UPerNet) on 4 folds with:
- SAM optimizer (per-model configurable)
- Combined dynamic loss (BCE + FocalTversky + Boundary)
- Multi-scale training (per-model configurable)
- CutMix + CopyPaste augmentations
- AMP + gradient clipping + early stopping
- 3 or 5-channel input (RGB or RGB + XY coordinates, per-model configurable)
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    FOLD_DIR, SAVE_DIR, VISUALIZATION_DIR,
    SEED, DEVICE, NUM_FOLDS,
    IMG_SIZE, NUM_WORKERS, THRESHOLD,
    VISUALIZE_EVERY_N_EPOCHS,
    SAM_RHO, GRAD_CLIP_MAX_NORM,
    CUTMIX_ALPHA, AUGMENTATION_PROB, COPYPASTE_MAX_OBJECTS,
    USE_OHEM, OHEM_RATIO,
    BCE_WEIGHT, TVERSKY_WEIGHT, BOUNDARY_WEIGHT,
    MODEL_CONFIGS,
)
from src.losses import CombinedLoss
from src.augmentations import get_train_transforms, get_val_transforms
from src.dataset import BinarySegDataset, build_image_dict
from src.models import create_model
from src.training_utils import (
    SAM,
    seed_everything,
    train_one_epoch,
    validate_one_epoch,
    visualize_batch,
)
from src.mixup_augmentations import SegmentationAugmentations


def collect_samples(images_dir: Path, masks_dir: Path) -> list:
    """Собирает пары (image_path, mask_path) из директорий через O(1) lookup."""
    image_dict = build_image_dict(images_dir)
    samples = []
    for mask_path in sorted(masks_dir.glob("*.png")):
        stem = mask_path.stem
        image_path = image_dict.get(stem)
        if image_path is not None:
            samples.append((image_path, mask_path))
    return samples


def train_single_fold(
    fold: int,
    model_name: str,
    model_config: dict,
    train_samples: list,
    val_samples: list,
    images_dir: Path,
    masks_dir: Path,
) -> dict:
    """Обучает одну модель на одном фолде.

    Args:
        fold: Номер фолда (1-4).
        model_name: Имя модели.
        model_config: Конфигурация модели.
        train_samples: Сэмплы для обучения.
        val_samples: Сэмплы для валидации.
        images_dir: Директория с изображениями.
        masks_dir: Директория с масками.

    Returns:
        История обучения (list of dicts).
    """
    lr = model_config["lr"]
    batch_size = model_config["batch_size"]
    num_epochs = model_config["num_epochs"]
    warmup_epochs = model_config["warmup_epochs"]
    early_stopping_patience = model_config["early_stopping_patience"]
    weight_decay = model_config["weight_decay"]
    encoder_name = model_config["encoder_name"]
    encoder_weights = model_config["encoder_weights"]

    # Per-model settings
    use_coordinates = model_config.get("use_coordinates", True)
    use_sam = model_config.get("use_sam", True)
    multi_scales = model_config.get("multi_scales", [IMG_SIZE])
    if multi_scales is None or len(multi_scales) == 0:
        multi_scales = [IMG_SIZE]
    input_channels = 5 if use_coordinates else 3

    print(f"\n{'='*60}")
    print(f"Training {model_name} on Fold {fold}")
    print(f"  LR={lr}, BS={batch_size}, Epochs={num_epochs}, Warmup={warmup_epochs}")
    print(f"  Early Stopping Patience={early_stopping_patience}")
    print(f"  Coords={use_coordinates}, SAM={use_sam}, Scales={multi_scales}")
    print(f"{'='*60}")

    # Создаем трансформы
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    # Создаем датасеты
    train_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        img_size=IMG_SIZE,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        augmentations=train_transforms,
        samples=train_samples,
        use_coordinates=use_coordinates,
    )

    train_dataset.custom_augs = SegmentationAugmentations(
        apply_prob=AUGMENTATION_PROB,
        cutmix_alpha=CUTMIX_ALPHA,
        copypaste_max_objects=COPYPASTE_MAX_OBJECTS,
    )

    val_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        img_size=IMG_SIZE,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        augmentations=val_transforms,
        samples=val_samples,
        use_coordinates=use_coordinates,
    )

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

    # Создаем модель
    model = create_model(model_name, in_channels=input_channels).to(DEVICE)

    # # PyTorch 2.0+ compilation для ускорения (15-20%)
    # try:
    #     model = torch.compile(model)
    #     print(f"  Model compiled successfully (PyTorch 2.0+)")
    # except Exception as e:
    #     print(f"  Model compilation skipped: {e}")

    # Создаем лосс, оптимизатор, scheduler
    loss_fn = CombinedLoss(
        bce_weight=BCE_WEIGHT,
        tversky_weight=TVERSKY_WEIGHT,
        boundary_weight=BOUNDARY_WEIGHT,
        use_ohem=USE_OHEM,
        ohem_ratio=OHEM_RATIO,
    ).to(DEVICE)

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = SAM(base_optimizer, rho=SAM_RHO) if use_sam else base_optimizer

    # LR Warmup: LinearLR + CosineAnnealingLR через SequentialLR
    opt = optimizer.base_optimizer if use_sam else optimizer
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=num_epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    grad_scaler = torch.amp.GradScaler()

    # Создаём подпапку для чекпоинтов этой модели/фолда
    checkpoint_subdir = SAVE_DIR / f"{model_name}_fold_{fold}"
    checkpoint_subdir.mkdir(parents=True, exist_ok=True)

    # Пути для сохранения (в подпапке)
    model_save_prefix = f"best_{model_name}_fold_{fold}"
    last_save_prefix = f"last_{model_name}_fold_{fold}"
    vis_dir = VISUALIZATION_DIR / f"{model_name}_fold_{fold}"

    best_val_score = -1.0
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # Обновляем веса лосса с учётом общего числа эпох
        loss_fn.update_weights(epoch, warmup_epochs, num_epochs)

        if epoch <= warmup_epochs:
            print(f"\n  Epoch {epoch}/{num_epochs} [WARMUP]: "
                  f"BCE={loss_fn.bce_weight}, Tversky({loss_fn.focal_tversky.alpha},"
                  f"{loss_fn.focal_tversky.beta})={loss_fn.tversky_weight}, "
                  f"Boundary={loss_fn.boundary_weight}")
        else:
            print(f"\n  Epoch {epoch}/{num_epochs} [MAIN]: "
                  f"BCE={loss_fn.bce_weight}, Tversky({loss_fn.focal_tversky.alpha},"
                  f"{loss_fn.focal_tversky.beta})={loss_fn.tversky_weight}, "
                  f"Boundary={loss_fn.boundary_weight}")

        # Обучение
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch,
            grad_scaler=grad_scaler,
            multi_scales=multi_scales,
            use_sam=use_sam,
            grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
        )

        # Валидация
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
        )

        scheduler.step()

        # Логирование
        row = {
            "epoch": epoch,
            "lr": optimizer.base_optimizer.param_groups[0]["lr"] if use_sam else optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)

        print(
            f"  Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={row['train_loss']:.4f} train_dice={row['train_dice']:.4f} train_iou={row['train_iou']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} val_iou={row['val_iou']:.4f}"
        )

        # Визуализация каждые N эпох
        if epoch % VISUALIZE_EVERY_N_EPOCHS == 0:
            # Берём первый батч из val_loader для визуализации
            model.eval()
            with torch.no_grad():
                val_iter = iter(val_loader)
                try:
                    vis_images, vis_masks = next(val_iter)
                    vis_images = vis_images.to(DEVICE)
                    vis_masks = vis_masks.to(DEVICE)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        vis_logits = model(vis_images)
                    vis_logits = vis_logits.float()
                    visualize_batch(
                        vis_images, vis_masks, vis_logits,
                        vis_dir, epoch, model_name, fold,
                        num_samples=4, threshold=THRESHOLD,
                    )
                except StopIteration:
                    pass

        # Сохранение last (каждую эпоху)
        optimizer_to_save = optimizer.base_optimizer if use_sam else optimizer
        torch.save(
            {
                "fold": fold,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_to_save.state_dict(),
                "val_dice": row["val_dice"],
                "config": {
                    "MODEL_NAME": model_name,
                    "ENCODER_NAME": encoder_name,
                    "ENCODER_WEIGHTS": encoder_weights,
                    "IMG_SIZE": IMG_SIZE,
                },
            },
            checkpoint_subdir / f"{last_save_prefix}.pth",
        )

        # Сохранение best и проверка early stopping
        # Используем композитную метрику: (val_dice + val_iou) / 2
        val_score = (row["val_dice"] + row["val_iou"]) / 2
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0

            # Сохраняем лучшую модель
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_to_save.state_dict(),
                    "val_dice": row["val_dice"],
                    "val_iou": row["val_iou"],
                    "val_score": val_score,
                    "config": {
                        "MODEL_NAME": model_name,
                        "ENCODER_NAME": encoder_name,
                        "ENCODER_WEIGHTS": encoder_weights,
                        "IMG_SIZE": IMG_SIZE,
                    },
                },
                checkpoint_subdir / f"{model_save_prefix}.pth",
            )
            print(f"  >>> Saved new best model with val_score={best_val_score:.4f} "
                  f"(dice={row['val_dice']:.4f}, iou={row['val_iou']:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n  Early stopping triggered after {epoch} epochs. "
                  f"Best val_score={best_val_score:.4f}")
            break

    # Сохранение истории
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_path = checkpoint_subdir / f"history_{model_name}_fold_{fold}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\n  Training finished. Best val_score={best_val_score:.4f}")
    print(f"  History saved to {history_path}")

    # Очистка памяти для защиты от OOM между моделями
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return history


def main():
    """Запускает полный цикл обучения ансамбля."""
    seed_everything(SEED)

    print(f"{'='*60}")
    print(f"Ensemble Training Pipeline")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Number of folds: {NUM_FOLDS}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Total experiments: {NUM_FOLDS} folds x {len(MODEL_CONFIGS)} models = {NUM_FOLDS * len(MODEL_CONFIGS)}")
    # Выводим конфигурацию каждой модели
    for mname, mcfg in MODEL_CONFIGS.items():
        coords = mcfg.get("use_coordinates", True)
        sam = mcfg.get("use_sam", True)
        scales = mcfg.get("multi_scales", [IMG_SIZE])
        if scales is None:
            scales = [IMG_SIZE]
        print(f"  {mname}: channels={5 if coords else 3}, SAM={sam}, scales={scales}")
    print(f"{'='*60}")

    # Цикл по фолдам
    for fold in range(1, NUM_FOLDS + 1):
        fold_dir = FOLD_DIR / f"fold_{fold}"
        train_dir = fold_dir / "train"
        val_dir = fold_dir / "val"

        train_images_dir = train_dir / "images"
        train_masks_dir = train_dir / "masks"
        val_images_dir = val_dir / "images"
        val_masks_dir = val_dir / "masks"

        # Проверяем существование директорий
        if not train_images_dir.exists() or not train_masks_dir.exists():
            print(f"WARNING: Train directory not found for fold {fold}: {train_dir}")
            continue
        if not val_images_dir.exists() or not val_masks_dir.exists():
            print(f"WARNING: Val directory not found for fold {fold}: {val_dir}")
            continue

        # Собираем сэмплы
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")
        train_samples = collect_samples(train_images_dir, train_masks_dir)
        val_samples = collect_samples(val_images_dir, val_masks_dir)
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")

        # Цикл по моделям
        for model_name, model_config in MODEL_CONFIGS.items():
            history = train_single_fold(
                fold=fold,
                model_name=model_name,
                model_config=model_config,
                train_samples=train_samples,
                val_samples=val_samples,
                images_dir=train_images_dir,  # dataset сам ищет по stem
                masks_dir=train_masks_dir,
            )

    print(f"\n{'='*60}")
    print(f"All training complete!")
    print(f"Checkpoints saved to: {SAVE_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
