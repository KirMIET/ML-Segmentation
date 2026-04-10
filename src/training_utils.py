import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.config import THRESHOLD, SAM_RHO, SEED


# =========================
# SAM OPTIMIZER
# =========================
class SAM:
    """SAM optimizer wrapper над базовым оптимизатором, совместимый с AMP GradScaler."""

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = SAM_RHO):
        self.rho = rho
        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def perturb(self, inv_scale: float = 1.0) -> bool:
        """Первый шаг: вычисляет возмущение. Возвращает False, если градиенты невалидны (Inf/NaN)."""
        grads = []
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad)

        if not grads:
            return True

        grad_norm = torch.norm(
            torch.stack([torch.norm(g, p=2) for g in grads]),
            p=2
        )

        true_grad_norm = grad_norm * inv_scale

        if not torch.isfinite(true_grad_norm) or true_grad_norm == 0:
            return False

        scale = self.rho / torch.clamp(true_grad_norm, min=1e-8)

        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (p.grad * inv_scale) * scale
                p.add_(e_w)
                p.__dict__['e_w'] = e_w

        return True

    @torch.no_grad()
    def unperturb(self) -> None:
        """Откатывает возмущение."""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if 'e_w' not in p.__dict__:
                    continue
                p.sub_(p.__dict__['e_w'])
                del p.__dict__['e_w']

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Применяет обновление базового оптимизатора."""
        self.base_optimizer.step()


# =========================
# UTILS
# =========================
def seed_everything(seed: int = SEED) -> None:
    """Устанавливает случайные сиды для воспроизводимости экспериментов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                           threshold: float = THRESHOLD, eps: float = 1e-7) -> float:
    """Вычисляет Dice score между предсказанными логитами и целевыми масками."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean().item()


def iou_score_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                          threshold: float = THRESHOLD, eps: float = 1e-7) -> float:
    """Вычисляет IoU score между предсказанными логитами и целевыми масками."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def visualize_batch(images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor,
                    save_path: Path, epoch: int, model_name: str, fold: int,
                    num_samples: int = 4, threshold: float = THRESHOLD) -> None:
    """Сохраняет визуализацию предсказаний модели для батча.
    
    Args:
        images: Входные изображения (B, C, H, W).
        masks: Ground truth маски (B, 1, H, W).
        logits: Предсказания модели (B, 1, H, W).
        save_path: Директория для сохранения.
        epoch: Текущая эпоха.
        model_name: Имя модели для префикса.
        fold: Номер фолда.
        num_samples: Количество сэмплов для визуализации.
        threshold: Порог для бинаризации.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    for i in range(min(num_samples, images.size(0))):
        img = images[i][:3].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)

        mask = masks[i].squeeze().cpu().numpy()
        pred = preds[i].squeeze().cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title(f"Prediction ({model_name} fold {fold})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path / f"epoch_{epoch:03d}.png", dpi=100)
    fig.clf()
    plt.close(fig)


# =========================
# TRAIN / VAL LOOPS
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    grad_scaler: torch.amp.GradScaler,
    multi_scales: list = None,
    use_sam: bool = True,
    grad_clip_max_norm: float = 1.0,
) -> dict:
    """Обучает модель одну эпоху с поддержкой AMP, SAM и multi-scale training.
    
    Args:
        model: Модель для обучения.
        loader: DataLoader с тренировочными данными.
        optimizer: Оптимизатор (SAM или обычный).
        loss_fn: Функция потерь.
        device: Устройство (cuda/cpu).
        epoch: Номер текущей эпохи.
        grad_scaler: GradScaler для AMP.
        multi_scales: Список размеров для multi-scale training.
        use_sam: Использовать ли SAM оптимизатор.
        grad_clip_max_norm: Максимальная норма градиента для clipping.
    """
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(loader, desc=f"  Train", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Multi-scale training: случайный размер на батч
        if multi_scales is not None and len(multi_scales) > 1:
            new_size = random.choice(multi_scales)
            if images.shape[-1] != new_size:
                images = F.interpolate(images, size=(new_size, new_size), 
                                      mode='bilinear', align_corners=False)
                masks = F.interpolate(masks, size=(new_size, new_size), mode='nearest')

        if use_sam and isinstance(optimizer, SAM):
            # SAM: первый forward + backward
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, masks)
            grad_scaler.scale(loss).backward()

            # Perturb
            inv_scale = 1.0 / grad_scaler.get_scale()
            is_finite = optimizer.perturb(inv_scale)

            if not is_finite:
                grad_scaler.unscale_(optimizer.base_optimizer)
                grad_scaler.step(optimizer.base_optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            # Второй forward с возмущёнными весами
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits_perturbed = model(images)
                loss_perturbed = loss_fn(logits_perturbed, masks)
            grad_scaler.scale(loss_perturbed).backward()

            # Откат возмущения
            optimizer.unperturb()

            # Gradient clipping + step
            grad_scaler.unscale_(optimizer.base_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            grad_scaler.step(optimizer.base_optimizer)
            grad_scaler.update()

            final_loss = loss_perturbed
            final_logits = logits_perturbed
        else:
            # Обычный training без SAM
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, masks)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            final_loss = loss
            final_logits = logits

        # Вычисление метрик
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits_float = final_logits.float()
            masks_float = masks.float()
            batch_loss = final_loss.item()
            batch_dice = dice_score_from_logits(logits_float.detach(), masks_float)
            batch_iou = iou_score_from_logits(logits_float.detach(), masks_float)

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
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> dict:
    """Валидирует модель одну эпоху без вычисления градиентов."""
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(loader, desc=f"  Val", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
            loss = loss_fn(logits, masks)

        logits_float = logits.float().detach()
        batch_loss = loss.item()
        batch_dice = dice_score_from_logits(logits_float, masks)
        batch_iou = iou_score_from_logits(logits_float, masks)

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
