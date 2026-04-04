import os
import random
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.mixup_augmentations import SegmentationAugmentations

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

# Визуализация предсказаний
VISUALIZATION_DIR = Path("./view_train_img")
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZE_EVERY_N_EPOCHS = 1

IMG_SIZE = 352
MULTI_SCALES = [256, 288, 352]
BATCH_SIZE = 16

NUM_EPOCHS = 20
WARMUP_EPOCHS = 4

LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 4
SEED = 42
THRESHOLD = 0.4

USE_COORDINATES = True
INPUT_CHANNELS = 5 if USE_COORDINATES else 3

# OHEM parameters
USE_OHEM = True
OHEM_RATIO = 0.25  # Keep top 25% hardest pixels

MODEL_NAME = "UnetPlusPlus"
ENCODER_NAME = "timm-efficientnet-b0"
ENCODER_WEIGHTS = "noisy-student"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры для early stopping
EARLY_STOPPING_PATIENCE = 5

# Параметры для gradient clipping
GRAD_CLIP_MAX_NORM = 1.0

# Параметры для CutMix
CUTMIX_ALPHA = 1.0

# SAM
USE_SAM = True
SAM_RHO = 0.02  


# =========================
# SAM OPTIMIZER
# =========================
class SAM:
    """SAM optimizer wrapper над базовым оптимизатором, совместимый с AMP GradScaler."""

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
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

        scale = self.rho / (true_grad_norm + 1e-12)

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
def seed_everything(seed: int = 42) -> None:
    """Устанавливает случайные сиды для воспроизводимости экспериментов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    """Ищет файл изображения по имени (stem) с различными расширениями."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    """Вычисляет Dice score между предсказанными логитами и целевыми масками."""
    probs = torch.sigmoid(logits)
    preds = (probs > THRESHOLD).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean().item()


def iou_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    """Вычисляет IoU score между предсказанными логитами и целевыми масками."""
    probs = torch.sigmoid(logits)
    preds = (probs > THRESHOLD).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def visualize_batch(images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor,
                    save_path: Path, epoch: int, num_samples: int = 4) -> None:
    """Сохраняет визуализацию предсказаний модели для батча."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    probs = torch.sigmoid(logits)
    preds = (probs > THRESHOLD).float()

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
        axes[i, 2].set_title(f"Prediction (Epoch {epoch})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path / f"epoch_{epoch:03d}.png", dpi=100)
    fig.clf()
    plt.close(fig)


# =========================
# AUGMENTATIONS
# =========================
def get_train_transforms(img_size: int | None = None) -> A.Compose:
    """Создаёт pipeline аугментаций для обучения с учётом специфики товаров на кассе.
    
    Args:
        img_size: Размер изображения. Если None, ресайз не применяется (для multi-scale).
    """
    transforms = [
        # Геометрические трансформации
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-10, 10),
            p=0.5,
        ),

        # Оптические искажения (уменьшены на 25%)
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=22.5, val_shift_limit=15, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

        # Размытие (движение товаров)
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        # A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.3),  # уменьшено на 25%
        # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.075, 0.375), p=0.3),  # уменьшено на 25%

        # Морфологические операции
        A.OneOf(
            [
                A.Sharpen(p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ],
            p=0.3,
        ),

        # Random Scale
        A.RandomScale(scale_limit=(-0.2, 0.2), p=0.5),
    ]

    # Добавляем Resize только если указан размер
    if img_size is not None:
        transforms.append(A.Resize(height=img_size, width=img_size))

    # Нормализация ImageNet
    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return A.Compose(transforms, is_check_shapes=False)


def get_val_transforms(img_size: int = 352) -> A.Compose:
    """Создаёт pipeline аугментаций для валидации (только ресайз и нормализация)."""
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


def create_coordinate_maps(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт нормализованные координатные карты X и Y."""
    x_coords = np.linspace(0, 1, width, dtype=np.float32)
    y_coords = np.linspace(0, 1, height, dtype=np.float32)
    x_map, y_map = np.meshgrid(x_coords, y_coords)
    return x_map.copy(), y_map.copy()


# =========================
# DATASET
# =========================
class BinarySegDataset(Dataset):
    """Dataset для бинарной сегментации с поддержкой Albumentations и кастомных тензорных аугментаций."""

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        img_size: int = 352,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        augmentations: A.Compose | None = None,
        samples: list | None = None,
        use_coordinates: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augmentations = augmentations
        self.custom_augs = None
        self.use_coordinates = use_coordinates

        self.preprocess_input = None
        if encoder_weights is not None:
            self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)

        # Если сэмплы переданы готовым списком - используем их
        if samples is not None:
            self.samples = samples
        else:
            # Иначе собираем сами
            self.samples = []
            for mask_path in sorted(self.masks_dir.glob("*.png")):
                stem = mask_path.stem
                image_path = find_image_for_stem(self.images_dir, stem)
                if image_path is not None:
                    self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No paired image/mask samples found")

        print(f"Dataset initialized with {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)

    def _get_single_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Вспомогательный метод: загружает 1 картинку, применяет Albumentations и возвращает тензоры."""
        image_path, mask_path = self.samples[idx]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        if self.augmentations is not None:
            # Ресайзим изображение и маску перед применением аугментаций
            image_rgb = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            transformed = self.augmentations(image=image_rgb, mask=mask)
            image_rgb = transformed["image"]
            mask = transformed["mask"]

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()

            # добавляем координатные каналы после аугментаций
            if self.use_coordinates:
                h, w = image_rgb.shape[1], image_rgb.shape[2]
                x_map, y_map = create_coordinate_maps(h, w)
                coords = np.stack([x_map, y_map], axis=0)
                coords = torch.from_numpy(coords).float()
                image_rgb = torch.cat([image_rgb, coords], dim=0)

        else:
            image_rgb = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            image_rgb = image_rgb.astype(np.float32)

            if self.preprocess_input is not None:
                image_rgb = self.preprocess_input(image_rgb)
            else:
                image_rgb = image_rgb / 255.0

            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.float32)

            image_rgb = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()

            # добавляем координатные каналы
            if self.use_coordinates:
                x_map, y_map = create_coordinate_maps(self.img_size, self.img_size)
                coords = np.stack([x_map, y_map], axis=0)
                coords = torch.from_numpy(coords).float()
                image_rgb = torch.cat([image_rgb, coords], dim=0)

            mask = torch.from_numpy(mask[None, ...]).float()

        return image_rgb, mask

    def __getitem__(self, idx: int):
        image_rgb, mask = self._get_single_sample(idx)

        if self.custom_augs is not None:
            random_idx = random.randint(0, len(self.samples) - 1)
            source_image, source_mask = self._get_single_sample(random_idx)

            image_rgb, mask = self.custom_augs(
                image=image_rgb,
                mask=mask,
                source_image=source_image,
                source_mask=source_mask
            )

        return image_rgb.contiguous(), mask.contiguous()


# =========================
# MODEL
# =========================
def build_model() -> nn.Module:
    """Создаёт модель сегментации на основе конфигурации."""
    if MODEL_NAME == "Unet":
        model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=INPUT_CHANNELS,  
            classes=1,
            activation=None,
        )
    elif MODEL_NAME == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=INPUT_CHANNELS, 
            classes=1,
            activation=None,
        )
    elif MODEL_NAME == "FPN":
        model = smp.FPN(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=INPUT_CHANNELS,  
            classes=1,
            activation=None,
        )
    elif MODEL_NAME == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=INPUT_CHANNELS, 
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
    return model


# =========================
# LOSS FUNCTIONS
# =========================
class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss — обобщение Dice с фокусировкой на сложных примерах."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha  # вес для False Positive
        self.beta = beta    # вес для False Negative
        self.gamma = gamma  # параметр фокусировки

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычисляет Focal Tversky Loss между логитами и масками."""
        probs = torch.sigmoid(logits)

        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        tversky = (TP + 1e-7) / (TP + self.alpha * FP + self.beta * FN + 1e-7)
        return torch.pow(1 - tversky, self.gamma)


class BoundaryLoss(nn.Module):
    """Boundary Loss — штрафует ошибки на границах объектов через Sobel-фильтр."""

    def __init__(self, sigma: float = 2):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычисляет Boundary Dice Loss, выделяя границы через Sobel-оператор."""
        probs = torch.sigmoid(logits / 0.1)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               device=probs.device).view(1, 1, 3, 3).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               device=probs.device).view(1, 1, 3, 3).float()

        pred_edge = F.conv2d(probs, sobel_x, padding=1).abs() + \
                    F.conv2d(probs, sobel_y, padding=1).abs()
        target_edge = F.conv2d(targets, sobel_x, padding=1).abs() + \
                      F.conv2d(targets, sobel_y, padding=1).abs()

        inter = (pred_edge * target_edge).sum()
        union = pred_edge.sum() + target_edge.sum()
        return 1 - (2 * inter + 1e-7) / (union + 1e-7)


class OHEMBCEWithLogitsLoss(nn.Module):
    """BCE Loss с OHEM """

    def __init__(self, ohem_ratio: float = 0.25, min_kept: int = 10000):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.min_kept = min_kept 

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_per_pixel = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        loss_flat = loss_per_pixel.view(-1)

        sorted_loss, _ = torch.sort(loss_flat, descending=True)

        num_pixels = loss_flat.size(0)
        num_kept = max(
            int(num_pixels * self.ohem_ratio),
            self.min_kept
        )
        num_kept = min(num_kept, num_pixels)

        kept_loss = sorted_loss[:num_kept]

        return kept_loss.mean()


class CombinedLoss(nn.Module):
    """Комбинированный лосс: BCE (с OHEM) + FocalTversky + Boundary"""

    def __init__(self,
                 bce_weight: float = 0.4,
                 tversky_weight: float = 0.5,
                 boundary_weight: float = 0.1,
                 use_ohem: bool = True,
                 ohem_ratio: float = 0.25):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight

        # BCE с OHEM или обычный
        if use_ohem:
            self.bce_loss = OHEMBCEWithLogitsLoss(ohem_ratio=ohem_ratio)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

        # alpha=0.3 (штраф за FP - захват фона), beta=0.7 (штраф за FN - пропуск товара)
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.boundary_loss = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0

        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(logits, targets)

        if self.tversky_weight > 0:
            loss += self.tversky_weight * self.focal_tversky(logits, targets)

        if self.boundary_weight > 0:
            loss += self.boundary_weight * self.boundary_loss(logits, targets)
            
        return loss


# =========================
# TRAIN / VAL LOOPS
# =========================
def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    epoch,
    num_epochs,
    grad_scaler,
    visualization_dir: Path | None = None,
    visualize_every_n: int = 1,
    use_sam: bool = False,
):
    """Обучает модель одну эпоху с поддержкой AMP и CutMix."""
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    saved_images = False
    saved_batch = None

    pbar = tqdm(loader, desc=f"Train", leave=False)

    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        new_size = random.choice(MULTI_SCALES)
        if images.shape[-1] != new_size:
            images = F.interpolate(images, size=(new_size, new_size), mode='bilinear', align_corners=False)
            masks = F.interpolate(masks, size=(new_size, new_size), mode='nearest')

        if use_sam:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, masks)
            grad_scaler.scale(loss).backward()

            inv_scale = 1.0 / grad_scaler.get_scale()

            is_finite = optimizer.perturb(inv_scale)

            if not is_finite:
                grad_scaler.unscale_(optimizer.base_optimizer)
                grad_scaler.step(optimizer.base_optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits_perturbed = model(images)
                loss_perturbed = loss_fn(logits_perturbed, masks)
            grad_scaler.scale(loss_perturbed).backward()

            optimizer.unperturb()

            grad_scaler.unscale_(optimizer.base_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)

            grad_scaler.step(optimizer.base_optimizer)
            grad_scaler.update()

            final_loss = loss_perturbed
            final_logits = logits_perturbed
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, masks)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            final_loss = loss
            final_logits = logits

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits_float = final_logits.float()
            masks_float = masks.float()
            batch_loss = final_loss.item()
            batch_dice = dice_score_from_logits(logits_float.detach(), masks_float)
            batch_iou = iou_score_from_logits(logits_float.detach(), masks_float)

        running_loss += batch_loss
        running_dice += batch_dice
        running_iou += batch_iou

        if not saved_images and visualization_dir is not None and epoch % visualize_every_n == 0:
            saved_batch = (images.clone(), masks.clone(), logits_float.detach().clone())
            saved_images = True

        pbar.set_postfix(loss=f"{batch_loss:.4f}", dice=f"{batch_dice:.4f}")

    n = len(loader)
    metrics = {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "iou": running_iou / n,
    }
    
    if saved_batch is not None and visualization_dir is not None and epoch % visualize_every_n == 0:
        imgs, msks, lgts = saved_batch
        visualize_batch(imgs, msks, lgts, visualization_dir, epoch, num_samples=4)

    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device, epoch, num_epochs):
    """Валидирует модель одну эпоху без вычисления градиентов."""
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(loader, desc=f"Val", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
            loss = loss_fn(logits, masks)

        logits_float = logits.float()
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


# =========================
# MAIN
# =========================
def main():
    """Запускает полный цикл обучения модели с улучшениями."""
    seed_everything(SEED)

    # Multi-scale training: для train_transforms не указываем размер (будет выбираться случайно)
    train_transforms = get_train_transforms(img_size=IMG_SIZE)
    val_transforms = get_val_transforms(IMG_SIZE)

    # Сначала просто собираем все пары файлов
    all_samples = []
    for mask_path in sorted(MASKS_DIR.glob("*.png")):
        stem = mask_path.stem
        image_path = find_image_for_stem(IMAGES_DIR, stem)
        if image_path is not None:
            all_samples.append((image_path, mask_path))

    # Перемешиваем и делим списки файлов
    random.Random(SEED).shuffle(all_samples)
    val_size = int(len(all_samples) * VAL_RATIO)

    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    print(f"Total: {len(all_samples)} | Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_dataset = BinarySegDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        img_size=IMG_SIZE,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        augmentations=train_transforms,
        samples=train_samples,
        use_coordinates=USE_COORDINATES,
    )

    train_dataset.custom_augs = SegmentationAugmentations(
        dataset_samples=train_samples,
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
        copypaste_max_objects=3,
        apply_prob=0.5,
    )

    val_dataset = BinarySegDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        img_size=IMG_SIZE,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        augmentations=val_transforms,
        samples=val_samples,
        use_coordinates=USE_COORDINATES,
    )

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

    loss_fn = CombinedLoss(
        bce_weight=0.4,
        tversky_weight=0.6,
        boundary_weight=0.0,
        use_ohem=USE_OHEM,
        ohem_ratio=OHEM_RATIO
    ).to(DEVICE)

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = SAM(base_optimizer, rho=SAM_RHO) if USE_SAM else base_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer if USE_SAM else optimizer, T_max=NUM_EPOCHS
    )

    grad_scaler = torch.amp.GradScaler()

    best_val_dice = -1.0
    history = []

    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        
        if epoch <= WARMUP_EPOCHS:
            loss_fn.bce_weight = 0.7       
            loss_fn.tversky_weight = 0.3   
            loss_fn.focal_tversky.alpha = 0.5 
            loss_fn.focal_tversky.beta = 0.5
            loss_fn.boundary_weight = 0.0 
            print(f"Epoch {epoch}: BCE=0.7, Tversky(0.5,0.5)=0.3, Boundary=0.0")
        else:
            loss_fn.bce_weight = 0.4
            loss_fn.tversky_weight = 0.5
            loss_fn.focal_tversky.alpha = 0.3 
            loss_fn.focal_tversky.beta = 0.7  
            loss_fn.boundary_weight = 0.1  
            print(f"Epoch {epoch}: BCE=0.4, Tversky(0.3,0.7)=0.5, Boundary=0.1")

        # Запускаем обучение с уже обновленными весами внутри loss_fn
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, DEVICE, epoch, NUM_EPOCHS, grad_scaler,
            visualization_dir=VISUALIZATION_DIR,
            visualize_every_n=VISUALIZE_EVERY_N_EPOCHS,
            use_sam=USE_SAM,
        )

        val_metrics = validate_one_epoch(model, val_loader, loss_fn, DEVICE, epoch, NUM_EPOCHS)

        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.base_optimizer.param_groups[0]["lr"] if USE_SAM else optimizer.param_groups[0]["lr"],
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

        # Сохранение last
        optimizer_to_save = optimizer.base_optimizer if USE_SAM else optimizer
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_to_save.state_dict(),
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

        # Сохранение best и проверка early stopping
        if row["val_dice"] > best_val_dice:
            best_val_dice = row["val_dice"]
            epochs_without_improvement = 0  # Сброс счётчика
            optimizer_to_save = optimizer.base_optimizer if USE_SAM else optimizer
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_to_save.state_dict(),
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
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping проверка
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs. Best val_dice={best_val_dice:.4f}")
            break

    # Сохранение истории
    import pandas as pd

    history_df = pd.DataFrame(history)
    history_df.to_csv(SAVE_DIR / "history.csv", index=False)
    print(f"Training finished. Best val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()
