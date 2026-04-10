import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    FOCAL_TVERSKY_ALPHA,
    FOCAL_TVERSKY_BETA,
    FOCAL_TVERSKY_GAMMA,
    BOUNDARY_SIGMA,
    OHEM_RATIO,
    OHEM_MIN_KEPT,
    # Warmup phase parameters
    WARMUP_BCE_WEIGHT,
    WARMUP_TVERSKY_WEIGHT,
    WARMUP_BOUNDARY_WEIGHT,
    WARMUP_FOCAL_TVERSKY_ALPHA,
    WARMUP_FOCAL_TVERSKY_BETA,
    # Main phase end values
    BCE_WEIGHT,
    TVERSKY_WEIGHT,
    BOUNDARY_WEIGHT,
)


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss — обобщение Dice с фокусировкой на сложных примерах."""

    def __init__(
        self,
        alpha: float = FOCAL_TVERSKY_ALPHA,
        beta: float = FOCAL_TVERSKY_BETA,
        gamma: float = FOCAL_TVERSKY_GAMMA,
    ):
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

    def __init__(self, sigma: float = BOUNDARY_SIGMA):
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
    """BCE Loss с OHEM (Online Hard Example Mining)."""

    def __init__(self, ohem_ratio: float = OHEM_RATIO, min_kept: int = OHEM_MIN_KEPT):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.min_kept = min_kept

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычисляет BCE Loss только на сложных примерах."""
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
    """Комбинированный лосс: BCE (с OHEM) + FocalTversky + Boundary.

    Поддерживает динамическую смену весов через метод update_weights().
    """

    def __init__(
        self,
        bce_weight: float = 0.4,
        tversky_weight: float = 0.5,
        boundary_weight: float = 0.1,
        use_ohem: bool = True,
        ohem_ratio: float = OHEM_RATIO,
    ):
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
        self.focal_tversky = FocalTverskyLoss()
        self.boundary_loss = BoundaryLoss()

    def update_weights(self, epoch: int, warmup_epochs: int, total_epochs: int) -> None:
        """Обновляет веса компонентов лосса в зависимости от эпохи.

        Args:
            epoch: Текущая эпоха (1-indexed).
            warmup_epochs: Количество эпох warmup.
            total_epochs: Общее количество эпох обучения.
        """
        # Ускоренный warmup: занимает половину от переданного warmup_epochs
        effective_warmup = max(1, warmup_epochs // 2)

        if epoch <= effective_warmup:
            # Warmup: больше BCE, симметричный Tversky
            self.bce_weight = WARMUP_BCE_WEIGHT
            self.tversky_weight = WARMUP_TVERSKY_WEIGHT
            self.focal_tversky.alpha = WARMUP_FOCAL_TVERSKY_ALPHA
            self.focal_tversky.beta = WARMUP_FOCAL_TVERSKY_BETA
            self.boundary_weight = WARMUP_BOUNDARY_WEIGHT
        else:
            # Плавный переход после warmup
            denominator = total_epochs - effective_warmup
            progress = (epoch - effective_warmup) / denominator if denominator > 0 else 1.0

            # BCE от WARMUP_BCE_WEIGHT до BCE_WEIGHT
            self.bce_weight = WARMUP_BCE_WEIGHT - (WARMUP_BCE_WEIGHT - BCE_WEIGHT) * progress
            # Tversky от WARMUP_TVERSKY_WEIGHT до TVERSKY_WEIGHT
            self.tversky_weight = WARMUP_TVERSKY_WEIGHT + (TVERSKY_WEIGHT - WARMUP_TVERSKY_WEIGHT) * progress
            # Boundary от WARMUP_BOUNDARY_WEIGHT до BOUNDARY_WEIGHT
            self.boundary_weight = WARMUP_BOUNDARY_WEIGHT + (BOUNDARY_WEIGHT - WARMUP_BOUNDARY_WEIGHT) * progress
            # FocalTversky alpha от WARMUP_FOCAL_TVERSKY_ALPHA до FOCAL_TVERSKY_ALPHA
            self.focal_tversky.alpha = WARMUP_FOCAL_TVERSKY_ALPHA - (WARMUP_FOCAL_TVERSKY_ALPHA - FOCAL_TVERSKY_ALPHA) * progress
            # FocalTversky beta от WARMUP_FOCAL_TVERSKY_BETA до FOCAL_TVERSKY_BETA
            self.focal_tversky.beta = WARMUP_FOCAL_TVERSKY_BETA + (FOCAL_TVERSKY_BETA - WARMUP_FOCAL_TVERSKY_BETA) * progress

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычисляет комбинированный лосс."""
        loss = 0.0

        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(logits, targets)

        if self.tversky_weight > 0:
            loss += self.tversky_weight * self.focal_tversky(logits, targets)

        if self.boundary_weight > 0:
            loss += self.boundary_weight * self.boundary_loss(logits, targets)

        return loss
