import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """BCE Loss с OHEM (Online Hard Example Mining)."""

    def __init__(self, ohem_ratio: float = 0.25, min_kept: int = 10000):
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

    def update_weights(self, epoch: int, warmup_epochs: int, total_epochs: int) -> None:
        """Обновляет веса компонентов лосса в зависимости от эпохи.
        
        Args:
            epoch: Текущая эпоха (1-indexed).
            warmup_epochs: Количество эпох warmup.
            total_epochs: Общее количество эпох обучения.
        """
        if epoch <= warmup_epochs:
            # Warmup: больше BCE, симметричный Tversky
            self.bce_weight = 0.7
            self.tversky_weight = 0.3
            self.focal_tversky.alpha = 0.5
            self.focal_tversky.beta = 0.5
            self.boundary_weight = 0.0
        else:
            # Плавный переход после warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            
            # BCE от 0.7 до 0.4
            self.bce_weight = 0.7 - (0.3 * progress)
            # Tversky от 0.3 до 0.5
            self.tversky_weight = 0.3 + (0.2 * progress)
            # Boundary от 0.0 до 0.1
            self.boundary_weight = 0.1 * progress
            # FocalTversky alpha от 0.5 до 0.3, beta от 0.5 до 0.7
            self.focal_tversky.alpha = 0.5 - (0.2 * progress)
            self.focal_tversky.beta = 0.5 + (0.2 * progress)

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
