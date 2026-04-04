import random
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def get_random_lambda(alpha: float = 0.4) -> float:
    """Генерирует случайный коэффициент λ из Beta(α, α) распределения для CutMix."""
    if alpha <= 0:
        return 1.0
    return np.random.beta(alpha, alpha)


def get_random_bbox(h: int, w: int, area_ratio: float = 0.3) -> Tuple[int, int, int, int]:
    """Генерирует случайную bounding box заданной площади относительно изображения для CutMix."""
    target_area = h * w * area_ratio
    aspect_ratio = random.uniform(0.5, 2.0)

    obj_h = int(np.sqrt(target_area * aspect_ratio))
    obj_w = int(np.sqrt(target_area / aspect_ratio))

    obj_h = min(obj_h, h)
    obj_w = min(obj_w, w)

    y1 = random.randint(0, h - obj_h)
    x1 = random.randint(0, w - obj_w)

    return x1, y1, x1 + obj_w, y1 + obj_h


def extract_objects_from_mask(
    image: torch.Tensor, mask: torch.Tensor
) -> List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]]:
    """Извлекает отдельные объекты (связные компоненты) из изображения по маске для CopyPaste."""
    objects = []
    # Переводим маску в numpy для поиска компонент
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

    if mask_np.sum() == 0:
        return objects

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_np, connectivity=8
    )

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 50:  # Игнорируем совсем мелкий шум
            continue

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[2], x + w), min(image.shape[1], y + h)

        obj_image = image[:, y1:y2, x1:x2].clone()
        component_mask = (labels[y1:y2, x1:x2] == i).astype(np.float32)
        obj_mask = torch.from_numpy(component_mask).unsqueeze(0).to(image.device)

        objects.append((obj_image, obj_mask, (x1, y1, x2, y2)))

    return objects


class CutMixSegmentation:
    """CutMix аугментация для бинарной сегментации с вырезанием и вставкой прямоугольных областей."""

    def __init__(self, alpha: float = 1.0, apply_prob: float = 0.5):
        self.alpha = alpha
        self.apply_prob = apply_prob

    def __call__(self, image1: torch.Tensor, mask1: torch.Tensor, image2: torch.Tensor, mask2: torch.Tensor):
        if random.random() > self.apply_prob:
            return image1, mask1

        lambda_val = get_random_lambda(self.alpha)
        h, w = image1.shape[1], image1.shape[2]
        x1, y1, x2, y2 = get_random_bbox(h, w, area_ratio=1 - lambda_val)

        mixed_image = image1.clone()
        mixed_mask = mask1.clone()

        # Копируем ТОЛЬКО RGB каналы. 
        # Координатные каналы (если есть) остаются от image1, так как пространственное 
        # положение этих пикселей в итоговом тензоре не изменилось.
        mixed_image[:3, y1:y2, x1:x2] = image2[:3, y1:y2, x1:x2]
        mixed_mask[:, y1:y2, x1:x2] = mask2[:, y1:y2, x1:x2]

        return mixed_image, mixed_mask


class CopyPasteSegmentation:
    """CopyPaste аугментация для сегментации товаров — копирование объектов из одного изображения в другое."""

    def __init__(self, max_objects: int = 3, apply_prob: float = 0.5):
        self.max_objects = max_objects
        self.apply_prob = apply_prob

    def _paste_object(
        self, target_image: torch.Tensor, target_mask: torch.Tensor,
        obj_image: torch.Tensor, obj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вставляет объект в случайную позицию целевого изображения со случайным скейлингом."""
        obj_h, obj_w = obj_image.shape[1], obj_image.shape[2]
        tgt_h, tgt_w = target_image.shape[1], target_image.shape[2]

        scale_factor = random.uniform(0.6, 1.2)
        new_h, new_w = int(obj_h * scale_factor), int(obj_w * scale_factor)

        if new_h > tgt_h or new_w > tgt_w:
            scale = min(tgt_h / obj_h, tgt_w / obj_w)
            new_h, new_w = int(obj_h * scale), int(obj_w * scale)

        # Ресайзим объект (работает для любого кол-ва каналов)
        obj_image = F.interpolate(obj_image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        obj_mask = F.interpolate(obj_mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
        obj_h, obj_w = new_h, new_w

        max_y = tgt_h - obj_h
        max_x = tgt_w - obj_w

        paste_y = random.randint(0, max_y) if max_y > 0 else 0
        paste_x = random.randint(0, max_x) if max_x > 0 else 0

        result_image = target_image.clone()
        result_mask = target_mask.clone()

        y1, y2 = paste_y, paste_y + obj_h
        x1, x2 = paste_x, paste_x + obj_w

        obj_mask_binary = (obj_mask > 0.5).float()

        # Копируем только RGB каналы. Координаты (каналы 3, 4) остаются от target_image
        result_image[:3, y1:y2, x1:x2] = (
            obj_image[:3] * obj_mask_binary + result_image[:3, y1:y2, x1:x2] * (1 - obj_mask_binary)
        )
        result_mask[:, y1:y2, x1:x2] = torch.maximum(result_mask[:, y1:y2, x1:x2], obj_mask)

        return result_image, result_mask

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor, 
        source_image: torch.Tensor, source_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Копирует объекты из source_image и вставляет в image."""
        if random.random() > self.apply_prob:
            return image, mask

        # Извлекаем объекты из уже аугментированного source_image
        objects = extract_objects_from_mask(source_image, source_mask)
        if not objects:
            return image, mask

        # Выбираем случайные объекты для вставки
        random.shuffle(objects)
        num_objects_to_paste = min(random.randint(1, self.max_objects), len(objects))

        result_image, result_mask = image.clone(), mask.clone()

        for i in range(num_objects_to_paste):
            obj_image, obj_mask, _ = objects[i]
            result_image, result_mask = self._paste_object(result_image, result_mask, obj_image, obj_mask)

        return result_image, result_mask


class SegmentationAugmentations:
    """Комбинирует CutMix и CopyPaste для удобного применения в Dataset."""

    def __init__(self, apply_prob: float = 0.5, **kwargs):
        self.apply_prob = apply_prob
        self.cutmix = CutMixSegmentation(alpha=kwargs.get("cutmix_alpha", 1.0), apply_prob=1.0)
        self.copypaste = CopyPasteSegmentation(max_objects=kwargs.get("copypaste_max_objects", 3), apply_prob=1.0)

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor,
        source_image: Optional[torch.Tensor] = None, source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if random.random() > self.apply_prob:
            return image, mask

        # Если доп. изображения нет - аугментации применить невозможно
        if source_image is None or source_mask is None:
            return image, mask

        # Выбираем 1 из 2 методов с равной вероятностью
        aug_choice = random.choice(["cutmix", "copypaste"])

        if aug_choice == "cutmix":
            return self.cutmix(image, mask, source_image, source_mask)
        else:
            return self.copypaste(image, mask, source_image, source_mask)