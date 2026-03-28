import random
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def get_random_lambda(alpha: float = 0.4) -> float:
    """Генерирует случайный коэффициент λ из Beta(α, α) распределения для MixUp/CutMix."""
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
    # Переводим в uint8 (0 и 1)
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

    if mask_np.sum() == 0:
        return objects

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_np, connectivity=8
    )

    for i in range(1, num_labels):  # Пропускаем фон (label 0)
        x, y, w, h, area = stats[i]
        if area < 100:  # Пропускаем слишком мелкие объекты
            continue

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[2], x + w), min(image.shape[1], y + h)

        # Вырезаем изображение
        obj_image = image[:, y1:y2, x1:x2].clone()

        # ИСПРАВЛЕНИЕ: Создаем маску ИМЕННО ЭТОГО объекта (где label == i)
        component_mask = (labels[y1:y2, x1:x2] == i).astype(np.float32)
        obj_mask = torch.from_numpy(component_mask).unsqueeze(0).to(image.device)

        objects.append((obj_image, obj_mask, (x1, y1, x2, y2)))

    return objects

class MixUpSegmentation:
    """MixUp аугментация для бинарной сегментации с линейной интерполяцией изображений и масок."""

    def __init__(self, alpha: float = 0.4, apply_prob: float = 0.5):
        """Инициализирует MixUp с параметром alpha и вероятностью применения."""
        self.alpha = alpha
        self.apply_prob = apply_prob

    def __call__(self, image1: torch.Tensor, mask1: torch.Tensor, image2: torch.Tensor, mask2: torch.Tensor):
        """Применяет MixUp к паре изображений и масок, возвращая смешанные тензоры."""
        if random.random() > self.apply_prob:
            return image1, mask1

        lambda_val = get_random_lambda(self.alpha)
        mixed_image = lambda_val * image1 + (1 - lambda_val) * image2
        mixed_mask = lambda_val * mask1 + (1 - lambda_val) * mask2

        return mixed_image, mixed_mask

class CutMixSegmentation:
    """CutMix аугментация для бинарной сегментации с вырезанием и вставкой прямоугольных областей."""

    def __init__(self, alpha: float = 1.0, apply_prob: float = 0.5):
        """Инициализирует CutMix с параметром alpha и вероятностью применения."""
        self.alpha = alpha
        self.apply_prob = apply_prob

    def __call__(self, image1: torch.Tensor, mask1: torch.Tensor, image2: torch.Tensor, mask2: torch.Tensor):
        """Применяет CutMix к паре изображений и масок, вставляя область из одного в другое."""
        if random.random() > self.apply_prob:
            return image1, mask1

        lambda_val = get_random_lambda(self.alpha)
        h, w = image1.shape[1], image1.shape[2]
        x1, y1, x2, y2 = get_random_bbox(h, w, area_ratio=1 - lambda_val)

        mixed_image = image1.clone()
        mixed_mask = mask1.clone()

        mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        mixed_mask[:, y1:y2, x1:x2] = mask2[:, y1:y2, x1:x2]

        return mixed_image, mixed_mask

class CopyPasteSegmentation:
    """CopyPaste аугментация для сегментации товаров — копирование объектов из одного изображения в другое."""

    def __init__(
        self,
        dataset_samples: Optional[List] = None,
        max_objects: int = 3,
        apply_prob: float = 0.5,
        encoder_name: str = "timm-efficientnet-b0",
        encoder_weights: str = "noisy-student",
    ):
        """Инициализирует CopyPaste с параметрами датасета, максимальным числом объектов и вероятностью."""
        self.dataset_samples = dataset_samples
        self.max_objects = max_objects
        self.apply_prob = apply_prob
        self.preprocess_input = None

        if encoder_weights is not None:
            self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Загружает и предобрабатывает сэмпл из датасета по индексу."""
        import cv2
        image_path, mask_path = self.dataset_samples[idx]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        if self.preprocess_input is not None:
            image_rgb = self.preprocess_input(image_rgb)
        else:
            image_rgb = image_rgb.astype(np.float32) / 255.0

        image = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask[None, ...]).float()
        return image, mask

    def _paste_object(
        self, target_image: torch.Tensor, target_mask: torch.Tensor,
        obj_image: torch.Tensor, obj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вставляет объект в случайную позицию целевого изображения со случайным скейлингом."""

        obj_h, obj_w = obj_image.shape[1], obj_image.shape[2]
        tgt_h, tgt_w = target_image.shape[1], target_image.shape[2]

        # ИСПРАВЛЕНИЕ: Добавляем случайный скейлинг для разнообразия размеров (важно для кассы!)
        scale_factor = random.uniform(0.6, 1.2)
        new_h, new_w = int(obj_h * scale_factor), int(obj_w * scale_factor)

        # Защита от выхода за границы
        if new_h > tgt_h or new_w > tgt_w:
            scale = min(tgt_h / obj_h, tgt_w / obj_w)
            new_h, new_w = int(obj_h * scale), int(obj_w * scale)

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

        result_image[:, y1:y2, x1:x2] = (
            obj_image * obj_mask_binary + result_image[:, y1:y2, x1:x2] * (1 - obj_mask_binary)
        )
        result_mask[:, y1:y2, x1:x2] = torch.maximum(result_mask[:, y1:y2, x1:x2], obj_mask)

        return result_image, result_mask

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Применяет CopyPaste — копирует 1-3 объекта из случайных сэмплов датасета в целевое изображение."""
        if random.random() > self.apply_prob or not self.dataset_samples:
            return image, mask

        num_objects = random.randint(1, self.max_objects)
        # Ограничиваем количество попыток, чтобы не зависнуть, если попались пустые картинки
        source_indices = random.sample(range(len(self.dataset_samples)), min(num_objects + 2, len(self.dataset_samples)))

        result_image, result_mask = image.clone(), mask.clone()
        pasted_count = 0

        for src_idx in source_indices:
            if pasted_count >= num_objects:
                break
            try:
                src_image, src_mask = self._load_sample(src_idx)
                objects = extract_objects_from_mask(src_image, src_mask)

                if not objects:
                    continue

                obj_image, obj_mask, _ = random.choice(objects)
                result_image, result_mask = self._paste_object(result_image, result_mask, obj_image, obj_mask)
                pasted_count += 1
            except Exception:
                continue

        return result_image, result_mask


class SegmentationAugmentations:
    """Комбинирует MixUp, CutMix и CopyPaste для удобного применения в Dataset."""

    def __init__(self, dataset_samples: Optional[List] = None, apply_prob: float = 0.5, **kwargs):
        """Инициализирует комбинированные аугментации с параметрами для каждой из них."""
        self.mixup = MixUpSegmentation(alpha=kwargs.get("mixup_alpha", 0.4), apply_prob=1.0)
        self.cutmix = CutMixSegmentation(alpha=kwargs.get("cutmix_alpha", 1.0), apply_prob=1.0)
        self.copypaste = CopyPasteSegmentation(
            dataset_samples=dataset_samples,
            max_objects=kwargs.get("copypaste_max_objects", 3),
            apply_prob=1.0,  # Контролируем вероятность внутри этого класса
            encoder_name=kwargs.get("encoder_name", "timm-efficientnet-b0"),
            encoder_weights=kwargs.get("encoder_weights", "noisy-student"),
        )
        self.apply_prob = apply_prob

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor,
        source_image: Optional[torch.Tensor] = None, source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Применяет случайную аугментацию (MixUp/CutMix/CopyPaste) к изображению и маске."""

        # Если не повезло - отдаем оригинал
        if random.random() > self.apply_prob:
            return image, mask

        # Собираем доступные опции
        choices = ["copypaste"]
        if source_image is not None and source_mask is not None:
            choices.extend(["mixup", "cutmix"])

        # Выбираем строго ОДНУ аугментацию
        aug_choice = random.choice(choices)

        if aug_choice == "mixup":
            return self.mixup(image, mask, source_image, source_mask)
        elif aug_choice == "cutmix":
            return self.cutmix(image, mask, source_image, source_mask)
        elif aug_choice == "copypaste":
            return self.copypaste(image, mask)

        return image, mask