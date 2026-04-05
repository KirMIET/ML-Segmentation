import random
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from src.augmentations import create_coordinate_maps


def build_image_dict(images_dir: Path) -> dict:
    """Сканирует директорию один раз и строит {stem: path} словарь для быстрого поиска.
    
    Args:
        images_dir: Путь к директории с изображениями.
        
    Returns:
        Словарь вида {'image_name': Path('image_name.png'), ...}.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts}


class BinarySegDataset(torch.utils.data.Dataset):
    """Dataset для бинарной сегментации с поддержкой 5-канального входа.
    
    Поддерживает multi-scale training — размер выбирается случайно в __getitem__.
    Нормализация выполняется вручную после Albumentations для корректной работы SMP.
    """

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

        # Быстрый поиск изображений через словарь O(1)
        self.image_dict = build_image_dict(self.images_dir)

        self.preprocess_input = None
        if encoder_weights is not None:
            self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)

        # Если сэмплы переданы готовым списком - используем их
        if samples is not None:
            self.samples = samples
        else:
            # Иначе собираем сами через O(1) lookup
            self.samples = []
            for mask_path in sorted(self.masks_dir.glob("*.png")):
                stem = mask_path.stem
                image_path = self.image_dict.get(stem)
                if image_path is not None:
                    self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No paired image/mask samples found in {images_dir} / {masks_dir}")

        print(f"  Dataset initialized with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_normalization(self, image_rgb: np.ndarray) -> np.ndarray:
        """Применяет SMP preprocess или деление на 255."""
        if self.preprocess_input is not None:
            return self.preprocess_input(image_rgb)
        else:
            return image_rgb / 255.0

    def _get_single_sample(self, idx: int, target_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Вспомогательный метод: загружает 1 картинку, применяет аугментации и возвращает тензоры.
        
        Args:
            idx: Индекс сэмпла.
            target_size: Целевой размер для ресайза (multi-scale).
        """
        image_path, mask_path = self.samples[idx]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        # Resize через cv2 перед применением трансформов
        image_rgb = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        if self.augmentations is not None and len(self.augmentations) > 0:
            transformed = self.augmentations(image=image_rgb, mask=mask)
            image_rgb = transformed["image"]
            mask = transformed["mask"]

        # Нормализация строго после Albumentations
        image_rgb = self._apply_normalization(image_rgb)

        # Ручная конвертация в тензор
        image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)[None, ...]).float()

        # Добавляем координатные каналы
        if self.use_coordinates:
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            x_map, y_map = create_coordinate_maps(h, w)
            coords = np.stack([x_map, y_map], axis=0)
            coords_tensor = torch.from_numpy(coords).float()
            image_tensor = torch.cat([image_tensor, coords_tensor], dim=0)

        return image_tensor, mask_tensor

    def __getitem__(self, idx: int):
        # Выбираем случайный размер для multi-scale training
        target_size = self.img_size  # по умолчанию

        image_tensor, mask_tensor = self._get_single_sample(idx, target_size)

        if self.custom_augs is not None:
            random_idx = random.randint(0, len(self.samples) - 1)
            source_image, source_mask = self._get_single_sample(random_idx, target_size)

            image_tensor, mask_tensor = self.custom_augs(
                image=image_tensor,
                mask=mask_tensor,
                source_image=source_image,
                source_mask=source_mask
            )

        return image_tensor.contiguous(), mask_tensor.contiguous()
