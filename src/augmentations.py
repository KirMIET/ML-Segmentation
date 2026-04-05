import cv2
import numpy as np
import albumentations as A


def get_train_transforms() -> A.Compose:
    """Создаёт pipeline аугментаций для обучения.
    
    НЕ включает Normalize и ToTensor — они выполняются вручную в dataset.py
    после Albumentations для корректной работы SMP preprocessing.
    """
    return A.Compose(
        [
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

            # Оптические искажения
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=22.5, val_shift_limit=15, p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

            # Размытие и шум
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(5, 25), p=0.3),

            # Морфологические операции
            A.OneOf(
                [
                    A.Sharpen(p=1.0),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ],
                p=0.3,
            ),

            # Random Scale
            # A.RandomScale(scale_limit=(-0.2, 0.2), p=0.5),
        ],
        is_check_shapes=False,
    )


def get_val_transforms() -> A.Compose:
    """Создаёт pipeline аугментаций для валидации.
    
    НЕ включает Normalize и ToTensor — они выполняются вручную в dataset.py
    после Albumentations для корректной работы SMP preprocessing.
    """
    return A.Compose([], is_check_shapes=False)


def create_coordinate_maps(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт нормализованные координатные карты X и Y.
    
    Args:
        height: Высота изображения.
        width: Ширина изображения.
        
    Returns:
        Кортеж (x_map, y_map) — нормализованные координатные карты [0, 1].
    """
    x_coords = np.linspace(0, 1, width, dtype=np.float32)
    y_coords = np.linspace(0, 1, height, dtype=np.float32)
    x_map, y_map = np.meshgrid(x_coords, y_coords)
    return x_map.copy(), y_map.copy()
