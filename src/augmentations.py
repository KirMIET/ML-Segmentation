import cv2
import numpy as np
import albumentations as A

from src.config import (
    HORIZONTAL_FLIP_P,
    VERTICAL_FLIP_P,
    RANDOM_ROTATE_90_P,
    ROTATE_LIMIT,
    ROTATE_P,
    AFFINE_SCALE,
    AFFINE_TRANSLATE_PERCENT,
    AFFINE_ROTATE,
    AFFINE_SHEAR,
    AFFINE_P,
    BRIGHTNESS_CONTRAST_LIMIT,
    BRIGHTNESS_CONTRAST_P,
    HUE_SHIFT_LIMIT,
    SAT_SHIFT_LIMIT,
    VAL_SHIFT_LIMIT,
    HUE_SATURATION_VALUE_P,
    RGB_SHIFT_LIMIT,
    RGB_SHIFT_P,
    GAUSSIAN_BLUR_LIMIT,
    GAUSSIAN_BLUR_P,
    GAUSS_NOISE_VAR_LIMIT,
    GAUSS_NOISE_P,
    SHARPEN_P,
    CLAHE_CLIP_LIMIT,
    CLAHE_P,
    ONE_OF_MORPH_P,
    RANDOM_SCALE_LIMIT,
    RANDOM_SCALE_P,
)


def get_train_transforms() -> A.Compose:
    """Создаёт pipeline аугментаций для обучения.

    НЕ включает Normalize и ToTensor — они выполняются вручную в dataset.py
    после Albumentations для корректной работы SMP preprocessing.
    """
    return A.Compose(
        [
            # Геометрические трансформации
            A.HorizontalFlip(p=HORIZONTAL_FLIP_P),
            A.VerticalFlip(p=VERTICAL_FLIP_P),
            A.RandomRotate90(p=RANDOM_ROTATE_90_P),
            A.Rotate(limit=ROTATE_LIMIT, p=ROTATE_P, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(
                scale=AFFINE_SCALE,
                translate_percent=AFFINE_TRANSLATE_PERCENT,
                rotate=AFFINE_ROTATE,
                shear=AFFINE_SHEAR,
                p=AFFINE_P,
            ),

            # Оптические искажения
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS_CONTRAST_LIMIT,
                contrast_limit=BRIGHTNESS_CONTRAST_LIMIT,
                p=BRIGHTNESS_CONTRAST_P,
            ),
            A.HueSaturationValue(
                hue_shift_limit=HUE_SHIFT_LIMIT,
                sat_shift_limit=SAT_SHIFT_LIMIT,
                val_shift_limit=VAL_SHIFT_LIMIT,
                p=HUE_SATURATION_VALUE_P,
            ),
            A.RGBShift(
                r_shift_limit=RGB_SHIFT_LIMIT,
                g_shift_limit=RGB_SHIFT_LIMIT,
                b_shift_limit=RGB_SHIFT_LIMIT,
                p=RGB_SHIFT_P,
            ),

            # Размытие и шум
            A.GaussianBlur(blur_limit=GAUSSIAN_BLUR_LIMIT, p=GAUSSIAN_BLUR_P),
            A.GaussNoise(var_limit=GAUSS_NOISE_VAR_LIMIT, p=GAUSS_NOISE_P),

            # Морфологические операции
            A.OneOf(
                [
                    A.Sharpen(p=SHARPEN_P),
                    A.CLAHE(clip_limit=CLAHE_CLIP_LIMIT, p=CLAHE_P),
                ],
                p=ONE_OF_MORPH_P,
            ),

            # Random Scale
            # A.RandomScale(scale_limit=RANDOM_SCALE_LIMIT, p=RANDOM_SCALE_P),
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
