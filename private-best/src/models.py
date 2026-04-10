import torch.nn as nn
import segmentation_models_pytorch as smp

from src.config import MODEL_CONFIGS, DEFAULT_IN_CHANNELS, DEFAULT_NUM_CLASSES


def create_model(model_name: str, in_channels: int = DEFAULT_IN_CHANNELS) -> nn.Module:
    """Фабричная функция для создания модели по имени.

    Читает параметры напрямую из MODEL_CONFIGS, чтобы избежать дублирования.

    Args:
        model_name: Имя модели из MODEL_CONFIGS.
        in_channels: Количество входных каналов (5 для RGB + координаты).

    Returns:
        Модель сегментации.

    Raises:
        ValueError: Если модель не поддерживается.
    """
    cfg = MODEL_CONFIGS[model_name]
    encoder_name = cfg["encoder_name"]
    encoder_weights = cfg["encoder_weights"]

    if model_name == "UnetPlusPlus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=DEFAULT_NUM_CLASSES,
            activation=None,
        )
    elif model_name == "FPN":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=DEFAULT_NUM_CLASSES,
            activation=None,
        )
    elif model_name == "SegFormer":
        return smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=DEFAULT_NUM_CLASSES,
            activation=None,
        )
    elif model_name == "UPerNet":
        return smp.UPerNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=DEFAULT_NUM_CLASSES,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
