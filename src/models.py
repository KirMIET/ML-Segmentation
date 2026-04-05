import torch.nn as nn
import segmentation_models_pytorch as smp


def create_unetpp_efficientnet_b3(in_channels: int = 5) -> nn.Module:
    """Создаёт UNet++ с encoder timm-efficientnet-b3."""
    return smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b3",
        encoder_weights="noisy-student",
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


def create_fpn_convnext_small(in_channels: int = 5) -> nn.Module:
    """Создаёт FPN с encoder timm-convnext_small."""
    return smp.FPN(
        encoder_name="tu-convnext_small",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


def create_segformer_mit_b3(in_channels: int = 5) -> nn.Module:
    """Создаёт SegFormer с encoder mit_b2."""
    return smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


def create_upernet_convnext_base(in_channels: int = 5) -> nn.Module:
    """Создаёт UPerNet с encoder convnext_base."""
    return smp.UPerNet(
        encoder_name="resnext50_32x4d",
        encoder_weights="swsl",
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


def create_model(model_name: str, in_channels: int = 5) -> nn.Module:
    """Фабричная функция для создания модели по имени.
    
    Args:
        model_name: Имя модели из MODEL_CONFIGS.
        in_channels: Количество входных каналов (5 для RGB + координаты).
        
    Returns:
        Модель сегментации.
        
    Raises:
        ValueError: Если модель не поддерживается.
    """
    if model_name == "UnetPlusPlus":
        return create_unetpp_efficientnet_b3(in_channels)
    elif model_name == "FPN":
        return create_fpn_convnext_small(in_channels)
    elif model_name == "SegFormer":
        return create_segformer_mit_b3(in_channels)
    elif model_name == "UPerNet":
        return create_upernet_convnext_base(in_channels)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
