import torch
from pathlib import Path

# ==========================================
# ВПИШИТЕ СЮДА ПУТЬ К ПРОБЛЕМНОМУ ЧЕКПОИНТУ
# ==========================================
CHECKPOINT_PATH = r"./weights/best_model_fold_4.pth"


def get_input_channels(state_dict):
    """Определяет количество входных каналов (3 или 5)."""
    for key, tensor in state_dict.items():
        # Ищем самую первую свертку энкодера
        if key.endswith(".weight") and tensor.ndim == 4:
            if "stem" in key or "conv1" in key or "patch_embed" in key:
                # Формат свертки: [out_channels, in_channels, kH, kW]
                return tensor.shape[1]
    return "Неизвестно"


def get_model_type(state_dict):
    """Определяет тип декодера (Unet, UnetPlusPlus, SegFormer, UPerNet)."""
    keys = list(state_dict.keys())
    keys_str = " ".join(keys)
    
    if "decoder.blocks.x_0_0" in keys_str:
        return "UnetPlusPlus"
    elif "decoder.blocks.0.0" in keys_str or "decoder.blocks.0.conv1" in keys_str:
        return "Unet"
    elif "decode_head.conv.weight" in keys_str or "decode_head.ppm" in keys_str:
        return "UPerNet"
    elif "decode_head.linear_c" in keys_str:
        return "SegFormer"
    elif "decoder.p5" in keys_str:
        return "FPN"
    else:
        return "Неизвестная модель (возможно кастомная)"


def get_encoder_guess(state_dict):
    """Пытается угадать энкодер по размеру первой свертки и именам слоев."""
    keys = list(state_dict.keys())
    
    # 1. Проверяем семейство EfficientNet (timm-efficientnet)
    if "encoder.conv_stem.weight" in state_dict:
        stem_shape = state_dict["encoder.conv_stem.weight"].shape
        out_channels = stem_shape[0] # Количество фильтров на выходе
        
        if out_channels == 32:
            return "timm-efficientnet-b0, b1 или b2 (stem=32)"
        elif out_channels == 40:
            return "timm-efficientnet-b3 (stem=40)  <-- ЧАСТАЯ ОШИБКА!"
        elif out_channels == 48:
            return "timm-efficientnet-b4 или b5 (stem=48) <-- ЧАСТАЯ ОШИБКА!"
        elif out_channels == 56:
            return "timm-efficientnet-b6 (stem=56)"
        elif out_channels == 64:
            return "timm-efficientnet-b7 (stem=64)"
            
    # 2. Проверяем семейство SegFormer (mit_bX)
    elif "encoder.patch_embeddings.0.proj.weight" in state_dict:
        embed_shape = state_dict["encoder.patch_embeddings.0.proj.weight"].shape
        dim = embed_shape[0]
        
        if dim == 32:
            return "mit_b0 или mit_b1 (embed_dim=32)"
        elif dim == 64:
            # Отличаем b2, b3, b4, b5 по глубине
            if "encoder.block.3.3.mlp.fc1.weight" not in keys:
                return "mit_b2 (embed_dim=64, shallow)"
            elif "encoder.block.3.17.mlp.fc1.weight" not in keys:
                return "mit_b3 (embed_dim=64, medium)"
            else:
                return "mit_b4 или mit_b5 (embed_dim=64, deep)"
                
    # 3. Проверяем семейство ResNet
    elif "encoder.conv1.weight" in state_dict:
        # Проверяем наличие layer4
        has_layer4 = any("encoder.layer4" in k for k in keys)
        if has_layer4:
            # Смотрим на размер bottleneck
            if "encoder.layer1.0.conv3.weight" in keys:
                return "resnet50, resnet101 или resnext (Bottleneck)"
            else:
                return "resnet18 или resnet34 (BasicBlock)"

    return "Не удалось точно определить (посмотрите размеры слоев ниже)"


def main():
    path = Path(CHECKPOINT_PATH)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        return

    print(f"🔍 Анализ файла: {path.name}")
    print("=" * 60)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return

    # --- 1. Читаем то, что НАПИСАНО в конфиге ---
    cfg = checkpoint.get("config", {})
    cfg_model = cfg.get("MODEL_NAME", "Не указано")
    cfg_encoder = cfg.get("ENCODER_NAME", "Не указано")
    cfg_coords = cfg.get("use_coordinates", True)
    cfg_channels = 5 if cfg_coords else 3

    print("[КОНФИГ ВНУТРИ ФАЙЛА (Может содержать ошибки)]")
    print(f"  Модель:  {cfg_model}")
    print(f"  Энкодер: {cfg_encoder}")
    print(f"  Каналы:  {cfg_channels} (use_coordinates={cfg_coords})")
    print("-" * 60)

    # --- 2. Анализируем РЕАЛЬНЫЕ ВЕСА ---
    if "model_state_dict" not in checkpoint:
        print("❌ В файле нет ключа 'model_state_dict'. Это не чекпоинт модели!")
        return
        
    state_dict = checkpoint["model_state_dict"]
    
    real_channels = get_input_channels(state_dict)
    real_model = get_model_type(state_dict)
    real_encoder = get_encoder_guess(state_dict)

    print("[РЕАЛЬНАЯ АРХИТЕКТУРА ИЗ ВЕСОВ (100% Правда)]")
    
    # Сравниваем каналы
    ch_mark = "✅ Совпадает" if str(real_channels) == str(cfg_channels) else "❌ ОШИБКА!"
    print(f"  Каналы:  {real_channels} {ch_mark}")
    
    # Сравниваем модель
    mod_mark = "✅ Совпадает" if real_model == cfg_model else "❌ ОШИБКА!"
    print(f"  Модель:  {real_model} {mod_mark}")
    
    # Выводим энкодер
    print(f"  Энкодер: {real_encoder}")
    
    print("=" * 60)
    
    # Если энкодер не совпадает, даем совет
    if "ОШИБКА" in ch_mark or "ОШИБКА" in mod_mark or ("b3" in real_encoder and "b4" in cfg_encoder) or ("b4" in real_encoder and "b3" in cfg_encoder):
        print("💡 ВЫВОД:")
        print("Конфиг не совпадает с реальными весами модели.")
        print("Вам нужно использовать 'ENCODER_OVERRIDES' в скрипте инференса,")
        print("чтобы принудительно задать параметры, которые написаны в блоке [РЕАЛЬНАЯ АРХИТЕКТУРА].")


if __name__ == "__main__":
    main()