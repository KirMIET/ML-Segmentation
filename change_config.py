import torch
from pathlib import Path

# ==========================================
# КОНФИГУРАЦИЯ ИСПРАВЛЕНИЯ
# ==========================================
# Путь к кривому чекпоинту
CHECKPOINT_PATH = r"./changes/change14/UPerNet_fold_4/best_UPerNet_fold_4.pth"

# Путь, куда сохранить исправленный (лучше с приставкой _fixed, чтобы не потерять оригинал)
OUTPUT_PATH = r"./changes/change14/UPerNet_fold_4/best_UPerNet_fold_4.pth"

# ==========================================
# ПРАВИЛЬНЫЕ ЗНАЧЕНИЯ (впишите те, что выдал скрипт-детектор)
# ==========================================
CORRECT_MODEL_NAME = "UPerNet"
CORRECT_ENCODER_NAME = "resnext50_32x4d"  # Например, меняем b4 на b3
CORRECT_USE_COORDINATES = False                # False = 3 канала, True = 5 каналов
CORRECT_IMG_SIZE = 352


def main():
    print(f"Загрузка: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Если в чекпоинте вообще не было конфига, создаем пустой словарь
    if "config" not in checkpoint:
        checkpoint["config"] = {}
        print("В чекпоинте не было конфига. Создаем новый.")
    
    cfg = checkpoint["config"]
    
    print("\n[СТАРЫЙ КОНФИГ]")
    print(f"  MODEL_NAME:      {cfg.get('MODEL_NAME', 'Нет')}")
    print(f"  ENCODER_NAME:    {cfg.get('ENCODER_NAME', 'Нет')}")
    print(f"  use_coordinates: {cfg.get('use_coordinates', 'Нет')}")
    print(f"  IMG_SIZE:        {cfg.get('IMG_SIZE', 'Нет')}")
    
    # Вносим исправления
    cfg["MODEL_NAME"] = CORRECT_MODEL_NAME
    cfg["ENCODER_NAME"] = CORRECT_ENCODER_NAME
    cfg["use_coordinates"] = CORRECT_USE_COORDINATES
    cfg["IMG_SIZE"] = CORRECT_IMG_SIZE
    
    checkpoint["config"] = cfg
    
    print("\n[НОВЫЙ КОНФИГ]")
    print(f"  MODEL_NAME:      {cfg['MODEL_NAME']}")
    print(f"  ENCODER_NAME:    {cfg['ENCODER_NAME']}")
    print(f"  use_coordinates: {cfg['use_coordinates']}")
    print(f"  IMG_SIZE:        {cfg['IMG_SIZE']}")
    
    # Сохраняем исправленный файл
    torch.save(checkpoint, OUTPUT_PATH)
    print(f"\n✅ Успешно сохранено в: {OUTPUT_PATH}")
    print("Теперь вы можете переименовать этот файл в оригинальное имя и использовать в ансамбле!")

if __name__ == "__main__":
    main()