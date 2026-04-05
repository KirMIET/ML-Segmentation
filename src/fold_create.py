import os
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np


def parse_sample_filename(path):
    path = Path(path)
    parts = path.stem.split("_")

    if len(parts) < 6:
        raise ValueError(f"Формат файла неверный: {path.name}")

    return {
        "camera_ip": parts[0],
        "sequence_code": f"{parts[1]}_{parts[2]}",
        "product_id": parts[3],
    }


def make_folds(dataset, n_splits=5, seed=42):
    """
    Кастомный splitter:
    - группируем по (camera_ip + sequence_code + product_id)
    - внутри каждой камеры делим группы на folds
    - в каждом fold у каждой камеры примерно 80/20 по группам
    """
    rng = random.Random(seed)

    group_to_indices = defaultdict(list)
    camera_to_groups = defaultdict(list)

    # Собираем группы
    for idx, (img_path, _) in enumerate(dataset.samples):
        meta = parse_sample_filename(img_path)

        group_id = f"{meta['camera_ip']}__{meta['sequence_code']}__{meta['product_id']}"
        group_to_indices[group_id].append(idx)

    # Собираем список групп для каждой камеры
    for group_id in group_to_indices.keys():
        camera_ip = group_id.split("__")[0]
        camera_to_groups[camera_ip].append(group_id)

    # Для каждой камеры делим её группы на n_splits частей
    camera_splits = {}
    for cam, groups in camera_to_groups.items():
        groups = list(groups)
        rng.shuffle(groups)
        camera_splits[cam] = [list(x) for x in np.array_split(groups, n_splits)]

    # Собираем folds
    folds = []
    for fold_id in range(n_splits):
        train_idx = []
        val_idx = []

        for cam, splits in camera_splits.items():
            for split_id, group_chunk in enumerate(splits):
                target = val_idx if split_id == fold_id else train_idx

                for group_id in group_chunk:
                    target.extend(group_to_indices[group_id])

        folds.append((
            np.array(sorted(train_idx), dtype=np.int64),
            np.array(sorted(val_idx), dtype=np.int64),
        ))

    return folds


def print_camera_distribution(dataset, train_idx, val_idx, fold_num=None):
    train_counter = Counter()
    val_counter = Counter()

    for idx in train_idx:
        img_path, _ = dataset.samples[idx]
        cam = parse_sample_filename(img_path)["camera_ip"]
        train_counter[cam] += 1

    for idx in val_idx:
        img_path, _ = dataset.samples[idx]
        cam = parse_sample_filename(img_path)["camera_ip"]
        val_counter[cam] += 1

    all_cameras = sorted(set(train_counter.keys()) | set(val_counter.keys()))

    prefix = f"[Fold {fold_num}] " if fold_num is not None else ""
    print(prefix + "Распределение по камерам:")
    print(prefix + "-" * 74)
    print(f"{'camera_ip':<18} {'train':>8} {'val':>8} {'total':>8} {'val_%':>8}")

    for cam in all_cameras:
        train_n = train_counter[cam]
        val_n = val_counter[cam]
        total_n = train_n + val_n
        val_pct = 100.0 * val_n / total_n if total_n > 0 else 0.0

        print(f"{cam:<18} {train_n:>8} {val_n:>8} {total_n:>8} {val_pct:>7.2f}")

    print(prefix + "-" * 74)


class SimpleDataset:
    """Простой dataset для работы с файлами напрямую"""
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Получаем список всех изображений
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Создаем samples - список кортежей (image_path, mask_path)
        self.samples = []
        for img_path in self.image_files:
            mask_path = self.masks_dir / (img_path.stem + ".jpg")
            if not mask_path.exists():
                # Пробуем другие расширения масок
                mask_path = self.masks_dir / (img_path.stem + ".png")
            self.samples.append((str(img_path), str(mask_path)))
    
    def __len__(self):
        return len(self.samples)


def copy_files_to_fold(dataset, indices, dest_images_dir, dest_masks_dir):
    """Копирует файлы изображений и масок в целевую директорию"""
    dest_images_dir = Path(dest_images_dir)
    dest_masks_dir = Path(dest_masks_dir)
    
    # Создаем директории если не существуют
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_masks_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in indices:
        img_path, mask_path = dataset.samples[idx]
        
        # Копируем изображение
        dest_img = dest_images_dir / Path(img_path).name
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)
        
        # Копируем маску
        dest_mask = dest_masks_dir / Path(mask_path).name
        if not dest_mask.exists():
            if Path(mask_path).exists():
                shutil.copy2(mask_path, dest_mask)


def main():
    # Параметры
    TRAIN_IMAGES_DIR = "dataset/best_dataset/train/images"
    TRAIN_MASKS_DIR = "dataset/best_dataset/train/masks"
    OUTPUT_DIR = "dataset/dataset_fold"
    N_SPLITS = 4
    SEED = 42
    
    # Создаем dataset
    print("Загрузка датасета...")
    full_dataset = SimpleDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    print(f"Найдено {len(full_dataset)} сэмплов")
    
    # Создаем фолды
    print(f"\nСоздание {N_SPLITS} фолдов...")
    folds = make_folds(full_dataset, n_splits=N_SPLITS, seed=SEED)
    
    # Очищаем выходную директорию
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Сохраняем каждый фолд
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"\n{'='*74}")
        print(f"Обработка Fold {fold}/{N_SPLITS}")
        print(f"{'='*74}")
        
        # Печатаем распределение по камерам
        print_camera_distribution(full_dataset, train_idx, val_idx, fold_num=fold)
        
        # Создаем директории для фолда
        fold_dir = Path(OUTPUT_DIR) / f"fold_{fold}"
        fold_train_images = fold_dir / "train" / "images"
        fold_train_masks = fold_dir / "train" / "masks"
        fold_val_images = fold_dir / "val" / "images"
        fold_val_masks = fold_dir / "val" / "masks"
        
        # Копируем файлы
        print(f"\nКопирование {len(train_idx)} train сэмплов...")
        copy_files_to_fold(full_dataset, train_idx, fold_train_images, fold_train_masks)
        
        print(f"Копирование {len(val_idx)} val сэмплов...")
        copy_files_to_fold(full_dataset, val_idx, fold_val_images, fold_val_masks)
        
        print(f"\nFold {fold} сохранен в {fold_dir}")
        print(f"  Train: {len(train_idx)} сэмплов")
        print(f"  Val: {len(val_idx)} сэмплов")
    
    print(f"\n{'='*74}")
    print(f"Все {N_SPLITS} фолдов успешно созданы в {OUTPUT_DIR}")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
