#!/usr/bin/env python3
"""
GUI приложение для проверки и разметки изображений датасета.
Отображает изображение, маску и наложенную маску, позволяет категоризировать пары изображений.
"""

import os
import json
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class DatasetChecker:
    """Класс приложения для проверки датасета."""

    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Checker")
        self.root.geometry("1400x800")

        # Пути к данным
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.train_dir = os.path.join(self.base_dir, "dataset", "full_dataset", "train")
        self.images_dir = os.path.join(self.train_dir, "images")
        self.masks_dir = os.path.join(self.train_dir, "masks")
        self.change_data_dir = os.path.join(self.base_dir, "dataset", "change_data")
        self.processed_file = os.path.join(self.change_data_dir, "processed.json")

        # Подпапки для категорий
        self.categories = {
            "ready_train": "ready_train",
            "change_mask": "change_mask",
            "delete": "delete"
        }

        # Создаем папки для change_data
        self._create_directories()

        # Загружаем список обработанных файлов
        self.processed_files = self._load_processed()

        # Получаем список всех пар изображений
        self.image_pairs = self._get_image_pairs()

        # Фильтруем уже обработанные
        self.remaining_pairs = self._filter_processed()

        # Текущий индекс
        self.current_index = 0

        # Текущие изображения
        self.current_image = None
        self.current_mask = None
        self.current_overlay = None
        self.current_filename = None

        # Создаем интерфейс
        self._create_ui()

        # Загружаем первое изображение
        self.load_current_pair()

    def _create_directories(self):
        """Создает необходимые директории для сохранения."""
        os.makedirs(self.change_data_dir, exist_ok=True)
        for category in self.categories.values():
            os.makedirs(os.path.join(self.change_data_dir, category, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.change_data_dir, category, "masks"), exist_ok=True)

    def _load_processed(self):
        """Загружает список обработанных файлов из JSON."""
        if os.path.exists(self.processed_file):
            try:
                with open(self.processed_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_processed(self):
        """Сохраняет список обработанных файлов в JSON."""
        with open(self.processed_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, indent=2, ensure_ascii=False)

    def _get_image_pairs(self):
        """Получает список пар изображение-маска."""
        pairs = []
        image_files = sorted(os.listdir(self.images_dir))

        for img_file in image_files:
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = os.path.splitext(img_file)[0]
                mask_file = f"{base_name}.png"

                img_path = os.path.join(self.images_dir, img_file)
                mask_path = os.path.join(self.masks_dir, mask_file)

                if os.path.exists(mask_path):
                    pairs.append({
                        "image": img_path,
                        "mask": mask_path,
                        "filename": base_name
                    })

        return pairs

    def _filter_processed(self):
        """Фильтрует уже обработанные пары."""
        return [
            pair for pair in self.image_pairs
            if pair["filename"] not in self.processed_files
        ]

    def _create_ui(self):
        """Создает пользовательский интерфейс."""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Конфигурация grid для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Фрейм для изображений
        images_frame = ttk.LabelFrame(main_frame, text="Изображения", padding="5")
        images_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.columnconfigure(2, weight=1)
        images_frame.rowconfigure(0, weight=1)

        # Фрейм для оригинального изображения
        image_frame = ttk.Frame(images_frame)
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame, text="Изображение", anchor="center")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Фрейм для маски
        mask_frame = ttk.Frame(images_frame)
        mask_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        mask_frame.columnconfigure(0, weight=1)
        mask_frame.rowconfigure(0, weight=1)

        self.mask_label = ttk.Label(mask_frame, text="Маска", anchor="center")
        self.mask_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Фрейм для наложенной маски
        overlay_frame = ttk.Frame(images_frame)
        overlay_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        overlay_frame.columnconfigure(0, weight=1)
        overlay_frame.rowconfigure(0, weight=1)

        self.overlay_label = ttk.Label(overlay_frame, text="Маска на изображении", anchor="center")
        self.overlay_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Фрейм для кнопок
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)

        # Кнопки действий
        self.btn_ready = ttk.Button(
            buttons_frame,
            text="Оставить как есть",
            command=self.on_ready,
            style="Accent.TButton"
        )
        self.btn_ready.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        self.btn_change = ttk.Button(
            buttons_frame,
            text="Поменять разметку",
            command=self.on_change,
            style="Warning.TButton"
        )
        self.btn_change.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        self.btn_delete = ttk.Button(
            buttons_frame,
            text="На удаление",
            command=self.on_delete,
            style="Danger.TButton"
        )
        self.btn_delete.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5)

        # Фрейм прогресса
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))

        self.progress_label = ttk.Label(progress_frame, text="Прогресс: 0 / 0")
        self.progress_label.pack(side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=max(len(self.image_pairs), 1)
        )
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Статус бар
        self.status_label = ttk.Label(main_frame, text="Готово", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # Привязка клавиш
        self.root.bind("<Return>", lambda e: self.on_ready())
        self.root.bind("<space>", lambda e: self.on_change())
        self.root.bind("<BackSpace>", lambda e: self.on_delete())

    def load_current_pair(self):
        """Загружает текущую пару изображений."""
        if self.current_index >= len(self.remaining_pairs):
            self._show_completion_message()
            return

        pair = self.remaining_pairs[self.current_index]
        self.current_filename = pair["filename"]

        # Загрузка изображения
        try:
            img = Image.open(pair["image"])
            img_size = self._resize_image(img)
            self.current_image = ImageTk.PhotoImage(img_size)
            self.image_label.configure(image=self.current_image, text="")
        except Exception as e:
            self.image_label.configure(image="", text=f"Ошибка загрузки: {e}")
            self.current_image = None

        # Загрузка маски
        try:
            mask = Image.open(pair["mask"])
            mask_size = self._resize_image(mask)
            self.current_mask = ImageTk.PhotoImage(mask_size)
            self.mask_label.configure(image=self.current_mask, text="")
        except Exception as e:
            self.mask_label.configure(image="", text=f"Ошибка загрузки: {e}")
            self.current_mask = None

        # Создание наложенной маски (overlay)
        try:
            img_orig = Image.open(pair["image"]).convert("RGB")
            mask_orig = Image.open(pair["mask"]).convert("L")  # Grayscale

            # Приводим к одному размеру
            img_orig = img_orig.resize((500, 500), Image.Resampling.LANCZOS)
            mask_orig = mask_orig.resize((500, 500), Image.Resampling.LANCZOS)

            # Создаем цветную маску (красный полупрозрачный)
            mask_color = Image.new("RGBA", img_orig.size, (255, 0, 0, 0))
            mask_data = []
            for pixel in mask_orig.getdata():
                alpha = int(pixel * 0.5)  # 50% прозрачности
                mask_data.append((255, 0, 0, alpha))
            mask_color.putdata(mask_data)

            # Накладываем маску на изображение
            overlay = Image.alpha_composite(img_orig.convert("RGBA"), mask_color)
            overlay = overlay.convert("RGB")

            self.current_overlay = ImageTk.PhotoImage(overlay)
            self.overlay_label.configure(image=self.current_overlay, text="")
        except Exception as e:
            self.overlay_label.configure(image="", text=f"Ошибка: {e}")
            self.current_overlay = None

        # Обновление прогресса
        processed_count = len(self.image_pairs) - len(self.remaining_pairs)
        total_count = len(self.image_pairs)
        self.progress_label.configure(text=f"Прогресс: {processed_count} / {total_count}")
        self.progress_bar["value"] = processed_count

        # Обновление статуса
        self.status_label.configure(text=f"Файл: {self.current_filename}")

    def _resize_image(self, img, max_size=500):
        """Изменяет размер изображения для отображения."""
        img_copy = img.copy()
        img_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img_copy

    def _save_pair(self, category):
        """Сохраняет текущую пару в указанную категорию."""
        if self.current_filename is None:
            return

        pair = self.remaining_pairs[self.current_index]

        # Копируем изображение в подпапку images
        dest_image_dir = os.path.join(self.change_data_dir, category, "images")
        dest_image = os.path.join(dest_image_dir, os.path.basename(pair["image"]))
        shutil.copy2(pair["image"], dest_image)

        # Копируем маску в подпапку masks
        dest_mask_dir = os.path.join(self.change_data_dir, category, "masks")
        dest_mask = os.path.join(dest_mask_dir, os.path.basename(pair["mask"]))
        shutil.copy2(pair["mask"], dest_mask)

        # Добавляем в обработанные
        self.processed_files.append(self.current_filename)
        self._save_processed()

        # Удаляем из оставшихся
        self.remaining_pairs.pop(self.current_index)

        # Обновляем статус
        category_names = {
            "ready_train": "Оставлено",
            "change_mask": "На переразметку",
            "delete": "На удаление"
        }
        self.status_label.configure(text=f"{category_names.get(category, category)}: {self.current_filename}")

        # Загружаем следующую пару
        self.load_current_pair()

    def on_ready(self):
        """Обработчик кнопки 'Оставить как есть'."""
        self._save_pair(self.categories["ready_train"])

    def on_change(self):
        """Обработчик кнопки 'Поменять разметку'."""
        self._save_pair(self.categories["change_mask"])

    def on_delete(self):
        """Обработчик кнопки 'На удаление'."""
        self._save_pair(self.categories["delete"])

    def _show_completion_message(self):
        """Показывает сообщение о завершении."""
        self.image_label.configure(image="", text="Все изображения обработаны!")
        self.mask_label.configure(image="", text="")
        self.overlay_label.configure(image="", text="")
        self.btn_ready.configure(state=tk.DISABLED)
        self.btn_change.configure(state=tk.DISABLED)
        self.btn_delete.configure(state=tk.DISABLED)
        self.progress_label.configure(text=f"Завершено: {len(self.image_pairs)} / {len(self.image_pairs)}")
        self.progress_bar["value"] = len(self.image_pairs)
        self.status_label.configure(text="Все изображения обработаны!")

        messagebox.showinfo("Завершено", "Все изображения из датасета были обработаны!")


def main():
    """Точка входа приложения."""
    root = tk.Tk()

    # Настройка стиля
    style = ttk.Style()
    style.theme_use("clam")

    # Настройка цветов кнопок
    style.configure("Accent.TButton", foreground="green")
    style.configure("Warning.TButton", foreground="orange")
    style.configure("Danger.TButton", foreground="red")

    app = DatasetChecker(root)
    root.mainloop()


if __name__ == "__main__":
    main()
