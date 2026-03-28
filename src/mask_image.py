#!/usr/bin/env python3
"""
GUI приложение для просмотра и перемещения изображений между папками change_data.
Отображает изображение и маску, позволяет переключаться между файлами и переносить их между папками.
"""

import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class MaskImageViewer:
    """Класс приложения для просмотра и перемещения изображений."""

    def __init__(self, root):
        self.root = root
        self.root.title("Mask Image Viewer")
        self.root.geometry("1200x700")

        # Пути к данным
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.change_data_dir = os.path.join(self.base_dir, "dataset", "change_data")

        # Подпапки с категориями
        self.categories = ["ready_train", "change_mask", "delete"]

        # Текущая выбранная категория
        self.current_category = tk.StringVar(value=self.categories[0])

        # Список файлов в текущей категории
        self.file_pairs = []
        self.current_index = 0

        # Текущие изображения
        self.current_image = None
        self.current_mask = None
        self.current_overlay = None
        self.current_filename = None

        # Создаем интерфейс
        self._create_ui()

        # Загружаем список файлов
        self._load_file_pairs()

        # Загружаем первое изображение
        self.load_current_pair()

    def _create_ui(self):
        """Создает пользовательский интерфейс."""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Конфигурация grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # === Верхняя панель: выбор категории ===
        category_frame = ttk.LabelFrame(main_frame, text="Папка", padding="5")
        category_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        for i, category in enumerate(self.categories):
            rb = ttk.Radiobutton(
                category_frame,
                text=category,
                variable=self.current_category,
                value=category,
                command=self.on_category_change
            )
            rb.grid(row=0, column=i, padx=20)

        # Фрейм для изображений
        images_frame = ttk.LabelFrame(main_frame, text="Изображения", padding="5")
        images_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)

        # Фрейм для изображения с наложенной маской
        overlay_frame = ttk.Frame(images_frame)
        overlay_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        overlay_frame.columnconfigure(0, weight=1)
        overlay_frame.rowconfigure(0, weight=1)

        self.overlay_label = ttk.Label(overlay_frame, text="Изображение с маской", anchor="center")
        self.overlay_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Фрейм для маски
        mask_frame = ttk.Frame(images_frame)
        mask_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        mask_frame.columnconfigure(0, weight=1)
        mask_frame.rowconfigure(0, weight=1)

        self.mask_label = ttk.Label(mask_frame, text="Маска", anchor="center")
        self.mask_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # === Навигация ===
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)
        nav_frame.columnconfigure(2, weight=1)

        self.btn_prev = ttk.Button(nav_frame, text="← Предыдущее", command=self.on_prev)
        self.btn_prev.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        self.lbl_position = ttk.Label(nav_frame, text="0 / 0", anchor="center")
        self.lbl_position.grid(row=0, column=1, sticky=(tk.W, tk.E))

        self.btn_next = ttk.Button(nav_frame, text="Следующее →", command=self.on_next)
        self.btn_next.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5)

        # === Перенос между папками ===
        move_frame = ttk.LabelFrame(main_frame, text="Перенести в папку", padding="5")
        move_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        move_frame.columnconfigure(0, weight=1)
        move_frame.columnconfigure(1, weight=1)
        move_frame.columnconfigure(2, weight=1)

        self.btn_move_ready = ttk.Button(
            move_frame,
            text="ready_train",
            command=lambda: self.on_move("ready_train"),
            style="Accent.TButton"
        )
        self.btn_move_ready.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        self.btn_move_change = ttk.Button(
            move_frame,
            text="change_mask",
            command=lambda: self.on_move("change_mask"),
            style="Warning.TButton"
        )
        self.btn_move_change.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        self.btn_move_delete = ttk.Button(
            move_frame,
            text="delete",
            command=lambda: self.on_move("delete"),
            style="Danger.TButton"
        )
        self.btn_move_delete.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5)

        # Статус бар
        self.status_label = ttk.Label(main_frame, text="Готово", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=4, column=0, sticky=(tk.W, tk.E))

        # Привязка клавиш
        self.root.bind("<Left>", lambda e: self.on_prev())
        self.root.bind("<Right>", lambda e: self.on_next())

    def _get_category_path(self, category):
        """Возвращает путь к папке категории."""
        return os.path.join(self.change_data_dir, category)

    def _load_file_pairs(self):
        """Загружает список пар изображение-маска из текущей категории."""
        category = self.current_category.get()
        category_path = self._get_category_path(category)

        images_path = os.path.join(category_path, "images")
        masks_path = os.path.join(category_path, "masks")

        self.file_pairs = []

        if not os.path.exists(images_path):
            return

        for img_file in sorted(os.listdir(images_path)):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = os.path.splitext(img_file)[0]
                mask_file = f"{base_name}.png"

                img_path = os.path.join(images_path, img_file)
                mask_path = os.path.join(masks_path, mask_file)

                if os.path.exists(mask_path):
                    self.file_pairs.append({
                        "image": img_path,
                        "mask": mask_path,
                        "filename": base_name,
                        "image_name": img_file,
                        "mask_name": mask_file
                    })

        self.current_index = 0

    def load_current_pair(self):
        """Загружает текущую пару изображений."""
        if not self.file_pairs:
            self.overlay_label.configure(image="", text="Нет изображений в папке")
            self.mask_label.configure(image="", text="")
            self.lbl_position.configure(text="0 / 0")
            self.status_label.configure(text="Нет файлов для отображения")
            self.current_image = None
            self.current_mask = None
            self.current_overlay = None
            self.current_filename = None
            return

        if self.current_index >= len(self.file_pairs):
            self.current_index = len(self.file_pairs) - 1

        pair = self.file_pairs[self.current_index]
        self.current_filename = pair["filename"]

        # Загрузка изображения с наложенной маской
        try:
            img_orig = Image.open(pair["image"]).convert("RGB")
            mask_orig = Image.open(pair["mask"]).convert("L")

            # Определяем размер для отображения
            max_size = 500
            img_orig = img_orig.resize((max_size, max_size), Image.Resampling.LANCZOS)
            mask_orig = mask_orig.resize((max_size, max_size), Image.Resampling.LANCZOS)

            # Создаем цветную маску (зелёная полупрозрачная)
            mask_color = Image.new("RGBA", img_orig.size, (0, 255, 0, 0))
            mask_data = []
            for pixel in mask_orig.getdata():
                alpha = int(pixel * 0.5)  # 50% прозрачности
                mask_data.append((0, 255, 0, alpha))
            mask_color.putdata(mask_data)

            # Накладываем маску на изображение
            overlay = Image.alpha_composite(img_orig.convert("RGBA"), mask_color)
            overlay = overlay.convert("RGB")

            self.current_overlay = ImageTk.PhotoImage(overlay)
            self.overlay_label.configure(image=self.current_overlay, text="")
        except Exception as e:
            self.overlay_label.configure(image="", text=f"Ошибка: {e}")
            self.current_overlay = None

        # Загрузка маски
        try:
            mask = Image.open(pair["mask"])
            mask = mask.resize((max_size, max_size), Image.Resampling.LANCZOS)
            self.current_mask = ImageTk.PhotoImage(mask)
            self.mask_label.configure(image=self.current_mask, text="")
        except Exception as e:
            self.mask_label.configure(image="", text=f"Ошибка: {e}")
            self.current_mask = None

        # Обновление позиции
        self.lbl_position.configure(text=f"{self.current_index + 1} / {len(self.file_pairs)}")

        # Обновление статуса
        category = self.current_category.get()
        self.status_label.configure(text=f"[{category}] {self.current_filename}")

        # Обновление кнопок
        self._update_buttons()

    def _update_buttons(self):
        """Обновляет состояние кнопок навигации."""
        if not self.file_pairs:
            self.btn_prev.configure(state=tk.DISABLED)
            self.btn_next.configure(state=tk.DISABLED)
            return

        self.btn_prev.configure(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.btn_next.configure(state=tk.NORMAL if self.current_index < len(self.file_pairs) - 1 else tk.DISABLED)

    def on_category_change(self):
        """Обработчик смены категории."""
        self._load_file_pairs()
        self.load_current_pair()

    def on_prev(self):
        """Обработчик кнопки 'Предыдущее'."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_pair()

    def on_next(self):
        """Обработчик кнопки 'Следующее'."""
        if self.current_index < len(self.file_pairs) - 1:
            self.current_index += 1
            self.load_current_pair()

    def on_move(self, target_category):
        """Обработчик переноса в другую папку."""
        if not self.file_pairs or self.current_filename is None:
            messagebox.showwarning("Предупреждение", "Нет файлов для переноса")
            return

        current_category = self.current_category.get()

        if current_category == target_category:
            messagebox.showinfo("Инфо", "Файл уже находится в этой папке")
            return

        # Подтверждение
        confirm = messagebox.askyesno(
            "Подтверждение",
            f"Перенести '{self.current_filename}' из '{current_category}' в '{target_category}'?"
        )
        if not confirm:
            return

        pair = self.file_pairs[self.current_index]

        # Пути
        src_images_dir = os.path.join(self._get_category_path(current_category), "images")
        src_masks_dir = os.path.join(self._get_category_path(current_category), "masks")
        dst_images_dir = os.path.join(self._get_category_path(target_category), "images")
        dst_masks_dir = os.path.join(self._get_category_path(target_category), "masks")

        # Перемещаем файлы
        try:
            shutil.move(
                os.path.join(src_images_dir, pair["image_name"]),
                os.path.join(dst_images_dir, pair["image_name"])
            )
            shutil.move(
                os.path.join(src_masks_dir, pair["mask_name"]),
                os.path.join(dst_masks_dir, pair["mask_name"])
            )

            # Удаляем из текущего списка
            self.file_pairs.pop(self.current_index)

            # Корректируем индекс
            if self.current_index >= len(self.file_pairs):
                self.current_index = max(0, len(self.file_pairs) - 1)

            # Обновляем отображение
            self.load_current_pair()

            self.status_label.configure(text=f"Перенесено в '{target_category}'")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось переместить файлы: {e}")

    def _show_empty(self):
        """Показывает пустое состояние."""
        self.overlay_label.configure(image="", text="Нет изображений в папке")
        self.mask_label.configure(image="", text="")
        self.lbl_position.configure(text="0 / 0")


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

    app = MaskImageViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
