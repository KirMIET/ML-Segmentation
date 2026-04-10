"""
Mask Viewer - Mini application for visualizing images with segmentation masks.
Displays: original image, mask overlay, and semi-transparent mask on image.
"""

import csv
import json
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

# Increase CSV field size limit for large masks
csv.field_size_limit(10000000)


class MaskViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Viewer - Image Segmentation")
        self.root.geometry("1400x900")
        
        # Paths
        self.csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "submissions",
            "submission_king.csv"
        )
        self.images_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset",
            "best_dataset",
            "test_images"
        )
        
        # Data
        self.samples = []  # List of (image_path, mask_array)
        self.current_idx = 0
        
        # Display settings
        self.display_size = 512  # Display size for images
        self.mask_alpha = 0.5  # Transparency for overlay
        
        # Load data
        self._load_data()
        
        # Build UI
        self._build_ui()
        
        # Show first sample
        if self.samples:
            self._show_sample(0)
    
    def _load_data(self):
        """Load image-mask pairs from CSV and images directory."""
        print("Loading masks from CSV...")

        # Build mapping of image IDs to mask data
        mask_data = {}
        errors = 0
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row_idx, row in enumerate(reader):
                if len(row) >= 2:
                    image_id = row[0]
                    try:
                        mask_array = json.loads(row[1])
                        mask_data[image_id] = np.array(mask_array, dtype=np.uint8)
                    except json.JSONDecodeError as e:
                        errors += 1
                        if errors <= 3:
                            print(f"JSON error for {image_id}: {e}")
                    except Exception as e:
                        errors += 1
                        if errors <= 3:
                            print(f"Error loading mask for {image_id}: {e}")

        print(f"Loaded {len(mask_data)} masks ({errors} errors)")
        
        # Match with images
        if not os.path.exists(self.images_dir):
            print(f"Images directory not found: {self.images_dir}")
            return
        
        print("Scanning images directory...")
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files:
            # CSV ImageId includes .jpg extension, match with full filename
            image_id_with_ext = img_file  # e.g., "xxx.jpg"
            if image_id_with_ext in mask_data:
                img_path = os.path.join(self.images_dir, img_file)
                self.samples.append({
                    'image_path': img_path,
                    'mask': mask_data[image_id_with_ext],
                    'filename': img_file
                })
        
        print(f"Found {len(self.samples)} image-mask pairs")
    
    def _build_ui(self):
        """Build the user interface."""
        # Top control panel
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Navigation buttons
        ttk.Button(control_frame, text="<< First", 
                  command=self._go_first).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="< Previous", 
                  command=self._go_previous).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next >", 
                  command=self._go_next).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Last >>", 
                  command=self._go_last).pack(side=tk.LEFT, padx=5)
        
        # Image counter
        self.counter_label = ttk.Label(control_frame, text="0 / 0")
        self.counter_label.pack(side=tk.LEFT, padx=20)
        
        # Filename label
        self.filename_label = ttk.Label(control_frame, text="", foreground="gray")
        self.filename_label.pack(side=tk.LEFT, padx=20)
        
        # Alpha control
        ttk.Label(control_frame, text="Mask Transparency:").pack(side=tk.LEFT, padx=(30, 5))
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_slider = ttk.Scale(control_frame, from_=0.1, to=1.0, 
                                variable=self.alpha_var, orient=tk.HORIZONTAL,
                                command=self._update_display)
        alpha_slider.pack(side=tk.LEFT, padx=5)
        self.alpha_label = ttk.Label(control_frame, text="0.50")
        self.alpha_label.pack(side=tk.LEFT, padx=5)
        
        # Jump to specific image
        ttk.Label(control_frame, text="Go to:").pack(side=tk.LEFT, padx=(30, 5))
        self.jump_var = tk.StringVar()
        jump_entry = ttk.Entry(control_frame, textvariable=self.jump_var, width=10)
        jump_entry.pack(side=tk.LEFT, padx=5)
        jump_entry.bind('<Return>', self._jump_to_image)
        ttk.Button(control_frame, text="Go", 
                  command=self._jump_to_image).pack(side=tk.LEFT, padx=5)
        
        # Main display area with three panels
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create three panels
        self.panels = []
        panel_titles = ["Original Image", "Mask", "Image + Mask Overlay"]
        
        for i, title in enumerate(panel_titles):
            panel_frame = ttk.LabelFrame(display_frame, text=title, padding=5)
            panel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            # Canvas for image display
            canvas = tk.Canvas(panel_frame, bg='gray', highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            
            self.panels.append({
                'frame': panel_frame,
                'canvas': canvas,
                'image_ref': None  # Keep reference to prevent garbage collection
            })
    
    def _show_sample(self, idx):
        """Display the sample at given index."""
        if not self.samples or idx < 0 or idx >= len(self.samples):
            return
        
        self.current_idx = idx
        sample = self.samples[idx]
        
        # Update counter and filename
        self.counter_label.config(text=f"{idx + 1} / {len(self.samples)}")
        self.filename_label.config(text=sample['filename'])
        
        # Load image
        try:
            image = Image.open(sample['image_path'])
        except Exception as e:
            print(f"Error loading image: {e}")
            return
        
        mask = sample['mask']
        
        # Panel 1: Original image
        self._display_image_in_panel(0, image)
        
        # Panel 2: Mask only
        mask_image = self._create_mask_image(mask)
        self._display_image_in_panel(1, mask_image)
        
        # Panel 3: Image with semi-transparent mask overlay
        overlay_image = self._create_overlay_image(image, mask)
        self._display_image_in_panel(2, overlay_image)
    
    def _display_image_in_panel(self, panel_idx, pil_image):
        """Resize and display a PIL image in the specified panel."""
        panel = self.panels[panel_idx]
        canvas = panel['canvas']
        
        # Calculate display size maintaining aspect ratio
        orig_width, orig_height = pil_image.size
        aspect_ratio = orig_width / orig_height
        
        if orig_width >= orig_height:
            display_width = self.display_size
            display_height = int(self.display_size / aspect_ratio)
        else:
            display_height = self.display_size
            display_width = int(self.display_size * aspect_ratio)
        
        # Resize image
        resized_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(display_width // 2, display_height // 2, image=photo, anchor=tk.CENTER)
        canvas.config(width=display_width, height=display_height)
        
        # Keep reference
        panel['image_ref'] = photo
    
    def _create_mask_image(self, mask):
        """Create a PIL image from mask array."""
        # Convert mask to 0-255 range
        mask_255 = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_255, mode='L')
        
        # Convert to RGB for display (grayscale)
        mask_image = mask_image.convert('RGB')
        
        return mask_image
    
    def _create_overlay_image(self, image, mask):
        """Create image with semi-transparent mask overlay."""
        # Convert image to RGBA if needed
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Create colored mask
        mask_colored = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        
        # Set mask color (green with alpha)
        mask_colored[:, :, 0] = 0    # R
        mask_colored[:, :, 1] = 255  # G
        mask_colored[:, :, 2] = 0    # B
        mask_colored[:, :, 3] = (mask * 255 * self.mask_alpha).astype(np.uint8)  # A
        
        mask_image = Image.fromarray(mask_colored, mode='RGBA')
        
        # Resize mask to match image if needed
        if image.size != mask_image.size:
            mask_image = mask_image.resize(image.size, Image.Resampling.LANCZOS)
        
        # Composite
        overlay = Image.alpha_composite(image, mask_image)
        
        return overlay
    
    def _update_display(self, value=None):
        """Update display with current settings."""
        # Update alpha label
        self.alpha_var.get()
        self.alpha_label.config(text=f"{self.alpha_var.get():.2f}")
        self.mask_alpha = self.alpha_var.get()
        
        # Refresh current sample
        self._show_sample(self.current_idx)
    
    def _go_first(self):
        self._show_sample(0)
    
    def _go_previous(self):
        if self.current_idx > 0:
            self._show_sample(self.current_idx - 1)
    
    def _go_next(self):
        if self.current_idx < len(self.samples) - 1:
            self._show_sample(self.current_idx + 1)
    
    def _go_last(self):
        self._show_sample(len(self.samples) - 1)
    
    def _jump_to_image(self, event=None):
        """Jump to specific image number."""
        try:
            idx = int(self.jump_var.get()) - 1
            if 0 <= idx < len(self.samples):
                self._show_sample(idx)
        except ValueError:
            pass


def main():
    root = tk.Tk()
    app = MaskViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
