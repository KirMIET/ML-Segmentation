"""
Central Configuration for Ensemble Segmentation Training Pipeline
==================================================================
All hardcoded parameters are collected here for easy modification.
Import this module in any script that needs these parameters.
"""

from pathlib import Path
import torch


# =========================
# PATHS
# =========================
FOLD_DIR = Path("dataset/dataset_fold")

SAVE_DIR = Path("./model_checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

VISUALIZATION_DIR = Path("./view_train_img")
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# GENERAL SETTINGS
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FOLDS = 2

# =========================
# DATASET SETTINGS
# =========================
IMG_SIZE = 352
NUM_WORKERS = 4
THRESHOLD = 0.4  # Threshold for binarization in metrics/visualization

# =========================
# VISUALIZATION SETTINGS
# =========================
VISUALIZE_EVERY_N_EPOCHS = 5

# =========================
# TRAINING SETTINGS
# =========================
SAM_RHO = 0.05
GRAD_CLIP_MAX_NORM = 1.0

# =========================
# AUGMENTATION SETTINGS (CutMix / CopyPaste)
# =========================
CUTMIX_ALPHA = 1.0
AUGMENTATION_PROB = 0.5
COPYPASTE_MAX_OBJECTS = 3

# =========================
# LOSS SETTINGS
# =========================
USE_OHEM = True
OHEM_RATIO = 0.2
OHEM_MIN_KEPT = 10000

# Initial loss weights (will be updated dynamically during training)
BCE_WEIGHT = 0.4
TVERSKY_WEIGHT = 0.5
BOUNDARY_WEIGHT = 0.1

# Focal Tversky parameters
FOCAL_TVERSKY_ALPHA = 0.3  # weight for False Positive
FOCAL_TVERSKY_BETA = 0.7   # weight for False Negative
FOCAL_TVERSKY_GAMMA = 0.75

# Boundary Loss parameters
BOUNDARY_SIGMA = 2.0

# Dynamic weight scheduling parameters (warmup phase weights)
WARMUP_BCE_WEIGHT = 0.7
WARMUP_TVERSKY_WEIGHT = 0.3
WARMUP_BOUNDARY_WEIGHT = 0.0
WARMUP_FOCAL_TVERSKY_ALPHA = 0.5
WARMUP_FOCAL_TVERSKY_BETA = 0.5

# =========================
# ALBUMENTATIONS PIPELINE PARAMETERS
# =========================
# Geometric transforms
HORIZONTAL_FLIP_P = 0.5
VERTICAL_FLIP_P = 0.3
RANDOM_ROTATE_90_P = 0.5
ROTATE_LIMIT = 45
ROTATE_P = 0.5

AFFINE_SCALE = (0.8, 1.2)
AFFINE_TRANSLATE_PERCENT = (-0.1, 0.1)
AFFINE_ROTATE = (-15, 15)
AFFINE_SHEAR = (-10, 10)
AFFINE_P = 0.5

# Optical distortions
BRIGHTNESS_CONTRAST_LIMIT = 0.15
BRIGHTNESS_CONTRAST_P = 0.3

HUE_SHIFT_LIMIT = 15
SAT_SHIFT_LIMIT = 22.5
VAL_SHIFT_LIMIT = 15
HUE_SATURATION_VALUE_P = 0.3

RGB_SHIFT_LIMIT = 15
RGB_SHIFT_P = 0.3

# Blur and noise
GAUSSIAN_BLUR_LIMIT = (3, 7)
GAUSSIAN_BLUR_P = 0.3

GAUSS_NOISE_VAR_LIMIT = (5, 25)
GAUSS_NOISE_P = 0.3

# Morphological operations
SHARPEN_P = 1.0
CLAHE_CLIP_LIMIT = 4.0
CLAHE_P = 1.0
ONE_OF_MORPH_P = 0.3

# Random Scale (commented out by default)
RANDOM_SCALE_LIMIT = (-0.2, 0.2)
RANDOM_SCALE_P = 0.5

# =========================
# MODEL CONFIGS
# =========================
# Each model config contains:
#   - model_name: Display name
#   - encoder_name: Encoder for segmentation_models_pytorch
#   - encoder_weights: Pretrained weights
#   - batch_size: Training batch size
#   - lr: Learning rate
#   - num_epochs: Max training epochs
#   - warmup_epochs: Linear warmup epochs
#   - early_stopping_patience: Patience for early stopping
#   - weight_decay: Optimizer weight decay
#   - use_coordinates: Whether to add XY coordinate channels (5-ch input)
#   - use_sam: Whether to use SAM optimizer
#   - multi_scales: List of scales for multi-scale training (None = single scale)

MODEL_CONFIGS = {
    "UnetPlusPlus": {
        "model_name": "UnetPlusPlus",
        "encoder_name": "timm-efficientnet-b4",
        "encoder_weights": "noisy-student",
        "batch_size": 12,
        "lr": 1.5e-3,
        "num_epochs": 40,
        "warmup_epochs": 3,
        "early_stopping_patience": 7,
        "weight_decay": 1e-4,
        "use_coordinates": False,
        "use_sam": True,
        "multi_scales": [256, 352],
    },

    # "FPN": {
    #     "model_name": "FPN",
    #     "encoder_name": "tu-convnext_small",
    #     "encoder_weights": "imagenet",
    #     "batch_size": 12,
    #     "lr": 5e-4,
    #     "num_epochs": 35,
    #     "warmup_epochs": 3,
    #     "early_stopping_patience": 7,
    #     "weight_decay": 1e-4,
    #     "use_coordinates": True,
    #     "use_sam": True,
    #     "multi_scales": [256, 352],
    # },

    "SegFormer": {
        "model_name": "SegFormer",
        "encoder_name": "mit_b3",
        "encoder_weights": "imagenet",
        "batch_size": 12,
        "lr": 1.5e-4,
        "num_epochs": 50,
        "warmup_epochs": 6,
        "early_stopping_patience": 10,
        "weight_decay": 1e-2,
        "use_coordinates": False,
        "use_sam": False,
        "multi_scales": None,
    },

    # "UPerNet": {
    #     "model_name": "UPerNet",
    #     "encoder_name": "resnext50_32x4d",
    #     "encoder_weights": "swsl",
    #     "batch_size": 8,
    #     "lr": 4.5e-4,
    #     "num_epochs": 35,
    #     "warmup_epochs": 3,
    #     "early_stopping_patience": 7,
    #     "weight_decay": 1e-4,
    #     "use_coordinates": False,
    #     "use_sam": False,
    #     "multi_scales": [256, 352],
    # },
}

# =========================
# MODEL FACTORY PARAMETERS (for src/models.py)
# =========================
# Default in_channels and num_classes used when creating models
DEFAULT_IN_CHANNELS = 5  # 3 (RGB) + 2 (XY coords) if use_coordinates=True
DEFAULT_NUM_CLASSES = 1
