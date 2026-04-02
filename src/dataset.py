"""
dataset.py — Data pipeline for BW2C colorization

Loads color images from disk, converts them to CIE Lab color space, and
returns (L, ab) tensor pairs for training and validation.

    L  (1 × H × W) — lightness channel  → model INPUT  (the "grayscale" image)
    ab (2 × H × W) — color channels     → model TARGET (what we want to predict)

Supported dataset layouts (both can coexist in the same folder):
    COCO 2017  : data/train/  with flat .jpg files
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color          # rgb2lab — the key color-space conversion
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ── Normalization constants (from Zhang et al. ECCV 2016) ────────────────────
# L  lives in [0, 100]   → subtract 50, divide by 100  → ≈ [-0.5,  0.5]
# ab live in [-110, 110] → divide by 110               → ≈ [-1.0,  1.0]
L_MEAN  = 50.0
L_NORM  = 100.0
AB_NORM = 110.0

# Supported image file extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Input spatial size expected by the model
IMG_SIZE = 384


def normalize_l(l_channel: np.ndarray) -> np.ndarray:
    """Map L from [0, 100] to roughly [-0.5, 0.5]."""
    return (l_channel - L_MEAN) / L_NORM


def unnormalize_l(l_channel: np.ndarray) -> np.ndarray:
    """Reverse normalize_l — used during inference/evaluation."""
    return l_channel * L_NORM + L_MEAN


def normalize_ab(ab_channels: np.ndarray) -> np.ndarray:
    """Map ab from [-110, 110] to [-1, 1]."""
    return ab_channels / AB_NORM


def unnormalize_ab(ab_channels: np.ndarray) -> np.ndarray:
    """Reverse normalize_ab — used during inference/evaluation."""
    return ab_channels * AB_NORM


def lab_to_rgb(l: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from separate (unnormalized) L and ab channels.

    Args:
        l  : (H, W)    — L channel in [0, 100]
        ab : (H, W, 2) — ab channels in [-110, 110]

    Returns:
        rgb : (H, W, 3) float32 in [0, 1]
    """
    lab = np.concatenate([l[..., np.newaxis], ab], axis=-1).astype(np.float32)
    rgb = color.lab2rgb(lab)               # skimage handles the math
    return rgb.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for image colorization.

    Scans a directory for image files and, for each image, returns:
        l_tensor  : FloatTensor (1, IMG_SIZE, IMG_SIZE)  — normalized L channel
        ab_tensor : FloatTensor (2, IMG_SIZE, IMG_SIZE)  — normalized ab channels

    The color ground truth is derived entirely from the image itself —
    no external labels or annotations are needed.

    Args:
        root_dir  : path to folder containing images (e.g. "data/train/")
        augment   : if True, applies random horizontal flip + random crop
        max_images: optional cap on dataset size (useful for quick experiments)
    """

    def __init__(self, root_dir: str, augment: bool = True, max_images: int = None):
        self.root_dir = Path(root_dir)
        self.augment  = augment

        # Collect all valid image paths
        self.image_paths = sorted([
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in IMG_EXTENSIONS
        ])

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in '{root_dir}'.\n"
                f"Expected extensions: {IMG_EXTENSIONS}\n"
                "Make sure you've downloaded COCO 2017 into data/train/"
            )

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        # Augmentation transforms applied to the PIL image BEFORE Lab conversion.
        # We only use spatial transforms here — color jitter would corrupt our
        # ground-truth ab channels, so we never apply it.
        self._aug_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(IMG_SIZE, pad_if_needed=True),
        ])

        # Resize used when augmentation is off (validation)
        self._resize = T.Resize((IMG_SIZE, IMG_SIZE))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]

        # ── 1. Load image as RGB PIL ──────────────────────────────────────────
        img = Image.open(img_path).convert("RGB")

        # ── 2. Resize / augment (still in RGB PIL space) ─────────────────────
        # Important: we do spatial transforms BEFORE converting to Lab so that
        # both L and ab go through the exact same spatial operation.
        if self.augment:
            # Ensure minimum size before random crop
            w, h = img.size
            if w < IMG_SIZE or h < IMG_SIZE:
                img = T.Resize(IMG_SIZE)(img)
            img = self._aug_transform(img)
        else:
            img = self._resize(img)

        # ── 3. RGB → CIE Lab ──────────────────────────────────────────────────
        # skimage.color.rgb2lab expects a float array in [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3) in [0,1]
        img_lab = color.rgb2lab(img_np)                     # (H, W, 3) Lab

        # ── 4. Separate L and ab ──────────────────────────────────────────────
        l_channel  = img_lab[:, :, 0]       # (H, W)    lightness
        ab_channels = img_lab[:, :, 1:]     # (H, W, 2) chrominance

        # ── 5. Normalize both ─────────────────────────────────────────────────
        l_norm  = normalize_l(l_channel)    # ≈ [-0.5, 0.5]
        ab_norm = normalize_ab(ab_channels) # ≈ [-1.0, 1.0]

        # ── 6. Convert to tensors ─────────────────────────────────────────────
        # PyTorch expects (C, H, W). L has 1 channel, ab has 2.
        l_tensor  = torch.from_numpy(l_norm).unsqueeze(0).float()   # (1, H, W)
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()  # (2, H, W)

        return l_tensor, ab_tensor


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    train_dir: str,
    val_dir:   str,
    batch_size: int  = 32,
    num_workers: int = 4,
    max_train: int   = None,
    max_val:   int   = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader).

    Args:
        train_dir   : path to training images   (e.g. "data/train/")
        val_dir     : path to validation images  (e.g. "data/val/")
        batch_size  : images per batch (reduce if you run out of GPU memory)
        num_workers : parallel data loading workers (0 = single-threaded, safe on Windows)
        max_train   : cap on training images   (None = use all)
        max_val     : cap on validation images (None = use all)

    Returns:
        train_loader, val_loader
    """
    train_dataset = ColorizationDataset(train_dir, augment=True,  max_images=max_train)
    val_dataset   = ColorizationDataset(val_dir,   augment=False, max_images=max_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),  # keep workers alive between epochs — saves respawn overhead
        prefetch_factor=2 if num_workers > 0 else None,  # prefetch 2 batches per worker ahead of GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"Training images  : {len(train_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"Batch size       : {batch_size}")
    print(f"Train batches    : {len(train_loader):,}")
    print(f"Val batches      : {len(val_loader):,}")

    return train_loader, val_loader


# ── Quick sanity check ────────────────────────────────────────────────────────
# Run this file directly to verify your dataset is loading correctly:
#   python src/dataset.py

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    from matplotlib.colors import LinearSegmentedColormap
    b_cmap = LinearSegmentedColormap.from_list("b_channel", ["blue", "white", "yellow"])

    TRAIN_DIR = "data/train"
    VAL_DIR   = "data/val"

    if not Path(TRAIN_DIR).exists() or not any(Path(TRAIN_DIR).iterdir()):
        print("ERROR: No images found in data/train/")
        print("Download the dataset first:\n  python src/download_data.py --val-only")
        sys.exit(1)

    try:
        train_loader, val_loader = get_dataloaders(
            TRAIN_DIR, VAL_DIR,
            batch_size=4,
            num_workers=0,      # 0 workers required for Windows interactive run
            max_train=20,
            max_val=8,
        )
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Grab one batch and verify shapes + value ranges
    l_batch, ab_batch = next(iter(train_loader))
    print(f"\nBatch check:")
    print(f"  L  shape : {l_batch.shape}   (expect: [B, 1, 256, 256])")
    print(f"  ab shape : {ab_batch.shape}  (expect: [B, 2, 256, 256])")
    print(f"  L  range : [{l_batch.min():.3f}, {l_batch.max():.3f}]  (expect ≈ [-0.5, 0.5])")
    print(f"  ab range : [{ab_batch.min():.3f}, {ab_batch.max():.3f}]  (expect ≈ [-1.0, 1.0])")

    # Visual check — reconstruct the first image from L + ab and display it
    l_np  = unnormalize_l(l_batch[0, 0].numpy())           # (256, 256)
    ab_np = unnormalize_ab(ab_batch[0].permute(1, 2, 0).numpy())  # (256, 256, 2)
    rgb   = lab_to_rgb(l_np, ab_np)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(l_np, cmap="gray");               axes[0].set_title("Input — L (grayscale)");    axes[0].axis("off")
    axes[1].imshow(ab_np[:, :, 0], cmap="RdYlGn_r"); axes[1].set_title("Target — a channel (green↔red)");  axes[1].axis("off")
    axes[2].imshow(ab_np[:, :, 1], cmap=b_cmap); axes[2].set_title("Target — b channel (blue↔yellow)"); axes[2].axis("off")
    axes[3].imshow(rgb);                             axes[3].set_title("Reconstructed RGB");         axes[3].axis("off")
    plt.tight_layout()
    plt.savefig("results/dataset_sanity_check.png", dpi=100)
    plt.show()
    print("\nSanity check passed! Saved to results/dataset_sanity_check.png")
