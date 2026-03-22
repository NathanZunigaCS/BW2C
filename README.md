# BW2C — Black & White to Color

**CS 4263 Deep Learning — University of Texas at San Antonio**  
Sean Jauregui · Nathan Zuniga

A deep learning system that colorizes grayscale images by predicting color information from grayscale inputs using a CNN with a pretrained ResNet-18 backbone.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Course Concepts Applied](#course-concepts-applied)
- [Results](#results)
- [References](#references)

---

## Overview

Colorizing black and white images has practical value in several domains:
- **Historical preservation** — restoring photographs where color was never recorded
- **Artistic assistance** — helping artists explore color choices on monochrome artwork
- **Visual emphasis** — bringing attention to subject matter in monotone scenes

Traditional machine learning approaches can produce acceptable results, but achieving high accuracy requires deep learning techniques. This project uses a **Convolutional Neural Network (CNN)** with a pretrained backbone and a U-Net style decoder to learn the complex mapping from grayscale (luminance) to color (chrominance).

---

## Architecture

### Color Space — CIE Lab

All processing is done in the **CIE Lab color space** rather than RGB. This is the foundational design decision of the project:

- **L channel** (Lightness, range 0–100) — this IS the grayscale image. It is the model's input.
- **a channel** (green ↔ red, range ≈ −128 to +128) — predicted by model
- **b channel** (blue ↔ yellow, range ≈ −128 to +128) — predicted by model

At inference time: original L + predicted ab → `lab2rgb()` → final color image.

Normalization follows the constants established in Zhang et al. (ECCV 2016):
- L normalized: `(L − 50) / 100` → roughly `[−0.5, 0.5]`
- ab normalized: `ab / 110` → roughly `[−1, 1]`

### Network: ResNet-18 Encoder + Dilated Bottleneck + U-Net Decoder

```
INPUT
  └─ L channel (1×256×256)
       └─ Repeated to 3 channels → (3×256×256)

ENCODER  [ResNet-18, pretrained on ImageNet — frozen in Phase 1]
  stem:    Conv(7×7, s=2) + BN + ReLU + MaxPool(3×3, s=2)  → 64ch,  64×64  [skip1]
  layer1:  ResBlock×2                                        → 64ch,  64×64
  layer2:  ResBlock×2, stride=2                              → 128ch, 32×32  [skip2]
  layer3:  ResBlock×2, stride=2                              → 256ch, 16×16  [skip3]
  layer4:  ResBlock×2, stride=2                              → 512ch,  8×8   [bottleneck]

BOTTLENECK  [Dilated convolutions — expands receptive field without spatial loss]
  DilatedConv(512→512, k=3, dilation=2) + BN + ReLU
  DilatedConv(512→512, k=3, dilation=2) + BN + ReLU         → 512ch, 8×8

DECODER  [U-Net style — skip connections preserve spatial detail]
  UpBlock1: ConvTranspose(512→256, s=2) → concat skip3 → Conv(512→256) + BN + ReLU + Dropout(0.2)  → 256ch, 16×16
  UpBlock2: ConvTranspose(256→128, s=2) → concat skip2 → Conv(256→128) + BN + ReLU + Dropout(0.2)  → 128ch, 32×32
  UpBlock3: ConvTranspose(128→64,  s=2) → concat skip1 → Conv(128→64)  + BN + ReLU                 →  64ch, 64×64
  UpBlock4: ConvTranspose(64→64,   s=2) →                Conv(64→64)   + BN + ReLU                 →  64ch, 128×128
  UpBlock5: ConvTranspose(64→32,   s=2) →                Conv(32→32)   + BN + ReLU                 →  32ch, 256×256
  Head:     Conv(32→2, k=1) + Tanh                                                                  →   2ch, 256×256

OUTPUT RECONSTRUCTION
  predicted ab × 110  →  unnormalize
  combine with original L
  skimage.color.lab2rgb()  →  final RGB image
```

### Training Strategy

Training is split into two phases to leverage transfer learning effectively:

| Phase | Encoder | Decoder | Learning Rate | Epochs |
|---|---|---|---|---|
| **Phase 1** — Warm-up | Frozen | Training | `3e-4` | 5–10 |
| **Phase 2** — Fine-tune | Unfrozen | Training | `1e-5` | 20–50 |

**Loss Function:** L1 loss on predicted vs. ground-truth ab channels.

$$\mathcal{L}_{L1} = \frac{1}{H \cdot W} \sum_{h,w} | ab_{pred}(h,w) - ab_{gt}(h,w) |$$

L1 is preferred over L2 because L2's squared penalty causes the model to "play it safe" and predict desaturated colors. L1 allows more color diversity.

---

## Datasets

| Dataset | Images | Purpose |
|---|---|---|
| [COCO 2017 Train](https://cocodataset.org/#download) | ~118,000 | Primary training — diverse subjects (people, animals, objects, food, outdoor scenes) |
| [MIT Places205](http://places.csail.mit.edu/) | Subset (~50k) | Supplementary — scene/landscape focused, improves outdoor colorization |

**No labels are used from either dataset.** Only the images are needed. The L channel is the input; the ab channels are the supervision signal — both derived automatically by converting each image to Lab space.

---

## Project Structure

```
BW2C/
├── data/
│   ├── train/          # training images (not committed — see .gitignore)
│   └── val/            # validation images
├── src/
│   ├── dataset.py      # Dataset class, Lab conversion, DataLoader setup
│   ├── model.py        # ResNetColorizer — encoder, bottleneck, decoder
│   ├── train.py        # Training loop (Phase 1 + Phase 2), checkpointing
│   ├── inference.py    # Run colorization on a single B&W image
│   └── evaluate.py     # PSNR, SSIM metrics + comparison grid output
├── checkpoints/        # Saved model weights (not committed)
├── results/            # Colorized output images
├── Presentation Assets/
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU training is very slow for this task)

### Installation

```bash
git clone https://github.com/NathanZunigaCS/BW2C.git
cd BW2C
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Dataset Download

**COCO 2017:**
```bash
# Download train2017 images (~18GB) from https://cocodataset.org/#download
# Extract to data/train/
```

**MIT Places (optional):**
```bash
# Request access at http://places.csail.mit.edu/
# Extract subset to data/train/ alongside COCO images
```

---

## Usage

### Training

```bash
# Phase 1 only (frozen encoder, fast)
python src/train.py --phase 1 --epochs 10 --lr 3e-4

# Phase 2 fine-tuning (requires Phase 1 checkpoint)
python src/train.py --phase 2 --epochs 30 --lr 1e-5 --checkpoint checkpoints/phase1_best.pth

# Full run (both phases back-to-back)
python src/train.py --full
```

### Inference

```bash
# Colorize a single image
python src/inference.py --input path/to/bw_image.jpg --output results/colorized.png

# Colorize all images in a folder
python src/inference.py --input_dir path/to/bw_folder/ --output_dir results/
```

### Evaluation

```bash
# Compute PSNR and SSIM on validation set + generate comparison grid
python src/evaluate.py --checkpoint checkpoints/phase2_best.pth --val_dir data/val/
```

---

## Course Concepts Applied

This project directly applies material from CS 4263 lectures:

| Concept | Lecture | Application |
|---|---|---|
| Convolutional layers (kernels, stride, padding) | DL-6 | Every layer of the encoder and decoder |
| Pooling (max, global) | DL-6 | ResNet-18 stem (MaxPool) and global context |
| Effective Receptive Field | DL-6 | Motivates bottleneck design and dilation |
| CNN architectures — ResNet, shortcut connections | DL-7 | Pretrained ResNet-18 backbone |
| Dropout regularization | DL-8 | `Dropout2d(0.2)` in decoder UpBlocks |
| Dilated convolutions | DL-8 | Bottleneck layers to expand receptive field |

---

## Results

*To be populated after training.*

| Metric | Value |
|---|---|
| PSNR (validation) | — |
| SSIM (validation) | — |

Sample outputs will be added to the `results/` folder.

---

## References

1. **Zhang, R., Isola, P., Efros, A.A.** — *Colorful Image Colorization*, ECCV 2016. https://arxiv.org/abs/1603.08511
2. **He, K., Zhang, X., Ren, S., Sun, J.** — *Deep Residual Learning for Image Recognition*, CVPR 2016. https://arxiv.org/abs/1512.03385
3. **Ronneberger, O., Fischer, P., Brox, T.** — *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015. https://arxiv.org/abs/1505.04597
4. **Yu, F., Koltun, V.** — *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016. https://arxiv.org/abs/1511.07122
5. **Srivastava, N., Hinton, G., et al.** — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 2014.
6. Lin, T.Y., et al. — *Microsoft COCO: Common Objects in Context*, ECCV 2014. https://arxiv.org/abs/1405.0312
7. Zhou, B., et al. — *Places: A 10 Million Image Database for Scene Recognition*, TPAMI 2017.
