# BW2C — Black & White to Color

**CS 4263 Deep Learning — University of Texas at San Antonio**  
Sean Jauregui · Nathan Zuniga

A deep learning system that automatically colorizes grayscale images using a pretrained ResNet-18 encoder, dilated bottleneck, and U-Net decoder trained on 318,000 images in the CIE Lab color space.

🎨 **[Live Demo on Streamlit Cloud](https://cs4263bw2c.streamlit.app/)**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Datasets](#datasets)
- [Model Progression](#model-progression)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Course Concepts Applied](#course-concepts-applied)
- [Results](#results)
- [References](#references)

---

## Overview

Colorizing black and white images has practical value across several domains:

- **Historical preservation** — restoring photographs where color was never recorded
- **Artistic assistance** — helping artists explore color choices on monochrome artwork
- **Visual emphasis** — bringing attention to subject matter in monotone scenes

This project implements a **CNN-based colorization pipeline** operating in the CIE Lab color space. The L (lightness) channel is the grayscale input; the model predicts the a and b (chrominance) channels, which are then combined with the original L to reconstruct a full-color image.

---

## Architecture

### Color Space — CIE Lab

All processing is done in the **CIE Lab** color space rather than RGB:

| Channel | Meaning | Range | Role |
|---|---|---|---|
| **L** | Lightness | 0–100 | Model **input** (the grayscale image) |
| **a** | Green ↔ Red | ≈ −128 to +128 | Model **output** |
| **b** | Blue ↔ Yellow | ≈ −128 to +128 | Model **output** |

Normalization follows Zhang et al. (ECCV 2016):
- L: `(L − 50) / 100` → ≈ [−0.5, 0.5]
- ab: `ab / 110` → ≈ [−1, 1]

### Network: ResNet-18 Encoder + Dilated Bottleneck + U-Net Decoder

```
INPUT
  └─ L channel (1×256×256) → repeated to 3ch → (3×256×256)

ENCODER  [ResNet-18, pretrained on ImageNet — frozen in Phase 1]
  stem:    Conv(7×7, s=2) + BN + ReLU + MaxPool  → 64ch,  128×128  [skip0]
  layer1:  ResBlock×2                             → 64ch,   64×64  [skip1]
  layer2:  ResBlock×2, stride=2                   → 128ch,  32×32  [skip2]
  layer3:  ResBlock×2, stride=2                   → 256ch,  16×16  [skip3]
  layer4:  ResBlock×2, stride=2                   → 512ch,   8×8

BOTTLENECK  [Dilated convolutions — expands receptive field without spatial loss]
  DilatedConv(512→512, k=3, dilation=2) + BN + ReLU  × 2         → 512ch, 8×8

DECODER  [U-Net — skip connections preserve fine spatial detail]
  UpBlock1: ConvTranspose(512→256, s=2) + concat skip3 → Conv + BN + ReLU + Dropout(0.2)  → 256ch, 16×16
  UpBlock2: ConvTranspose(256→128, s=2) + concat skip2 → Conv + BN + ReLU + Dropout(0.2)  → 128ch, 32×32
  UpBlock3: ConvTranspose(128→64,  s=2) + concat skip1 → Conv + BN + ReLU                 →  64ch, 64×64
  UpBlock4: ConvTranspose(64→64,   s=2) + concat skip0 → Conv + BN + ReLU                 →  64ch, 128×128
  UpBlock5: ConvTranspose(64→32,   s=2)               → Conv + BN + ReLU                 →  32ch, 256×256
  Head:     Conv(32→2, k=1) + Tanh                                                        →   2ch, 256×256

OUTPUT
  predicted ab × 110  →  unnormalize
  combine with original L  →  lab2rgb()  →  final RGB image
```

**Trainable parameters:** ~21M total (encoder ~11M, decoder ~10M)

---

## Training Strategy

Training is split into two phases to leverage transfer learning:

| Phase | Encoder | Learning Rate | Purpose |
|---|---|---|---|
| **Phase 1** — Warm-up | Frozen | `3e-4` | Train decoder only; fast convergence |
| **Phase 2** — Fine-tune | Unfrozen | `1e-5` (cosine decay → `1e-7`) | End-to-end refinement |

### Loss Function

**Primary:** L1 loss on predicted vs. ground-truth normalized ab channels:

$$\mathcal{L}_{L1} = \frac{1}{H \cdot W} \sum_{h,w} \left| ab_{\text{pred}}(h,w) - ab_{\text{gt}}(h,w) \right|$$

L1 is preferred over MSE/L2 because L2's squared penalty causes the model to collapse toward desaturated (gray) predictions to minimize average error.

**Auxiliary (v5):** Colorfulness penalty — directly penalizes near-gray ab predictions to counteract L1's natural desaturation bias:

$$\mathcal{L}_{\text{color}} = -\frac{1}{BHW} \sum \sqrt{a_{\text{pred}}^2 + b_{\text{pred}}^2 + \epsilon}$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{L1} + 0.05 \cdot \mathcal{L}_{\text{color}}$$

### AMP (Automatic Mixed Precision)

Training uses float16 AMP for ~40% GPU memory reduction and faster throughput. The colorfulness penalty runs outside the AMP autocast block (forced float32) to avoid float16 epsilon underflow (`1e-8 → 0`, causing NaN gradients in the sqrt).

---

## Datasets

| Dataset | Images | Purpose |
|---|---|---|
| [COCO 2017 Train](https://cocodataset.org/#download) | ~118,000 | Diverse subjects: people, animals, objects, food, outdoor scenes |
| [ImageNet](https://www.image-net.org/) (subset) | ~200,000 | Additional variety for generalization |
| **Total** | **~318,000** | Combined training set |

No labels are used from either dataset — only the images. The L channel is the input; the ab channels are the supervision signal, derived automatically by converting each image to Lab space.

Validation is performed on the **COCO 2017 val** set (5,000 images).

---

## Model Progression

| Version | Backbone | Dataset | Add. Loss | Epochs | Best Val L1 | Train Time |
|---|---|---|---|---|---|---|
| **v1** | ResNet-18 | COCO 118k | — | 5+20 | 0.07057 | ~8h |
| **v2** | ResNet-18 | COCO 118k | — | 5+20 | 0.07057 | ~8.5h |
| **v3** | ResNet-18 | COCO+ImageNet 318k | — | 5+12 | 0.06922 | 19h 23m |
| **v4** | ResNet-34 | COCO+ImageNet 318k | VGG perceptual | 5+15 | 0.06824 | 48h 32m |
| **v5** ✅ | ResNet-18 | COCO+ImageNet 318k | Colorfulness penalty | 5+15 | 0.06947 | 20h 44m |

**v5 is the current deployed model.**

Key lessons from the progression:
- **v2 → v3**: Expanding training data from 118k → 318k images gave the largest single validation improvement (−0.00135 L1)
- **v3 → v4**: ResNet-34 + VGG perceptual loss improved raw L1 but introduced a cool-tone color bias (ImageNet's perceptual feature distribution skews toward cooler/neutral tones), making visual results worse on many scenes despite a better number
- **v4 → v5**: Replaced VGG with a lightweight colorfulness penalty — no cool-tone bias, comparable L1 to v3, and visually richer/warmer color predictions. Batch size doubled (32→64) for faster throughput

---

## Project Structure

```
BW2C/
├── src/
│   ├── dataset.py      # Dataset class, Lab conversion, DataLoader (supports multi-dir)
│   ├── model.py        # ResNetColorizer — encoder (resnet18/34), bottleneck, decoder
│   ├── train.py        # Training loop: Phase 1 + Phase 2, AMP, checkpointing, JSON logs
│   ├── inference.py    # Colorize a single image from checkpoint
│   └── evaluate.py     # PSNR metric + validation loop
├── checkpoints/        # v3 best checkpoint (ResNet-18, val L1=0.06922)
├── checkpoints_v5/     # v5 best checkpoint (ResNet-18 + colorfulness, val L1=0.06947)
├── results/            # v3 training metrics JSON
├── results_v5/         # v5 training metrics and summary JSON
├── v1_results/         # Sample outputs from each training run
├── v2_results/
├── v3_results/
├── v4_results/
├── v5_results/
├── streamlit_app.py    # 5-tab Streamlit demo: Inference, Metrics, Training Log, Summary, Model Comparison
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU strongly recommended (CPU inference works but is slow)

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

Download and extract into `data/train/`:

```
data/
├── train/
│   ├── train2017/        # COCO 2017 train images (~18 GB)
│   └── train_ImageNet/   # ImageNet subset
└── val/
    └── val2017/          # COCO 2017 val images (~1 GB)
```

---

## Usage

### Live Demo

The easiest way to try the model is the hosted Streamlit app (no setup required):

**[https://cs4263bw2c.streamlit.app/](https://cs4263bw2c.streamlit.app/)**

Upload any grayscale (or color) image and click **Colorize**.

### Training

```bash
# Windows — TORCHDYNAMO_DISABLE=1 required (no Triton support)
$env:TORCHDYNAMO_DISABLE=1

# v5-style training (ResNet-18 + colorfulness penalty, isolated output)
python src/train.py `
  --train-dir data/train/train2017 data/train/train_ImageNet `
  --val-dir data/val/val2017 `
  --batch-size 64 --num-workers 4 `
  --phase1-epochs 5 --phase2-epochs 15 `
  --lr2 1e-5 --amp --pretrained `
  --backbone resnet18 `
  --output-dir checkpoints_v5 `
  --results-dir results_v5
```

Key flags:

| Flag | Description |
|---|---|
| `--backbone resnet18\|resnet34` | Encoder architecture |
| `--amp` | Enable AMP float16 (recommended on CUDA) |
| `--pretrained` | Load ImageNet weights for encoder |
| `--output-dir` | Where checkpoints are saved |
| `--results-dir` | Where training_metrics.json and training_summary.json are saved |
| `--perceptual-weight 0.1` | Enable VGG perceptual loss (omit for v5-style) |

### Inference (Single Image)

```bash
python src/inference.py --input path/to/image.jpg --output results/colorized.png --checkpoint checkpoints_v5/best.pth
```

### Evaluation

```bash
python src/evaluate.py --checkpoint checkpoints_v5/best.pth --val-dir data/val/val2017
```

### Run Streamlit Locally

```bash
streamlit run streamlit_app.py
```

---

## Course Concepts Applied

| Concept | Application |
|---|---|
| **Transfer Learning** | ResNet-18 encoder pretrained on ImageNet; Phase 1 freezes encoder to train decoder only |
| **U-Net / Skip Connections** | Decoder receives encoder feature maps at 4 spatial scales to preserve spatial detail |
| **Dilated Convolutions** | Bottleneck uses dilation=2 to expand receptive field without downsampling |
| **CIE Lab Color Space** | Decouples luminance (input) from chrominance (prediction target) |
| **AMP Mixed Precision** | float16 forward pass with float32 auxiliary losses; GradScaler for stable training |
| **Two-Phase Training** | Frozen encoder warm-up → full fine-tune with lower LR + cosine annealing |
| **L1 Loss** | Preferred over MSE for color tasks — avoids variance-collapsing gray predictions |
| **Colorfulness Penalty** | Auxiliary loss maximizing predicted chroma to counteract L1's desaturation bias |
| **Data Augmentation** | Random horizontal flip, random crop during training |
| **Batch Normalization** | Applied in both encoder (via ResNet) and all decoder conv blocks |
| **Dropout** | 0.2 dropout in first two decoder blocks to reduce overfitting |

---

## Results

### Quantitative — All Versions

| Version | Backbone | Dataset | Additional Loss | Batch | Ph1 Ep | Ph2 Ep | Best Val L1 (ab) | Train Time |
|---|---|---|---|---|---|---|---|---|
| **v1** | ResNet-18 | COCO 118k | — | 32 | 5 | 20 | 0.07057 | ~8h |
| **v2** | ResNet-18 | COCO 118k | — | 32 | 5 | 20 | 0.07057 | ~8.5h |
| **v3** | ResNet-18 | COCO+ImageNet 318k | — | 32 | 5 | 12 | 0.06922 | 19h 23m |
| **v4** | ResNet-34 | COCO+ImageNet 318k | VGG perceptual (0.1×) | 32 | 5 | 15 | 0.06824 | 48h 32m |
| **v5** ✅ | ResNet-18 | COCO+ImageNet 318k | Colorfulness penalty (0.05×) | 64 | 5 | 15 | 0.06947 | 20h 44m |

### Key Findings

- **Data scale (v2→v3)** was the single largest improvement driver, cutting Val L1 by −0.00135 with no architectural change.
- **v4 achieved the best raw L1 (0.06824)** but required 48.5h of training (2.3× longer than v5) and introduced a systematic cool-tone bias from VGG's ImageNet feature distribution, making visual quality worse on warm-toned scenes.
- **v5 recovers near-v3 L1 efficiency** while producing warmer, more vivid colorizations. The colorfulness penalty adds negligible compute and fully eliminates the cool-tone bias. Doubling batch size (32→64) reduced wall-clock time relative to v3 despite 3 additional fine-tune epochs.
- v1 and v2 produced identical Val L1, confirming the COCO-only 118k data ceiling; more epochs did not help.

### Qualitative

Visual outputs from all versions are stored in `v1_results/` through `v5_results/`. The **Model Comparison** tab in the Streamlit app displays all five versions side-by-side for direct visual comparison.

---

## Conclusion

This project successfully developed and iteratively refined a supervised CNN-based image colorization pipeline across five model versions. Starting from a ResNet-18 + U-Net baseline trained on COCO 2017, each version introduced a targeted change — expanded training data, a deeper encoder, VGG perceptual loss, and a colorfulness penalty — with each decision directly motivated by observed failure modes in the prior version.

The final deployed model (v5) demonstrates that a lightweight colorfulness penalty is a more practical solution to L1's desaturation bias than VGG perceptual loss: it achieves comparable Val L1 to v3 (0.06947 vs. 0.06922), trains in under 21 hours (vs. 48.5h for v4), and produces warmer, more coherent colorizations without the cool-tone bias introduced by v4.

**Limitations:**
- Desaturation persists on high-frequency textured regions (dense foliage, fabric) where ab is ambiguous from luminance alone.
- Rare or unusual colors (neon signage, brightly painted vehicles) are underrepresented in COCO+ImageNet and are often predicted conservatively.
- Predicted ab at 256×256 must be upscaled to original resolution, which can introduce minor color boundary bleeding near sharp edges.

**Future Work:**
- **Transformer encoder** — replacing ResNet with a ViT or Swin Transformer for longer-range spatial context, critical for uniform-color regions like sky and water.
- **Class-conditioned colorization** — incorporating scene labels (as in Iizuka et al. [4]) or text prompts (as in Nishio and Miyata [3]) to resolve inherent color ambiguity.
- **Adversarial refinement** — adding a lightweight PatchGAN discriminator (following Isola et al. [2]) as a post-training fine-tune to sharpen color boundaries.
- **Domain-specific perceptual loss** — training a perceptual network on colorization data rather than ImageNet VGG to avoid the cool-tone bias observed in v4.

---

## References

[1] R. Zhang, P. Isola, and A. A. Efros, "Colorful image colorization," in *Computer Vision – ECCV 2016*, B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds. Cham, Switzerland: Springer, 2016.

[2] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017.

[3] R. Nishio and T. Miyata, "Zero-shot class conditioned image colorization by using pretrained diffusion models," *Nonlinear Theory and Its Applications, IEICE*, vol. 16, no. 4, 2025, doi: 10.1587/nolta.16.896.

[4] S. Iizuka, E. Simo-Serra, and H. Ishikawa, "Let there be color!: Joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification," *ACM Trans. Graph.*, vol. 35, no. 4, 2016.

[5] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*. Cham, Switzerland: Springer, 2015.

[6] "What Is LAB Color? - TruHu Blog," TruHu Blog - Good color made easy, Jan. 22, 2024. https://truhu.app/blog/what-is-lab-color/.

[7] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in *Proc. ECCV*, 2014.

[8] O. Russakovsky et al., "ImageNet large scale visual recognition challenge," *Int. J. Comput. Vis.*, vol. 115, no. 3, pp. 211–252, 2015.

[9] J. Johnson, A. Alahi, and L. Fei-Fei, "Perceptual losses for real-time style transfer and super-resolution," in *Proc. ECCV*, 2016.