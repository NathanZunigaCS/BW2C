"""train.py — training loop for BW2C colorization (v4: ResNet-34 + VGG perceptual loss)"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model import ResNetColorizer, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="Train BW2C colorization model")
    p.add_argument("--train-dir", type=str, nargs='+', default=["data/train"])
    p.add_argument("--val-dir", type=str, default="data/val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--phase1-epochs", type=int, default=5)
    p.add_argument("--phase2-epochs", type=int, default=20)
    p.add_argument("--lr1", type=float, default=3e-4)
    p.add_argument("--lr2", type=float, default=1e-5)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable AMP mixed precision (faster on CUDA)")
    p.add_argument("--perceptual-weight", type=float, default=0.0,
                   help="Weight for VGG perceptual loss (0 = disabled). Recommended: 0.1")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Limit batches per epoch (quick NaN smoke-test, e.g. --max-batches 20)")
    return p.parse_args()


def colorfulness_penalty(pred_ab: torch.Tensor) -> torch.Tensor:
    """Encourage chromatic predictions by penalizing low-saturation outputs.
    Motivated by Zhang et al. ECCV 2016 — L1 regression suppresses vivid colors
    by averaging toward gray; this term counteracts that bias.
    pred_ab: (B, 2, H, W) normalized to [-1, 1]
    """
    chroma = torch.sqrt(pred_ab[:, 0] ** 2 + pred_ab[:, 1] ** 2 + 1e-4)  # 1e-4 safe in float16
    return -chroma.mean()  # negative = maximize chroma


def lab_to_rgb_batch(L_norm: torch.Tensor, ab_norm: torch.Tensor) -> torch.Tensor:
    """Differentiable CIE Lab → sRGB conversion for use in perceptual loss.
    L_norm : (B, 1, H, W) normalized L  — range [-0.5, 0.5] (dataset convention)
    ab_norm: (B, 2, H, W) normalized ab — range [-1,   1]   (dataset convention)
    Returns: (B, 3, H, W) sRGB in [0, 1]
    """
    L  = L_norm  * 100.0 + 50.0    # [0, 100]
    ab = ab_norm * 110.0            # [-110, 110]
    a  = ab[:, 0:1]
    b  = ab[:, 1:2]

    # Lab → XYZ (D65 illuminant, CIE standard observer)
    fy = (L + 16.0) / 116.0
    fx =  a / 500.0 + fy
    fz = fy - b / 200.0

    eps   = 6.0 / 29.0                       # ≈ 0.2069
    kappa = 3.0 * eps * eps                  # ≈ 0.1284

    def f_inv(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.pow(3), kappa * (t - 16.0 / 116.0))

    X = 0.950456 * f_inv(fx)
    Y = 1.000000 * f_inv(fy)
    Z = 1.089058 * f_inv(fz)

    # XYZ → linear sRGB (IEC 61966-2-1 matrix)
    R_lin =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    B_lin =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    # sRGB gamma encode
    def gamma(c: torch.Tensor) -> torch.Tensor:
        c = c.clamp(0.0, 1.0)
        # torch.where evaluates BOTH branches for all pixels, then masks.
        # Gradient of c.pow(1/2.4) at c=0 is inf; mask * inf = NaN (IEEE 754).
        # Fix: clamp the base of pow to 0.0031308 so its gradient is always finite.
        c_pow = c.clamp(min=0.0031308)
        return torch.where(c <= 0.0031308,
                           12.92 * c,
                           1.055 * c_pow.pow(1.0 / 2.4) - 0.055)

    return torch.cat([gamma(R_lin), gamma(G_lin), gamma(B_lin)], dim=1)  # (B,3,H,W)


class PerceptualLoss(nn.Module):
    """VGG-16 perceptual loss using relu1_2 and relu2_2 feature activations.
    Encourages structural and chromatic fidelity beyond per-pixel L1.
    Based on Johnson et al. (2016) 'Perceptual Losses for Real-Time Style Transfer'.
    The VGG weights are frozen — no gradients flow through them.
    """
    def __init__(self):
        super().__init__()
        vgg   = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        feats = vgg.features
        self.slice1 = nn.Sequential(*list(feats.children())[:5])   # up to relu1_2
        self.slice2 = nn.Sequential(*list(feats.children())[5:10]) # up to relu2_2
        for p in self.parameters():
            p.requires_grad = False
        # ImageNet channel-wise normalization expected by VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
        """pred_rgb, gt_rgb: (B, 3, H, W) in [0, 1]"""
        pred = (pred_rgb - self.mean) / self.std
        gt   = (gt_rgb   - self.mean) / self.std
        f1_p = self.slice1(pred);   f1_g = self.slice1(gt)
        f2_p = self.slice2(f1_p);   f2_g = self.slice2(f1_g)
        return F.l1_loss(f1_p, f1_g) + F.l1_loss(f2_p, f2_g)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None,
                perceptual_fn=None, perc_weight=0.1, max_batches=None):
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (l, ab) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        l = l.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            # AMP: only model forward + L1 run in float16
            with torch.amp.autocast('cuda'):
                pred_ab = model(l)
                loss = criterion(pred_ab, ab)
            # All auxiliary losses in float32 — float16 epsilon underflow causes NaN gradients
            with torch.amp.autocast('cuda', enabled=False):
                pred_ab_f = pred_ab.float()
                loss = loss + 0.05 * colorfulness_penalty(pred_ab_f)
                if perceptual_fn is not None:
                    pred_rgb = lab_to_rgb_batch(l.float(), pred_ab_f)
                    gt_rgb   = lab_to_rgb_batch(l.float(), ab.float())
                    loss = loss + perc_weight * perceptual_fn(pred_rgb, gt_rgb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_ab = model(l)
            loss = criterion(pred_ab, ab) + 0.05 * colorfulness_penalty(pred_ab)
            if perceptual_fn is not None:
                pred_rgb = lab_to_rgb_batch(l, pred_ab)
                gt_rgb   = lab_to_rgb_batch(l, ab)
                loss = loss + perc_weight * perceptual_fn(pred_rgb, gt_rgb)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * l.size(0)
        count += l.size(0)

    return running_loss / max(count, 1)


def eval_epoch(model, dataloader, criterion, device, max_batches=None):
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (l, ab) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            l = l.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)

            pred_ab = model(l)
            loss = criterion(pred_ab, ab)

            running_loss += loss.item() * l.size(0)
            count += l.size(0)

    return running_loss / max(count, 1)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # cuDNN auto-tunes convolution algorithms for fixed input size — free ~5-10% speed
    torch.backends.cudnn.benchmark = True

    # AMP scaler — only active when --amp flag is passed
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == "cuda") else None
    if scaler:
        print("AMP mixed precision enabled")

    model = ResNetColorizer(pretrained=args.pretrained)
    model.to(device)

    # torch.compile: fuses ops and generates optimized CUDA kernels (~10-20% faster training)
    # Requires PyTorch 2.0+. Falls back gracefully if unavailable.
    if device.type == "cuda":
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            print("torch.compile not available, skipping")

    print("Model created, trainable params:", count_parameters(model))

    train_loader, val_loader = get_dataloaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    criterion = nn.L1Loss()

    # Perceptual loss — loaded only when --perceptual-weight > 0
    perceptual_fn = None
    if args.perceptual_weight > 0.0:
        print(f"VGG perceptual loss enabled (weight={args.perceptual_weight})")
        perceptual_fn = PerceptualLoss().to(device).eval()  # eval() prevents BatchNorm training-mode drift

    start_epoch = 0
    best_val = float("inf")
    training_run_start = time.time()
    epoch_log = []  # collects per-epoch timing + loss for summary

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume and args.checkpoint and os.path.exists(args.checkpoint):
        print("Resuming from checkpoint", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_val = ckpt.get("best_val", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1

    # Phase 1: freeze encoder
    if args.phase1_epochs > 0:
        model.freeze_encoder()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr1)

        for epoch in range(start_epoch, start_epoch + args.phase1_epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                     perceptual_fn=perceptual_fn, perc_weight=args.perceptual_weight,
                                     max_batches=args.max_batches)
            val_loss = eval_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
            t1 = time.time()

            print(f"[Phase1] Epoch {epoch+1}/{start_epoch + args.phase1_epochs} | train: {train_loss:.5f} | val: {val_loss:.5f} | {t1-t0:.1f}s")

            epoch_log.append({"phase": 1, "epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": round(t1-t0, 1)})

            ckpt_path = os.path.join(args.output_dir, f"best_phase1_epoch{epoch+1}.pth")
            save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "best_val": val_loss}, ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "best_val": best_val}, os.path.join(args.output_dir, "best.pth"))

        # Phase 2 start epoch numbering
        start_epoch += args.phase1_epochs

    # Phase 2: unfreeze full model
    model.unfreeze_encoder()
    optimizer = Adam(model.parameters(), lr=args.lr2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs, eta_min=1e-7)

    for epoch in range(start_epoch, start_epoch + args.phase2_epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                 perceptual_fn=perceptual_fn, perc_weight=args.perceptual_weight,
                                 max_batches=args.max_batches)
        val_loss = eval_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
        t1 = time.time()

        print(f"[Phase2] Epoch {epoch-start_epoch+1}/{args.phase2_epochs} | train: {train_loss:.5f} | val: {val_loss:.5f} | {t1-t0:.1f}s")

        scheduler.step()
        epoch_log.append({"phase": 2, "epoch": epoch-start_epoch+1, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": round(t1-t0, 1)})

        ckpt_path = os.path.join(args.output_dir, f"best_phase2_epoch{epoch+1}.pth")
        save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "best_val": val_loss}, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "best_val": best_val}, os.path.join(args.output_dir, "best.pth"))

        result_file = Path("results/training_metrics.json")
        stats = {"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": round(t1-t0, 1)}
        if result_file.exists():
            existing = json.loads(result_file.read_text())
        else:
            existing = []
        existing.append(stats)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(existing, indent=2))

    total_elapsed = time.time() - training_run_start
    summary = {
        "total_elapsed_s": round(total_elapsed, 1),
        "total_elapsed_human": f"{int(total_elapsed//3600)}h {int((total_elapsed%3600)//60)}m {int(total_elapsed%60)}s",
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "best_val_l1_ab": round(best_val, 6),
        "batch_size": args.batch_size,
        "train_dirs": args.train_dir,
        "amp": args.amp,
        "perceptual_weight": args.perceptual_weight,
        "epochs": epoch_log,
    }
    summary_file = Path("results/training_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Training summary saved to {summary_file}")
    print("Training complete. Best val L1(ab) =", best_val)


if __name__ == "__main__":
    main()
