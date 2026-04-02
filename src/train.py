"""train.py — training loop for BW2C colorization"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from model import ResNetColorizer, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="Train BW2C colorization model")
    p.add_argument("--train-dir", type=str, default="data/train")
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
    return p.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    count = 0

    for l, ab in dataloader:
        l = l.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            # AMP: run forward in float16, keep loss scaling stable
            with torch.amp.autocast('cuda'):
                pred_ab = model(l)
                loss = criterion(pred_ab, ab)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_ab = model(l)
            loss = criterion(pred_ab, ab)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * l.size(0)
        count += l.size(0)

    return running_loss / max(count, 1)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for l, ab in dataloader:
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

    start_epoch = 0
    best_val = float("inf")

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
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss = eval_epoch(model, val_loader, criterion, device)
            t1 = time.time()

            print(f"[Phase1] Epoch {epoch+1}/{start_epoch + args.phase1_epochs} | train: {train_loss:.5f} | val: {val_loss:.5f} | {t1-t0:.1f}s")

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

    for epoch in range(start_epoch, start_epoch + args.phase2_epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        t1 = time.time()

        print(f"[Phase2] Epoch {epoch-start_epoch+1}/{args.phase2_epochs} | train: {train_loss:.5f} | val: {val_loss:.5f} | {t1-t0:.1f}s")

        ckpt_path = os.path.join(args.output_dir, f"best_phase2_epoch{epoch+1}.pth")
        save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "best_val": val_loss}, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "best_val": best_val}, os.path.join(args.output_dir, "best.pth"))

        result_file = Path("results/training_metrics.json")
        stats = {"epoch": epoch+1, "train_loss":train_loss, "val_loss":val_loss}
        if result_file.exists():
            existing = json.loads(result_file.read_text())
        else:
            existing = []
        existing.append(stats)
        result_file.write_text(json.dumps(existing, indent=2))

    print("Training complete. Best val L1(ab) =", best_val)


if __name__ == "__main__":
    main()
