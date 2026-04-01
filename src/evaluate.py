"""evaluate.py — evaluate colorization model on validation set"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_dataloaders, unnormalize_ab, lab_to_rgb
from src.model import ResNetColorizer


def compute_psnr(img1, img2, max_val=1.0):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(description="Evaluate BW2C colorization model")
    parser.add_argument("--val-dir", type=str, default="data/val")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    model = ResNetColorizer(pretrained=args.pretrained).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_loader = get_dataloaders(
        train_dir="data/train",
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    criterion = nn.L1Loss(reduction="mean")

    total_l1 = 0.0
    total_psnr = 0.0
    num_batches = 0

    with torch.no_grad():
        for l, ab in tqdm(val_loader, desc="Evaluating"):
            l = l.to(device)
            ab = ab.to(device)

            pred_ab = model(l)
            l1 = criterion(pred_ab, ab).item()
            total_l1 += l1

            # PSNR in RGB domain (reconstruct one batch for estimated perceptual quality)
            # using small sample to reduce compute costs.
            l_cpu = l.cpu()
            ab_true_cpu = ab.cpu()
            ab_pred_cpu = pred_ab.cpu()

            rgb_true = []
            rgb_pred = []
            for i in range(l_cpu.size(0)):
                l_np = (l_cpu[i, 0].numpy() * 100.0) + 50.0  # unnormalize l to [0,100]
                ab_true_np = unnormalize_ab(ab_true_cpu[i].permute(1, 2, 0).numpy())
                ab_pred_np = unnormalize_ab(ab_pred_cpu[i].permute(1, 2, 0).numpy())

                rgb_true.append(lab_to_rgb(l_np, ab_true_np))
                rgb_pred.append(lab_to_rgb(l_np, ab_pred_np))

            rgb_true_t = torch.tensor(np.stack(rgb_true)).permute(0, 3, 1, 2)
            rgb_pred_t = torch.tensor(np.stack(rgb_pred)).permute(0, 3, 1, 2)
            total_psnr += compute_psnr(rgb_pred_t, rgb_true_t).item()

            num_batches += 1

    print(f"Mean L1(ab) on val: {total_l1/num_batches:.6f}")
    print(f"Mean PSNR on val: {total_psnr/num_batches:.2f} dB")


if __name__ == "__main__":
    main()
