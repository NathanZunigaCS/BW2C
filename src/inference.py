"""inference.py — run colorization on test images"""

import argparse
import os

import numpy as np
from PIL import Image
from skimage import color
import torch

from dataset import normalize_l, unnormalize_ab
from model import ResNetColorizer


def load_model(checkpoint_path, device, pretrained=False):
    model = ResNetColorizer(pretrained=pretrained)
    model = model.to(device)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint {checkpoint_path}")
    model.eval()
    return model


def colorize_image(model, image_path, output_path, device, img_size=256):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size), resample=Image.BICUBIC)

    img_np = np.array(image, dtype=np.float32) / 255.0
    img_lab = color.rgb2lab(img_np)

    l_channel = img_lab[:, :, 0]  # [0,100]
    l_norm = normalize_l(l_channel)
    l_tensor = torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).to(device).float()

    with torch.no_grad():
        ab_pred_norm = model(l_tensor)

    ab_pred_norm = ab_pred_norm.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    ab_pred = unnormalize_ab(ab_pred_norm)

    lab_pred = np.concatenate([l_channel[:, :, np.newaxis], ab_pred], axis=-1).astype(np.float32)
    rgb_pred = color.lab2rgb(lab_pred)

    rgb_pred = np.clip(rgb_pred, 0.0, 1.0)
    rgb_img = (rgb_pred * 255.0).astype(np.uint8)
    Image.fromarray(rgb_img).save(output_path)
    print(f"Saved colorized image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Colorize grayscale images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--input", type=str, required=True, help="Input image or folder")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images to process")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, pretrained=args.pretrained)

    os.makedirs(args.output, exist_ok=True)

    def list_images(dir_path):
        return [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    if os.path.isdir(args.input):
        images = list_images(args.input)

        # Handle nested COCO extraction pattern (data/val/val2017/val2017)
        if not images:
            for child in sorted(os.listdir(args.input)):
                child_path = os.path.join(args.input, child)
                if os.path.isdir(child_path):
                    images.extend(list_images(child_path))

        if not images:
            raise FileNotFoundError(f"No images found in '{args.input}' or its immediate subdirectories.")

        if args.limit > 0:
            images = images[:args.limit]

        for inp in images:
            file_name = os.path.basename(inp)
            out = os.path.join(args.output, f"colorized_{file_name}")
            colorize_image(model, inp, out, device, img_size=args.size)
    else:
        base = os.path.basename(args.input)
        if not base:
            base = "colorized.png"
        else:
            base = f"colorized_{base}"
        out = os.path.join(args.output, base)
        colorize_image(model, args.input, out, device, img_size=args.size)


if __name__ == "__main__":
    main()
