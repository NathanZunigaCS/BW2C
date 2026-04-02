# streamlit_app.py
import os
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage import color
from src.model import ResNetColorizer
from src.dataset import normalize_l, unnormalize_ab, lab_to_rgb
from src.evaluate import compute_psnr  # re-export or duplicate easy function

CHECKPOINT = "checkpoints/best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

@st.cache_resource
def load_model(pretrained=False):
    model = ResNetColorizer(pretrained=pretrained).to(DEVICE)
    if Path(CHECKPOINT).exists():
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def colorize_image(model, image: Image.Image):
    # Keep original size for display; only resize L to 256x256 for model input
    img_full = image.convert("RGB")
    orig_w, orig_h = img_full.size

    img_np_full = np.asarray(img_full, dtype=np.float32) / 255.0
    lab_full = color.rgb2lab(img_np_full)
    l_full = lab_full[:, :, 0]  # original resolution L, used for final output

    # Resize L to model input size
    l_pil = Image.fromarray(l_full.astype(np.float32)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
    l_small = np.asarray(l_pil, dtype=np.float32)
    l_norm = normalize_l(l_small)
    l_tensor = torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        ab_pred_small = model(l_tensor).cpu().numpy()[0].transpose(1, 2, 0)  # (256,256,2)
    ab_pred_small = unnormalize_ab(ab_pred_small)

    # Saturation boost: push muted predictions toward richer color
    # 1.4x amplifies weak ab values (orange, red, purple) without blowing out greens/blues
    ab_pred_small = np.clip(ab_pred_small * 1.4, -110, 110)

    # Upscale predicted ab back to original resolution
    ab_a = Image.fromarray(ab_pred_small[:, :, 0].astype(np.float32)).resize((orig_w, orig_h), resample=Image.BICUBIC)
    ab_b = Image.fromarray(ab_pred_small[:, :, 1].astype(np.float32)).resize((orig_w, orig_h), resample=Image.BICUBIC)
    ab_pred_full = np.stack([np.asarray(ab_a), np.asarray(ab_b)], axis=-1)

    output_lab = np.concatenate([l_full[:, :, np.newaxis], ab_pred_full], axis=-1).astype(np.float32)
    output_rgb = np.clip(color.lab2rgb(output_lab), 0, 1)
    output_img = Image.fromarray((output_rgb * 255).astype(np.uint8))
    return output_img

def get_training_metrics():
    stats = {}
    # Option 1: load existing metrics JSON if available
    mpath = Path("results/training_metrics.json")
    if mpath.exists():
        try:
            stats = json.loads(mpath.read_text())
        except Exception:
            stats = {"error": "Unable to parse results/training_metrics.json"}
    else:
        stats["note"] = "No training_metrics.json found."
    return stats

def get_latest_eval_metrics():
    out = {}
    val_dir = Path("data/val/val2017")
    if val_dir.exists():
        out["status"] = "computed"
        out["len"] = sum(1 for _ in val_dir.rglob("*.jpg"))
    else:
        out["status"] = "missing val folder"
    return out

def main():
    st.title("BW2C Colorization Demo")
    st.write("Simple Streamlit frontend for grayscale-to-color model with metrics tabs.")

    model = load_model(pretrained=False)
    st.sidebar.success(f"Model loaded ({DEVICE})")

    tabs = st.tabs(["Inference", "Metrics", "Training log"])
    with tabs[0]:
        st.subheader("Inference")
        uploaded = st.file_uploader("Upload grayscale image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            input_img = Image.open(uploaded).convert("RGB")
            st.image(input_img, caption="Input image", use_container_width=True)
            if st.button("Colorize"):
                with st.spinner("Running model..."):
                    out_img = colorize_image(model, input_img)
                    st.image(out_img, caption="Colorized output", use_container_width=True)
                    out_path = Path("results/streamlit_colorized.png")
                    out_path.parent.mkdir(exist_ok=True, parents=True)
                    out_img.save(out_path)
                    st.success(f"Saved output to {out_path}")

    with tabs[1]:
        st.subheader("Evaluation / Data summary")
        st.write("Validation dataset status:")
        val_path = Path("data/val/val2017")
        if val_path.exists():
            n_val = sum(1 for _ in val_path.rglob("*.jpg"))
            st.write(f"val image count: {n_val}")
        else:
            st.write("val folder not found: data/val/val2017")
        st.write("Basic metrics:")
        eval_metrics = get_latest_eval_metrics()
        st.json(eval_metrics)

    with tabs[2]:
        st.subheader("Training metrics (if available)")
        tm = get_training_metrics()
        st.json(tm)

if __name__ == "__main__":
    main()