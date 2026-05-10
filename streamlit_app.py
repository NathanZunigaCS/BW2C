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

CHECKPOINT = "checkpoints_v5/best.pth"  # v5 model (ResNet-18 + colorfulness penalty)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

@st.cache_resource
def load_model(pretrained=False):
    model = ResNetColorizer(pretrained=pretrained, backbone='resnet18').to(DEVICE)
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
    mpath = Path("results_v5/training_metrics.json")
    if mpath.exists():
        try:
            stats = json.loads(mpath.read_text())
        except Exception:
            stats = {"error": "Unable to parse results_v5/training_metrics.json"}
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

    tabs = st.tabs(["Inference", "Metrics", "Training Log", "Training Summary"])
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
        st.write("Model: ResNet-18 encoder + U-Net decoder, trained on COCO 2017 + ImageNet (318k images)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Val L1 (ab)", "0.06947")
        col2.metric("Train Images", "~318,000")
        col3.metric("Val Images", "5,000")
        st.markdown("---")
        st.markdown("""
**Evaluation notes:**
- Validation performed on COCO 2017 val set (5,000 images) at end of each epoch
- Loss metric: Mean Absolute Error on normalized *ab* channels (range \u00b1110)
- Best checkpoint saved at phase\u202f2 epoch\u202f11 (overall epoch\u202f16)
- Training: 5 encoder-frozen phase\u20091 epochs + 15 full fine-tune phase\u20092 epochs
- Colorfulness penalty (0.05\u00d7) added to push predictions away from desaturated gray
        """)

    with tabs[2]:
        st.subheader("Per-epoch training metrics")
        tm = get_training_metrics()
        if isinstance(tm, list) and len(tm) > 0:
            import pandas as pd
            df = pd.DataFrame(tm)
            cols = [c for c in ["epoch", "train_loss", "val_loss", "elapsed_s"] if c in df.columns]
            df = df[cols]
            st.caption("Phase 2 epochs shown (phase 1 logs not persisted). Epochs 6–17 = phase 2 epochs 1–12.")
            st.dataframe(df, use_container_width=True)
            if "val_loss" in df.columns and "train_loss" in df.columns:
                chart_df = df[["train_loss", "val_loss"]].reset_index(drop=True)
                chart_df.index = df["epoch"].values
                chart_df.index.name = "epoch"
                st.line_chart(chart_df)
        else:
            st.json(tm)

    with tabs[3]:
        st.subheader("Training run summary")
        summary_path = Path("results_v5/training_summary.json")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                st.error("Could not parse training_summary.json")
                summary = None
            if summary:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Training Time", summary.get("total_elapsed_human", "—"))
                col2.metric("Best Val L1(ab)", f"{summary.get('best_val_l1_ab', '—'):.5f}" if isinstance(summary.get("best_val_l1_ab"), float) else "—")
                col3.metric("Total Epochs", (summary.get("phase1_epochs", 0) or 0) + (summary.get("phase2_epochs", 0) or 0))

                st.markdown("---")
                col4, col5, col6 = st.columns(3)
                col4.metric("Batch Size", summary.get("batch_size", "—"))
                col5.metric("AMP Enabled", "Yes" if summary.get("amp") else "No")
                train_dirs = summary.get("train_dirs", [])
                col6.metric("Train Directories", len(train_dirs) if isinstance(train_dirs, list) else 1)

                if isinstance(train_dirs, list):
                    st.caption("Train directories: " + ", ".join(train_dirs))

                epochs = summary.get("epochs", [])
                if epochs:
                    import pandas as pd
                    df_e = pd.DataFrame(epochs)
                    cols_e = [c for c in ["phase", "epoch", "train_loss", "val_loss", "elapsed_s"] if c in df_e.columns]
                    df_e = df_e[cols_e]
                    st.markdown("**Per-epoch breakdown**")
                    st.dataframe(df_e, use_container_width=True)
                    if "elapsed_s" in df_e.columns:
                        st.markdown("**Epoch duration (seconds)**")
                        st.bar_chart(df_e.set_index(df_e.index)["elapsed_s"])
        else:
            st.info("No training_summary.json found. Run training first to generate it.")

if __name__ == "__main__":
    main()