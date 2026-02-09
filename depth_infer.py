from __future__ import annotations

"""
depth_infer.py – Minimal, self-contained depth inference with Metric3D.

Preprocessing follows the official hubconf.py exactly:
  1. Keep-ratio resize to fit (616, 1064)
  2. Scale intrinsics by the same factor
  3. Pad borders with ImageNet mean
  4. Normalize with ImageNet mean/std
  5. Inference  →  un-pad  →  upsample to original size
  6. De-canonical transform: depth *= scaled_fx / 1000.0

Usage from a notebook / script:
    from depth_infer import load_model, infer_depth, visualize_4panel, load_image
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Model loading ────────────────────────────────────────────────────────

def load_model(
    framework: str = "metric3d_vit_small",
    repo: str = "yvanyin/metric3d",
    checkpoint: str | None = None,
    device: torch.device = DEVICE,
):
    """
    Load a Metric3D model from torch.hub and optionally swap in a fine-tuned
    checkpoint.  Returns the model in eval mode on *device*.
    """
    model = torch.hub.load(repo, framework, pretrain=(checkpoint is None))
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"✓ Loaded checkpoint  {checkpoint}")
    else:
        print(f"✓ Using hub pretrained weights for {framework}")
    return model.to(device).eval()


# ── 2. Image I/O ────────────────────────────────────────────────────────────

def load_image(path: str, frame_index: int = 0) -> np.ndarray:
    """
    Load an image (BGR→RGB) from a file *or* grab a single frame from a video.

    Returns
    -------
    rgb : np.ndarray  (H, W, 3) uint8, RGB order
    """
    p = Path(path)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    if p.suffix.lower() in video_exts:
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── 3. Metric3D pre/post-processing (matches hubconf.py exactly) ────────────

_IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
_IMAGENET_STD  = torch.tensor([58.395,  57.12,  57.375]).float()[:, None, None]
_PAD_VALUE     = [123.675, 116.28, 103.53]          # border fill = ImageNet mean
_CANONICAL_FOCAL = 1000.0


def _prep_image(
    rgb: np.ndarray,
    intrinsic: list[float],
    input_size: tuple[int, int] = (616, 1064),
) -> tuple[torch.Tensor, list[int], list[float]]:
    """
    Metric3D canonical-camera preprocessing (mirrors hubconf.__main__).

    1. Keep-ratio resize so the image fits inside *input_size*.
    2. Scale intrinsics by the same factor.
    3. Pad with ImageNet-mean border to reach exact *input_size*.
    4. Normalise with ImageNet mean / std.

    Returns
    -------
    img_tensor      : (1, 3, H, W) float32 normalised, ready for the model
    pad_info        : [pad_top, pad_bottom, pad_left, pad_right]
    intrinsic_scaled: [fx', fy', cx', cy'] after the keep-ratio resize
    """
    h, w = rgb.shape[:2]

    # ── keep-ratio resize ──
    scale = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # scale intrinsics by the same factor
    intrinsic_scaled = [v * scale for v in intrinsic]

    # ── pad to input_size with ImageNet mean ──
    pad_h = input_size[0] - new_h
    pad_w = input_size[1] - new_w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb_padded = cv2.copyMakeBorder(
        rgb_resized,
        pad_h_half, pad_h - pad_h_half,
        pad_w_half, pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=_PAD_VALUE,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # ── normalise ──
    img_t = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float()
    img_t = (img_t - _IMAGENET_MEAN) / _IMAGENET_STD
    img_t = img_t.unsqueeze(0)  # (1, 3, H, W)

    return img_t, pad_info, intrinsic_scaled


def _postprocess_depth(
    pred_depth: torch.Tensor,
    pad_info: list[int],
    original_hw: tuple[int, int],
    intrinsic_scaled: list[float],
) -> np.ndarray:
    """
    Un-pad  →  upsample to original size  →  de-canonical transform.

    Returns
    -------
    depth : (H, W) float32 numpy array in metric units
    """
    d = pred_depth.squeeze()  # (H_pad, W_pad)

    # ── remove padding ──
    h_pad, w_pad = d.shape
    top, bottom, left, right = pad_info
    d = d[top : h_pad - bottom if bottom > 0 else h_pad,
          left: w_pad - right  if right  > 0 else w_pad]

    # ── upsample to original resolution ──
    oh, ow = original_hw
    d = F.interpolate(
        d[None, None, :, :], size=(oh, ow), mode="bilinear", align_corners=False
    ).squeeze()

    # ── de-canonical transform ──
    # The model predicts in a canonical camera space with focal = 1000.
    # To get real metric depth: depth *= (real_focal / canonical_focal)
    canonical_to_real = intrinsic_scaled[0] / _CANONICAL_FOCAL
    d = d * canonical_to_real

    d = torch.clamp(d, 0, 300)
    return d.cpu().numpy()


# ── 4. Inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_depth(
    model: torch.nn.Module,
    rgb: np.ndarray,
    intrinsic: list[float],
    global_scale: float = 1.0,
    input_size: tuple[int, int] = (616, 1064),
    device: torch.device = DEVICE,
) -> np.ndarray:
    """
    Run Metric3D inference on a single RGB image.

    Parameters
    ----------
    model : loaded Metric3D model (eval mode)
    rgb   : (H, W, 3) uint8 RGB numpy array
    intrinsic : [fx, fy, cx, cy]
    global_scale : extra multiplicative factor on the output depth
    input_size : Metric3D network input resolution (ViT: 616×1064, ConvNeXt: 544×1216)

    Returns
    -------
    depth : (H, W) float32 numpy array – metric depth in metres
    """
    oh, ow = rgb.shape[:2]
    img_t, pad_info, intrinsic_scaled = _prep_image(rgb, intrinsic, input_size)
    img_t = img_t.to(device)

    # hubconf uses model.inference(); fall back to model() for compatibility
    if hasattr(model, "inference"):
        pred_depth, confidence, output_dict = model.inference({"input": img_t})
    else:
        pred_depth, confidence, output_dict = model({"input": img_t})

    if torch.isnan(pred_depth).any():
        raise ValueError("Model produced NaN depth values")

    depth = _postprocess_depth(pred_depth, pad_info, (oh, ow), intrinsic_scaled)

    if global_scale != 1.0:
        depth = depth * global_scale

    return depth


# ── 5. Visualisation ─────────────────────────────────────────────────────────

def visualize_4panel(
    rgb: np.ndarray,
    depth: np.ndarray,
    save_path: str | None = None,
    depth_gt: np.ndarray | None = None,
    cmap: str = "inferno",
    max_depth: float | None = None,
    title: str = "Depth Estimation",
    dpi: int = 150,
):
    """
    Create a professional 4-panel figure:

        ┌───────────┬───────────┐
        │  RGB      │  Depth    │
        ├───────────┼───────────┤
        │  Overlay  │ Histogram │  (or GT + Error if depth_gt given)
        └───────────┴───────────┘

    Returns
    -------
    fig : matplotlib Figure
    """
    valid = depth[depth > 0]
    if max_depth is None:
        max_depth = float(np.percentile(valid, 95)) if valid.size else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # ── Panel 1: RGB ──
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input RGB", fontsize=12)
    axes[0, 0].axis("off")

    # ── Panel 2: Predicted depth ──
    im = axes[0, 1].imshow(depth, cmap=cmap, vmin=0, vmax=max_depth)
    axes[0, 1].set_title("Predicted Depth (m)", fontsize=12)
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    if depth_gt is not None:
        # ── Panel 3: Ground-truth ──
        im_gt = axes[1, 0].imshow(depth_gt, cmap=cmap, vmin=0, vmax=max_depth)
        axes[1, 0].set_title("Ground-Truth Depth (m)", fontsize=12)
        axes[1, 0].axis("off")
        fig.colorbar(im_gt, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # ── Panel 4: Absolute error ──
        mask = (depth_gt > 0) & (depth > 0)
        error = np.abs(depth - depth_gt) * mask
        err_max = float(np.percentile(error[mask], 95)) if mask.any() else 1.0
        im_err = axes[1, 1].imshow(error, cmap="hot", vmin=0, vmax=err_max)
        mae = float(error[mask].mean()) if mask.any() else 0.0
        axes[1, 1].set_title(f"Absolute Error  (MAE={mae:.4f} m)", fontsize=12)
        axes[1, 1].axis("off")
        fig.colorbar(im_err, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        # ── Panel 3: Depth overlay on RGB ──
        axes[1, 0].imshow(rgb)
        overlay = axes[1, 0].imshow(depth, cmap=cmap, alpha=0.55, vmin=0, vmax=max_depth)
        axes[1, 0].set_title("Depth Overlay", fontsize=12)
        axes[1, 0].axis("off")
        fig.colorbar(overlay, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # ── Panel 4: Histogram ──
        axes[1, 1].hist(valid, bins=120, color="#4a90d9", edgecolor="none", alpha=0.85)
        axes[1, 1].set_xlabel("Depth (m)")
        axes[1, 1].set_ylabel("Pixel count")
        axes[1, 1].set_title("Depth Distribution", fontsize=12)
        med = float(np.median(valid)) if valid.size else 0
        axes[1, 1].axvline(med, color="red", ls="--", lw=1.2, label=f"median = {med:.3f} m")
        axes[1, 1].legend(fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved → {save_path}")

    return fig