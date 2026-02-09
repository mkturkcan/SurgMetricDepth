# SurgMetricDepth

Metric depth models for surgery applications

SurgMetricDepth provides a minimal inference wrapper around [Metric3D](https://github.com/YvanYin/Metric3D) for metric (absolute-scale) monocular depth estimation in surgical scenes. It handles canonical-camera preprocessing, supports loading fine-tuned checkpoints, and includes a 4-panel visualisation utility.

## Features

- **Metric depth inference** from a single RGB image or video frame
- **Full video pipeline** — process every frame of a video, producing a side-by-side (RGB | depth colourmap) output video
- **Centre-crop** — remove a configurable percentage from each edge (top/bottom/left/right) before inference, with automatic intrinsic adjustment
- **Temporal smoothing** — exponential moving average (EMA) across frames to reduce frame-to-frame depth flickering
- **Custom checkpoint loading** for fine-tuned surgical models (e.g. Hamlyn dataset)
- **Canonical-camera preprocessing** matching the official Metric3D pipeline (keep-ratio resize, intrinsic scaling, ImageNet-mean padding, de-canonical transform)
- **4-panel visualisation**: input RGB, predicted depth, depth overlay / ground-truth, depth histogram / absolute error
- Supports both **ViT** (`metric3d_vit_small`, input 616x1064) and **ConvNeXt** (input 544x1216) backbones

## Project Structure

```
SurgMetricDepth/
├── depth_infer.py    # Core inference module (model loading, preprocessing, inference, visualisation)
├── sample.py         # Single-image depth estimation example
├── sample_video.py   # Full-video depth estimation example
├── LICENSE           # AGPL-3.0
├── .gitignore
└── README.md
```

## Instructions

### Setup

```bash
conda create -n open-mmlab9 python=3.9 -y
conda activate open-mmlab9
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmcv
git clone https://github.com/YvanYin/Metric3D.git
cd Metric3D
pip install -r requirements_v2.txt
cd ..
```

### Model Presets

Both sample scripts include three model presets — uncomment the one you want:

| Preset | `MODEL_NAME` | `CHECKPOINT` | Notes |
|---|---|---|---|
| **ViT-Small + Hamlyn** (default) | `metric3d_vit_small` | `Metric3D_hamlyn.pth` | Fine-tuned on surgical data |
| ViT-Small pretrained | `metric3d_vit_small` | `None` | Official Metric3D weights |
| ViT-Giant2 pretrained | `metric3d_vit_giant2` | `None` | Largest backbone, highest accuracy |

The Hamlyn checkpoint can be downloaded from Hugging Face:

```bash
wget https://huggingface.co/mehmetkeremturkcan/SurgMetricDepth/resolve/main/Metric3D_hamlyn.pth
```

### Usage

#### Single Image

Edit the configuration block in `sample.py` to point to your image and camera intrinsics, then run:

```bash
python sample.py
```

| Parameter | Description | Default |
|---|---|---|
| `IMAGE_PATH` | Path to an image (`.png`, `.jpg`, ...) or video (`.mp4`, `.avi`, ...) | `"sample.png"` |
| `FRAME_INDEX` | Video frame to extract (ignored for images) | `0` |
| `INTRINSIC` | Camera intrinsics as `[fx, fy, cx, cy]` | `[391.0, 391.0, 320.0, 240.0]` |
| `GLOBAL_SCALE` | Multiplicative factor applied to the output depth | `1.0` |
| `SAVE_PATH` | Output path for the visualisation figure, or `None` to only display | `"depth_output.png"` |

#### Video

Edit the configuration block in `sample_video.py` and run:

```bash
python sample_video.py
```

| Parameter | Description | Default |
|---|---|---|
| `VIDEO_PATH` | Path to the input video | `"surgery.mp4"` |
| `VIDEO_OUTPUT` | Output path for the side-by-side video | `"depth_video.mp4"` |
| `INTRINSIC` | Camera intrinsics as `[fx, fy, cx, cy]` | `[391.0, 391.0, 320.0, 240.0]` |
| `GLOBAL_SCALE` | Multiplicative factor applied to the output depth | `1.0` |
| `OUTPUT_FPS` | FPS of the output video (`None` = match input) | `30` |
| `CROP_PCT` | Centre-crop as `[top%, bottom%, left%, right%]` (0 = no crop) | `[0, 0, 0, 0]` |
| `TEMPORAL_ALPHA` | EMA smoothing weight (see below) | `0.4` |

**Centre-crop** removes a percentage of pixels from each edge before inference. This is useful for surgical endoscope footage where the border contains a black vignette or overlaid UI elements. Camera intrinsics are automatically adjusted (principal point shifted) to match the cropped region.

**Temporal smoothing** applies an exponential moving average (EMA) to the depth maps across consecutive frames:

```
smoothed_t = alpha * depth_t + (1 - alpha) * smoothed_{t-1}
```

- `alpha = 1.0` — no smoothing; each frame is independent (raw model output)
- `alpha = 0.4` (default) — moderate smoothing; balances responsiveness with stability
- `alpha = 0.1` — heavy smoothing; very stable but slow to react to scene changes

Per-frame monocular depth estimates can flicker because the model has no temporal context — it processes each frame independently. EMA smoothing alleviates this by blending each new prediction with the running average, suppressing high-frequency jitter while still tracking gradual depth changes.

#### Python API

```python
from depth_infer import load_model, load_image, infer_depth, infer_video, visualize_4panel

model = load_model("metric3d_vit_small", "yvanyin/metric3d", "your_checkpoint.pth")

# Single image
rgb = load_image("frame.png")
depth = infer_depth(model, rgb, [fx, fy, cx, cy])
fig = visualize_4panel(rgb, depth, save_path="output.png")

# Full video with crop and smoothing
infer_video(
    model, "surgery.mp4", [fx, fy, cx, cy],
    output_path="depth_video.mp4",
    crop_pct=[10, 10, 5, 5],    # remove 10% top/bottom, 5% left/right
    temporal_alpha=0.4,          # EMA smoothing
)
```

`infer_depth` returns a `(H, W)` float32 NumPy array of metric depth values in metres.

If you have ground-truth depth, pass it to `visualize_4panel` via the `depth_gt` parameter to get an error map and MAE instead of the overlay and histogram panels.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
