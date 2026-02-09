# SurgMetricDepth

Metric depth models for surgery applications

SurgMetricDepth provides a minimal inference wrapper around [Metric3D](https://github.com/YvanYin/Metric3D) for metric (absolute-scale) monocular depth estimation in surgical scenes. It handles canonical-camera preprocessing, supports loading fine-tuned checkpoints, and includes a 4-panel visualisation utility.

## Features

- **Metric depth inference** from a single RGB image or video frame
- **Custom checkpoint loading** for fine-tuned surgical models (e.g. Hamlyn dataset)
- **Canonical-camera preprocessing** matching the official Metric3D pipeline (keep-ratio resize, intrinsic scaling, ImageNet-mean padding, de-canonical transform)
- **4-panel visualisation**: input RGB, predicted depth, depth overlay / ground-truth, depth histogram / absolute error
- Supports both **ViT** (`metric3d_vit_small`, input 616x1064) and **ConvNeXt** (input 544x1216) backbones

## Project Structure

```
SurgMetricDepth/
├── depth_infer.py   # Core inference module (model loading, preprocessing, inference, visualisation)
├── sample.py        # Example script showing end-to-end usage
├── LICENSE          # AGPL-3.0
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

### Usage

Edit the configuration block in `sample.py` to point to your image/video and camera intrinsics, then run:

```bash
python sample.py
```

#### Configuration

| Parameter | Description | Default |
|---|---|---|
| `IMAGE_PATH` | Path to an image (`.png`, `.jpg`, ...) or video (`.mp4`, `.avi`, ...) | `"sample.png"` |
| `FRAME_INDEX` | Video frame to extract (ignored for images) | `0` |
| `INTRINSIC` | Camera intrinsics as `[fx, fy, cx, cy]` | `[391.0, 391.0, 320.0, 240.0]` |
| `MODEL_NAME` | Metric3D hub model name | `"metric3d_vit_small"` |
| `REPO` | `torch.hub` repository | `"yvanyin/metric3d"` |
| `CHECKPOINT` | Path to a `.pth` fine-tuned checkpoint, or `None` for pretrained weights | `"scene_reconstruction_Metric3D_ckpt_hamlyn_20250513_epoch8.pth"` |
| `GLOBAL_SCALE` | Multiplicative factor applied to the output depth | `1.0` |
| `SAVE_PATH` | Output path for the visualisation figure, or `None` to only display | `"depth_output.png"` |

#### Python API

```python
from depth_infer import load_model, load_image, infer_depth, visualize_4panel

model = load_model("metric3d_vit_small", "yvanyin/metric3d", "your_checkpoint.pth")
rgb = load_image("frame.png")
depth = infer_depth(model, rgb, [fx, fy, cx, cy])
fig = visualize_4panel(rgb, depth, save_path="output.png")
```

`infer_depth` returns a `(H, W)` float32 NumPy array of metric depth values in metres.

If you have ground-truth depth, pass it to `visualize_4panel` via the `depth_gt` parameter to get an error map and MAE instead of the overlay and histogram panels.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
