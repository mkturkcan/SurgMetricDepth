from depth_infer import load_model, infer_video

# ━━━━━━━━━━━  CONFIGURATION (edit these)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO_PATH     = "surgery.mp4"            # input video file
INTRINSIC      = [391.0, 391.0, 320.0, 240.0]  # [fx, fy, cx, cy]

# ── Model presets (uncomment one) ──
# ViT-Small fine-tuned on Hamlyn  (download: https://huggingface.co/mehmetkeremturkcan/SurgMetricDepth/resolve/main/Metric3D_hamlyn.pth)
MODEL_NAME, CHECKPOINT = "metric3d_vit_small", "Metric3D_hamlyn.pth"
# ViT-Small pretrained (no fine-tune)
# MODEL_NAME, CHECKPOINT = "metric3d_vit_small", None
# ViT-Giant2 pretrained (no fine-tune)
# MODEL_NAME, CHECKPOINT = "metric3d_vit_giant2", None

REPO           = "yvanyin/metric3d"       # torch.hub repo

GLOBAL_SCALE   = 0.05 #1.0                      # multiplier on output depth
VIDEO_OUTPUT   = "depth_video.mp4"        # output side-by-side video
OUTPUT_FPS     = 30                       # output FPS (None = match input)
CROP_PCT       = [0, 0, 0, 0]            # [top%, bottom%, left%, right%] to crop
TEMPORAL_ALPHA = 0.4                      # EMA smoothing: 1.0 = none, lower = smoother
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1) Load model
model = load_model(MODEL_NAME, REPO, CHECKPOINT)

# 2) Run video depth inference
infer_video(
    model, VIDEO_PATH, INTRINSIC,
    output_path=VIDEO_OUTPUT,
    crop_pct=CROP_PCT,
    global_scale=GLOBAL_SCALE,
    output_fps=OUTPUT_FPS,
    temporal_alpha=TEMPORAL_ALPHA,
)
