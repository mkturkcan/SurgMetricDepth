from depth_infer import load_model, load_image, infer_depth, visualize_4panel

# ━━━━━━━━━━━  CONFIGURATION (edit these)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE_PATH   = "sample.png"       # or a .mp4/.avi → grabs one frame
FRAME_INDEX  = 0                          # which frame (ignored for images)
INTRINSIC    = [391.0, 391.0, 320.0, 240.0]  # [fx, fy, cx, cy]

# ── Model presets (uncomment one) ──
# ViT-Small fine-tuned on Hamlyn  (download: https://huggingface.co/mehmetkeremturkcan/SurgMetricDepth/resolve/main/Metric3D_hamlyn.pth)
MODEL_NAME, CHECKPOINT = "metric3d_vit_small", "Metric3D_hamlyn.pth"
# ViT-Small pretrained (no fine-tune)
# MODEL_NAME, CHECKPOINT = "metric3d_vit_small", None
# ViT-Giant2 pretrained (no fine-tune)
# MODEL_NAME, CHECKPOINT = "metric3d_vit_giant2", None

REPO         = "yvanyin/metric3d"         # torch.hub repo

GLOBAL_SCALE = 0.05 #1.0                        # multiplier on output depth
SAVE_PATH    = "depth_output.png"         # set to None to just display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1) Load model
model = load_model(MODEL_NAME, REPO, CHECKPOINT)

# 2) Load image (or video frame)
rgb = load_image(IMAGE_PATH, frame_index=FRAME_INDEX)
print(f"Image shape: {rgb.shape}")

# 3) Run depth estimation
depth = infer_depth(model, rgb, INTRINSIC, global_scale=GLOBAL_SCALE)
print(f"Depth range: {depth[depth>0].min():.3f} – {depth.max():.3f} m")

# 4) Visualise
fig = visualize_4panel(rgb, depth, save_path=SAVE_PATH, title="Metric3D Depth")
fig  # display in notebook
