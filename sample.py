from depth_infer import load_model, load_image, infer_depth, visualize_4panel

# ━━━━━━━━━━━  CONFIGURATION (edit these)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE_PATH   = "sample.png"       # or a .mp4/.avi → grabs one frame
FRAME_INDEX  = 0                          # which frame (ignored for images)
INTRINSIC    = [391.0, 391.0, 320.0, 240.0]  # [fx, fy, cx, cy]

MODEL_NAME   = "metric3d_vit_small"       # hub model name
REPO         = "yvanyin/metric3d"         # torch.hub repo
CHECKPOINT   = 'scene_reconstruction_Metric3D_ckpt_hamlyn_20250513_epoch8.pth'                       # .pth path or None for pretrained

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