import os
import cv2
import numpy as np

DISP_PATH = "outputs/stereo_depth_rectified/disparity_filled.npy"
Q_PATH = "data/stereo/results/Q.npy"

disp = np.load(DISP_PATH).astype(np.float32)
Q = np.load(Q_PATH).astype(np.float64)

h, w = disp.shape

# -------------------------------
# VALID DISPARITY MASK
# -------------------------------
valid = np.isfinite(disp) & (disp > 20)

# -------------------------------
# ROI (OBJECT REGION ONLY)
# adjust if needed
# -------------------------------
roi = np.zeros_like(valid)

x1, y1 = 350, 40
x2, y2 = 1120, 650

x1 = max(0, min(w, x1))
x2 = max(0, min(w, x2))
y1 = max(0, min(h, y1))
y2 = max(0, min(h, y2))

roi[y1:y2, x1:x2] = True

valid = valid & roi

# -------------------------------
# STATS
# -------------------------------
valid_count = np.count_nonzero(valid)
total_count = np.count_nonzero(roi)
coverage = valid_count / total_count * 100.0

d = disp[valid]

print("\n=== DISPARITY STATS (OBJECT ONLY) ===")
print(f"Valid disparity pixels: {valid_count:,} ({coverage:.1f}%)")

if len(d) == 0:
    print("No valid disparities in ROI.")
    exit()

print(
    f"disparity (px): "
    f"10%={np.percentile(d, 10):.1f}  "
    f"median={np.percentile(d, 50):.1f}  "
    f"90%={np.percentile(d, 90):.1f}"
)

# -------------------------------
# DEPTH (Z)
# -------------------------------
points_3d = cv2.reprojectImageTo3D(disp, Q)
z = points_3d[:, :, 2]

z_valid = z[valid]
z_valid = z_valid[np.isfinite(z_valid)]

if len(z_valid) == 0:
    print("No valid Z values.")
    exit()

if np.median(z_valid) < 0:
    z_valid = -z_valid

print(
    f"implied Z (mm): "
    f"near={np.percentile(z_valid, 10):.0f}  "
    f"median={np.percentile(z_valid, 50):.0f}  "
    f"far={np.percentile(z_valid, 90):.0f}"
)