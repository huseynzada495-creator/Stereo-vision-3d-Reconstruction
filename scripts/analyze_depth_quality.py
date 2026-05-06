import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DISP_PATH = "outputs/stereo_depth_rectified/disparity_wls.npy"
IMG_PATH = "outputs/stereo_depth_rectified/rectified_left.jpg"
Q_PATH = "outputs/stereo_depth_rectified/Q.npy"
OUT_DIR = "outputs/depth_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMG_PATH)
disp = np.load(DISP_PATH).astype(np.float32)
Q = np.load(Q_PATH).astype(np.float64)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

green = (hue >= 35) & (hue <= 95) & (sat > 20) & (val > 10) & (val < 245)
brown_orange = (hue >= 5) & (hue <= 35) & (sat > 10) & (val > 20) & (val < 245)
white_shirt = (sat < 80) & (val > 80) & (val < 245)

mask = green | brown_orange | white_shirt

roi = np.zeros(mask.shape, dtype=bool)
roi[60:650, 500:980] = True
mask &= roi

valid = mask & np.isfinite(disp) & (disp > 0)

points3d = cv2.reprojectImageTo3D(disp, Q)
points = points3d[valid]

if np.median(points[:, 2]) < 0:
    points *= -1

z = points[:, 2]
d = disp[valid]

print("\n=== DISPARITY STATS ON OBJECT MASK ===")
print("count:", len(d))
print("disp min/max:", np.min(d), np.max(d))
print("disp percentiles:", np.percentile(d, [1, 5, 25, 50, 75, 95, 99]))

print("\n=== Z STATS ON OBJECT MASK ===")
print("z min/max:", np.min(z), np.max(z))
print("z percentiles:", np.percentile(z, [1, 5, 25, 50, 75, 95, 99]))

# Local disparity consistency
smooth = cv2.medianBlur(disp.astype(np.float32), 5)
diff = np.abs(disp - smooth)

stable = valid & (diff < 2.0)
unstable = valid & (diff >= 2.0)

print("\n=== LOCAL CONSISTENCY ===")
print("stable pixels:", np.count_nonzero(stable))
print("unstable pixels:", np.count_nonzero(unstable))

debug = img.copy()
debug[unstable] = (0, 0, 255)
debug[stable] = (0, 255, 0)
cv2.imwrite(os.path.join(OUT_DIR, "stable_green_unstable_red.jpg"), debug)

# Z outlier view
z_img = np.zeros(disp.shape, dtype=np.uint8)
z_valid = z
lo, hi = np.percentile(z_valid, [5, 95])
z_norm = np.clip((points3d[:, :, 2] * (-1 if np.median(points3d[valid][:, 2]) < 0 else 1) - lo) * 255 / (hi - lo), 0, 255)
z_img[valid] = z_norm[valid].astype(np.uint8)
cv2.imwrite(os.path.join(OUT_DIR, "z_on_object_mask.jpg"), z_img)

plt.figure()
plt.hist(d, bins=80)
plt.title("Disparity histogram on object mask")
plt.xlabel("Disparity")
plt.ylabel("Pixel count")
plt.savefig(os.path.join(OUT_DIR, "disparity_hist.png"))

plt.figure()
plt.hist(z, bins=80)
plt.title("Z histogram on object mask")
plt.xlabel("Z depth")
plt.ylabel("Pixel count")
plt.savefig(os.path.join(OUT_DIR, "z_hist.png"))

print("\nSaved analysis images to:", OUT_DIR)