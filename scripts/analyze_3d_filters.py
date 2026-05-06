import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

IMG_PATH = "outputs/stereo_depth_rectified/rectified_left.jpg"
DISP_PATH = "outputs/stereo_depth_rectified/disparity_wls.npy"
Q_PATH = "outputs/stereo_depth_rectified/Q.npy"
OUT_DIR = "outputs/filter_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMG_PATH)
disp = np.load(DISP_PATH).astype(np.float32)
Q = np.load(Q_PATH).astype(np.float64)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

green = (hue >= 35) & (hue <= 95) & (sat > 20) & (val > 20) & (val < 245)
teddy = (hue >= 5) & (hue <= 35) & (sat > 15) & (val > 20) & (val < 245)
white = (sat < 90) & (val > 80) & (val < 230)

roi_all = np.zeros(disp.shape, dtype=bool)
roi_all[60:650, 500:980] = True

roi_shirt = np.zeros(disp.shape, dtype=bool)
roi_shirt[280:520, 600:900] = True

mask = ((green | teddy) & roi_all) | (white & roi_shirt)
valid = mask & np.isfinite(disp) & (disp >= 8) & (disp <= 127)

points3d = cv2.reprojectImageTo3D(disp, Q)
points = points3d[valid]
colors = img[valid][:, ::-1] / 255.0

if np.median(points[:, 2]) < 0:
    points *= -1

x, y, z = points[:, 0], points[:, 1], points[:, 2]

print("\nTOTAL VALID:", len(points))
print("\nZ percentiles:", np.percentile(z, [1, 5, 10, 25, 50, 75, 90, 95, 99]))
print("X percentiles:", np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99]))
print("Y percentiles:", np.percentile(y, [1, 5, 10, 25, 50, 75, 90, 95, 99]))

tests = [
    ("z_500_3000", (z >= 500) & (z <= 3000)),
    ("z_500_3500", (z >= 500) & (z <= 3500)),
    ("z_500_4000", (z >= 500) & (z <= 4000)),
    ("box_700_600", (np.abs(x) < 700) & (np.abs(y) < 600)),
    ("box_900_800", (np.abs(x) < 900) & (np.abs(y) < 800)),
    ("box_1200_1000", (np.abs(x) < 1200) & (np.abs(y) < 1000)),
    ("z3500_box900_800", (z >= 500) & (z <= 3500) & (np.abs(x) < 900) & (np.abs(y) < 800)),
]

for name, keep in tests:
    kept = np.count_nonzero(keep)
    print(f"{name}: {kept} / {len(points)} = {100*kept/len(points):.2f}%")

    debug = np.zeros(valid.shape, dtype=np.uint8)
    coords = np.where(valid)
    debug[coords[0][keep], coords[1][keep]] = 255
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_kept_mask.jpg"), debug)

    pcd = o3d.geometry.PointCloud()
    p = points[keep].copy()
    c = colors[keep].copy()
    p = p - np.mean(p, axis=0)
    p = p / 1000.0
    pcd.points = o3d.utility.Vector3dVector(p.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    o3d.io.write_point_cloud(os.path.join(OUT_DIR, f"{name}.ply"), pcd)

plt.figure()
plt.hist(z, bins=100)
plt.title("Z distribution")
plt.xlabel("Z")
plt.ylabel("count")
plt.savefig(os.path.join(OUT_DIR, "z_distribution.png"))

plt.figure()
plt.hist(x, bins=100)
plt.title("X distribution")
plt.xlabel("X")
plt.ylabel("count")
plt.savefig(os.path.join(OUT_DIR, "x_distribution.png"))

plt.figure()
plt.hist(y, bins=100)
plt.title("Y distribution")
plt.xlabel("Y")
plt.ylabel("count")
plt.savefig(os.path.join(OUT_DIR, "y_distribution.png"))

print("\nSaved to:", OUT_DIR)
print("Open the .ply files in MeshLab/Open3D and compare.")