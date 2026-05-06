import cv2
import numpy as np
import open3d as o3d
import os

base_dir = r"C:\3dproj"

params_path = os.path.join(base_dir, "outputsbest", "stereo_params.yml")
left_path = os.path.join(base_dir, "outputsbest", "stereo_pairs_objects", "left", "image_05.png")
right_path = os.path.join(base_dir, "outputsbest", "stereo_pairs_objects", "right", "image_05.png")

save_dir = os.path.join(base_dir, "outputsbest", "reconstruction_results")
os.makedirs(save_dir, exist_ok=True)

fs = cv2.FileStorage(params_path, cv2.FILE_STORAGE_READ)

if not fs.isOpened():
    raise RuntimeError("Could not open stereo_params.yml")

K1 = fs.getNode("K1").mat()
D1 = fs.getNode("D1").mat()
K2 = fs.getNode("K2").mat()
D2 = fs.getNode("D2").mat()

R1 = fs.getNode("R1").mat()
R2 = fs.getNode("R2").mat()
P1 = fs.getNode("P1").mat()
P2 = fs.getNode("P2").mat()
Q = fs.getNode("Q").mat()

width = int(fs.getNode("image_width").real())
height = int(fs.getNode("image_height").real())

fs.release()

imgL = cv2.imread(left_path)
imgR = cv2.imread(right_path)

if imgL is None or imgR is None:
    raise RuntimeError("Could not load stereo images.")

imgL = cv2.resize(imgL, (width, height))
imgR = cv2.resize(imgR, (width, height))

map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, (width, height), cv2.CV_32FC1
)

map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, (width, height), cv2.CV_32FC1
)

rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

window_size = 7
num_disp = 128

left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=8,
    speckleWindowSize=150,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_HH
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(
    matcher_left=left_matcher
)

wls_filter.setLambda(20000)
wls_filter.setSigmaColor(1.5)

dispL = left_matcher.compute(grayL, grayR)
dispR = right_matcher.compute(grayR, grayL)

filtered_disp = wls_filter.filter(
    dispL,
    grayL,
    None,
    dispR
).astype(np.float32) / 16.0

np.save(os.path.join(save_dir, "disparity_wls_fixed.npy"), filtered_disp)

disp_vis = cv2.normalize(
    filtered_disp,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
    cv2.CV_8U
)

disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
cv2.imwrite(os.path.join(save_dir, "filtered_disparity_fixed.png"), disp_color)

check = cv2.hconcat([rectL, rectR])

for y in range(0, height, 40):
    cv2.line(check, (0, y), (2 * width, y), (0, 255, 0), 1)

cv2.imwrite(os.path.join(save_dir, "rectification_check.png"), check)

points_3d = cv2.reprojectImageTo3D(
    filtered_disp,
    Q,
    handleMissingValues=True
)

colors = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)

z = points_3d[:, :, 2]

mask = np.isfinite(points_3d).all(axis=2)
mask &= filtered_disp > 3.0
mask &= z > 450
mask &= z < 7000

gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
mask &= gray > 10

h, w = mask.shape

roi = np.zeros_like(mask, dtype=bool)
roi[60:min(650, h), 500:min(980, w)] = True
mask &= roi

hsv = cv2.cvtColor(rectL, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]
sat = hsv[:, :, 1]
val = hsv[:, :, 2]

green = (hue >= 35) & (hue <= 95) & (sat > 20) & (val > 20) & (val < 245)
brown = (hue >= 5) & (hue <= 35) & (sat > 15) & (val > 20) & (val < 245)
white = (sat < 90) & (val > 80) & (val < 230)

color_mask = green | brown | white
mask &= color_mask

kernel = np.ones((3, 3), np.uint8)
mask_u8 = mask.astype(np.uint8) * 255
mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
mask = mask_u8.astype(bool)

output_points = points_3d[mask]
output_colors = colors[mask].astype(np.float64) / 255.0

if len(output_points) == 0:
    raise RuntimeError("No valid 3D points found.")

if np.median(output_points[:, 2]) < 0:
    output_points *= -1

z = output_points[:, 2]
z_low, z_high = np.percentile(z, [2, 98])
keep = (z >= z_low) & (z <= z_high)

output_points = output_points[keep]
output_colors = output_colors[keep]

z = output_points[:, 2]
z_center = np.median(z)
z_band = 1000.0
z_new = np.clip(z, z_center - z_band, z_center + z_band)
scale = z_new / (z + 1e-6)
output_points = output_points * scale[:, None]

output_points = output_points - np.mean(output_points, axis=0)
output_points = output_points / 1000.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(output_points.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(output_colors.astype(np.float64))

pcd = pcd.voxel_down_sample(0.003)

if len(pcd.points) > 500:
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=1.0
    )

labels = np.array(
    pcd.cluster_dbscan(
        eps=0.10,
        min_points=10,
        print_progress=True
    )
)

if labels.size > 0 and labels.max() >= 0:
    valid_labels = labels[labels >= 0]
    largest_label = np.bincount(valid_labels).argmax()
    keep_idx = np.where(labels == largest_label)[0]
    pcd = pcd.select_by_index(keep_idx)

ply_path = os.path.join(save_dir, "fixed_3d_reconstruction.ply")
o3d.io.write_point_cloud(ply_path, pcd)

print("Saved disparity:", os.path.join(save_dir, "filtered_disparity_fixed.png"))
print("Saved point cloud:", ply_path)
print("Final points:", len(pcd.points))

cv2.imshow("Filtered Disparity Fixed", disp_color)
cv2.imshow("Rectification Check", cv2.resize(check, (1280, 360)))

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Fixed 3D Reconstruction",
    width=1200,
    height=900
)

cv2.waitKey(0)
cv2.destroyAllWindows()