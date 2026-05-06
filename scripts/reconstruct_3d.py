import os
import argparse
import cv2
import numpy as np
import open3d as o3d

INPUT_DIR = "outputs/stereo_depth_rectified"
OUTPUT_DIR = "outputs/3d_reconstruction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEPTH_FILES = {
    "raw": "disparity_raw.npy",
    "wls": "disparity_wls.npy",
    "wls_object_smooth": "disparity_wls_object_smooth.npy",
    "clean": "disparity_clean.npy",
    "filled": "disparity_filled.npy",
    "dense": "disparity_dense.npy",
    "dense_filled": "disparity_dense_filled.npy",
}


def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def percent(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


def create_object_color_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h, w = hue.shape

    green = (hue >= 35) & (hue <= 95) & (sat > 20) & (val > 20) & (val < 245)
    teddy = (hue >= 5) & (hue <= 35) & (sat > 15) & (val > 20) & (val < 245)
    white = (sat < 90) & (val > 80) & (val < 230)

    roi_all = np.zeros((h, w), dtype=bool)
    roi_all[60:min(650, h), 500:min(980, w)] = True

    roi_shirt = np.zeros((h, w), dtype=bool)
    roi_shirt[280:min(520, h), 600:min(900, w)] = True

    object_mask = ((green | teddy) & roi_all) | (white & roi_shirt)

    kernel = np.ones((3, 3), np.uint8)
    object_mask_u8 = object_mask.astype(np.uint8) * 255
    object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "object_color_mask.jpg"), object_mask_u8)
    return object_mask_u8.astype(bool)


def make_mask(img, disp, min_disp, max_disp, object_only=False, object_color_mask=False):
    h, w = disp.shape

    mask = np.isfinite(disp)
    mask &= disp >= min_disp
    mask &= disp <= max_disp

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask &= gray > 10

    if object_only:
        total_mask = np.zeros_like(mask, dtype=bool)
        total_mask[60:min(650, h), 500:min(980, w)] = True
        mask &= total_mask
    else:
        total_mask = np.ones_like(mask, dtype=bool)

    if object_color_mask:
        color_mask = create_object_color_mask(img)
        mask &= color_mask
        total_mask &= color_mask

    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_2d_mask.jpg"), mask.astype(np.uint8) * 255)
    return mask, total_mask


def remove_gradient_edges(mask, disp, threshold):
    disp_clean = disp.copy()
    disp_clean[~np.isfinite(disp_clean)] = 0

    dx = cv2.Sobel(disp_clean, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(disp_clean, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(dx, dy)

    return mask & (grad < threshold)


def remove_table_color(points, colors):
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    table = (r > 145) & (g > 100) & (b < 130)

    return points[~table], colors[~table]


def apply_center_box(points, colors, xlim=1200, ylim=1200):
    x = points[:, 0]
    y = points[:, 1]

    keep = (
        (x > -xlim) & (x < xlim) &
        (y > -ylim) & (y < ylim)
    )

    return points[keep], colors[keep]


def pull_flying_points(points, z_band=900.0):
    z = points[:, 2]
    z_center = np.median(z)

    z_new = np.clip(z, z_center - z_band, z_center + z_band)

    scale = z_new / (z + 1e-6)
    points = points * scale[:, None]

    print("\n=== PULL FLYING POINTS ===")
    print("Z center:", z_center)
    print("Z band:", z_band)

    return points


def keep_largest_cluster(pcd, eps=0.08, min_points=15):
    if len(pcd.points) == 0:
        return pcd

    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=True
        )
    )

    if labels.size == 0 or labels.max() < 0:
        print("Cluster filter: no clusters found")
        return pcd

    valid_labels = labels[labels >= 0]
    largest_label = np.bincount(valid_labels).argmax()
    keep_idx = np.where(labels == largest_label)[0]

    return pcd.select_by_index(keep_idx)


def print_z_distribution(points):
    z = points[:, 2]

    print("\n=== Z DISTRIBUTION AFTER FLIP ===")
    print("Min Z:", np.min(z))
    print("Max Z:", np.max(z))
    print("Median Z:", np.median(z))

    p = np.percentile(z, [1, 5, 25, 50, 75, 95, 99])
    print("Percentiles:", p)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--depth", choices=list(DEPTH_FILES.keys()), default="wls")

    parser.add_argument("--object-only", action="store_true")
    parser.add_argument("--object-color-mask", action="store_true")
    parser.add_argument("--keep-table", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-gradient", action="store_true")
    parser.add_argument("--no-z-trim", action="store_true")

    parser.add_argument("--use-box", action="store_true")
    parser.add_argument("--box-x", type=float, default=1200.0)
    parser.add_argument("--box-y", type=float, default=1200.0)

    parser.add_argument("--pull-flying", action="store_true")
    parser.add_argument("--z-band", type=float, default=900.0)

    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--cluster-eps", type=float, default=0.08)
    parser.add_argument("--cluster-min-points", type=int, default=15)

    parser.add_argument("--min-disp", type=float, default=8.0)
    parser.add_argument("--max-disp", type=float, default=127.0)

    parser.add_argument("--grad-thresh", type=float, default=35.0)

    parser.add_argument("--z-low-percent", type=float, default=5.0)
    parser.add_argument("--z-high-percent", type=float, default=95.0)

    parser.add_argument("--z-min", type=float, default=450.0)
    parser.add_argument("--z-max", type=float, default=6000.0)

    parser.add_argument("--voxel", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--flip-y", action="store_true")

    args = parser.parse_args()

    disp_path = os.path.join(INPUT_DIR, DEPTH_FILES[args.depth])
    img_path = os.path.join(INPUT_DIR, "rectified_left.jpg")
    q_path = os.path.join(INPUT_DIR, "Q.npy")

    disp = load_npy(disp_path).astype(np.float32)
    Q = load_npy(q_path).astype(np.float64)

    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    if img.shape[:2] != disp.shape:
        img = cv2.resize(img, (disp.shape[1], disp.shape[0]))

    valid, total_mask = make_mask(
        img,
        disp,
        args.min_disp,
        args.max_disp,
        object_only=args.object_only,
        object_color_mask=args.object_color_mask,
    )

    total_pixels = np.count_nonzero(total_mask)
    initial_valid = np.count_nonzero(valid)

    if not args.no_gradient:
        valid = remove_gradient_edges(valid, disp, args.grad_thresh)

    after_gradient = np.count_nonzero(valid)

    disp_for_3d = disp.copy()
    disp_for_3d[~valid] = 0

    points_3d = cv2.reprojectImageTo3D(
        disp_for_3d,
        Q,
        handleMissingValues=True
    )

    finite = np.isfinite(points_3d).all(axis=2)
    finite &= valid

    points = points_3d[finite]
    colors = img[finite][:, ::-1].astype(np.float64) / 255.0

    if len(points) == 0:
        raise RuntimeError("No valid 3D points.")

    if np.median(points[:, 2]) < 0:
        points *= -1

    if args.flip_y:
        points[:, 1] *= -1

    print_z_distribution(points)

    before_z_hard = len(points)

    z = points[:, 2]
    keep_z_hard = (z >= args.z_min) & (z <= args.z_max)

    points = points[keep_z_hard]
    colors = colors[keep_z_hard]

    after_z_hard = len(points)

    if len(points) == 0:
        raise RuntimeError(
            "No points left after hard Z clamp. "
            "Your Q may output meters instead of millimeters. "
            "Try --z-min 0.45 --z-max 6.0"
        )

    before_pull = len(points)

    if args.pull_flying:
        points = pull_flying_points(points, z_band=args.z_band)

    after_pull = len(points)

    before_box = len(points)

    if args.use_box:
        points, colors = apply_center_box(
            points,
            colors,
            args.box_x,
            args.box_y
        )

    after_box = len(points)

    if len(points) == 0:
        raise RuntimeError("No points left after box filter.")

    before_table = len(points)

    if not args.keep_table:
        points, colors = remove_table_color(points, colors)

    after_table = len(points)

    if len(points) == 0:
        raise RuntimeError("No points left after table filtering.")

    if not args.no_z_trim:
        z = points[:, 2]
        z_low, z_high = np.percentile(
            z,
            [args.z_low_percent, args.z_high_percent]
        )

        keep = (z >= z_low) & (z <= z_high)

        points = points[keep]
        colors = colors[keep]

    after_z_trim = len(points)

    if len(points) == 0:
        raise RuntimeError("No points left after Z trim.")

    points = points - np.mean(points, axis=0)
    points = points / 1000.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    before_cluster = len(pcd.points)

    if args.cluster:
        pcd = keep_largest_cluster(
            pcd,
            eps=args.cluster_eps,
            min_points=args.cluster_min_points,
        )

    after_cluster = len(pcd.points)

    before_stat = len(pcd.points)

    if not args.no_filter and len(pcd.points) > 500:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=50,
            std_ratio=0.8
        )

    final_points = len(pcd.points)

    suffix = "object_only" if args.object_only else "full"
    table_suffix = "keep_table" if args.keep_table else "no_table"
    cluster_suffix = "cluster" if args.cluster else "no_cluster"
    color_suffix = "color_mask" if args.object_color_mask else "roi"

    out_ply = os.path.join(
        OUTPUT_DIR,
        f"point_cloud_{args.depth}_{suffix}_{color_suffix}_{table_suffix}_{cluster_suffix}.ply",
    )

    o3d.io.write_point_cloud(out_ply, pcd)

    print("\n=== 3D RECONSTRUCTION ===")
    print("Depth:", args.depth)
    print("Object only:", args.object_only)
    print("Object color mask:", args.object_color_mask)
    print("Keep table:", args.keep_table)
    print("Pull flying:", args.pull_flying)
    print("Z band:", args.z_band)
    print("Use box:", args.use_box)
    print("Box X/Y:", args.box_x, args.box_y)
    print("No gradient:", args.no_gradient)
    print("No Z trim:", args.no_z_trim)
    print("No stat filter:", args.no_filter)
    print("Cluster:", args.cluster)
    print("Min disparity:", args.min_disp)
    print("Hard Z clamp:", args.z_min, args.z_max)

    print("\n=== COVERAGE ===")
    print(f"Initial valid disparity: {initial_valid} / {total_pixels} = {percent(initial_valid, total_pixels):.2f}%")
    print(f"After gradient filter : {after_gradient} / {total_pixels} = {percent(after_gradient, total_pixels):.2f}%")
    print(f"Before hard Z clamp   : {before_z_hard} / {total_pixels} = {percent(before_z_hard, total_pixels):.2f}%")
    print(f"After hard Z clamp    : {after_z_hard} / {total_pixels} = {percent(after_z_hard, total_pixels):.2f}%")
    print(f"Before pull flying    : {before_pull} / {total_pixels} = {percent(before_pull, total_pixels):.2f}%")
    print(f"After pull flying     : {after_pull} / {total_pixels} = {percent(after_pull, total_pixels):.2f}%")
    print(f"Before box filter     : {before_box} / {total_pixels} = {percent(before_box, total_pixels):.2f}%")
    print(f"After box filter      : {after_box} / {total_pixels} = {percent(after_box, total_pixels):.2f}%")
    print(f"After table filter    : {after_table} / {total_pixels} = {percent(after_table, total_pixels):.2f}%")
    print(f"After Z trim          : {after_z_trim} / {total_pixels} = {percent(after_z_trim, total_pixels):.2f}%")
    print(f"Before cluster        : {before_cluster} / {total_pixels} = {percent(before_cluster, total_pixels):.2f}%")
    print(f"After cluster         : {after_cluster} / {total_pixels} = {percent(after_cluster, total_pixels):.2f}%")
    print(f"Before stat filter    : {before_stat} / {total_pixels} = {percent(before_stat, total_pixels):.2f}%")
    print(f"Final points          : {final_points} / {total_pixels} = {percent(final_points, total_pixels):.2f}%")

    print("\nSaved:", out_ply)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="3D Reconstruction",
        width=1200,
        height=900
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([0, 0, 0])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()