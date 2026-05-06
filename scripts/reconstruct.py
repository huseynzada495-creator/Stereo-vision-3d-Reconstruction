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


def save_mask(name, mask):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, mask.astype(np.uint8) * 255)
    return path


def create_loose_object_mask(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Main ROI: keep more than before
    roi = np.zeros((h, w), dtype=bool)
    roi[int(h * 0.05):int(h * 0.98), int(w * 0.25):int(w * 0.98)] = True

    # Keep colored object parts + bright/white object parts
    colored = (sat > 15) & (val > 20) & (val < 250)
    white_gray = (sat < 120) & (val > 60) & (val < 245)
    non_black = gray > 8

    mask = roi & non_black & (colored | white_gray)

    kernel = np.ones((7, 7), np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    return mask_u8.astype(bool)


def make_mask(img, disp, min_disp, max_disp, use_object_mask=True):
    h, w = disp.shape

    mask = np.isfinite(disp)
    mask &= disp >= min_disp
    mask &= disp <= max_disp

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask &= gray > 8

    if use_object_mask:
        object_mask = create_loose_object_mask(img)
        mask &= object_mask
        save_mask("object_color_mask.jpg", object_mask)
    else:
        object_mask = np.ones_like(mask, dtype=bool)

    save_mask("final_2d_mask.jpg", mask)
    return mask, object_mask


def remove_gradient_edges(mask, disp, threshold):
    dx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(dx, dy)
    grad[~np.isfinite(grad)] = 999999
    return mask & (grad < threshold)


def remove_table_color(points, colors):
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    # Less aggressive table removal
    table = (r > 150) & (g > 110) & (b < 135)
    return points[~table], colors[~table]


def apply_center_box(points, colors, xlim=2500, ylim=2500):
    x = points[:, 0]
    y = points[:, 1]

    keep = (
        (x > -xlim) & (x < xlim) &
        (y > -ylim) & (y < ylim)
    )

    return points[keep], colors[keep]


def keep_largest_cluster(pcd, eps=0.12, min_points=10):
    if len(pcd.points) == 0:
        return pcd

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
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
    print("\n=== Z DISTRIBUTION ===")
    print("Min Z:", np.min(z))
    print("Max Z:", np.max(z))
    print("Median Z:", np.median(z))
    print("Percentiles:", np.percentile(z, [1, 5, 25, 50, 75, 95, 99]))


def make_mesh_from_pointcloud(pcd, mesh_path, depth=8):
    if len(pcd.points) < 500:
        print("Mesh skipped: not enough points")
        return None

    mesh_pcd = o3d.geometry.PointCloud(pcd)
    mesh_pcd.estimate_normals()
    mesh_pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        mesh_pcd,
        depth=depth
    )

    densities = np.asarray(densities)

    # Keep more mesh than before
    keep = densities > np.quantile(densities, 0.03)
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(mesh_path, mesh)
    return mesh


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--depth", choices=list(DEPTH_FILES.keys()), default="dense_filled")

    parser.add_argument("--no-object-mask", action="store_true")
    parser.add_argument("--keep-table", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-gradient", action="store_true", default=True)
    parser.add_argument("--no-z-trim", action="store_true", default=False)

    parser.add_argument("--use-box", action="store_true", default=False)
    parser.add_argument("--box-x", type=float, default=2500.0)
    parser.add_argument("--box-y", type=float, default=2500.0)

    parser.add_argument("--cluster", action="store_true", default=False)
    parser.add_argument("--cluster-eps", type=float, default=0.12)
    parser.add_argument("--cluster-min-points", type=int, default=10)

    parser.add_argument("--min-disp", type=float, default=2.0)
    parser.add_argument("--max-disp", type=float, default=127.0)

    parser.add_argument("--grad-thresh", type=float, default=60.0)

    # Wider Z range: keeps more of the object
    parser.add_argument("--z-low-percent", type=float, default=1.0)
    parser.add_argument("--z-high-percent", type=float, default=99.0)

    parser.add_argument("--z-min", type=float, default=200.0)
    parser.add_argument("--z-max", type=float, default=30000.0)

    parser.add_argument("--voxel", type=float, default=0.003)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--flip-y", action="store_true", default=True)

    parser.add_argument("--make-mesh", action="store_true", default=True)
    parser.add_argument("--mesh-depth", type=int, default=8)
    parser.add_argument("--view-mesh", action="store_true")

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

    disp[~np.isfinite(disp)] = 0
    disp[disp < 0] = 0

    disp_preview = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "used_disparity_preview.png"),
        disp_preview.astype(np.uint8)
    )

    valid, total_mask = make_mask(
        img,
        disp,
        args.min_disp,
        args.max_disp,
        use_object_mask=not args.no_object_mask,
    )

    total_pixels = np.count_nonzero(total_mask)
    initial_valid = np.count_nonzero(valid)

    if not args.no_gradient:
        valid = remove_gradient_edges(valid, disp, args.grad_thresh)

    after_gradient = np.count_nonzero(valid)

    disp_for_3d = disp.copy()
    disp_for_3d[~valid] = 0

    points_3d = cv2.reprojectImageTo3D(disp_for_3d, Q, handleMissingValues=True)

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
        raise RuntimeError("No points left after hard Z clamp.")

    before_box = len(points)

    if args.use_box:
        points, colors = apply_center_box(points, colors, args.box_x, args.box_y)

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
        z_low, z_high = np.percentile(z, [args.z_low_percent, args.z_high_percent])
        keep = (z >= z_low) & (z <= z_high)
        points = points[keep]
        colors = colors[keep]

    after_z_trim = len(points)

    # Center and scale
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

    # Less harsh than before
    if not args.no_filter and len(pcd.points) > 500:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.8)

    final_points = len(pcd.points)

    base_name = f"{args.depth}_more_complete"

    out_ply = os.path.join(OUTPUT_DIR, f"point_cloud_{base_name}.ply")
    out_mesh = os.path.join(OUTPUT_DIR, f"mesh_{base_name}.ply")

    o3d.io.write_point_cloud(out_ply, pcd)

    mesh = None
    if args.make_mesh:
        mesh = make_mesh_from_pointcloud(pcd, out_mesh, depth=args.mesh_depth)

    print("\n=== 3D RECONSTRUCTION ===")
    print("Depth:", args.depth)
    print("Object mask:", not args.no_object_mask)
    print("Keep table:", args.keep_table)
    print("Use box:", args.use_box)
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
    print(f"Before box filter     : {before_box} / {total_pixels} = {percent(before_box, total_pixels):.2f}%")
    print(f"After box filter      : {after_box} / {total_pixels} = {percent(after_box, total_pixels):.2f}%")
    print(f"After table filter    : {after_table} / {total_pixels} = {percent(after_table, total_pixels):.2f}%")
    print(f"After Z trim          : {after_z_trim} / {total_pixels} = {percent(after_z_trim, total_pixels):.2f}%")
    print(f"Before cluster        : {before_cluster} / {total_pixels} = {percent(before_cluster, total_pixels):.2f}%")
    print(f"After cluster         : {after_cluster} / {total_pixels} = {percent(after_cluster, total_pixels):.2f}%")
    print(f"Before stat filter    : {before_stat} / {total_pixels} = {percent(before_stat, total_pixels):.2f}%")
    print(f"Final points          : {final_points} / {total_pixels} = {percent(final_points, total_pixels):.2f}%")

    print("\nSaved:")
    print("Point cloud:", out_ply)
    print("Mesh:", out_mesh)
    print("Mask:", os.path.join(OUTPUT_DIR, "final_2d_mask.jpg"))
    print("Disparity preview:", os.path.join(OUTPUT_DIR, "used_disparity_preview.png"))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Reconstruction", width=1200, height=900)

    if args.view_mesh and mesh is not None:
        vis.add_geometry(mesh)
    else:
        vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([0, 0, 0])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()