import os
import argparse
import numpy as np
import open3d as o3d

INPUT_DIR = "outputs/3d_reconstruction"
OUTPUT_DIR = "outputs/3d_reconstruction_cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_cloud(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing point cloud: {path}")

    pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        raise RuntimeError("Loaded point cloud is empty.")

    return pcd


def keep_largest_cluster(pcd, eps, min_points):
    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=True,
        )
    )

    if labels.size == 0 or labels.max() < 0:
        print("No cluster found. Keeping original cloud.")
        return pcd

    valid_labels = labels[labels >= 0]
    largest_label = np.bincount(valid_labels).argmax()
    keep_idx = np.where(labels == largest_label)[0]

    print("Clusters found:", labels.max() + 1)
    print("Kept cluster:", largest_label)
    print("Kept points:", len(keep_idx))

    return pcd.select_by_index(keep_idx)


def radius_filter(pcd, nb_points, radius):
    clean, ind = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )
    print("After radius filter:", len(clean.points))
    return clean


def statistical_filter(pcd, nb_neighbors, std_ratio):
    clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    print("After statistical filter:", len(clean.points))
    return clean


def z_trim(pcd, low_percent, high_percent):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    z = points[:, 2]
    z_low, z_high = np.percentile(z, [low_percent, high_percent])

    keep = (z >= z_low) & (z <= z_high)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points[keep])
    new_pcd.colors = o3d.utility.Vector3dVector(colors[keep])

    print("After Z trim:", len(new_pcd.points))
    print("Z range kept:", z_low, z_high)

    return new_pcd


def voxel_downsample(pcd, voxel):
    if voxel <= 0:
        return pcd

    pcd = pcd.voxel_down_sample(voxel)
    print("After voxel downsample:", len(pcd.points))
    return pcd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input .ply file path",
    )

    parser.add_argument("--output", default=None)

    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--cluster-eps", type=float, default=0.08)
    parser.add_argument("--cluster-min-points", type=int, default=15)

    parser.add_argument("--radius-filter", action="store_true")
    parser.add_argument("--radius", type=float, default=0.04)
    parser.add_argument("--radius-nb", type=int, default=8)

    parser.add_argument("--stat-filter", action="store_true")
    parser.add_argument("--stat-nb", type=int, default=40)
    parser.add_argument("--stat-std", type=float, default=1.0)

    parser.add_argument("--z-trim", action="store_true")
    parser.add_argument("--z-low", type=float, default=1.0)
    parser.add_argument("--z-high", type=float, default=99.0)

    parser.add_argument("--voxel", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=6.0)

    args = parser.parse_args()

    pcd = load_cloud(args.input)

    print("\n=== 3D ONLY CLEANER ===")
    print("Input:", args.input)
    print("Initial points:", len(pcd.points))

    pcd = voxel_downsample(pcd, args.voxel)

    if args.z_trim:
        pcd = z_trim(pcd, args.z_low, args.z_high)

    if args.cluster:
        pcd = keep_largest_cluster(
            pcd,
            eps=args.cluster_eps,
            min_points=args.cluster_min_points,
        )

    if args.radius_filter:
        pcd = radius_filter(
            pcd,
            nb_points=args.radius_nb,
            radius=args.radius,
        )

    if args.stat_filter:
        pcd = statistical_filter(
            pcd,
            nb_neighbors=args.stat_nb,
            std_ratio=args.stat_std,
        )

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(OUTPUT_DIR, base + "_cleaned.ply")
    else:
        out_path = args.output

    o3d.io.write_point_cloud(out_path, pcd)

    print("\nFinal points:", len(pcd.points))
    print("Saved:", out_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Only Cleaned Point Cloud", width=1200, height=900)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([0, 0, 0])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()