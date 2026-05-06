import os
import glob
import argparse
import cv2
import numpy as np
import open3d as o3d

LEFT_DIR = "data/stereo/final_object/cam_left"
RIGHT_DIR = "data/stereo/final_object/cam_right"
CALIB_DIR = "data/stereo/results"
OUT_DIR = "outputs/multiview_reconstruction"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_DISP = 0
NUM_DISP = 16 * 12
BLOCK_SIZE = 7


def find_image(folder, side, pair_id):
    for ext in ("jpg", "jpeg", "png", "bmp"):
        p = os.path.join(folder, f"{side}_{pair_id}.{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing {side}_{pair_id}")


def load_npy(name):
    return np.load(os.path.join(CALIB_DIR, name))


def compute_disparity(rectL, rectR):
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)

    grayL = cv2.GaussianBlur(grayL, (3, 3), 0)
    grayR = cv2.GaussianBlur(grayR, (3, 3), 0)

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * 3 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=4,
        speckleWindowSize=100,
        speckleRange=24,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp16 = left_matcher.compute(grayL, grayR)

    if hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        dispR16 = right_matcher.compute(grayR, grayL)

        wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls.setLambda(3000.0)
        wls.setSigmaColor(1.0)

        disp16 = wls.filter(disp16, grayL, None, dispR16)

    disp = disp16.astype(np.float32) / 16.0
    disp[disp < 2] = 0
    disp[disp > 240] = 0
    return disp


def make_valid_mask(img, disp):
    h, w = disp.shape

    valid = np.isfinite(disp)
    valid &= disp >= 2
    valid &= disp <= 240

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid &= gray > 20

    roi = np.zeros_like(valid, dtype=bool)
    roi[80:610, 470:980] = True
    valid &= roi

    dx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(dx, dy)
    valid &= grad < 25

    return valid


def remove_table(points, colors):
    rgb = np.clip(colors * 255, 0, 255).astype(np.uint8)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    table_like = (r > 145) & (g > 105) & (b < 115)

    return points[~table_like], colors[~table_like]


def pair_to_cloud(pair_id, maps, Q, voxel):
    map1x, map1y, map2x, map2y = maps

    lp = find_image(LEFT_DIR, "left", pair_id)
    rp = find_image(RIGHT_DIR, "right", pair_id)

    imgL = cv2.imread(lp)
    imgR = cv2.imread(rp)

    h, w = map1x.shape[:2]
    imgL = cv2.resize(imgL, (w, h))
    imgR = cv2.resize(imgR, (w, h))

    rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    disp = compute_disparity(rectL, rectR)
    valid = make_valid_mask(rectL, disp)

    points3d = cv2.reprojectImageTo3D(disp, Q)

    finite = np.isfinite(points3d).all(axis=2)
    finite &= valid

    points = points3d[finite]
    colors = rectL[finite][:, ::-1].astype(np.float64) / 255.0

    if np.median(points[:, 2]) < 0:
        points[:, 2] *= -1

    points, colors = remove_table(points, colors)

    z = points[:, 2]
    z_low, z_high = np.percentile(z, [10, 90])
    keep = (z >= z_low) & (z <= z_high)

    points = points[keep]
    colors = colors[keep]

    points = points - np.mean(points, axis=0)
    points = points / 1000.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    if len(pcd.points) > 500:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=2.0,
        )

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
    )

    out = os.path.join(OUT_DIR, f"cloud_pair_{pair_id}.ply")
    o3d.io.write_point_cloud(out, pcd)

    print(f"Pair {pair_id}: {len(pcd.points)} points saved -> {out}")

    return pcd


def register_clouds(source, target, voxel):
    threshold = max(voxel * 8, 0.05)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    return result.transformation, result.fitness, result.inlier_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["5", "7", "8", "9", "10"])
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--point-size", type=float, default=5.0)
    args = parser.parse_args()

    maps = (
        load_npy("map1x.npy"),
        load_npy("map1y.npy"),
        load_npy("map2x.npy"),
        load_npy("map2y.npy"),
    )
    Q = load_npy("Q.npy").astype(np.float64)

    clouds = []

    for pair_id in args.pairs:
        pcd = pair_to_cloud(pair_id, maps, Q, args.voxel)
        clouds.append(pcd)

    merged = clouds[0]

    for i in range(1, len(clouds)):
        T, fitness, rmse = register_clouds(clouds[i], merged, args.voxel)
        print(f"Register pair {args.pairs[i]} -> fitness={fitness:.3f}, rmse={rmse:.4f}")

        aligned = clouds[i].transform(T)
        merged += aligned

        merged = merged.voxel_down_sample(args.voxel)
        merged.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )

    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=40,
        std_ratio=2.0,
    )

    out_merged = os.path.join(OUT_DIR, "merged_multiview.ply")
    o3d.io.write_point_cloud(out_merged, merged)

    print("\nDONE")
    print("Merged points:", len(merged.points))
    print("Saved:", out_merged)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Multi-view Reconstruction")
    vis.add_geometry(merged)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([0, 0, 0])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()