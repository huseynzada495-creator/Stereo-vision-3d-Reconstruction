import cv2
import numpy as np
import glob
import os
import json

left_folder = r"C:\3dproj\data\stereo\cam_left\images_full"
right_folder = r"C:\3dproj\data\stereo\cam_right\images_full"
output_folder = r"C:\3dproj\data\stereo\results"
os.makedirs(output_folder, exist_ok=True)

checkerboard = (5, 4)
square_size = 25.0

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    1e-4,
)

left_images = sorted(
    glob.glob(os.path.join(left_folder, "*.jpg")) +
    glob.glob(os.path.join(left_folder, "*.jpeg")) +
    glob.glob(os.path.join(left_folder, "*.png"))
)

right_images = sorted(
    glob.glob(os.path.join(right_folder, "*.jpg")) +
    glob.glob(os.path.join(right_folder, "*.jpeg")) +
    glob.glob(os.path.join(right_folder, "*.png"))
)

print("Left images found:", len(left_images))
print("Right images found:", len(right_images))

if len(left_images) == 0 or len(right_images) == 0:
    raise RuntimeError("No images found.")

if len(left_images) != len(right_images):
    raise RuntimeError("Left/right image count mismatch.")

objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []
used_pairs = []

image_size = None

for i, (lp, rp) in enumerate(zip(left_images, right_images)):
    img_l = cv2.imread(lp)
    img_r = cv2.imread(rp)

    if img_l is None or img_r is None:
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray_l.shape[::-1]

    if gray_l.shape[::-1] != image_size or gray_r.shape[::-1] != image_size:
        print("Skipping size mismatch:", lp, rp)
        continue

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard, None)

    print(f"Pair {i+1}: left={ret_l}, right={ret_r}")

    if ret_l and ret_r:
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)
        used_pairs.append((os.path.basename(lp), os.path.basename(rp)))

        dbg_l = img_l.copy()
        dbg_r = img_r.copy()
        cv2.drawChessboardCorners(dbg_l, checkerboard, corners_l, ret_l)
        cv2.drawChessboardCorners(dbg_r, checkerboard, corners_r, ret_r)

        cv2.imwrite(os.path.join(output_folder, f"corners_left_{i+1}.jpg"), dbg_l)
        cv2.imwrite(os.path.join(output_folder, f"corners_right_{i+1}.jpg"), dbg_r)

print("\nValid stereo checkerboard pairs:", len(objpoints))

if len(objpoints) < 8:
    raise RuntimeError("Not enough valid checkerboard pairs. Need at least 8 good pairs.")

# =========================
# MONO CALIBRATION
# =========================

ret_l, K1, D1, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_left, image_size, None, None
)

ret_r, K2, D2, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_right, image_size, None, None
)

print("\nLeft RMS:", ret_l)
print("Right RMS:", ret_r)
print("K1:\n", K1)
print("D1:\n", D1)
print("K2:\n", K2)
print("D2:\n", D2)

# =========================
# STEREO CALIBRATION
# =========================

flags = cv2.CALIB_FIX_INTRINSIC

ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1,
    D1,
    K2,
    D2,
    image_size,
    criteria=criteria,
    flags=flags,
)

print("\nStereo RMS:", ret_stereo)
print("R:\n", R)
print("T:\n", T)

# =========================
# STEREO RECTIFICATION
# =========================
# IMPORTANT: use original K1/K2 here, not getOptimalNewCameraMatrix

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1,
    D1,
    K2,
    D2,
    image_size,
    R,
    T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.5,
)

print("\nStereo rectification done.")
print("P1:\n", P1)
print("P2:\n", P2)
print("Q:\n", Q)

# =========================
# RECTIFICATION MAPS
# =========================

map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, image_size, cv2.CV_32FC1
)

map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, image_size, cv2.CV_32FC1
)

# =========================
# SAVE EVERYTHING SEPARATELY
# =========================

np.save(os.path.join(output_folder, "K1.npy"), K1)
np.save(os.path.join(output_folder, "dist1.npy"), D1)
np.save(os.path.join(output_folder, "K2.npy"), K2)
np.save(os.path.join(output_folder, "dist2.npy"), D2)

np.save(os.path.join(output_folder, "R.npy"), R)
np.save(os.path.join(output_folder, "T.npy"), T)
np.save(os.path.join(output_folder, "E.npy"), E)
np.save(os.path.join(output_folder, "F.npy"), F)

np.save(os.path.join(output_folder, "R1.npy"), R1)
np.save(os.path.join(output_folder, "R2.npy"), R2)
np.save(os.path.join(output_folder, "P1.npy"), P1)
np.save(os.path.join(output_folder, "P2.npy"), P2)
np.save(os.path.join(output_folder, "Q.npy"), Q)

np.save(os.path.join(output_folder, "map1x.npy"), map1x)
np.save(os.path.join(output_folder, "map1y.npy"), map1y)
np.save(os.path.join(output_folder, "map2x.npy"), map2x)
np.save(os.path.join(output_folder, "map2y.npy"), map2y)

np.savez(
    os.path.join(output_folder, "stereo_calibration_data.npz"),
    K1=K1,
    D1=D1,
    K2=K2,
    D2=D2,
    R=R,
    T=T,
    E=E,
    F=F,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q,
    roi1=roi1,
    roi2=roi2,
    map1x=map1x,
    map1y=map1y,
    map2x=map2x,
    map2y=map2y,
    image_size=image_size,
)

report = {
    "left_rms_error": float(ret_l),
    "right_rms_error": float(ret_r),
    "stereo_rms_error": float(ret_stereo),
    "valid_pairs": len(objpoints),
    "image_size": image_size,
    "checkerboard": checkerboard,
    "square_size": square_size,
    "used_pairs": used_pairs,
}

with open(os.path.join(output_folder, "stereo_report.json"), "w") as f:
    json.dump(report, f, indent=2)

# =========================
# VISUAL RECTIFICATION CHECK
# =========================

img_l = cv2.imread(left_images[0])
img_r = cv2.imread(right_images[0])

rect_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
rect_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)

cv2.imwrite(os.path.join(output_folder, "rectified_left.jpg"), rect_l)
cv2.imwrite(os.path.join(output_folder, "rectified_right.jpg"), rect_r)

combined = np.hstack((rect_l, rect_r))

for y in range(0, combined.shape[0], 40):
    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

cv2.imwrite(os.path.join(output_folder, "rectification_check.jpg"), combined)

print("\nDONE.")
print("Saved all calibration files to:", output_folder)
print("Check:", os.path.join(output_folder, "rectification_check.jpg"))