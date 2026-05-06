import os
import glob
import cv2
import numpy as np

# =========================
# SETTINGS
# =========================

MODE = "final"

PAIR_ID = 0
FINAL_PAIR_ID = "5"

LEFT_DIR = "data/stereo/cam_left/images_full"
RIGHT_DIR = "data/stereo/cam_right/images_full"

STEREO_DIR = "data/stereo/results"
OUT_DIR = "outputs/stereo_depth_rectified"
os.makedirs(OUT_DIR, exist_ok=True)

# Cleaner map settings
MIN_DISP = 0
NUM_DISP = 16 * 8
BLOCK_SIZE = 5

MIN_VALID_DISP = 2.0
MAX_VALID_DISP = 127.0


# =========================
# HELPERS
# =========================

def collect(folder):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def find_final_image(side, pair_id):
    folder = f"data/stereo/final_object/cam_{side}"
    for ext in ("jpg", "jpeg", "png", "bmp"):
        path = os.path.join(folder, f"{side}_{pair_id}.{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find final {side} image for ID {pair_id}")


def load_npy(name):
    path = os.path.join(STEREO_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing calibration file: {path}")
    return np.load(path)


def normalize_and_colorize(disp):
    valid = np.isfinite(disp) & (disp >= MIN_VALID_DISP)
    out = np.zeros_like(disp, dtype=np.uint8)

    if np.count_nonzero(valid) > 0:
        lo, hi = np.percentile(disp[valid], [5, 95])
        hi = max(hi, lo + 1e-6)
        out[valid] = np.clip((disp[valid] - lo) * 255.0 / (hi - lo), 0, 255)

    color = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    color[~valid] = 0
    return out, color


def draw_lines(img, step=40):
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
    return out


def print_disp_info(name, disp):
    valid = disp[np.isfinite(disp) & (disp >= MIN_VALID_DISP)]
    print(f"\n=== {name} DISPARITY ===")
    print("Valid:", len(valid))
    if len(valid):
        print("Mean :", valid.mean())
        print("Max  :", valid.max())
        print("Min  :", valid.min())


def feature_match(imgL, imgR, method="SIFT"):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
        ratio = 0.85
    elif method == "ORB":
        detector = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=10,
        )
        norm = cv2.NORM_HAMMING
        ratio = 0.90
    else:
        raise ValueError("Use SIFT or ORB")

    kpL, desL = detector.detectAndCompute(grayL, None)
    kpR, desR = detector.detectAndCompute(grayR, None)

    if desL is None or desR is None:
        return kpL, kpR, []

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(desL, desR, k=2)

    good = []

    for pair in knn:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            xL, yL = kpL[m.queryIdx].pt
            xR, yR = kpR[m.trainIdx].pt

            if abs(yL - yR) < 5.0:
                good.append(m)

    return kpL, kpR, good


def fill_disparity(disp, radius=5, dilate_kernel=5, dilate_iter=2):
    disp_filled = disp.copy()

    valid = np.isfinite(disp_filled) & (disp_filled >= MIN_VALID_DISP)
    invalid = ~valid

    disp_norm = np.zeros_like(disp_filled, dtype=np.uint8)

    if np.count_nonzero(valid) == 0:
        return disp_filled

    lo, hi = np.percentile(disp_filled[valid], [5, 95])
    hi = max(hi, lo + 1e-6)

    disp_norm[valid] = np.clip(
        (disp_filled[valid] - lo) * 255.0 / (hi - lo),
        0,
        255
    ).astype(np.uint8)

    hole_mask = invalid.astype(np.uint8) * 255

    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    near_valid = cv2.dilate(valid.astype(np.uint8), kernel, iterations=dilate_iter)

    hole_mask[near_valid == 0] = 0

    inpainted = cv2.inpaint(
        disp_norm,
        hole_mask,
        radius,
        cv2.INPAINT_TELEA
    )

    disp_filled[hole_mask > 0] = (
        inpainted[hole_mask > 0].astype(np.float32) / 255.0
    ) * (hi - lo) + lo

    #disp_filled = cv2.medianBlur(disp_filled.astype(np.float32), 5)

    disp_filled[disp_filled < MIN_VALID_DISP] = 0
    disp_filled[disp_filled > MAX_VALID_DISP] = 0

    return disp_filled


# =========================
# LOAD IMAGE PAIR
# =========================

if MODE == "calib":
    left_paths = collect(LEFT_DIR)
    right_paths = collect(RIGHT_DIR)

    if not left_paths or not right_paths:
        raise RuntimeError("No calibration images found.")

    if len(left_paths) != len(right_paths):
        raise RuntimeError("Left/right calibration image count mismatch.")

    LEFT_PATH = left_paths[PAIR_ID]
    RIGHT_PATH = right_paths[PAIR_ID]
else:
    LEFT_PATH = find_final_image("left", FINAL_PAIR_ID)
    RIGHT_PATH = find_final_image("right", FINAL_PAIR_ID)

imgL = cv2.imread(LEFT_PATH)
imgR = cv2.imread(RIGHT_PATH)

if imgL is None or imgR is None:
    raise FileNotFoundError(f"Could not load:\n{LEFT_PATH}\n{RIGHT_PATH}")

if imgL.shape != imgR.shape:
    raise RuntimeError(
        f"Left/right image sizes differ.\n"
        f"Left : {imgL.shape}\n"
        f"Right: {imgR.shape}"
    )

print("\n=== INPUT ===")
print("Mode :", MODE)
print("Left :", LEFT_PATH)
print("Right:", RIGHT_PATH)
print("Shape:", imgL.shape)


# =========================
# LOAD CALIBRATION
# =========================

map1x = load_npy("map1x.npy")
map1y = load_npy("map1y.npy")
map2x = load_npy("map2x.npy")
map2y = load_npy("map2y.npy")
Q = load_npy("Q.npy")


# =========================
# RECTIFY
# =========================

rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

cv2.imwrite(os.path.join(OUT_DIR, "raw_left.jpg"), imgL)
cv2.imwrite(os.path.join(OUT_DIR, "raw_right.jpg"), imgR)
cv2.imwrite(os.path.join(OUT_DIR, "rectified_left.jpg"), rectL)
cv2.imwrite(os.path.join(OUT_DIR, "rectified_right.jpg"), rectR)
cv2.imwrite(os.path.join(OUT_DIR, "rectified_left_lines.jpg"), draw_lines(rectL))
cv2.imwrite(os.path.join(OUT_DIR, "rectified_right_lines.jpg"), draw_lines(rectR))


# =========================
# SIFT / ORB MATCHING CHECK
# =========================

for method in ["SIFT", "ORB"]:
    kpL, kpR, good = feature_match(rectL, rectR, method)

    print(
        f"{method} | "
        f"left kp={len(kpL)} | right kp={len(kpR)} | good matches={len(good)}"
    )

    match_img = cv2.drawMatches(
        rectL,
        kpL,
        rectR,
        kpR,
        good[:1000],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(
        os.path.join(OUT_DIR, f"{method.lower()}_matches_rectified.jpg"),
        match_img,
    )


# =========================
# PREPROCESS
# =========================

#grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
#grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
grayL = rectL[:, :, 1]   # green channel
grayR = rectR[:, :, 1]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
grayL = clahe.apply(grayL)
grayR = clahe.apply(grayR)

grayL = cv2.GaussianBlur(grayL, (3, 3), 0)
grayR = cv2.GaussianBlur(grayR, (3, 3), 0)


# =========================
# CLEAN SGBM DISPARITY
# =========================

left_matcher = cv2.StereoSGBM_create(
    minDisparity=MIN_DISP,
    numDisparities=NUM_DISP,
    blockSize=BLOCK_SIZE,
    P1=8 * 3 * BLOCK_SIZE * BLOCK_SIZE,
    P2=32 * 3 * BLOCK_SIZE * BLOCK_SIZE,
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

disp_raw16 = left_matcher.compute(grayL, grayR)
disp_raw = disp_raw16.astype(np.float32) / 16.0
disp_raw[disp_raw < 0] = 0

disp_wls = disp_raw.copy()
use_wls = False

if (
    hasattr(cv2, "ximgproc")
    and hasattr(cv2.ximgproc, "createRightMatcher")
    and hasattr(cv2.ximgproc, "createDisparityWLSFilter")
):
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    disp_right16 = right_matcher.compute(grayR, grayL)

    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)

    filtered16 = wls.filter(disp_raw16, grayL, None, disp_right16)
    disp_wls = filtered16.astype(np.float32) / 16.0
    disp_wls[disp_wls < 0] = 0
    disp_wls[disp_wls > MAX_VALID_DISP] = 0
    # Remove wrong isolated disparity on smooth teddy/head regions
    # normalize disparity to uint8
valid = disp_wls > 0

disp_u8 = np.zeros_like(disp_wls, dtype=np.uint8)

if np.count_nonzero(valid) > 0:
    lo, hi = np.percentile(disp_wls[valid], [5, 95])
    hi = max(hi, lo + 1e-6)

    disp_u8[valid] = np.clip(
        (disp_wls[valid] - lo) * 255.0 / (hi - lo),
        0, 255
    ).astype(np.uint8)

    # median blur in uint8
    smooth_u8 = cv2.medianBlur(disp_u8, 3)

    # convert back to float disparity
    smooth = np.zeros_like(disp_wls, dtype=np.float32)
    smooth[valid] = (
        smooth_u8[valid].astype(np.float32) / 255.0
    ) * (hi - lo) + lo

    # consistency filter
    diff = np.abs(disp_wls - smooth)
    bad = (disp_wls > 0) & (diff > 15)
    disp_wls[bad] = 0

    use_wls = True
# =========================
# OBJECT-GUIDED WLS SMOOTHING
# =========================

hsv = cv2.cvtColor(rectL, cv2.COLOR_BGR2HSV)
hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

green = (hue >= 35) & (hue <= 95) & (sat > 20) & (val > 20) & (val < 245)
teddy = (hue >= 5) & (hue <= 35) & (sat > 15) & (val > 20) & (val < 245)
white = (sat < 90) & (val > 80) & (val < 230)

roi_all = np.zeros_like(green, dtype=bool)
roi_all[60:650, 500:980] = True

roi_shirt = np.zeros_like(green, dtype=bool)
roi_shirt[280:520, 600:900] = True

object_mask = ((green | teddy) & roi_all) | (white & roi_shirt)
valid_obj = object_mask & (disp_wls > 0)

disp_smooth = cv2.bilateralFilter(
    disp_wls.astype(np.float32),
    d=9,
    sigmaColor=20,
    sigmaSpace=20,
)

disp_wls_object_smooth = disp_wls.copy()
disp_wls_object_smooth[valid_obj] = (
    0.70 * disp_wls[valid_obj] +
    0.30 * disp_smooth[valid_obj]
)

np.save(os.path.join(OUT_DIR, "disparity_wls_object_smooth.npy"), disp_wls_object_smooth)

u8, color = normalize_and_colorize(disp_wls_object_smooth)
cv2.imwrite(os.path.join(OUT_DIR, "disparity_wls_object_smooth.jpg"), u8)
cv2.imwrite(os.path.join(OUT_DIR, "disparity_wls_object_smooth_color.jpg"), color)

# =========================
# DENSE SGBM DISPARITY
# =========================

dense_matcher = cv2.StereoSGBM_create(
    minDisparity=MIN_DISP,
    numDisparities=NUM_DISP,
    blockSize=BLOCK_SIZE,
    P1=8 * 3 * BLOCK_SIZE * BLOCK_SIZE,
    P2=32 * 3 * BLOCK_SIZE * BLOCK_SIZE,
    disp12MaxDiff=2,
    preFilterCap=63,
    uniquenessRatio=1,
    speckleWindowSize=30,
    speckleRange=48,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

disp_dense_raw16 = dense_matcher.compute(grayL, grayR)
disp_dense = disp_dense_raw16.astype(np.float32) / 16.0
disp_dense[disp_dense < 0] = 0

disp_dense_wls = disp_dense.copy()
use_dense_wls = False

if (
    hasattr(cv2, "ximgproc")
    and hasattr(cv2.ximgproc, "createRightMatcher")
    and hasattr(cv2.ximgproc, "createDisparityWLSFilter")
):
    dense_right_matcher = cv2.ximgproc.createRightMatcher(dense_matcher)
    disp_dense_right16 = dense_right_matcher.compute(grayR, grayL)

    dense_wls = cv2.ximgproc.createDisparityWLSFilter(dense_matcher)
    dense_wls.setLambda(2000.0)
    dense_wls.setSigmaColor(0.8)

    dense_filtered16 = dense_wls.filter(disp_dense_raw16, grayL, None, disp_dense_right16)
    disp_dense_wls = dense_filtered16.astype(np.float32) / 16.0
    disp_dense_wls[disp_dense_wls < 0] = 0
    use_dense_wls = True


# =========================
# MASKS
# =========================

sobelx = cv2.Sobel(grayL, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(grayL, cv2.CV_32F, 0, 1, ksize=3)
texture = cv2.magnitude(sobelx, sobely)

texture_mask = texture > 2.0
texture_vis = np.zeros_like(grayL)
texture_vis[texture_mask] = 255
cv2.imwrite(os.path.join(OUT_DIR, "texture_mask.jpg"), texture_vis)

h, w = grayL.shape
object_mask = np.zeros((h, w), dtype=np.uint8)
object_mask[40:650, 350:1120] = 255
cv2.imwrite(os.path.join(OUT_DIR, "object_mask.jpg"), object_mask)


# =========================
# CLEAN DISPARITY
# =========================

disp_clean = disp_wls.copy()

disp_clean[disp_clean < 5.0] = 0
disp_clean[disp_clean > MAX_VALID_DISP] = 0

valid_mask = (disp_clean > 0).astype(np.uint8)

kernel = np.ones((2, 2), np.uint8)
#valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)

disp_clean[valid_mask == 0] = 0

#disp_clean = cv2.bilateralFilter(disp_clean.astype(np.float32), 3, 10, 10)
disp_clean[disp_clean < 5.0] = 0
disp_clean[disp_clean > MAX_VALID_DISP] = 0


# =========================
# STANDARD FILLED DISPARITY
# =========================

disp_filled = fill_disparity(
    disp_clean,
    radius=5,
    dilate_kernel=5,
    dilate_iter=2,
)


# =========================
# DENSE FILLED DISPARITY
# =========================

disp_dense_clean = disp_dense_wls.copy()

disp_dense_clean[disp_dense_clean < MIN_VALID_DISP] = 0
disp_dense_clean[disp_dense_clean > MAX_VALID_DISP] = 0

# light cleanup only
dense_valid_mask = (disp_dense_clean > 0).astype(np.uint8)
dense_kernel = np.ones((2, 2), np.uint8)
dense_valid_mask = cv2.morphologyEx(dense_valid_mask, cv2.MORPH_CLOSE, dense_kernel)
disp_dense_clean[dense_valid_mask == 0] = 0

disp_dense_filled = fill_disparity(
    disp_dense_clean,
    radius=9,
    dilate_kernel=9,
    dilate_iter=4,
)

# extra aggressive fill for nearby holes
disp_dense_filled = fill_disparity(
    disp_dense_filled,
    radius=7,
    dilate_kernel=11,
    dilate_iter=2,
)


# =========================
# SAVE OUTPUTS
# =========================

print("\nWLS filtering used:", use_wls)
print("Dense WLS filtering used:", use_dense_wls)

np.save(os.path.join(OUT_DIR, "disparity_raw.npy"), disp_raw)
np.save(os.path.join(OUT_DIR, "disparity_wls.npy"), disp_wls)
np.save(os.path.join(OUT_DIR, "disparity_clean.npy"), disp_clean)
np.save(os.path.join(OUT_DIR, "disparity_filled.npy"), disp_filled)
np.save(os.path.join(OUT_DIR, "disparity_dense.npy"), disp_dense_wls)
np.save(os.path.join(OUT_DIR, "disparity_dense_filled.npy"), disp_dense_filled)
np.save(os.path.join(OUT_DIR, "Q.npy"), Q)

for name, disp in [
    ("disparity_raw", disp_raw),
    ("disparity_wls", disp_wls),
    ("disparity_clean", disp_clean),
    ("disparity_filled", disp_filled),
    ("disparity_dense", disp_dense_wls),
    ("disparity_dense_filled", disp_dense_filled),
]:
    u8, color = normalize_and_colorize(disp)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}.jpg"), u8)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_color.jpg"), color)

print_disp_info("RAW", disp_raw)
print_disp_info("WLS", disp_wls)
print_disp_info("CLEAN", disp_clean)
print_disp_info("FILLED", disp_filled)
print_disp_info("DENSE", disp_dense_wls)
print_disp_info("DENSE_FILLED", disp_dense_filled)

print("\nSaved outputs to:", OUT_DIR)
print("Check:")
print(" - disparity_filled_color.jpg")
print(" - disparity_dense_color.jpg")
print(" - disparity_dense_filled_color.jpg")


# =========================
# DISPLAY
# =========================

_, wls_color = normalize_and_colorize(disp_wls)
_, clean_color = normalize_and_colorize(disp_clean)
_, filled_color = normalize_and_colorize(disp_filled)
_, dense_color = normalize_and_colorize(disp_dense_wls)
_, dense_filled_color = normalize_and_colorize(disp_dense_filled)

cv2.imshow("Rectified Left Lines", draw_lines(rectL))
cv2.imshow("Rectified Right Lines", draw_lines(rectR))
cv2.imshow("WLS Disparity", wls_color)
cv2.imshow("Clean Disparity", clean_color)
cv2.imshow("Filled Disparity", filled_color)
cv2.imshow("Dense Disparity", dense_color)
cv2.imshow("Dense Filled Disparity", dense_filled_color)

print("\nPress q to quit.")
while True:
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()