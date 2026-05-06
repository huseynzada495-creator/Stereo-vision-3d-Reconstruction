import os
import cv2
import numpy as np

# =========================
# SETTINGS
# =========================

PAIR_ID = "5"

LEFT_PATH = f"data/stereo/final_object/cam_left/left_{PAIR_ID}.jpg"
RIGHT_PATH = f"data/stereo/final_object/cam_right/right_{PAIR_ID}.jpg"

OUTPUT_DIR = "outputs/final_object_direct"
STEREO_DIR = "data/stereo/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_RECTIFICATION = True
RECTIFY_MODE = "saved_maps"
ALPHA = 1.0

MIN_DISP = 0
NUM_DISP = 16 * 12
BLOCK_SIZE = 7


def load_npy(name):
    path = os.path.join(STEREO_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing calibration file: {path}")
    return np.load(path)


def normalize_and_colorize(disp):
    valid = np.isfinite(disp) & (disp > 0)
    out = np.zeros_like(disp, dtype=np.uint8)

    if np.count_nonzero(valid) > 0:
        dmin = float(np.percentile(disp[valid], 5))
        dmax = float(np.percentile(disp[valid], 95))
        dmax = max(dmax, dmin + 1e-6)
        out[valid] = np.clip((disp[valid] - dmin) * 255.0 / (dmax - dmin), 0, 255)

    color = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    color[~valid] = 0
    return out, color

def fill_disparity_holes(disp, min_disp=2.0, max_disp=240.0):
    disp = disp.astype(np.float32).copy()

    valid = np.isfinite(disp) & (disp >= min_disp) & (disp <= max_disp)
    invalid = ~valid

    if np.count_nonzero(valid) == 0:
        return disp

    # Normalize valid disparity to 8-bit for inpainting
    lo, hi = np.percentile(disp[valid], [5, 95])
    hi = max(hi, lo + 1e-6)

    disp_u8 = np.zeros_like(disp, dtype=np.uint8)
    disp_u8[valid] = np.clip((disp[valid] - lo) * 255.0 / (hi - lo), 0, 255)

    # Inpaint invalid holes
    mask_u8 = invalid.astype(np.uint8) * 255
    filled_u8 = cv2.inpaint(disp_u8, mask_u8, 5, cv2.INPAINT_TELEA)

    # Convert back to disparity scale
    filled = disp.copy()
    filled[invalid] = (filled_u8[invalid].astype(np.float32) / 255.0) * (hi - lo) + lo

    filled[filled < min_disp] = 0
    filled[filled > max_disp] = 0

    return filled

def draw_horizontal_lines(img, step=80):
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
    return out


def create_object_roi_mask(shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    x1 = int(w * 0.03)
    x2 = int(w * 0.99)
    y1 = int(h * 0.03)
    y2 = int(h * 0.99)

    mask[y1:y2, x1:x2] = 255
    return mask, (x1, y1, x2, y2)


def print_disp_info(label, disp):
    valid = disp[np.isfinite(disp) & (disp > 0)]
    print(f"\n=== {label} DISPARITY INFO ===")
    if len(valid) > 0:
        print(f"Valid disparity pixels: {len(valid)}")
        print(f"Min disparity         : {valid.min():.4f}")
        print(f"Mean disparity        : {valid.mean():.4f}")
        print(f"Max disparity         : {valid.max():.4f}")
    else:
        print("No valid disparity values found.")


# =========================
# LOAD FINAL OBJECT PAIR
# =========================

imgL0 = cv2.imread(LEFT_PATH)
imgR0 = cv2.imread(RIGHT_PATH)

if imgL0 is None or imgR0 is None:
    raise FileNotFoundError(f"Could not load:\n{LEFT_PATH}\n{RIGHT_PATH}")

print("\n=== INPUT FINAL OBJECT PAIR ===")
print("Left :", LEFT_PATH, imgL0.shape)
print("Right:", RIGHT_PATH, imgR0.shape)

if imgL0.shape != imgR0.shape:
    raise RuntimeError(
        f"Left/right image sizes differ.\n"
        f"Left : {imgL0.shape}\n"
        f"Right: {imgR0.shape}"
    )

# =========================
# RECTIFICATION
# =========================

Q = load_npy("Q.npy")

if USE_RECTIFICATION:
    if RECTIFY_MODE == "saved_maps":
        map1x = load_npy("map1x.npy")
        map1y = load_npy("map1y.npy")
        map2x = load_npy("map2x.npy")
        map2y = load_npy("map2y.npy")

        imgL = cv2.remap(imgL0, map1x, map1y, cv2.INTER_LINEAR)
        imgR = cv2.remap(imgR0, map2x, map2y, cv2.INTER_LINEAR)

        print("\n=== SELECTED PIPELINE ===")
        print("Rectification used: True")
        print("Rectification mode: saved calibration maps")

    elif RECTIFY_MODE == "recompute":
        K1 = load_npy("K1.npy")
        D1 = load_npy("dist1.npy")
        K2 = load_npy("K2.npy")
        D2 = load_npy("dist2.npy")
        R = load_npy("R.npy")
        T = load_npy("T.npy")

        h, w = imgL0.shape[:2]
        img_size = (w, h)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=ALPHA,
        )

        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, img_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, img_size, cv2.CV_32FC1
        )

        imgL = cv2.remap(imgL0, map1x, map1y, cv2.INTER_LINEAR)
        imgR = cv2.remap(imgR0, map2x, map2y, cv2.INTER_LINEAR)

        print("\n=== SELECTED PIPELINE ===")
        print("Rectification used: True")
        print("Rectification mode: recompute")
        print(f"alpha             : {ALPHA}")

    else:
        raise RuntimeError("RECTIFY_MODE must be 'saved_maps' or 'recompute'")
else:
    imgL = imgL0.copy()
    imgR = imgR0.copy()
    print("\n=== SELECTED PIPELINE ===")
    print("Rectification used: False")

print("Output size:", imgL.shape)

cv2.imwrite(os.path.join(OUTPUT_DIR, "left_full.jpg"), imgL)
cv2.imwrite(os.path.join(OUTPUT_DIR, "right_full.jpg"), imgR)
cv2.imwrite(os.path.join(OUTPUT_DIR, "left_lines.jpg"), draw_horizontal_lines(imgL))
cv2.imwrite(os.path.join(OUTPUT_DIR, "right_lines.jpg"), draw_horizontal_lines(imgR))
np.save(os.path.join(OUTPUT_DIR, "Q.npy"), Q)

# =========================
# PREPROCESS
# =========================

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
grayL_eq = clahe.apply(grayL)
grayR_eq = clahe.apply(grayR)

grayL_eq = cv2.GaussianBlur(grayL_eq, (3, 3), 0)
grayR_eq = cv2.GaussianBlur(grayR_eq, (3, 3), 0)

cv2.imwrite(os.path.join(OUTPUT_DIR, "left_equalized.jpg"), grayL_eq)
cv2.imwrite(os.path.join(OUTPUT_DIR, "right_equalized.jpg"), grayR_eq)

# =========================
# ROI
# =========================

roi_mask, (rx1, ry1, rx2, ry2) = create_object_roi_mask(grayL_eq.shape)

roi_preview = imgL.copy()
cv2.rectangle(roi_preview, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
cv2.imwrite(os.path.join(OUTPUT_DIR, "disparity_roi_preview.jpg"), roi_preview)

# =========================
# DISPARITY
# =========================

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

dispL_raw16 = left_matcher.compute(grayL_eq, grayR_eq)
dispL = dispL_raw16.astype(np.float32) / 16.0
dispL[dispL < 0] = 0

dispWLS = dispL.copy()
use_wls = False

if (
    hasattr(cv2, "ximgproc")
    and hasattr(cv2.ximgproc, "createRightMatcher")
    and hasattr(cv2.ximgproc, "createDisparityWLSFilter")
):
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    dispR_raw16 = right_matcher.compute(grayR_eq, grayL_eq)

    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(3000.0)
    wls.setSigmaColor(1.0)

    filtered16 = wls.filter(dispL_raw16, grayL_eq, None, dispR_raw16)
    dispWLS = filtered16.astype(np.float32) / 16.0
    dispWLS[dispWLS < 0] = 0
    use_wls = True

print("\nWLS filtering used:", use_wls)

# =========================
# ROI + TEXTURE CLEANING
# =========================

disp_roi = dispWLS.copy()
disp_roi[roi_mask == 0] = 0

sobelx = cv2.Sobel(grayL_eq, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(grayL_eq, cv2.CV_32F, 0, 1, ksize=3)
texture = cv2.magnitude(sobelx, sobely)
texture_mask = texture > 1.5

disp_clean = disp_roi.copy()
disp_clean[disp_clean < 0.5] = 0
disp_clean[disp_clean > 240] = 0
disp_clean[~texture_mask] = 0

valid_mask = (disp_clean > 0).astype(np.uint8)

kernel = np.ones((2, 2), np.uint8)
valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)

disp_clean[valid_mask == 0] = 0
disp_clean = cv2.medianBlur(disp_clean.astype(np.float32), 5)

consistent = (disp_clean > 0).astype(np.uint8) * 255
# =========================
# DENSE DISPARITY (for more 3D points)
# =========================

disp_dense = dispWLS.copy()

# keep more pixels (less strict)
disp_dense[disp_dense < 0.5] = 0
disp_dense[disp_dense > 240] = 0

# still keep ROI
disp_dense[roi_mask == 0] = 0

# save dense version
np.save(os.path.join(OUTPUT_DIR, "disparity_dense.npy"), disp_dense)

u8, color = normalize_and_colorize(disp_dense)
cv2.imwrite(os.path.join(OUTPUT_DIR, "disparity_dense.jpg"), u8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "disparity_dense_color.jpg"), color)
texture_vis = np.zeros_like(grayL_eq)
texture_vis[texture_mask] = 255
cv2.imwrite(os.path.join(OUTPUT_DIR, "texture_mask.jpg"), texture_vis)

# =========================
# SAVE DISPARITY OUTPUTS
# =========================

np.save(os.path.join(OUTPUT_DIR, "disparity_raw.npy"), dispL)
np.save(os.path.join(OUTPUT_DIR, "disparity_wls.npy"), dispWLS)
np.save(os.path.join(OUTPUT_DIR, "disparity_roi.npy"), disp_roi)
np.save(os.path.join(OUTPUT_DIR, "disparity_clean.npy"), disp_clean)

for name, disp in [
    ("disparity_raw", dispL),
    ("disparity_wls", dispWLS),
    ("disparity_roi", disp_roi),
    ("disparity_clean", disp_clean),
    ("disparity_dense", disp_dense),   # ADD THIS
]:
    u8, color = normalize_and_colorize(disp)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.jpg"), u8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_color.jpg"), color)

cv2.imwrite(os.path.join(OUTPUT_DIR, "disparity_consistency_mask.jpg"), consistent)

print_disp_info("RAW", dispL)
print_disp_info("WLS", dispWLS)
print_disp_info("ROI", disp_roi)
print_disp_info("CLEAN", disp_clean)

print("\nSaved outputs to:", OUTPUT_DIR)
print("Best files to check first:")
print(" - left_lines.jpg / right_lines.jpg")
print(" - texture_mask.jpg")
print(" - disparity_wls_color.jpg")
print(" - disparity_roi_color.jpg")
print(" - disparity_clean_color.jpg")
print(" - Q.npy")

# =========================
# DISPLAY
# =========================

cv2.imshow("ROI Preview", roi_preview)
cv2.imshow("Left Lines", draw_horizontal_lines(imgL))
cv2.imshow("Right Lines", draw_horizontal_lines(imgR))

_, wls_color = normalize_and_colorize(dispWLS)
_, roi_color = normalize_and_colorize(disp_roi)
_, clean_color = normalize_and_colorize(disp_clean)

cv2.imshow("WLS Disparity Color", wls_color)
cv2.imshow("ROI Disparity Color", roi_color)
cv2.imshow("Clean Disparity Color", clean_color)
cv2.imshow("Texture Mask", texture_vis)

print("\nPress q to quit.")
while True:
    if (cv2.waitKey(50) & 0xFF) == ord("q"):
        break

cv2.destroyAllWindows()