import os
import cv2
import numpy as np

PAIR_ID = 0

LEFT_DIR = r"C:\3dproj\data\stereo\cam_left\images_full"
RIGHT_DIR = r"C:\3dproj\data\stereo\cam_right\images_full"
STEREO_DIR = r"C:\3dproj\data\stereo\results"
OUT_DIR = r"C:\3dproj\outputs\epipolar_check"

os.makedirs(OUT_DIR, exist_ok=True)

checkerboard = (5, 4)

left_images = sorted([
    os.path.join(LEFT_DIR, f)
    for f in os.listdir(LEFT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

right_images = sorted([
    os.path.join(RIGHT_DIR, f)
    for f in os.listdir(RIGHT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

imgL = cv2.imread(left_images[PAIR_ID])
imgR = cv2.imread(right_images[PAIR_ID])

if imgL is None or imgR is None:
    raise RuntimeError("Could not load images.")

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

F = np.load(os.path.join(STEREO_DIR, "F.npy"))

retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard)
retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard)

if not (retL and retR):
    raise RuntimeError("Checkerboard not detected.")

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    1e-4,
)

cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

ptsL = cornersL.reshape(-1, 2)
ptsR = cornersR.reshape(-1, 2)

# =========================
# NUMERICAL EPIPOLAR ERROR
# =========================

def point_line_distance(lines, points):
    """
    line: ax + by + c = 0
    distance = |ax + by + c| / sqrt(a^2 + b^2)
    """
    a = lines[:, 0]
    b = lines[:, 1]
    c = lines[:, 2]

    x = points[:, 0]
    y = points[:, 1]

    return np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)


linesR = cv2.computeCorrespondEpilines(
    ptsL.reshape(-1, 1, 2),
    1,
    F
).reshape(-1, 3)

linesL = cv2.computeCorrespondEpilines(
    ptsR.reshape(-1, 1, 2),
    2,
    F
).reshape(-1, 3)

err_right = point_line_distance(linesR, ptsR)
err_left = point_line_distance(linesL, ptsL)

all_errors = np.hstack([err_left, err_right])

print("\n=== EPIPOLAR GEOMETRY NUMERICAL CHECK ===")
print("Mean error px:", np.mean(all_errors))
print("Median error px:", np.median(all_errors))
print("Max error px:", np.max(all_errors))
print("Std error px:", np.std(all_errors))

if np.mean(all_errors) < 0.5:
    print("✅ Epipolar geometry is VERY GOOD")
elif np.mean(all_errors) < 1.0:
    print("✅ Epipolar geometry is GOOD")
elif np.mean(all_errors) < 2.0:
    print("⚠️ Epipolar geometry is usable but not perfect")
else:
    print("❌ Epipolar geometry is poor")

# =========================
# DRAW EPIPOLAR LINES
# =========================

def draw_lines(img_line, img_points, lines, line_points, other_points):
    h, w = img_line.shape[:2]

    img_line_color = img_line.copy()
    img_points_color = img_points.copy()

    for line, pt_line, pt_other in zip(lines, line_points, other_points):
        a, b, c = line

        if abs(b) < 1e-6:
            continue

        x0 = 0
        y0 = int(-c / b)

        x1 = w
        y1 = int(-(a * x1 + c) / b)

        color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.line(img_line_color, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img_line_color, tuple(np.int32(pt_line)), 5, color, -1)
        cv2.circle(img_points_color, tuple(np.int32(pt_other)), 5, color, -1)

    return img_line_color, img_points_color


epiR, _ = draw_lines(imgR, imgL, linesR, ptsR, ptsL)
epiL, _ = draw_lines(imgL, imgR, linesL, ptsL, ptsR)

cv2.imwrite(os.path.join(OUT_DIR, "epipolar_lines_left.jpg"), epiL)
cv2.imwrite(os.path.join(OUT_DIR, "epipolar_lines_right.jpg"), epiR)

combined = np.hstack((epiL, epiR))
cv2.imwrite(os.path.join(OUT_DIR, "epipolar_combined.jpg"), combined)

print("\nSaved:")
print(os.path.join(OUT_DIR, "epipolar_combined.jpg"))

cv2.imshow("Epipolar Geometry", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()