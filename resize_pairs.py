import cv2, glob, os

left_files = sorted(glob.glob("data/stereo/cam_left/images/*"))
right_files = sorted(glob.glob("data/stereo/cam_right/images/*"))

os.makedirs("data/stereo/cam_left/images_matched", exist_ok=True)
os.makedirs("data/stereo/cam_right/images_matched", exist_ok=True)

for i, (lp, rp) in enumerate(zip(left_files, right_files)):
    left = cv2.imread(lp)
    right = cv2.imread(rp)

    if left is None or right is None:
        print("Could not read:", lp, rp)
        continue

    h, w = left.shape[:2]
    right_resized = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(f"data/stereo/cam_left/images_matched/img_{i}.jpg", left)
    cv2.imwrite(f"data/stereo/cam_right/images_matched/img_{i}.jpg", right_resized)

print("done", len(left_files))
