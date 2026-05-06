import cv2
import os

LEFT_CAM = 1
RIGHT_CAM = 2

OUT_LEFT = "data/stereo/cam_left/images_alina"
OUT_RIGHT = "data/stereo/cam_right/images_alina"

TARGET_COUNT = 60
WIDTH = 1280
HEIGHT = 960
FPS = 30

PREVIEW_WIDTH = 1280  # display width only

os.makedirs(OUT_LEFT, exist_ok=True)
os.makedirs(OUT_RIGHT, exist_ok=True)

capL = cv2.VideoCapture(LEFT_CAM, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(RIGHT_CAM, cv2.CAP_DSHOW)

if not capL.isOpened():
    raise RuntimeError("Left camera did not open")

if not capR.isOpened():
    raise RuntimeError("Right camera did not open")

for cap in [capL, capR]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

count = 0

cv2.namedWindow("Stereo Capture", cv2.WINDOW_NORMAL)

print("Stereo capture started.")
print("Press SPACE to save pair.")
print("Press q to quit.")
print("IMPORTANT: keep checkerboard still when pressing SPACE.")

while True:
    okL_grab = capL.grab()
    okR_grab = capR.grab()

    if not okL_grab or not okR_grab:
        print("Grab failed.")
        break

    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()

    if not retL or not retR:
        print("Retrieve failed.")
        break

    if frameL.shape[:2] != frameR.shape[:2]:
        frameR = cv2.resize(frameR, (frameL.shape[1], frameL.shape[0]))

    preview = cv2.hconcat([frameL, frameR])

    cv2.putText(
        preview,
        f"Captured {count}/{TARGET_COUNT} | SPACE save | q quit",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Resize preview for display only
    scale = PREVIEW_WIDTH / preview.shape[1]
    preview_show = cv2.resize(
        preview,
        (PREVIEW_WIDTH, int(preview.shape[0] * scale))
    )

    cv2.imshow("Stereo Capture", preview_show)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord(" "):
        capL.grab()
        capR.grab()

        retL, frameL = capL.retrieve()
        retR, frameR = capR.retrieve()

        if not retL or not retR:
            print("Save failed: could not retrieve fresh pair.")
            continue

        if frameL.shape[:2] != frameR.shape[:2]:
            frameR = cv2.resize(frameR, (frameL.shape[1], frameL.shape[0]))

        filename = f"img_{count:03d}.jpg"

        left_path = os.path.join(OUT_LEFT, filename)
        right_path = os.path.join(OUT_RIGHT, filename)

        cv2.imwrite(left_path, frameL)
        cv2.imwrite(right_path, frameR)

        print(f"Saved pair {count}: {filename}")

        count += 1

        if count >= TARGET_COUNT:
            print("Done capturing.")
            break

capL.release()
capR.release()
cv2.destroyAllWindows()
