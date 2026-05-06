import cv2
import numpy as np

IMG_PATH = "outputs/stereo_depth_rectified/rectified_left.jpg"

# ROI (same as your object area)
ROI = (60, 650, 500, 980)  # y1, y2, x1, x2


def print_stats(name, data):
    data = data[data > 0]
    if len(data) == 0:
        return

    print(f"\n=== {name} ===")
    print("min :", np.percentile(data, 1))
    print("5%  :", np.percentile(data, 5))
    print("25% :", np.percentile(data, 25))
    print("50% :", np.percentile(data, 50))
    print("75% :", np.percentile(data, 75))
    print("95% :", np.percentile(data, 95))
    print("99% :", np.percentile(data, 99))


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise RuntimeError("Image not found")

    y1, y2, x1, x2 = ROI
    roi = img[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # =========================
    # PRINT DISTRIBUTION
    # =========================
    print_stats("HUE", h)
    print_stats("SATURATION", s)
    print_stats("VALUE", v)

    # =========================
    # INTERACTIVE TRACKBARS
    # =========================
    cv2.namedWindow("mask")

    def nothing(x):
        pass

    # Hue
    cv2.createTrackbar("H_min", "mask", 0, 179, nothing)
    cv2.createTrackbar("H_max", "mask", 179, 179, nothing)

    # Saturation
    cv2.createTrackbar("S_min", "mask", 0, 255, nothing)
    cv2.createTrackbar("S_max", "mask", 255, 255, nothing)

    # Value
    cv2.createTrackbar("V_min", "mask", 0, 255, nothing)
    cv2.createTrackbar("V_max", "mask", 255, 255, nothing)

    while True:
        hmin = cv2.getTrackbarPos("H_min", "mask")
        hmax = cv2.getTrackbarPos("H_max", "mask")
        smin = cv2.getTrackbarPos("S_min", "mask")
        smax = cv2.getTrackbarPos("S_max", "mask")
        vmin = cv2.getTrackbarPos("V_min", "mask")
        vmax = cv2.getTrackbarPos("V_max", "mask")

        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, smax, vmax])

        mask = cv2.inRange(hsv, lower, upper)

        preview = roi.copy()
        preview[mask == 0] = 0

        cv2.imshow("mask", mask)
        cv2.imshow("preview", preview)

        key = cv2.waitKey(30)
        if key == 27:  # ESC
            print("\nFINAL VALUES:")
            print(f"H: {hmin} - {hmax}")
            print(f"S: {smin} - {smax}")
            print(f"V: {vmin} - {vmax}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()