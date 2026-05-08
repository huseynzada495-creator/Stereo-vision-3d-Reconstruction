"""
Entry point for the stereo vision 3D reconstruction project.

The main pipeline scripts are stored in the scripts/ directory:
1. scripts/calibrate_stereo_metric.py
2. scripts/stereo_depth_rectified.py
3. scripts/reconstruct_3d.py
"""

import subprocess
import sys


def run_step(command: list[str]) -> None:
    print(f"\nRunning: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    print("Stereo Vision 3D Reconstruction Pipeline")

    print("\nRun individual stages manually:")
    print("  python scripts/calibrate_stereo_metric.py")
    print("  python scripts/stereo_depth_rectified.py")
    print("  python scripts/reconstruct_3d.py --depth filled --min-disp 5 --z-min 0 --z-max 10000 --no-z-trim --no-filter --point-size 5")


if __name__ == "__main__":
    main()
