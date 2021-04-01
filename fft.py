import sys
import os
import cv2 as cv
from utils import UTILS


if __name__ == "__main__":
    filename = "moonlanding.png"
    mode = 1
    if "-m" in sys.argv:
        i = sys.argv.index("-m")
        modes = ["1", "2", "3", "4"]
        if i + 1 < len(sys.argv) and sys.argv[i+1] in modes:
            mode = int(sys.argv[i+1])
        else:
            print("Please enter a valid mode [1, 2, 3, 4]")
            exit(1)
    if "-i" in sys.argv:
        i = sys.argv.index("-i")
        if i + 1 < len(sys.argv) and os.path.isfile(sys.argv[i+1]):
            filename = sys.argv[i+1]
        else:
            print("Please enter a valid file")
            exit(1)
    canopen = cv.haveImageReader(filename)
    if canopen:
        image_array = cv.imread(filename)
    else:
        print("The specified file cannot be read")
        exit(1)
    if mode == 1:
        img, ft = UTILS.get_FT(filename)
        UTILS.plot_FT(img, ft)
    elif mode == 2:
        img, ft = UTILS.get_FT(filename)
        UTILS.denoise(ft)