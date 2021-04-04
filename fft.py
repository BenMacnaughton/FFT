import sys
import os
import cv2 as cv
from utils import utils


if __name__ == "__main__":
    #Default arguments
    filename = "moonlanding.png"
    mode = 1
    #Process user input
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
    #check that the file can be processed
    canopen = cv.haveImageReader(filename)
    if canopen:
        image_array = cv.imread(filename)
    else:
        print("The specified file cannot be read")
        exit(1)
    #fft mode
    if mode == 1:
        img, ft = utils.get_ft(filename)
        utils.plot_ft(img, ft)
    #Denoising mode
    elif mode == 2:
        img, ft = utils.get_ft(filename)
        zeros, fraction = utils.denoise(ft)
        print("Number of non-zero coefficients: " + str(zeros) + ", % of original: " + str(fraction*100))
        utils.plot_denoised(img, ft)
    #Compression mode
    elif mode == 3:
        img, ft = utils.get_ft(filename)
        compressions = [0, 25, 50, 75, 85, 95]
        utils.plot_compressed(img, ft, compressions, filename)
    #Time complexity mode
    elif mode == 4:
        utils.plot_complexity()