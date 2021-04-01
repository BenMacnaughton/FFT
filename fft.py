import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import os
import cv2 as cv
import math

'''
Inverse Fourier Transform
'''
def IDFT(frequencies):
    N = frequencies.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    exponent = np.exp(2J * np.pi * k * n / N)

    inverse = np.dot(exponent, frequencies)
    return inverse / N

'''
Fast Inverse Fourier Transform
'''
def FIFT(frequencies):
    N = frequencies.shape[0]

    if N % 2 != 0:
        raise ValueError("must be a power of two")

    elif N <= 2:
        return DFT(frequencies)

    else:
        evens = FFT(frequencies[::2])
        odds = FFT(frequencies[1::2])
        extra_e = np.exp(2J * np.pi * np.arange(N) / N)

        return np.concatenate([evens + extra_e[:int(N / 2)] * odds, evens + extra_e[int(N / 2):] * odds])

'''
Two-Dimensional Inverse Discrete Fourier Transform
'''
def TDIDFT(frequencies):
    N, M = frequencies.shape
    n = np.arange(N)
    l = n.reshape((N, 1))
    exponent = np.exp(2J * np.pi * l * n / N)

    transform = np.zeros((N, M), dtype=complex)
    for n in range(N):
        transform[n] = FFT(frequencies[n])

    return np.dot(exponent, transform) / N

'''
Discrete Fourier Transform
Takes in 1D array of signals and performs brute force fourier transform
Returns the fourier transform
'''
def DFT(signals):
    # get the length of the array
    N = signals.shape[0]
    # create array from 1 to N-1
    n = np.arange(N)
    # create vertical array from 1 to N-1
    k = n.reshape((N, 1))
    # 2-D array of all the exponents
    exponent = np.exp(-2J * np.pi * k * n / N)
    # Adds from 0 to N-1 each each signal by the resultant exponents
    return np.dot(exponent, signals)

'''
Fast Fourier Transform using Cooley-Tukey method
Takes in 1D array of signals to be transformed
Returns FT
'''
def FFT(signals):
    N = signals.shape[0]

    if N % 2 != 0:
        raise ValueError("must be a power of two")

    # set limit of 2 before performiing DFT on a signal
    elif N <= 2:
        return DFT(signals)

    else:
        # even and odd arrays
        evens = FFT(signals[::2])
        odds = FFT(signals[1::2])
        # array of extra exponents for the odd part
        extra_e = np.exp(-2J * np.pi * np.arange(N) / N)
        # need arrays to be same size
        # X_k where 0 <= k <= (N-1) / 2 done first
        # second half done after
        return np.concatenate([evens + extra_e[:int(N / 2)] * odds, evens + extra_e[int(N / 2):] * odds])

'''
Two-Dimensional Fourier Transform
Takes in a 2D numpy array of size N x M
Performs FFT on each row and assembles a 2D transform
Returns 2D Transform
'''
def TDDFT(signals):
    N, M = signals.shape
    n = np.arange(N)
    l = n.reshape((N, 1))
    exponent = np.exp(-2J * np.pi * l * n / N)

    transform = np.zeros((N, M), dtype=complex)
    for n in range(N):
        transform[n] = FFT(signals[n])

    return np.dot(exponent, transform)

'''
Plots an image next to its FT
'''
def plot_FT(img, ft):
    #Set up dispay
    disp, cols = plt.subplots(1, 2)
    #Input orignal
    cols[0].imshow(img, plt.cm.gray)
    cols[0].set_title('Original image')
    #Input FT
    cols[1].imshow(np.abs(ft), norm=colors.LogNorm())
    cols[1].set_title('LogNorm of FT')
    disp.suptitle('Mode 1')
    plt.show()


'''
Returns the opened image and its FT
'''
def getFT(img_name):
    #Read original image
    img = plt.imread(img_name).astype(float)
    shape = img.shape
    #reshape to be of dimensions 2^n
    reshaped = resize(shape[0]), resize(shape[1])
    newimg = np.zeros(reshaped)
    newimg[:shape[0], :shape[1]] = img
    return img, TDDFT(newimg)

'''
Returns the next highest integer that is a power of 2
'''
def resize(val):
    i = int(math.log(val, 2))
    return int(pow(2, i+1))

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
        print("The specifed file cannot be read")
        exit(1)
    if mode == 1:
        img, ft = getFT(filename)
        plot_FT(img, ft)