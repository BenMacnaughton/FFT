import matplotlib.colors as mpl
import numpy as np
from math import exp, pi
import sys
import os



def DFT(signals):
    # get the length of the array
    N = signals.shape[0]
    # create array from 1 to N-1
    n = np.arange(N)
    # create vertical array from 1 to N-1
    k = np.reshape((N, 1))
    # 2-D array of all the exponents
    exponent = np.exp(-2J * np.pi * k * n / N)
    # Adds from 0 to N-1 each each signal by the resultant exponents
    return np.dot(exponent, signals)

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
        return np.concatenate(evens + extra_e[:int(N / 2)] * odds, evens + extra_e[int(N / 2):] * odds)

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
    print(filename)
    print(mode)