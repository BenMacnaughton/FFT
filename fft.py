import matplotlib.colors as mpl
import numpy as np
from math import exp, pi
import sys
import os

def FFT(signals):
    pass

def DFT(signals):
    N, M = signals.shape
    transform = np.zeros((N, M))
    exponent = exp(-2J * pi)

    for n in range(N):
        for m in range(M):
            for l in range(N):
                for k in range(M):  
                    transform[n, m] += signals[n, m]* exponent**(k*m/M + l*n/N)
    
    return transform
    
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