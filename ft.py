import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import os
import cv2 as cv
import math

class FT:

    '''
    Inverse Fourier Transform
    '''
    @staticmethod
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
    @staticmethod
    def FIFT(frequencies):
        N = frequencies.shape[0]

        if N % 2 != 0:
            raise ValueError("must be a power of two")

        elif N <= 2:
            return FT.DFT(frequencies)

        else:
            evens = FT.FFT(frequencies[::2])
            odds = FT.FFT(frequencies[1::2])
            extra_e = np.exp(2J * np.pi * np.arange(N) / N)

            return np.concatenate([evens + extra_e[:int(N / 2)] * odds, evens + extra_e[int(N / 2):] * odds])

    '''
    Two-Dimensional Inverse Discrete Fourier Transform
    '''
    @staticmethod
    def TDIDFT(frequencies):
        N, M = frequencies.shape
        n = np.arange(N)
        l = n.reshape((N, 1))
        exponent = np.exp(2J * np.pi * l * n / N)

        transform = np.zeros((N, M), dtype=complex)
        for n in range(N):
            transform[n] = FT.FFT(frequencies[n])

        return np.dot(exponent, transform) / N

    '''
    Discrete Fourier Transform
    Takes in 1D array of signals and performs brute force fourier transform
    Returns the fourier transform
    '''
    @staticmethod
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
    @staticmethod
    def FFT(signals):
        N = signals.shape[0]

        if N % 2 != 0:
            raise ValueError("must be a power of two")

        # set limit of 2 before performiing DFT on a signal
        elif N <= 2:
            return FT.DFT(signals)

        else:
            # even and odd arrays
            evens = FT.FFT(signals[::2])
            odds = FT.FFT(signals[1::2])
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
    @staticmethod
    def TDDFT(signals):
        N, M = signals.shape
        n = np.arange(N)
        l = n.reshape((N, 1))
        exponent = np.exp(-2J * np.pi * l * n / N)

        transform = np.zeros((N, M), dtype=complex)
        for n in range(N):
            transform[n] = FT.FFT(signals[n])

        return np.dot(exponent, transform)