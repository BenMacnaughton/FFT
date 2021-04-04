import numpy as np

class ft:

    '''
    Inverse Fourier Transform
    '''
    @staticmethod
    def idft(frequencies):
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
    def ifft(frequencies):
        N = frequencies.shape[0]

        if N % 2 != 0:
            raise ValueError("must be a power of two")

        elif N <= 2:
            return ft.idft(frequencies)

        else:
            evens = ft.ifft(frequencies[::2])
            odds = ft.ifft(frequencies[1::2])
            extra_e = np.exp(2J * np.pi * np.arange(N) / N)

            return np.concatenate([evens + extra_e[:int(N / 2)] * odds, evens + extra_e[int(N / 2):] * odds])

    '''
    Two-Dimensional Inverse Discrete Fourier Transform
    '''
    @staticmethod
    def ifft2(frequencies):
        N, M = frequencies.shape
        n = np.arange(N)
        l = n.reshape((N, 1))
        exponent = np.exp(2J * np.pi * l * n / N)

        transform = np.zeros((N, M), dtype=complex)
        for n in range(N):
            transform[n] = ft.ifft(frequencies[n])

        return np.dot(exponent, transform) / N

    '''
    Discrete Fourier Transform
    Takes in 1D array of signals and performs brute force fourier transform
    Returns the fourier transform
    '''
    @staticmethod
    def dft(signals):
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
    Returns ft
    '''
    @staticmethod
    def fft(signals):
        N = signals.shape[0]

        if N % 2 != 0:
            raise ValueError("must be a power of two")

        # set limit of 2 before performiing dft on a signal
        elif N <= 16:
            return ft.dft(signals)

        else:
            # even and odd arrays
            evens = ft.fft(signals[::2])
            odds = ft.fft(signals[1::2])
            # array of extra exponents for the odd part
            extra_e = np.exp(-2J * np.pi * np.arange(N) / N)
            # need arrays to be same size
            # X_k where 0 <= k <= (N-1) / 2 done first
            # second half done after
            return np.concatenate([evens + extra_e[:int(N / 2)] * odds, evens + extra_e[int(N / 2):] * odds])

    '''
    Fast Two-Dimensional Fourier Transform
    Takes in a 2D numpy array of size N x M
    Performs fft on each row and assembles a 2D transform
    Returns 2D Transform
    '''
    @staticmethod
    def fft2(signals):
        N, M = signals.shape
        n = np.arange(N)
        l = n.reshape((N, 1))
        exponent = np.exp(-2J * np.pi * l * n / N)

        transform = np.zeros((N, M), dtype=complex)
        for n in range(N):
            transform[n] = ft.fft(signals[n])

        return np.dot(exponent, transform)

    '''
    Slow Two-Dimensional Fourier Transform
    Takes in a 2D numpy array of size N x M
    Performs fft on each row and assembles a 2D transform
    Returns 2D Transform
    '''
    @staticmethod
    def dft2(signals):
        N, M = signals.shape
        n = np.arange(N)
        l = n.reshape((N, 1))
        exponent = np.exp(-2J * np.pi * l * n / N)

        transform = np.zeros((N, M), dtype=complex)
        for n in range(N):
            transform[n] = ft.dft(signals[n])

        return np.dot(exponent, transform)