import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
from ft import FT

class UTILS:

    '''
    Plots an image next to its FT
    '''
    @staticmethod
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
    Plots an image next to its denoised version
    '''
    @staticmethod
    def plot_denoised(img, ft):
        #Set up dispay
        disp, cols = plt.subplots(1, 2)
        denoised = FT.TDIDFT(ft).real
        shape = img.shape
        #Input orignal
        cols[0].imshow(img[:shape[0], :shape[1]], plt.cm.gray)
        cols[0].set_title('Original image')
        #Input denoised
        cols[1].imshow(denoised[:shape[0], :shape[1]], plt.cm.gray)
        cols[1].set_title('Denoised copy')
        disp.suptitle('Mode 2')
        plt.show()

    '''
    Denoises an image by setting high frequencies to 0
    Returns the number of non zero values and the fraction of non zero values
    '''
    @staticmethod
    def denoise(ft):
        mean, stddev = UTILS.get_stddev(ft)
        scale = 15
        zeros = 0
        N = ft.shape[0]
        M = ft.shape[1]
        for i in range(N):
            for j in range(M):
                if ft[i][j] > mean + scale * stddev:
                    ft[i][j] = 0
                    zeros += 1
        return N*M - zeros, (N*M - zeros)/(N*M)

    '''
    Returns the opened image and its FT
    '''
    @staticmethod
    def get_FT(img_name):
        #Read original image
        img = plt.imread(img_name).astype(float)
        shape = img.shape
        #reshape to be of dimensions 2^n
        reshaped = UTILS.resize(shape[0]), UTILS.resize(shape[1])
        newimg = np.zeros(reshaped)
        newimg[:shape[0], :shape[1]] = img
        return img, FT.TDDFT(newimg)

    '''
    Denoises an image by setting high frequencies to 0
    Returns the number of non zero values and the fraction of non zero values
    '''
    @staticmethod
    def get_local_mean(x, y, data):
        max_x = data.shape[0] - 1
        max_y = data.shape[1] - 1
        i = 0
        mean = 0
        if x - 1 >= 0:
            mean += data[x-1, y]
            i += 1
        if x - 1 >= 0 and y + 1 <= max_y:
            mean += data[x - 1, y + 1]
            i += 1
        if x - 1 >= 0 and y - 1 >= 0:
            mean += data[x - 1, y - 1]
            i += 1
        if  y - 1 >= 0:
            mean += data[x, y - 1]
            i += 1
        if y - 1 >= 0 and x + 1 <= max_x:
            mean += data[x + 1, y - 1]
            i += 1
        if x + 1 <= max_x:
            mean += data[x + 1, y]
            i += 1
        if y + 1 <= max_y:
            mean += data[x, y + 1]
            i += 1
        if x + 1 <= max_x and y + 1 <= max_y:
            mean += data[x + 1, y + 1]
            i += 1
        return mean/i

    '''
    Returns the next highest integer that is a power of 2
    '''
    @staticmethod
    def resize(val):
        i = int(math.log(val, 2))
        return int(pow(2, i+1))

    '''
    Returns the mean and standard deviation of a set of values
    '''
    @staticmethod
    def get_stddev(data):
        N = data.shape[0]
        M = data.shape[1]
        mean = 0
        for i in range(N):
            for j in range(M):
                mean += data[i][j]
        mean /= (N*M)
        stddev = 0
        for i in range(N):
            for j in range(M):
                stddev += (data[i][j] - mean) ** 2
        return mean, (stddev / (N*M)) ** (1/2)