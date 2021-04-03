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
        #Set up display
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
    Plots a series of 6 images with varying levels of compression
    '''
    @staticmethod
    def plot_compressed(img, ft, compressions, filename):
        #Set up display
        disp, cols = plt.subplots(2, 3)
        shape = img.shape
        for i in range(2):
            for j in range(3):
                comp = i*3 + j
                compressed = UTILS.compress(ft, compressions[comp])
                UTILS.save_FT(compressed, compressions[comp], filename)
                compressed = FT.TDIDFT(compressed).real
                cols[i][j].imshow(compressed[:shape[0], :shape[1]], plt.cm.gray)
                cols[i][j].set_title(str(compressions[comp]) + "% compressed")
        disp.suptitle('Mode 3')
        plt.show()

    '''
    Denoises an image by setting high frequencies to 0
    Returns the number of non zero values and the fraction of non zero values
    '''
    @staticmethod
    def denoise(ft):
        ratio = 0.0905
        zeros = 0
        N = ft.shape[0]
        M = ft.shape[1]
        ft[int(N*ratio):-int(N*ratio), :] = 0
        ft[:, int(M*ratio):-int(M*ratio)] = 0
        zeros = ratio*N*M
        return int(N*M - zeros), 1 - ratio

    '''
    Compresses an image by a factor of the compression input
    and saves the compressed image
    '''
    @staticmethod
    def compress(ft, compression):
        lb = np.percentile(ft, (100 - compression)//2)
        ub = np.percentile(ft, compression//2)
        compressed = ft * np.logical_or(ft <= lb, ft >= ub)
        return compressed

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
    Returns the next highest integer that is a power of 2
    '''
    @staticmethod
    def resize(val):
        i = int(math.log(val, 2))
        return int(pow(2, i+1))

    '''
    Compresses an image by a factor of the compression input
    and saves the compressed image
    '''
    @staticmethod
    def save_FT(ft, compression, filename):
        saved_name = filename.split(".")[0]
        saved_name += "_compressed_" + str(compression) + ".csv"
        np.savetxt(saved_name, ft, delimiter=",")
