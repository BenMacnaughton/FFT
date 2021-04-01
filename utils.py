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
    Denoises an image by setting high frequencies to 0
    '''
    @staticmethod
    def denoise(ft):
        print(ft[0][0])

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