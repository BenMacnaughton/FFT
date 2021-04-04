import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
from ft import ft
import time
import statistics

class utils:

    '''
    Plots an image next to its ft
    '''
    @staticmethod
    def plot_FT(img, transform):
        #Set up dispay
        disp, cols = plt.subplots(1, 2)
        #Input orignal
        cols[0].imshow(img, plt.cm.gray)
        cols[0].set_title('Original image')
        #Input ft
        cols[1].imshow(np.abs(transform), norm=colors.LogNorm())
        cols[1].set_title('LogNorm of ft')
        disp.suptitle('Mode 1')
        plt.show()

    '''
    Plots an image next to its denoised version
    '''
    @staticmethod
    def plot_denoised(img, transform):
        #Set up display
        disp, cols = plt.subplots(1, 2)
        denoised = ft.ifft2(transform).real
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
    def plot_compressed(img, transform, compressions, filename):
        #Set up display
        disp, cols = plt.subplots(2, 3)
        shape = img.shape
        for i in range(2):
            for j in range(3):
                comp = i*3 + j
                compressed = utils.compress(transform, compressions[comp])
                utils.save_ft(compressed, compressions[comp], filename)
                compressed = ft.ifft2(compressed).real
                cols[i][j].imshow(compressed[:shape[0], :shape[1]], plt.cm.gray)
                cols[i][j].set_title(str(compressions[comp]) + "% compressed")
        disp.suptitle('Mode 3')
        plt.show()

    '''
    Plots
    '''
    @staticmethod
    def plot_complexity():
        x = []
        dft_y = []
        fft_y = []
        dft_stdevs = []
        fft_stdevs = []
        size = 2**5
        while size <= 2**8:
            x.append(size)
            a = np.random.rand(utils.resize(size), utils.resize(size))
            dft_times = []
            fft_times = []
            for _ in range(10):
                start = time.time()
                ft.dft2(a)
                stop = time.time()
                dft_times.append(stop - start)
                start = time.time()
                ft.fft2(a)
                stop = time.time()
                fft_times.append(stop - start)
            #Get means and stdevs for both  algos
            dft_mean = statistics.mean(dft_times)
            dft_stdev = statistics.stdev(dft_times)
            fft_mean = statistics.mean(fft_times)
            fft_stdev = statistics.stdev(fft_times)
            dft_y.append(dft_mean)
            fft_y.append(fft_mean)
            dft_stdevs.append(dft_stdev)
            fft_stdevs.append(fft_stdev)
            #Output to the user
            print("Problem size = " + str(size))
            print("dft mean = " + str(dft_mean) + " dft stdev = " + str(dft_stdev))
            print("fft mean = " + str(fft_mean) + " fft stdev = " + str(fft_stdev))
            #Proceed to next size
            size *= 2
        #Plot outputs
        plt.errorbar(x, dft_y, yerr=dft_stdevs, fmt='r--')
        plt.errorbar(x, fft_y, yerr=fft_stdevs, fmt='b--')
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
        ub = np.percentile(ft, 50 + compression//2)
        compressed = ft * np.logical_or(ft <= lb, ft >= ub)
        print("Compression level = " + str(compression) + ", number of non zero values = " + str((100 - compression) * ft.size))
        return compressed

    '''
    Returns the opened image and its ft
    '''
    @staticmethod
    def get_ft(img_name):
        #Read original image
        img = plt.imread(img_name).astype(float)
        shape = img.shape
        #reshape to be of dimensions 2^n
        reshaped = utils.resize(shape[0]), utils.resize(shape[1])
        newimg = np.zeros(reshaped)
        newimg[:shape[0], :shape[1]] = img
        return img, ft.fft2(newimg)

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
    def save_ft(ft, compression, filename):
        saved_name = filename.split(".")[0]
        saved_name += "_compressed_" + str(compression) + ".txt"
        file = open(saved_name, "w")
        for i in range(ft.shape[0]):
            file.write("[ ")
            for j in range(ft.shape[1]):
                if ft[i,j] != 0:
                    val = str(ft[i,j]).replace("(", "").replace(")", "")
                    file.write(val + " ")
                else: file.write("0 ")
            file.write("]\n")
        file.close()
