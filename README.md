# fft
## Instructions for running the program
Using a Python version of 3.5+ run the command python3 fft.py [-m mode] [-i image]
### Syntax
- mode (optional):

  - [1] (Default) for fast mode where the image is converted into its FFT form and displayed

  - [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed

  - [3] for compressing and saving the image

  - [4] for plotting the runtime graphs for the report

- image (optional): filename of the image we wish to take the DFT of. (Default: moonlaning.png)
### External libraries used
- Numpy
- Open_cv
- MatlPlotLib (colors and pyplot)
