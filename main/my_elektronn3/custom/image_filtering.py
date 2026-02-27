''' Different filtering applied to simple generated or real data to improve segmentation '''

import numpy as np
from scipy.ndimage import median_filter, convolve, gaussian_filter, minimum_filter, maximum_filter

''' Note all input arrays are (B, D, H, W) '''

class MedianFiltering:
    '''
    Apply median filter to image/patch. Targets untouched

    inp: (B,D,H,W)
    size: eg. (1,3,3) or (3,1,1)
    '''
    def __init__(self, size):
        self.size = size
        
    def __call__(self, inp, targs):
        assert inp.ndim == 4

        inp_f = np.zeros_like(inp)

        for i in range(inp.shape[0]):
            inp_f[i] = median_filter(inp[i], self.size)

        return inp_f, targs

class GaussianFiltering:
    '''
    Apply Gaussian filter to image/patch. Targets untouched

    inp: (B,D,H,W)
    sigma: eg. (0,1,1) or (2,0,0)
    '''
    def __init__(self, sigma):
        self.sigma = sigma
        
    def __call__(self, inp, targs):
        assert inp.ndim == 4

        inp_f = np.zeros_like(inp)

        for i in range(inp.shape[0]):
            inp_f[i] = gaussian_filter(inp[i], self.sigma)

        return inp_f, targs

class ConvZFiltering:
    '''
    Apply convolution in z-direction to image/patch. Targets untouched

    inp: (B,D,H,W)
    kern: eg. [.3, .7]
    '''
    def __init__(self, kern):
        self.kern = kern
        
    def __call__(self, inp, targs):
        assert inp.ndim == 4

        inp_f = np.zeros_like(inp)

        for i in range(inp.shape[0]):
            inp_f[i] = convolve(inp[i], np.array(self.kern)[:,np.newaxis,np.newaxis])

        return inp_f, targs