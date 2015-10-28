from PIL import Image
import numpy as np
import scipy.fftpack as fp
import functools
import matplotlib.pyplot as plt
# import pdb

blocksize = (8, 8)

def blockproc(im, fun):

    xblocks = np.split(im, range(blocksize[0], im.shape[0], blocksize[0]), axis=0)
    xblocks_proc = []
    for xb in xblocks:
        yblocks = np.split(xb, range(blocksize[1], im.shape[1], blocksize[1]), axis=1)
        yblocks_proc = []
        for yb in yblocks:
            yb_proc = fun(yb)
            yblocks_proc.append(yb_proc)
        xblocks_proc.append(np.concatenate(yblocks_proc, axis=1))

    proc = np.concatenate(xblocks_proc, axis=0)

    return proc

def dct2(im):
     return fp.dct(fp.dct(im, norm='ortho', axis=0), norm='ortho', axis=1)

def idct2(im):
    return fp.idct(fp.idct(im, norm='ortho', axis=0), norm='ortho', axis=1)

def dft2(im):
    return fp.fft2(im)

def idft2(im):
    return fp.ifft2(im)

def dct_compress(im, limit):
    return compress(dct2(im), limit)

def dft_compress(im, limit):
    return compress(dft2(im), limit)

def compress(im, limit):
    M = im.shape[0]
    N = im.shape[1]
    MN = M * N
    selected = []
    cleared = 0

    for x in range(MN):
        m = x % M
        n = int(x / M)
        amp = abs(im[m][n])

        i = 0
        finished = False
        while i < limit and finished == False:
            if i >= len(selected):
                selected.append(x);
                finished = True
            else:
                sel_m = selected[i] % M
                sel_n = int(selected[i] / M)
                sel_amp = abs(im[sel_m][sel_n])

                if amp > sel_amp:
                    if (len(selected) >= limit):
                        prev = selected.pop()
                        prev_m = prev % M
                        prev_n = int(prev / M)
                        im[prev_m][prev_n] = 0
                        cleared = cleared + 1
                    selected.insert(i, x)
                    finished = True
            i = i + 1

        if i == limit and finished == False:
            im[m][n] = 0
            cleared = cleared + 1

    return im

def calc_distortion(im_proc, im):
    difference = 0.0
    original = 0.0
    for m in range(0, im.shape[0]):
        for n in range(0, im.shape[1]):
            original = original + abs(im[m][n])**2.0
            difference = difference + abs(im[m][n] - im_proc[m][n])**2.0

    return difference / original

def plot_distortion(im, sample_times, fun, ifun):
    samples = blocksize[0] * blocksize[1]
    distortion = []

    for n in sample_times:
        print('Finding distortion for N={}'.format(n))
        im_cmpr = blockproc(blockproc(im, lambda im: fun(im, n)), ifun)
        dist = calc_distortion(im_cmpr, im)
        distortion.append(dist)

    return distortion

image = Image.open('lena.png').convert('L')
im = np.array(image, dtype=np.float)

print('Creating dct inversed image')
im_dct = blockproc(im, dct2)
im_idct = blockproc(im_dct, idct2)
image_idct = Image.fromarray(im_idct)
image_idct.show(title='DCT');

print('Creating dft inversed image')
im_dft = blockproc(im, dft2)
im_idft = blockproc(im_dft, idft2)
image_idft = Image.fromarray(im_idft.real)
image_idft.show('DFT')

print('Creating dct compressed image')
im_dct_cmpr = blockproc(im, lambda im: dct_compress(im, 8))
im_dct_icmpr = blockproc(im_dct_cmpr, idct2)
image_dct_icmpr = Image.fromarray(im_dct_icmpr)
image_dct_icmpr.show(title='dct compressed')

print('Creating dft compressed image')
im_dft_cmpr = blockproc(im, lambda im: dft_compress(im, 8))
im_dft_icmpr = blockproc(im_dft_cmpr, fp.ifft2)
image_dft_icmpr = Image.fromarray(im_dft_icmpr.real)
image_dft_icmpr.show(title='dft compressed')

sample_times = [1, 2, 3, 4, 6, 8, 16, 24, 32, 40, 48, 56, 63]
# sample_times = [1, 2]

print('Plotting distortion for dct compression')
dct_cmpr_dist = plot_distortion(im, sample_times, dct_compress, idct2)
dct_plot, = plt.plot(sample_times, dct_cmpr_dist, color='r', label='dct')

print('Plotting distortion for dft compression')
dft_cmpr_dist = plot_distortion(im, sample_times, dft_compress, fp.ifft2)
dft_plot, = plt.plot(sample_times, dft_cmpr_dist, color='b', label='dft')

plt.xlabel('N')
plt.ylabel('D')
plt.legend([dct_plot, dft_plot], ['dct', 'dft'])
plt.show()
