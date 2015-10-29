from PIL import Image
import numpy as np
import scipy.fftpack as fp
import matplotlib.pyplot as plt
import argparse
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
    selected = []

    for n in range(N):
        for m in range(M):
            amp = abs(im[m][n])

            i = 0
            finished = False
            while i < limit and not finished:
                if i >= len(selected):
                    selected.append((m, n))
                    finished = True
                else:
                    sel_m = selected[i][0]
                    sel_n = selected[i][1]
                    sel_amp = abs(im[sel_m][sel_n])

                    if amp > sel_amp:
                        if (len(selected) >= limit):
                            prev = selected.pop()
                            prev_m = prev[0]
                            prev_n = prev[1]
                            im[prev_m][prev_n] = 0
                        selected.insert(i, (m, n))
                        finished = True
                i = i + 1

            if i == limit and not finished:
                im[m][n] = 0

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
    distortion = []

    for n in sample_times:
        print('Finding distortion for N={}'.format(n))
        im_cmpr = blockproc(blockproc(im, lambda im: fun(im, n)), ifun)
        dist = calc_distortion(im_cmpr, im)
        distortion.append(dist)

    return distortion

argparser = argparse.ArgumentParser(description='Do assignment 2')
available_tasks = ['all', 'transform-dct', 'transform-dft', 'transform-all',
                   'compress-dct', 'comptress-dft', 'compress-all',
                   'plot-distortion']
argparser.add_argument('tasks', default='all', nargs='*', choices=available_tasks,  help='Wich tasks to execute')
argparser.add_argument('--imgcmd', default=None, help='Command to use for showing images')
argparser.add_argument('--img', default='lena.png', help='Image to transform')
argparser.add_argument('--blocksize', default=8, help='Size of processing blocks')
args = argparser.parse_args()

blocksize = (args.blocksize, args.blocksize)
tasks = []
if 'all' in args.tasks:
    tasks = available_tasks
else:
    tasks = args.tasks

if 'transform-all' in tasks:
    tasks.extend(['transform-dct', 'transform-dft'])

if 'compress-all' in tasks:
    tasks.extend(['compress-dct', 'compress-dft'])

image = Image.open(args.img).convert('L')
im = np.array(image, dtype=np.float)

if 'transform-dct' in tasks:
    print('Creating dct inversed image')
    im_dct = blockproc(im, dct2)
    im_idct = blockproc(im_dct, idct2)
    image_idct = Image.fromarray(im_idct)
    image_idct.show(title='DCT', command=args.imgcmd)
    image_idct.convert('RGB').save('transform-dct.png')

if 'transform-dft' in tasks:
    print('Creating dft inversed image')
    im_dft = blockproc(im, dft2)
    im_idft = blockproc(im_dft, idft2)
    image_idft = Image.fromarray(im_idft.real)
    image_idft.show('DFT', command=args.imgcmd)
    image_idft.convert('RGB').save('transform-dft.png')

if 'compress-dct' in tasks:
    print('Creating dct compressed image')
    im_dct_cmpr = blockproc(im, lambda im: dct_compress(im, 8))
    im_dct_icmpr = blockproc(im_dct_cmpr, idct2)
    image_dct_icmpr = Image.fromarray(im_dct_icmpr)
    image_dct_icmpr.show(title='dct compressed', command=args.imgcmd)
    image_dct_icmpr.convert('RGB').save('compress-dct.png')

if 'compress-dft' in tasks:
    print('Creating dft compressed image')
    im_dft_cmpr = blockproc(im, lambda im: dft_compress(im, 8))
    im_dft_icmpr = blockproc(im_dft_cmpr, fp.ifft2)
    image_dft_icmpr = Image.fromarray(im_dft_icmpr.real)
    image_dft_icmpr.show(title='dft compressed', command=args.imgcmd)
    image_dft_icmpr.convert('RGB').save('compress-dft.png')

if 'plot-distortion' in tasks:
    sample_times = [1, 2, 3, 4, 6, 8, 16, 24, 32, 40, 48, 56, 63]
    # sample_times = [1, 16]

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
