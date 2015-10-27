from PIL import Image
import numpy as np
import scipy.fftpack as fp

def blockproc(im, fun):
    blocksize = (8, 8)

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

def compress(im):
    dct = fp.dct(fp.dct(im, norm='ortho', axis=0), norm='ortho', axis=1)
    return dct

image = Image.open('lena.png').convert('L')
im = np.array(image, dtype=np.float)
im_dct = blockproc(im, dct2)
im_dft = blockproc(im, fp.fft2)
im_compress = blockproc(im, compress)

#imagedct = Image.fromarray(imdct)
#imagedct.show()

im_idct = blockproc(im_dct, idct2)
image_idct = image.fromarray(im_idct)
image_idct.show()

# im_idft = blockproc(im_dft, fp.ifft2)
# image_idft = Image.fromarray(im_idft)
# image_idft.show()

im_uncompress = blockproc(im_compress, idct2)
image_uncompress = image.fromarray(im_uncompress)
image_uncompress.show()
