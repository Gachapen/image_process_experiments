import numpy as np
import pywt
from PIL import Image

np.set_printoptions(
    linewidth=2000
)


def find_dwt_matrix(size, wavelet_type):
    wavelet = pywt.Wavelet(wavelet_type)
    matrix = np.zeros(shape=(size, size))

    for i in range(size // 2):
        row_l = i
        row_h = i + size // 2
        for j in range(wavelet.dec_len):
            col = ((i * 2) - (wavelet.dec_len - 1 - j)) % size

            matrix[row_l][col] = wavelet.dec_lo[j]
            matrix[row_h][col] = wavelet.dec_hi[j]

    return matrix


def find_multilevel_dwt_matrix(size, levels, wavelet_type):
    matrix = find_dwt_matrix(size, wavelet_type)

    for lvl in range(1, levels):
        lvl_size = size // 2**(lvl)
        lvl_mat = find_dwt_matrix(lvl_size, wavelet_type)
        lvl_identity = np.identity(size - lvl_size)
        mat_l = np.concatenate(
            (
                lvl_mat,
                np.zeros((size - lvl_size, lvl_size))
            ),
            axis=0
        )
        mat_r = np.concatenate(
            (
                np.zeros((lvl_size, size - lvl_size)),
                lvl_identity
            ),
            axis=0
        )
        mat = np.concatenate((mat_l, mat_r), axis=1)

        matrix = np.dot(mat, matrix)

    return matrix


def task_1():
    dwt_haar_4 = find_dwt_matrix(4, 'haar')
    dwt_haar_8 = find_dwt_matrix(8, 'haar')
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 3.0, 3.0])
    x_dwt = np.dot(dwt_haar_8, x)

    print()
    print("DWT Matrix of size 4:")
    print(dwt_haar_4)

    print()
    print("DWT Matrix of size 8:")
    print(dwt_haar_8)

    print()
    print("x:")
    print(x)

    print()
    print("DWT of x with matrix:")
    print(x_dwt)

    print()
    print("DWT of x with pywt:")
    print(pywt.dwt(x, 'haar', mode='ppd'))


def task_2():
    dwt_haar_8_2lvl = find_multilevel_dwt_matrix(8, 2, 'haar')
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 3.0, 3.0])
    x_dwt = np.dot(dwt_haar_8_2lvl, x)

    dwt_haar_4 = find_dwt_matrix(4, 'haar')
    dwt_haar_8 = find_dwt_matrix(8, 'haar')
    x_dwt_man_lvl1 = np.dot(dwt_haar_8, x)
    x_dwt_man = np.concatenate(
        (
            np.dot(dwt_haar_4, x_dwt_man_lvl1[:4]),
            x_dwt_man_lvl1[4:]
        ),
        axis=0
    )

    print()
    print("2-level DWT matrix:")
    print(dwt_haar_8_2lvl)

    print()
    print("x:")
    print(x)

    print()
    print("2-level DWT of x with matrix:")
    print(x_dwt)

    print()
    print("2-level DWT of x manually:")
    print(x_dwt_man)

    print()
    print("2-level DWT of x with pywt:")
    print(pywt.wavedec(x, 'haar', mode='ppd', level=3))


def task_3(wavelet='haar', levels=1):
    img = Image.open('lena.png').convert('L')
    img_data = np.array(img, dtype=np.float)
    dwt = pywt.wavedec2(img_data, wavelet, mode='ppd', level=levels)

    dwt_data = dwt[0]

    for i in range(0, levels):
        lvl = i + 1
        data = dwt[lvl]

        shape = dwt_data.shape
        dwt_l = np.concatenate(
            (
                dwt_data,
                np.resize(data[1], shape)
            ),
            axis=0
        )
        dwt_r = np.concatenate(
            (
                np.resize(data[0], shape),
                np.resize(data[2], shape)
            ),
            axis=0
        )
        dwt_data = np.concatenate((dwt_l, dwt_r), axis=1)

    dwt_img = Image.fromarray(dwt_data)
    dwt_img.convert('RGB').save("img/lena_{}_{}_dwt.png".format(levels, wavelet))

    for lvl in range(1, levels+1):
        for j in range(0, 3):
            dwt[lvl][j].fill(0)

    img_data_idwt = pywt.waverec2(dwt, wavelet, mode='ppd')
    img_idwt = Image.fromarray(img_data_idwt)
    img_idwt.convert('RGB').save("img/lena_{}_{}.png".format(levels, wavelet))


def task_4(wavelet='haar'):
    for lvl in {2, 3, 4}:
        task_3(wavelet=wavelet, levels=lvl)


def task_5():
    wavelets = [
        'bior2.8',
        'coif3',
        'db15',
        'rbio3.9',
        'sym6'
    ]

    for wavelet in wavelets:
        task_3(wavelet=wavelet)
        task_4(wavelet=wavelet)


task_1()
task_2()
task_3()
task_4()
task_5()
