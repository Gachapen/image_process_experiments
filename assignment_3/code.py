import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile as wav
from scipy import ndimage as nd

two_pi = math.pi * 2.0


def write_wav(file, sample_rate, samples):
    wav.write(file, sample_rate, ((2**15 - 1) * samples).astype(np.int16))


def task_1():
    def signal(t):
        return 0.5 * np.sin(two_pi * 200.0 * t) \
               + 0.2 * np.sin(two_pi * 455.0 * t) \
               - 0.3 * np.cos(two_pi * 672.0 * t)

    sample_rate = 8192
    length = 1.0
    sample_points = np.arange(0.0, length, 1.0 / sample_rate)
    num_samples = len(sample_points)

    sampled_signal = signal(sample_points)
    write_wav('signal.wav', sample_rate, sampled_signal)

    signal_dft = np.fft.fft(sampled_signal)
    dft_range = range(-num_samples // 2, num_samples // 2)

    plt.plot(dft_range, abs(np.fft.fftshift(signal_dft)))
    plt.title("Signal power spectrum")
    plt.xlabel("k")
    plt.ylabel("|Y_k|")
    plt.show()

    filter_dft = np.concatenate(
        (
            np.zeros(501),
            np.ones(num_samples - 501 - 500),
            np.zeros(500)
        )
    )
    filter = np.fft.ifft(filter_dft)

    plt.plot(range(0, num_samples), abs(filter_dft))
    plt.ylim((0, 1.1))
    plt.title("H power spectrum")
    plt.xlabel("k")
    plt.ylabel("|H_k|")
    plt.show()

    plt.plot(range(0, num_samples), filter)
    plt.xlim((-100, num_samples + 100))
    plt.title("h filter")
    plt.xlabel("k")
    plt.ylabel("h_k")
    plt.show()

    num_nonzero = np.count_nonzero(filter)
    print("There are {} nonzero coefficients in the filter".format(num_nonzero))

    filtered_signal = nd.convolve(sampled_signal, filter, mode='wrap')
    write_wav('filtered.wav', sample_rate, filtered_signal)

    plt.plot(range(0, num_samples), abs(np.fft.fft(filtered_signal)))
    # plt.xlim((-100, num_samples + 100))
    plt.title("w power spectrum")
    plt.xlabel("k")
    plt.ylabel("|W_k|")
    plt.show()


def task_2():
    def signal(t):
        return np.sin(two_pi * 137.0 * t) + 0.4 * np.sin(two_pi * 147.0 * t)

    sample_rate = 1000
    length = 1.0
    sample_points = np.arange(0.0, length, 1.0 / sample_rate)
    num_samples = len(sample_points)

    sampled_signal = signal(sample_points)

    signal_dft = np.fft.fft(sampled_signal)

    plt.plot(range(0, num_samples // 2), abs(signal_dft[0:num_samples//2]))
    plt.title("F magnitudes")
    plt.xlabel("k")
    plt.ylabel("|F_k|")
    plt.show()

    window = 200
    window_dft = np.fft.fft(sampled_signal[0:window])

    plt.plot(np.arange(0, window // 2), abs(window_dft[0:window//2]))
    plt.title("F window (W) magnitudes")
    plt.xlabel("k")
    plt.ylabel("|W_k|")
    plt.show()


task_1()
task_2()
