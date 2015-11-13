import scipy.io.wavfile as wav
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.ndimage as nd

two_pi = math.pi * 2.0


def write_wav(file, sample_rate, samples):
    wav.write(file, sample_rate, ((2**15 - 1) * samples).astype(np.int16))


def task_1():
    def signal(t):
        return 0.5 * np.sin(two_pi * 200.0 * t) + 0.2 * np.sin(two_pi * 455.0 * t) - 0.3 * np.cos(two_pi * 672.0 * t)


    sample_rate = 8192
    length = 1.0
    sample_points = np.arange(0.0, length, 1.0 / sample_rate)
    num_samples = len(sample_points)

    sampled_signal = signal(sample_points)
    write_wav('signal.wav', sample_rate, sampled_signal)

    signal_dft = np.fft.fft(sampled_signal)

    plt.plot(range(0, sample_rate), abs(signal_dft))
    plt.show()

    filter_dft = np.concatenate((np.zeros(501), np.ones(num_samples - 501 - 500), np.zeros(500)))
    filter = np.fft.ifft(filter_dft)

    plt.plot(range(0, sample_rate), abs(filter_dft))
    plt.show()

    plt.plot(range(0, sample_rate), filter)
    plt.show()

    filtered_signal = nd.convolve(sampled_signal, filter, mode='wrap')
    write_wav('filtered.wav', sample_rate, filtered_signal)


def task_2():
    def signal(t):
        return np.sin(two_pi * 137.0 * t) + 0.4 * np.sin(two_pi * 147.0 * t)


    sample_rate = 1000
    length = 1.0
    sample_points = np.arange(0.0, length, 1.0 / sample_rate)
    num_samples = len(sample_points)

    sampled_signal = signal(sample_points)

    signal_dft = np.fft.fft(sampled_signal)

    plt.plot(range(0, num_samples / 2), abs(signal_dft[0:num_samples/2]))
    plt.show()

    window = 200
    window_ratio = float(num_samples) / float(window)

    plt.plot(np.arange(0, num_samples / 2, window_ratio), abs(np.fft.fft(sampled_signal[0:window])[0:window/2]))
    plt.show()


task_1()
task_2()
