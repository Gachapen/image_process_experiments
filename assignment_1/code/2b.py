# Written by Magnus Bjerke Vik for IMT4202 at HiG (GUC)

import math
import matplotlib.pyplot as plt

x1 = [1, 1, 1, 1]
x2 = [1, 0, 1, 0, 1, 0, 1, 0]
x3 = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

sequences = [x1, x2, x3]
colors = ['green', 'red', 'blue']
linestyles = ['-', '-', '-']

for i, sequence in enumerate(sequences):
    num_samples = len(sequence)

    magnitudes = []
    neg_two_pi = -2.0 * math.pi

    for k in range(9):
        dft = 0.0
        neg_two_pi_k = neg_two_pi * float(k)

        for m, sample in enumerate(sequence):
            param = neg_two_pi_k * float(m) / float(num_samples)
            dft = dft + sample * (math.cos(param) + 1.0j * math.sin(param))

        magnitude = abs(dft)
        magnitudes.append(magnitude)

    plt.plot(magnitudes, color=colors[i], linestyle=linestyles[i])
    plt.xlabel('k')
    plt.ylabel('Magnitude')
    plt.show()

