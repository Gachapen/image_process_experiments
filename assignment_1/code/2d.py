# Written by Magnus Bjerke Vik for IMT4202 at HiG (GUC)
# Based on http://askubuntu.com/questions/202355/how-to-play-a-fixed-frequency-sound-using-python

import math
import matplotlib.pyplot as plt

bitrate = 4
length = 1.0

num_samples = int(float(bitrate) * length)
two_pi = 2.0 * math.pi
raw_samples = []

for i in range(num_samples):
    t = (float(i) / float(num_samples)) * length
    wave = math.sin(two_pi * t)
    raw_samples.append(wave)

cks = []
neg_two_pi = two_pi * -1.0

min_hz = -2
max_hz = 2
hz_range = range(min_hz, max_hz + 1)

for k in hz_range:
    ck = 0.0
    neg_two_pi_k = neg_two_pi * float(k)

    for m in range(num_samples):
        param = neg_two_pi_k * float(m) / float(num_samples)
        calc = raw_samples[m] * (math.cos(param) + 1.0j * math.sin(param))
        ck = ck + calc

    ck = ck / float(num_samples)
    print("ck (k={}): {}".format(k, ck))
    cks.append(abs(ck))

plt.plot(hz_range, cks)
plt.xlabel('Hz')
plt.ylabel('|ck|')
plt.show()
