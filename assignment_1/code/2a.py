# Written by Magnus Bjerke Vik for IMT4202 at HiG (GUC)
# Based on http://askubuntu.com/questions/202355/how-to-play-a-fixed-frequency-sound-using-python

import math
import pyaudio
import matplotlib.pyplot as plt

PyAudio = pyaudio.PyAudio

bitrate = 44000
frequency = 440.0
amplitude = 1.2
length = 1.0

num_samples = int(float(bitrate) * length)
frequency_parameter = 2.0 * math.pi * frequency
raw_samples = []
byte_samples = bytearray(num_samples)

for i in range(num_samples):
    t = (float(i) / float(num_samples)) * length
    wave = math.sin(frequency_parameter * t) * amplitude
    raw_samples.append(wave)
    byte_wave = int((wave / amplitude + 1.0) / 2.0 * 255.0)
    byte_samples[i] = max(0, min(255, byte_wave))

wavedata = bytes(byte_samples)
audio = PyAudio()
stream = audio.open(format = audio.get_format_from_width(1), channels = 1, rate = bitrate, output = True)
stream.write(wavedata)
stream.stop_stream()
stream.close()
audio.terminate()

energies = []
neg_two_pi = -2.0 * math.pi

for k in range(int(frequency) * 2):
    ck = 0.0
    neg_two_pi_k = neg_two_pi * float(k)
    for m in range(num_samples):
        param = neg_two_pi_k * float(m) / float(num_samples)
        ck = ck + raw_samples[m] * (math.cos(param) + 1.0j * math.sin(param))
    ck = 1.0 / float(num_samples) * ck
    energy = float(num_samples) * math.pow(abs(ck), 2.0)
    energies.append(energy)

plt.plot(energies)
plt.xlabel('Hz')
plt.ylabel('Energy')
plt.show()
