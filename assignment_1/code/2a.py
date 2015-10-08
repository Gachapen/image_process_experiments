# Written by Magnus Bjerke Vik for IMT4202 at HiG (GUC)
# Based on http://askubuntu.com/questions/202355/how-to-play-a-fixed-frequency-sound-using-python

import math
import pyaudio
import matplotlib.pyplot as plt

PyAudio = pyaudio.PyAudio

bitrate = 44000
frequency = 440.0
amplitude = 1.2
length = 1.0 # Not part of the task, but nice to have

num_samples = int(float(bitrate) * length)
frequency_parameter = 2.0 * math.pi * frequency

raw_samples = [] # The raw samples as real values
byte_samples = bytearray(num_samples) # The samples converted to byte sized integers

sample_range = range(num_samples) # So we don't have to generate the range several times

for i in sample_range:
    t = (float(i) / float(num_samples)) * length
    wave = math.sin(frequency_parameter * t) * amplitude

    raw_samples.append(wave)

    byte_wave = int((wave + 1.0) / 2.0 * 255.0) # Tranform (-1, 1) to (0, 256]
    byte_samples[i] = max(0, min(255, byte_wave)) # Clamp values to (0, 256]

audio = PyAudio()
stream = audio.open(format = audio.get_format_from_width(1), channels = 1, rate = bitrate, output = True)
stream.write(bytes(byte_samples))
stream.stop_stream()
stream.close()
audio.terminate()

energies = []
neg_two_pi = -2.0 * math.pi
min_hz = -512
max_hz = 512
frequency_range = range(min_hz, max_hz + 1)

for k in frequency_range:
    ck = 0.0
    neg_two_pi_k = neg_two_pi * float(k)
    for m in sample_range:
        param = neg_two_pi_k * float(m) / float(num_samples)
        ck = ck + raw_samples[m] * (math.cos(param) + 1.0j * math.sin(param))
    ck = ck / float(num_samples)
    energy = float(num_samples) * math.pow(abs(ck), 2.0)
    energies.append(energy)

plt.plot(frequency_range, energies)
plt.xlabel('Hz')
plt.ylabel('Energy')
plt.show()
