import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os, random
import scipy
import scipy.fftpack


wav_fname = random.choice(os.listdir("./AudioMNIST/data/01/"))
print(wav_fname)

wav_fname = './AudioMNIST/data/01/2_01_11.wav'

samplerate, data = wavfile.read(wav_fname)


l_audio = len(data.shape)
print ("Channels", l_audio)

if l_audio == 2:
	signal = signal.sum(axis=1) / 2

N = data.shape[0]

print ("Complete Samplings N", N)
secs = N / float(samplerate)
print ("secs", secs)
Ts = 1.0/samplerate # sampling interval in time
print ("Timestep between samples Ts", Ts)


t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
FFT = abs(scipy.fft.fft(data))
FFT_side = FFT[range(N//2)] # one side FFT range
freqs = scipy.fftpack.fftfreq(data.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)] # one side frequency range
fft_freqs_side = np.array(freqs_side)
plt.subplot(311)

p1 = plt.plot(t, data, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(312)

p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.subplot(313)
p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')

plt.show()

