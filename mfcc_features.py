import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os, random
import scipy
import scipy.fftpack

import librosa
from librosa.feature import mfcc


wav_fname = random.choice(os.listdir("./AudioMNIST/data/01/"))
print(wav_fname)

wav_fname = './AudioMNIST/data/01/2_01_11.wav'

samplerate, data = librosa.load(wav_fname)




S = librosa.feature.melspectrogram(y=samplerate, sr=data, n_mels=128,
                                   fmax=8000)

mfccs = librosa.feature.mfcc(y = samplerate, sr=data)

import matplotlib.pyplot as plt

from matplotlib import cm

fig, ax = plt.subplots()
mfcc_data= np.swapaxes(mfccs, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('MFCC')

plt.show()



