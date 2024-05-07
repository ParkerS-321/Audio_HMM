import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os, random
import scipy
import scipy.fftpack

import librosa
from librosa.feature import mfcc
from hmmlearn import hmm
import re
import glob

from HMM_class import HMMTrain
from pathlib import Path

import pickle


input_train = '/Users/blackhawk/Desktop/Audio_HMM/AudioMNIST/train/'
input_test = '/Users/blackhawk/Desktop/Audio_HMM/AudioMNIST/test/'


#MODEL TRAINING

models = []

for dir in os.listdir(input_train):
	label = dir
	X = np.array([])
	labels = []
	dir_path = os.path.join(input_train, dir)
	for file in os.listdir(dir_path):
		if file.endswith('.wav'):
			file_path = os.path.join(dir_path, file)
			sample_freq, audio = librosa.load(file_path)
			#MFCC Features
			mfcc_features = mfcc(y=sample_freq, sr=audio)
			if len(X) == 0:
				X = mfcc_features[:,:13]
			else:
				X = np.append(X, mfcc_features[:,:13], axis=0)
		
			labels.append(label)
	
	#print('X.shape = ', X.shape)

	hmm_trainer = HMMTrain()
	hmm_trainer.train(X)
	models.append((hmm_trainer, label))
	hmm_trainer = None




with open('models.pkl', 'wb') as file:
	pickle.dump(models, file)
