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
from sklearn.metrics import accuracy_score
import pickle


with open('models.pkl', 'rb') as file:
	models = pickle.load(file)


input_test = '/Users/blackhawk/Desktop/Audio_HMM/AudioMNIST/test/'

y_true = []
y_pred = []

count = 0

for dir in os.listdir(input_test):
	dir_path = os.path.join(input_test, dir)
	for file in os.listdir(dir_path):
		if file.endswith('.wav'):
			file_path = os.path.join(dir_path,file)
			sample_freq, audio = librosa.load(file_path)
			mfcc_features = mfcc(y=sample_freq, sr=audio)[:,:13]
			#print(mfcc_features.shape)
			scores= []			
			for item in models:
				hmm_model, label = item
				score = hmm_model.get_score(mfcc_features)
				scores.append(score)
				index = np.array(scores).argmax()
		
			#print(models[index][1])
			label = dir			
			if models[index][1] == label:
				count+=1
			y_pred.append(models[index][1])
			y_true.append(label)
						
			#print(models[index][1], label)


acc = accuracy_score(y_true, y_pred)
print(acc)

