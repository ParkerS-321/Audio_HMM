import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os, random
import scipy
import scipy.fftpack
from hmmlearn import hmm

import librosa
from librosa.feature import mfcc



#Class to store the 10 HMM models

class HMMTrain(object):
	def __init__(self,model_name='GaussianHMM', n_components=4):
		self.model_name = model_name
		self.n_components = n_components
		
		self.models = []			
		if self.model_name == 'GaussianHMM':
			self.model=hmm.GaussianHMM(n_components=4)
		else:
			print("Please choose GaussianHMM")		
	

	def train(self, X):
		self.models.append(self.model.fit(X))
		
	def get_score(self, input_data):
		return self.model.score(input_data)




