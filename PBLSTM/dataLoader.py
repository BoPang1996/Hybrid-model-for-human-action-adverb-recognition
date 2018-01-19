import torch.utils.data as data
from getLabel import *
import os
import pickle
import numpy as np

edict = {}
edict['Angry'] = 0
edict['Disgust'] = 1
edict['Fear'] = 2
edict['Happy'] = 3
edict['Neutral'] = 4
edict['Sad'] = 5
edict['Surprise'] = 6

class ADHA(data.Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, feature_path, label_path, train_test_split_path, expression_path, train):
		self.samples = []
		splits = os.listdir(train_test_split_path)
		if train:
			Name = pickle.load(open(train_test_split_path + '/train.pickle','r'))
		else:
			Name = pickle.load(open(train_test_split_path + '/test.pickle', 'r'))
		actions = os.listdir(feature_path)
		n = 0
		for action in actions:
			samples = os.listdir(feature_path + '/' + action)
			for sample in samples:
				if sample[2:-7].decode() in Name:
					self.samples.append([action, sample])
					n = n + 1
		print("There are " + str(n) + ' samples')
		self.label_path = label_path
		self.feature_path = feature_path
		self.expression_path = expression_path


	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		label = readLabel(sample, int(sample[1][0]), self.label_path)
		#label_act,label_adv = readLabel(sample, int(sample[1][0]), self.label_path)
		feature = pickle.load(open(self.feature_path + '/' + sample[0] + '/' + sample[1]))
		if self.expression_path != None:
			expression = np.array([edict[open(self.expression_path + '/' + sample[2:-7] + '.txt','r').readlines()[0:-1]]])
		else:
			expression = np.array([1])
		return feature, expression, label

class ADHA_singleClass(data.Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, feature_path, label_path, train_test_split_path, train, expression_path, action):
		self.samples = []
		if train:
			Name = pickle.load(open(train_test_split_path + '/train.pickle','r'))
		else:
			Name = pickle.load(open(train_test_split_path + '/test.pickle', 'r'))
		n = 0
		samples = os.listdir(feature_path + '/' + action)
		for sample in samples:
			if sample[2:-7].decode() in Name:
				self.samples.append([action, sample])
				n = n + 1
		print("There are " + str(n) + ' samples')
		self.label_path = label_path
		self.feature_path = feature_path
		self.expression_path = expression_path


	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		label = readLabel(sample, int(sample[1][0]), self.label_path)
		#label_act,label_adv = readLabel(sample, int(sample[1][0]), self.label_path)
		feature = pickle.load(open(self.feature_path + '/' + sample[0] + '/' + sample[1]))
		if self.expression_path != None:
			expression = np.array([edict[open(self.expression_path + '/' + sample[2:-7] + '.txt','r').readlines()[0:-1]]])
		else:
			expression = np.array([1])
		return feature, expression, label

