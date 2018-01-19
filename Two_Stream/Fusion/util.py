import pickle, os
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import argparse
from getLabel import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import Optimizer
import imageio
import math
import numpy as np
import cv2

def mAP(pred, gt, n_class):
	AP = 0
	count = 0
	for i in range(n_class):
		conf = pred[:, i]
		conf = -conf
		gt_n = gt[:, i]
		sort_ind = np.argsort(conf)
		sort_arr = np.sort(conf)
		tp = np.zeros([pred.shape[0]])
		fp = np.zeros([pred.shape[0]])
		for rank, ind in enumerate(sort_ind):
			if gt_n[rank] == 1:
				tp[ind] = 1.
			else:
				fp[ind] = 1.
		if (tp == np.zeros(len(tp))).all():
			#print(tp)
			continue
		count = count + 1
		npos = np.sum(tp)
		tp_cum = np.cumsum(tp)
		fp_cum = np.cumsum(fp)
		rec = tp_cum / float(npos)
		prec = np.divide(tp_cum, tp_cum + fp_cum)
		tmp_ap = 0.
		for k, cont in enumerate(tp):
			if cont == 1.:
				tmp_ap = tmp_ap + prec[k]
		AP = AP + tmp_ap / np.sum(tp)
	if count == 0:
		return 0
	mAP = AP / float(count)
	return mAP

def hit_k(predict, label, k):
	total = len(predict)
	right = 0
	for id, sample in enumerate(predict):
		b = zip(sample, range(len(sample)))
		b.sort(key=lambda x: x[0],reverse=True)
		ranked_predict = [x[1] for x in b]
		for i in range(k):
			if label[id][ranked_predict[i]] == 1:
				right += 1
				break
	return float(right) / float(total)

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename):
	torch.save(state, filename + '/' + 'model.pkl')
	if is_best:
		shutil.copyfile(filename + '/' + 'model.pkl', filename + '/' + 'best.pkl')


class ADHA_oneclass_of(Dataset):
	def __init__(self, input_path, label_path, train_test_split_path, expression_path, train, cla, transform):
		self.samples = []
		self.decay = 80
		splits = os.listdir(train_test_split_path)
		if train:
			Name = pickle.load(open(train_test_split_path + '/train.pickle', 'r'))
		else:
			Name = pickle.load(open(train_test_split_path + '/test.pickle', 'r'))
		action = cla
		samples = os.listdir(input_path + '/' + action)
		for sample in samples:
			if sample[2:].decode() in Name:
				self.samples.append([action, sample])
		self.label_path = label_path
		self.input_path = input_path
		self.transform = transform
		self.expression_path = expression_path

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		label = readLabel(sample, int(sample[1][0]), self.label_path)

		person = int(sample[1][0])
		of = torch.FloatTensor(20, 224, 224)
		for i in range(10):
			if os.path.isfile(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_horizontal' + '.jpg'):
				img = cv2.imread(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_horizontal' + '.jpg')
				x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
				x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
				img = Image.fromarray(x)
				transformed_img = self.transform(img)
				img.close()
			else:
				transformed_img = transformed_img
			of[2 * (i), :, :] = transformed_img

			if os.path.isfile(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_vertical' + '.jpg'):
				img = cv2.imread(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_vertical' + '.jpg')
				x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
				x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
				img = Image.fromarray(x)
				transformed_img = self.transform(img)
				img.close()
			else:
				transformed_img = transformed_img

			of[2 * (i) + 1, :, :] = transformed_img

		if self.expression_path != None:
			expression = np.array(
				[edict[open(self.expression_path + '/' + sample[2:-7] + '.txt', 'r').readlines()[0:-1]]])
		else:
			expression = np.array([1])

		sample = (of, expression, label)
		return sample

class ADHA_of(Dataset):

	def __init__(self, input_path, label_path, train_test_split_path, expression_path, train, transform):
		self.samples = []
		self.decay = 80
		splits = os.listdir(train_test_split_path)
		if train:
			Name = pickle.load(open(train_test_split_path + '/train.pickle', 'r'))
		else:
			Name = pickle.load(open(train_test_split_path + '/test.pickle', 'r'))
		actions = os.listdir(input_path)
		for action in actions:
			samples = os.listdir(input_path + '/' + action)
			for sample in samples:
				if sample[2:].decode() in Name:
					self.samples.append([action, sample])
		self.label_path = label_path
		self.input_path = input_path
		self.transform = transform
		self.expression_path = expression_path

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		label = readLabel(sample, int(sample[1][0]), self.label_path)

		person = int(sample[1][0])
		of = torch.FloatTensor(20, 224, 224)
		for i in range(10):
			if os.path.isfile(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_horizontal' + '.jpg'):
				img = cv2.imread(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_horizontal' + '.jpg')
				x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
				x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
				img = Image.fromarray(x)
				transformed_img = self.transform(img)
				img.close()
			else:
				transformed_img = transformed_img
			of[2 * (i), :, :] = transformed_img

			if os.path.isfile(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_vertical' + '.jpg'):
				img = cv2.imread(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'of' + '/' + str(
					i * 8) + '_vertical' + '.jpg')
				x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
				x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
				img = Image.fromarray(x)
				transformed_img = self.transform(img)
				img.close()
			else:
				transformed_img = transformed_img

			of[2 * (i) + 1, :, :] = transformed_img

		if self.expression_path != None:
			expression = np.array(
				[edict[open(self.expression_path + '/' + sample[2:-7] + '.txt', 'r').readlines()[0:-1]]])
		else:
			expression = np.array([1])

		sample = (of, expression, label)
		return sample

class ADHA_oneclass_rgb(Dataset):
	def __init__(self, input_path, label_path, train_test_split_path, expression_path, train,cla, transform):
		self.samples = []
		self.decay = 80
		splits = os.listdir(train_test_split_path)
		if train:
			Name = pickle.load(open(train_test_split_path + '/train.pickle','r'))
		else:
			Name = pickle.load(open(train_test_split_path + '/test.pickle', 'r'))
		action = cla
		samples = os.listdir(input_path + '/' + action)
		for sample in samples:
			if sample[2:].decode() in Name:
				self.samples.append([action, sample])
		self.label_path = label_path
		self.input_path = input_path
		self.transform = transform
		self.expression_path = expression_path



	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		label = readLabel(sample, int(sample[1][0]), self.label_path)

		person = int(sample[1][0])
		img = cv2.imread(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'rgb' + '/' + sample[1][2:] + '.jpg')
		x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
		img = Image.fromarray(x)
		transformed_img = self.transform(img)
		img.close()

		if self.expression_path != None:
			expression = np.array([edict[open(self.expression_path + '/' + sample[2:-7] + '.txt','r').readlines()[0:-1]]])
		else:
			expression = np.array([1])

		sample = (transformed_img, expression, label)
		return sample

class ADHA_rgb(Dataset):

	def __init__(self, input_path, label_path, train_test_split_path, expression_path, train,transform):
		self.samples = []
		self.decay = 80
		splits = os.listdir(train_test_split_path)
		if train:
			Name = pickle.load(open(train_test_split_path + '/train.pickle','r'))
		else:
			Name = pickle.load(open(train_test_split_path + '/test.pickle', 'r'))
		actions = os.listdir(input_path)
		for action in actions:
			samples = os.listdir(input_path + '/' + action)
			for sample in samples:
				if sample[2:].decode() in Name:
					self.samples.append([action, sample])
		self.label_path = label_path
		self.input_path = input_path
		self.transform = transform
		self.expression_path = expression_path



	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		label = readLabel(sample, int(sample[1][0]), self.label_path)

		person = int(sample[1][0])
		img = cv2.imread(self.input_path + '/' + sample[0] + '/' + sample[1] + '/' + 'rgb' + '/' + sample[1][2:] + '.jpg')
		x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
		img = Image.fromarray(x)
		transformed_img = self.transform(img)
		img.close()

		if self.expression_path != None:
			expression = np.array([edict[open(self.expression_path + '/' + sample[2:-7] + '.txt','r').readlines()[0:-1]]])
		else:
			expression = np.array([1])

		sample = (transformed_img, expression, label)
		return sample

# class UCF_training_set(Dataset):
# 	def __init__(self, dic, dic_frame, root_dir, transform=None):
#
# 		dic_training = {}
# 		dic_nb_frame = {}
# 		for key in dic_frame:
# 			n = key.split('_', 1)[1].split('.', 1)[0]
# 			dic_nb_frame[n] = dic_frame[key]
# 		for key in dic:
# 			n, g = key.split('_', 1)
# 			if n == 'HandStandPushups':
# 				key2 = 'HandstandPushups_' + g
# 			else:
# 				key2 = key
#
# 			nb_frame = dic_nb_frame[key2]
# 			new_key = key2 + '[@]' + str(nb_frame)
# 			dic_training[new_key] = dic[key]
#
# 		self.keys = dic_training.keys()
# 		self.values = dic_training.values()
# 		self.root_dir = root_dir
# 		self.transform = transform
#
# 	def __len__(self):
# 		return len(self.keys)
#
# 	def __getitem__(self, idx):
#
# 		video_name, nb_frame = self.keys[idx].split('[@]')
# 		idx = randint(1, int(nb_frame))
#
# 		# rgb image
# 		if video_name.split('_')[0] == 'HandstandPushups':
# 			n, g = video_name.split('_', 1)
# 			name = 'HandStandPushups_' + g
# 			path = self.root_dir + 'HandstandPushups' + '/separated_images/v_' + name + '/v_' + name + '_'
# 		else:
# 			path = self.root_dir + video_name.split('_')[
# 				0] + '/separated_images/v_' + video_name + '/v_' + video_name + '_'
#
# 		img = Image.open(path + str(idx) + '.jpg')
# 		label = self.values[idx]
# 		label = int(label) - 1
#
# 		img = img.resize([224, 224])
# 		transformed_img = self.transform(img)
# 		img.close()
#
# 		sample = (transformed_img, label)
#
# 		return sample
#
#
# class UCF_testing_set(Dataset):
# 	def __init__(self, ucf_list, root_dir, transform=None):
#
# 		self.ucf_list = ucf_list
# 		self.root_dir = root_dir
# 		self.transform = transform
#
# 	def __len__(self):
# 		return len(self.ucf_list)
#
# 	def __getitem__(self, idx):
# 		# img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
# 		key, label = self.ucf_list[idx].split('[@]')
# 		video_name, idx = key.split('-')
# 		label = int(label) - 1
#
# 		# open image
# 		if video_name.split('_')[0] == 'HandstandPushups':
# 			n, g = video_name.split('_', 1)
# 			name = 'HandStandPushups_' + g
# 			path = self.root_dir + 'HandstandPushups' + '/separated_images/v_' + name + '/v_' + name + '_'
# 		else:
# 			path = self.root_dir + video_name.split('_')[
# 				0] + '/separated_images/v_' + video_name + '/v_' + video_name + '_'
#
# 		img = Image.open(path + str(idx) + '.jpg')
# 		img = img.resize([224, 224])
#
# 		if self.transform:
# 			transformed_img = self.transform(img)
# 			sample = (video_name, transformed_img, label)
# 		else:
# 			sample = (video_name, img, label)
#
# 		img.close()
# 		return sample


def record_info(info, mode):
	if mode == 'train':
		result = (
			'Epoch {epoch}'
			'Step {step}'
			'Loss {loss} '
			'mAP_act {mAP_act}'
			'mAP_adv {mAP_adv}'
			'Prec@1_act {top1_act} '
			'Prec@5_act {top5_act}'
			'Prec@1_adv {top1_adv} '
			'Prec@5_adv {top5_adv}'.format(epoch=info['Epoch'],step=info['Step'],loss=info['Loss'], mAP_act=info['mAP_act'], mAP_adv=info['mAP_adv'],  top1_act=info['Prec@1_act'],
			                         top5_act=info['Prec@5_act'], top1_adv=info['Prec@1_adv'], top5_adv=info['Prec@5_adv']))
		print result


	if mode == 'test':
		result = (
			'Step {step}'
			'mAP_act {mAP_act}'
			'mAP_adv {mAP_adv}'
			'Prec@1_act {top1_act} '
			'Prec@5_act {top5_act}'
			'Prec@1_adv {top1_adv} '
			'Prec@5_adv {top5_adv}'.format(step=info['Step'],
			                                 mAP_act=info['mAP_act'], mAP_adv=info['mAP_adv'], top1_act=info['Prec@1_act'], top5_act=info['Prec@5_act'], top1_adv=info['Prec@1_adv'], top5_adv=info['Prec@5_adv']))
		print result

