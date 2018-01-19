import numpy as np
import pickle
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from util import *
from network import resnet101_t2_rgb
from network import resnet101_t2_of
from network import classifier

torch.cuda.set_device(1)


parser = argparse.ArgumentParser(description='PyTorch Sub-JHMDB rgb frame training')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
model_path_rgb = '/Disk8/poli/models/ADHA/Two_stream/T2/spatial'
model_path_of = '/Disk8/poli/models/ADHA/Two_stream/T2/motion'
log_path = '/Disk8/poli/logs/ADHA/Two_stream/T2'
withexpression = True
expression_path = None
result_save_path = None
shuffle = False

def main():
	global arg
	arg = parser.parse_args()
	print arg

	# Prepare DataLoader
	data_loader_of = Data_Loader_of(
		BATCH_SIZE=arg.batch_size,
		num_workers=4,
		input_path='/Disk8/HMDB/Two_stream_input',
		label_path='/Disk8/HMDB/labels/result',
		train_test_split_path='/Disk8/HMDB/train_test_split',
		expression_path = expression_path
	)

	data_loader_rgb = Data_Loader_rgb(
		BATCH_SIZE=arg.batch_size,
		num_workers=4,
		input_path='/Disk8/HMDB/Two_stream_input',
		label_path='/Disk8/HMDB/labels/result',
		train_test_split_path='/Disk8/HMDB/train_test_split',
		expression_path=expression_path
	)

	test_loader_of = data_loader_of.validate()
	test_loader_rgb = data_loader_rgb.validate()
	# Model
	spatial_cnn = Spatial_CNN(
		batch_size=arg.batch_size,
		test_loader_rgb=test_loader_rgb,
		test_loader_of=test_loader_of,
		withexpression=withexpression
	)
	# Training
	spatial_cnn.run()


class Spatial_CNN():
	def __init__(self, batch_size, test_loader_rgb, test_loader_of,withexpression):
		self.batch_size = batch_size
		self.test_loader_rgb = test_loader_rgb
		self.test_loader_of = test_loader_of
		self.best_prec1 = 0
		self.withexpression = withexpression

	def run(self):

		self.model_1_of = resnet101_t2_of(pretrained=True, nb_classes=101, channel=20).cuda()
		self.model_2_of = classifier(withexpression=self.withexpression,nb_classes=51).cuda()
		self.model_of = nn.Sequential(
			self.model_1_of,
			self.model_2_of
		)

		self.model_1_rgb = resnet101_t2_rgb(pretrained=True, nb_classes=101).cuda()
		self.model_2_rgb = classifier(withexpression = self.withexpression, nb_classes=51).cuda()
		self.model_rgb = nn.Sequential(
			self.model_1_rgb,
			self.model_2_rgb
		)
		# Loss function and optimizer


		cudnn.benchmark = True

		if os.path.isfile(model_path_of + '/' + 'best.pkl'):
			print("==> loading checkpoint '{}'".format(model_path_of + '/' + 'best.pkl'))
			checkpoint = torch.load(model_path_of + '/' + 'best.pkl')
			self.best_prec1 = checkpoint['best_prec1']
			self.model_of.load_state_dict(checkpoint['state_dict'])
			print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
			      .format(model_path_of + '/' + 'best.pkl', checkpoint['epoch'], self.best_prec1))
		else:
			print("==> no checkpoint found at '{}'".format(model_path_of + '/' + 'best.pkl'))

		if os.path.isfile(model_path_rgb + '/' + 'best.pkl'):
			print("==> loading checkpoint '{}'".format(model_path_rgb + '/' + 'best.pkl'))
			checkpoint = torch.load(model_path_rgb + '/' + 'best.pkl')
			self.best_prec1 = checkpoint['best_prec1']
			self.model_rgb.load_state_dict(checkpoint['state_dict'])
			print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
			      .format(model_path_rgb + '/' + 'best.pkl', checkpoint['epoch'], self.best_prec1))
		else:
			print("==> no checkpoint found at '{}'".format(model_path_rgb + '/' + 'best.pkl'))




		print('==> [validation stage]')
		map = self.validate_1epoch()


	def validate_1epoch(self):

		# switch to evaluate mode
		self.model_of.eval()
		self.model_rgb.eval()
		dic_video_level_preds = {}
		labels_act = []
		labels_adv = []
		preds_adv_of = []
		preds_adv_rgb = []

		loaders = self.test_loader_of
		N_step = 0
		for loader in loaders:
			if len(loader) > N_step:
				N_step = len(loader)

		for i in range(N_step):
			loaderIters = [iter(loader) for loader in loaders]
			for loaderID, loader in enumerate(loaderIters):
				if i >= len(loaders[loaderID]):
					continue

				(data, expression, label) = loader.next()
				label_act = label[0]
				label_adv = label[1]
				label_act = label_act.cuda(async=True)
				label_adv = label_adv.cuda(async=True)
				data_var = Variable(data, volatile=True).cuda(async=True)
				expression_var = Variable(expression).cuda()
				label_act = Variable(label_act, volatile=True).cuda(async=True)
				label_adv = Variable(label_adv, volatile=True).cuda(async=True)

				# compute output
				self.model_2_of.load_state_dict(torch.load(model_path_of + '/' + 'params' + str(loaderID) + '.pkl'))
				output_adv = self.model_of([data_var.float(), expression_var.float()])

				# metric
				for Nsample in range(len(label_act)):
					labels_act.append(label_act[Nsample].data.cpu().numpy())
					labels_adv.append(label_adv[Nsample].data.cpu().numpy())
					preds_adv_of.append(output_adv[Nsample].data.cpu().numpy())

		loaders = self.test_loader_rgb
		N_step = 0
		for loader in loaders:
			if len(loader) > N_step:
				N_step = len(loader)

		for i in range(N_step):
			loaderIters = [iter(loader) for loader in loaders]
			for loaderID, loader in enumerate(loaderIters):
				if i >= len(loaders[loaderID]):
					continue

				(data, expression, label) = loader.next()
				label_act = label[0]
				label_adv = label[1]
				label_act = label_act.cuda(async=True)
				label_adv = label_adv.cuda(async=True)
				data_var = Variable(data, volatile=True).cuda(async=True)
				expression_var = Variable(expression).cuda()
				label_act = Variable(label_act, volatile=True).cuda(async=True)
				label_adv = Variable(label_adv, volatile=True).cuda(async=True)

				# compute output
				self.model_2_rgb.load_state_dict(torch.load(model_path_rgb + '/' + 'params' + str(loaderID) + '.pkl'))
				output_adv = self.model_rgb([data_var.float(), expression_var.float()])

				# metric
				for Nsample in range(len(label_act)):
					labels_act.append(label_act[Nsample].data.cpu().numpy())
					labels_adv.append(label_adv[Nsample].data.cpu().numpy())
					preds_adv_rgb.append(output_adv[Nsample].data.cpu().numpy())


		preds_adv = np.divide((preds_adv_of + preds_adv_rgb), 2)

		mAP_adv = mAP(np.array(preds_adv), np.array(labels_adv), 51)

		prec1_adv = hit_k(np.array(preds_adv), np.array(labels_adv), 1)
		prec5_adv = hit_k(np.array(preds_adv), np.array(labels_adv), 5)

		info = {
		        'Step': [i],
		        'mAP_act': ['-'],
		        'mAP_adv': [round(mAP_adv, 5)],
		        'Prec@1_act': ['-'],
		        'Prec@5_act': ['-'],
		        'Prec@1_adv': [round(prec1_adv, 4)],
		        'Prec@5_adv': [round(prec5_adv, 4)]
		        }
		record_info(info, 'test')
		
		if result_save_path != None:
			if not shuffle:
				pickle.dump(preds_adv, open(result_save_path + '/' + 'adverb_result.pickle','wb'))
				pickle.dump(labels_adv, open(result_save_path + '/' + 'adverb_label.pickle','wb'))
				print("Save out the result")
		else:
			print("Using shuffle can not save out the result")
			
		return mAP_adv



class Data_Loader_of():
	def __init__(self, BATCH_SIZE, num_workers,input_path, label_path, train_test_split_path,expression_path):
		self.BATCH_SIZE = BATCH_SIZE
		self.num_workers = num_workers
		# load data dictionary

		self.training_set=[ADHA_oneclass_of(input_path=input_path, label_path=label_path, train_test_split_path=train_test_split_path,expression_path=expression_path,train=True, cla=i,
		                                     transform=transforms.Compose([
			                                     # transforms.RandomCrop(224),
			                                     transforms.RandomHorizontalFlip(),
			                                     transforms.ToTensor(),
			                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                                          std=[0.229, 0.224, 0.225])
		                                     ])) for i in embedding_action]
		self.validation_set =[ADHA_oneclass_of( input_path=input_path, label_path=label_path, train_test_split_path=train_test_split_path, expression_path=expression_path, train=False, cla=i,
		                                     transform=transforms.Compose([
			                                     # transforms.RandomCrop(224),
			                                     transforms.RandomHorizontalFlip(),
			                                     transforms.ToTensor(),
			                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                                          std=[0.229, 0.224, 0.225])
		                                     ])) for i in embedding_action]
		print '==> Training data :', len(self.training_set)
		print '==> Validation data :', len(self.validation_set)

	def train(self):
		train_loader = [DataLoader(
			dataset=self.training_set[i],
			batch_size=self.BATCH_SIZE,
			shuffle=shuffle,
			num_workers=self.num_workers) for i in range(len(self.training_set))]
		return train_loader

	def validate(self):
		test_loader = [DataLoader(
			dataset=self.validation_set[i],
			batch_size=self.BATCH_SIZE,
			shuffle=shuffle,
			num_workers=self.num_workers) for i in range(len(self.training_set))]
		return test_loader

class Data_Loader_rgb():
	def __init__(self, BATCH_SIZE, num_workers,input_path, label_path, train_test_split_path,expression_path):
		self.BATCH_SIZE = BATCH_SIZE
		self.num_workers = num_workers
		# load data dictionary

		self.training_set=[ADHA_oneclass_rgb(input_path=input_path, label_path=label_path, train_test_split_path=train_test_split_path,expression_path=expression_path, train=True, cla=i,
		                                     transform=transforms.Compose([
			                                     # transforms.RandomCrop(224),
			                                     transforms.RandomHorizontalFlip(),
			                                     transforms.ToTensor(),
			                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                                          std=[0.229, 0.224, 0.225])
		                                     ])) for i in embedding_action]
		self.validation_set = [ADHA_oneclass_rgb(input_path=input_path, label_path=label_path, train_test_split_path=train_test_split_path,expression_path=expression_path, train=False, cla=i,
		                                     transform=transforms.Compose([
			                                     # transforms.RandomCrop(224),
			                                     transforms.RandomHorizontalFlip(),
			                                     transforms.ToTensor(),
			                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                                          std=[0.229, 0.224, 0.225])
		                                     ])) for i in embedding_action]
		print '==> Training data :', len(self.training_set)
		print '==> Validation data :', len(self.validation_set)

	def train(self):
		train_loader = [DataLoader(
			dataset=self.training_set[i],
			batch_size=self.BATCH_SIZE,
			shuffle=shuffle,
			num_workers=self.num_workers) for i in range(len(self.training_set))]
		return train_loader

	def validate(self):
		test_loader = [DataLoader(
			dataset=self.validation_set[i],
			batch_size=self.BATCH_SIZE,
			shuffle=shuffle,
			num_workers=self.num_workers) for i in range(len(self.training_set))]
		return test_loader



if __name__ == '__main__':
	main()