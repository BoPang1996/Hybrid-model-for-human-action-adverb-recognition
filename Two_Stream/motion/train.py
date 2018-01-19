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
from network import *

torch.cuda.set_device(1)


parser = argparse.ArgumentParser(description='PyTorch Sub-JHMDB rgb frame training')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=2, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', default=False, type=bool, help='evaluate model on validation set')
parser.add_argument('--resume', default=False, type=bool, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
model_path = '/Disk8/poli/models/ADHA/Two_stream/T1/motion'
log_path = '/Disk8/poli/logs/ADHA/Two_stream/T1'
withexpression = True

def main():
	global arg
	arg = parser.parse_args()
	print arg

	# Prepare DataLoader
	data_loader = Data_Loader(
		BATCH_SIZE=arg.batch_size,
		num_workers=4,
		input_path='/Disk8/HMDB/Two_stream_input',
		label_path='/Disk8/HMDB/labels/result', 
		train_test_split_path='/Disk8/HMDB/train_test_split',
		expression_path = None
	)

	train_loader = data_loader.train()
	test_loader = data_loader.validate()
	# Model
	spatial_cnn = Spatial_CNN(
		nb_epochs=arg.epochs,
		lr=arg.lr,
		batch_size=arg.batch_size,
		resume=arg.resume,
		start_epoch=arg.start_epoch,
		evaluate=arg.evaluate,
		train_loader=train_loader,
		test_loader=test_loader,
		withexpression = withexpression
	)
	# Training
	spatial_cnn.run()


class Spatial_CNN():
	def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, withexpression):
		self.nb_epochs = nb_epochs
		self.lr = lr
		self.batch_size = batch_size
		self.resume = resume
		self.start_epoch = start_epoch
		self.evaluate = evaluate
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.best_prec1 = 0
		self.withexpression = withexpression
		
	def run(self):
		self.model = resnet101(withexpression=self.withexpression, pretrained=True, nb_classes=101).cuda()
		# Loss function and optimizer
		self.criterion = nn.MSELoss().cuda()
		self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
		self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=0, verbose=True)

		cudnn.benchmark = True
		if self.resume:
			if os.path.isfile(model_path + '/' + 'best.pkl'):
				print("==> loading checkpoint '{}'".format(model_path + '/' + 'best.pkl'))
				checkpoint = torch.load(model_path + '/' + 'best.pkl')
				#self.start_epoch = checkpoint['epoch']
				self.best_prec1 = checkpoint['best_prec1']
				self.model.load_state_dict(checkpoint['state_dict'])
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
				      .format(model_path + '/' + 'best.pkl', checkpoint['epoch'], self.best_prec1))
			else:
				print("==> no checkpoint found at '{}'".format(self.resume))
		if self.evaluate:
			map = self.validate_1epoch()
			exit()

		for self.epoch in range(self.start_epoch, self.nb_epochs):
			print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
			self.train_1epoch()
			print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
			map = self.validate_1epoch()
			self.scheduler.step(map)

			is_best = map > self.best_prec1
			if is_best:
				self.best_prec1 = map

			save_checkpoint({
				'epoch': self.epoch,
				'state_dict': self.model.state_dict(),
				'best_prec1': self.best_prec1,
				'optimizer': self.optimizer.state_dict()
			}, is_best,filename=model_path)

	def train_1epoch(self):
		# switch to train mode
		self.model.train()
		end = time.time()
		# mini-batch training
		progress = self.train_loader
		for i, (data, expression, label) in enumerate(progress):
			# measure data loading time
			label_act = label[0]
			label_adv = label[1]
			expression_var = Variable(expression).cuda()
			input_var = Variable(data).cuda()
			target_act = Variable(label_act).cuda()
			target_adv = Variable(label_adv).cuda()

			# compute output
			output_act, output_adv = self.model(input_var.float(), expression_var.float())
			loss_act = self.criterion(output_act, target_act.float())
			loss_adv = self.criterion(output_adv, target_adv.float())
			loss = loss_act + loss_adv

			# measure accuracy and record loss
			mAP_act = mAP(output_act.data.cpu().numpy(), label_act.numpy(), 32)
			mAP_adv = mAP(output_adv.data.cpu().numpy(), label_adv.numpy(), 51)
			prec1_act= hit_k(output_act.data.cpu().numpy(), label_act.numpy(), 1)
			prec5_act = hit_k(output_act.data.cpu().numpy(), label_act.numpy(), 5)
			prec1_adv = hit_k(output_adv.data.cpu().numpy(), label_adv.numpy(), 1)
			prec5_adv = hit_k(output_adv.data.cpu().numpy(), label_adv.numpy(), 5)



			# compute gradient and do SGD step
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			# measure elapsed time
			end = time.time()

			info = {'Epoch': [self.epoch],
			        'Step': [i],
			        'Loss': [round(loss.data.cpu().numpy(), 5)],
			        'mAP_act':[round(mAP_act,5)],
			        'mAP_adv': [round(mAP_adv, 5)],
			        'Prec@1_act': [round(prec1_act, 4)],
			        'Prec@5_act': [round(prec5_act, 4)],
			        'Prec@1_adv': [round(prec1_adv, 4)],
			        'Prec@5_adv': [round(prec5_adv, 4)]
			        }
			record_info(info, 'train')

	def validate_1epoch(self):

		# switch to evaluate mode
		self.model.eval()
		dic_video_level_preds = {}
		labels_act = []
		labels_adv = []
		preds_act = []
		preds_adv = []

		progress = self.test_loader
		for i, (data, expression, label) in tqdm(enumerate(progress)):
			label_act = label[0]
			label_adv = label[1]
			label_act = label_act.cuda(async=True)
			label_adv = label_adv.cuda(async=True)
			data_var = Variable(data, volatile=True).cuda(async=True)
			expression_var = Variable(expression).cuda()
			label_act = Variable(label_act, volatile=True).cuda(async=True)
			label_adv = Variable(label_adv, volatile=True).cuda(async=True)

			# compute output
			output_act, output_adv = self.model(data_var.float(),expression_var.float())

			# metric
			for Nsample in range(len(label_act)):
				labels_act.append(label_act[Nsample].data.cpu().numpy())
				labels_adv.append(label_adv[Nsample].data.cpu().numpy())
				preds_act.append(output_act[Nsample].data.cpu().numpy())
				preds_adv.append(output_adv[Nsample].data.cpu().numpy())

		mAP_act = mAP(np.array(preds_act), np.array(labels_act), 32)
		mAP_adv = mAP(np.array(preds_adv), np.array(labels_adv), 51)
		prec1_act = hit_k(np.array(preds_act), np.array(labels_act), 1)
		prec5_act = hit_k(np.array(preds_act), np.array(labels_act), 5)
		prec1_adv = hit_k(np.array(preds_adv), np.array(labels_adv), 1)
		prec5_adv = hit_k(np.array(preds_adv), np.array(labels_adv), 5)

		info = {'Epoch': [self.epoch],
		        'Step': [i],
		        'mAP_act': [round(mAP_act, 5)],
		        'mAP_adv': [round(mAP_adv, 5)],
		        'Prec@1_act': [round(prec1_act, 4)],
		        'Prec@5_act': [round(prec5_act, 4)],
		        'Prec@1_adv': [round(prec1_adv, 4)],
		        'Prec@5_adv': [round(prec5_adv, 4)]
		        }
		record_info(info, 'test')
		return mAP_adv



class Data_Loader():
	def __init__(self, BATCH_SIZE, num_workers,input_path, label_path, train_test_split_path,expression_path):
		self.BATCH_SIZE = BATCH_SIZE
		self.num_workers = num_workers
		# load data dictionary


		self.training_set = ADHA(input_path=input_path, label_path=label_path, train_test_split_path=train_test_split_path, expression_path=None, train=True,
		                                     transform=transforms.Compose([
			                                     # transforms.RandomCrop(224),
			                                     transforms.RandomHorizontalFlip(),
			                                     transforms.ToTensor(),
			                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                                          std=[0.229, 0.224, 0.225])
		                                     ]))
		self.validation_set = ADHA(input_path=input_path, label_path=label_path, train_test_split_path=train_test_split_path, expression_path=None, train=False,
		                                      transform=transforms.Compose([
			                                      transforms.ToTensor(),
			                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                                           std=[0.229, 0.224, 0.225])
		                                      ]))
		print '==> Training data :', len(self.training_set)
		print '==> Validation data :', len(self.validation_set)

	def train(self):
		train_loader = DataLoader(
			dataset=self.training_set,
			batch_size=self.BATCH_SIZE,
			shuffle=True,
			num_workers=self.num_workers)
		return train_loader

	def validate(self):
		test_loader = DataLoader(
			dataset=self.validation_set,
			batch_size=self.BATCH_SIZE,
			shuffle=False,
			num_workers=self.num_workers)
		return test_loader




if __name__ == '__main__':
	main()