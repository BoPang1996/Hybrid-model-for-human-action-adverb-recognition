import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np


class LSTM_T1(nn.Module):
	def __init__(self, featureN, withexpression = False):
		super(LSTM_T1, self).__init__()
		# the hidden_size is 51
		if featureN == 1:
			self.rnn = nn.LSTM(
		        input_size=2048,
		        hidden_size=2048,
		        num_layers=3,
		        batch_first=True,  # e.g. (batch, time_step, input_size)
	        )
		else:
			self.rnn = nn.LSTM(
				input_size=4096,
				hidden_size=2048,
				num_layers=3,
				batch_first=True,  # e.g. (batch, time_step, input_size)
			)

		if withexpression == False:
			self.FC_action = nn.Linear(2048, 2048)
			self.FC_action2 = nn.Linear(2048, 32)
			self.SM_action = nn.Softmax()

			self.FC_adverb = nn.Linear(2048, 2048)
			self.FC_adverb2 = nn.Linear(2048, 51)
			self.SM_adverb = nn.Softmax()
		else:
			self.FC_action = nn.Linear(2049, 2048)
			self.FC_action2 = nn.Linear(2048, 32)
			self.SM_action = nn.Softmax()

			self.FC_adverb = nn.Linear(2049, 2048)
			self.FC_adverb2 = nn.Linear(2048, 51)
			self.SM_adverb = nn.Softmax()
		
		self.withexpression = withexpression

	def forward(self, input, expression=None):
		r_out, (h_n, h_c) = self.rnn(input, None)
		
		if self.withexpression == False:
			out_action = self.FC_action(r_out[:, -1, :])
			out_action = self.FC_action2(out_action)
			out_action = self.SM_action(out_action)

			out_adverb = self.FC_adverb(r_out[:, -1, :])
			out_adverb = self.FC_adverb2(out_adverb)
			out_adverb = self.SM_adverb(out_adverb)
		else:
			out_action = self.FC_action(torch.cat((r_out[:, -1, :],expression), 1))
			out_action = self.FC_action2(out_action)
			out_action = self.SM_action(out_action)

			out_adverb = self.FC_adverb(torch.cat((r_out[:, -1, :],expression), 1))
			out_adverb = self.FC_adverb2(out_adverb)
			out_adverb = self.SM_adverb(out_adverb)

		return out_action, out_adverb


class LSTM_T2_1(nn.Module):
	def __init__(self, featureN):
		super(LSTM_T2_1, self).__init__()
		# the hidden_size is 51
		if featureN == 1:
			self.rnn = nn.LSTM(
		        input_size=2048,
		        hidden_size=2048,
		        num_layers=3,
		        batch_first=True,  # e.g. (batch, time_step, input_size)
	        )
		else:
			self.rnn = nn.LSTM(
				input_size=4096,
				hidden_size=2048,
				num_layers=3,
				batch_first=True,  # e.g. (batch, time_step, input_size)
			)


	def forward(self, input):
		expression = input[1]
		input = input[0]
		r_out, (h_n, h_c) = self.rnn(input, None)

		return [r_out, expression]


class LSTM_T2_2(nn.Module):
	def __init__(self, withexpression = False):
		super(LSTM_T2_2, self).__init__()
		# the hidden_size is 51
		if not withexpression:
			self.FC_adverb = nn.Linear(2048, 1024)
			self.FC_adverb2 = nn.Linear(1024, 51)
			self.SM_adverb = nn.Softmax()
		else:
			self.FC_adverb = nn.Linear(2049, 1024)
			self.FC_adverb2 = nn.Linear(1024, 51)
			self.SM_adverb = nn.Softmax()
		self.withexpression = withexpression

	def forward(self, input):
		expression = input[1]
		input = input[0]
		if self.withexpression == False:
			out_adverb = self.FC_adverb(input[:, -1, :])
			out_adverb = self.FC_adverb2(out_adverb)
			out_adverb = self.SM_adverb(out_adverb)
		else:
			print(torch.cat((input[:, -1, :], expression), 1))
			out_adverb = self.FC_adverb(torch.cat((input[:, -1, :], expression), 1))
			out_adverb = self.FC_adverb2(out_adverb)
			out_adverb = self.SM_adverb(out_adverb)

		return out_adverb