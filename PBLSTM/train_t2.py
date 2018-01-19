import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from model import *
from dataLoader import *
from metrics import *

LR =0.0005
EPOCH = 20
batch_size = 128
featureN = 1
withexpression = True
model_path = '/Disk8/poli/models/ADHA/pose/task2_feature1'
resume = False


saveInter = 2
feature_path = '/Disk8/HMDB/pose_feature'
train_test_split_path = '/Disk8/HMDB/train_test_split'
label_path = '/Disk8/HMDB/labels/result'
expression_path = None

torch.cuda.set_device(1)

rnn_1 = LSTM_T2_1(featureN)
rnn_2 = LSTM_T2_2(withexpression)



rnn_1.cuda()
rnn_2.cuda()

rnn = nn.Sequential(
	rnn_1,
	rnn_2
        )
if resume:
	print("loading model")
	rnn.load_state_dict(torch.load(model_path + '/' + 'params.pkl'))

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
#loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
loss_func = nn.MSELoss()

ADHA_singleloaders = [torch.utils.data.DataLoader(
         ADHA_singleClass(feature_path=feature_path , label_path=label_path, train_test_split_path=train_test_split_path, expression_path=expression_path, train=True,action=i),
         batch_size= batch_size, shuffle= True, num_workers= 2) for i in embedding_action]

# training and testing
for epoch in range(EPOCH):
	N_step = 0
	for loader in ADHA_singleloaders:
		if len(loader) > N_step:
			N_step = len(loader)
	for step in range(N_step):   # gives batch data
		loaderIters = [iter(loader) for loader in ADHA_singleloaders]
		for loaderID, loader in enumerate(loaderIters):
			if step >= len(ADHA_singleloaders[loaderID]):
				continue

			(x, expression, y) = loader.next()
			if featureN == 1:
				x = x[:,1,:,:]
				b_x = Variable(x)  # reshape x to (batch, time_step, input_size)
			else:
				x_ = x
				x = []
				for i,x__ in enumerate(x_):
					x.append(np.concatenate(x__.numpy(), axis=1))
				x = np.array(x)
				b_x = Variable(torch.from_numpy(x))  # reshape x to (batch, time_step, input_size)

			b_expression = Variable(expression)
			b_y_action = Variable(y[0])   # batch y
			b_y_adverb = Variable(y[1])

			b_x = b_x.cuda()
			b_expression = b_expression.cuda()
			b_y_action = b_y_action.cuda()
			b_y_adverb = b_y_adverb.cuda()
			if step != 0:
				rnn_2.load_state_dict(torch.load(model_path + '/' + 'params' + str(loaderID) + '.pkl'))


			out_adverb = rnn([b_x.float(),b_expression.float()])               # rnn output
			loss = loss_func(out_adverb, b_y_adverb.float())

			#print(b_y_adverb.data.cpu().numpy()[0])
			map_adverb = mAP(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 51)
			hit_1_adverb = hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 1)
			hit_5_adverb = hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 5)
			print("Epoch: " + str(epoch) + " step: " + str(step) + " loaderID: " + str(loaderID) + " | map_adverb: " + str(map_adverb) + ' | hit_1_adverb: ' + str(hit_1_adverb) + ' | hit_5_adverb: ' + str(hit_5_adverb) )


			optimizer.zero_grad()           # clear gradients for this training step
			loss.backward()                 # backpropagation, compute gradients
			optimizer.step()                # apply gradients


			torch.save(rnn_2.state_dict(), model_path + '/' + 'params' + str(loaderID) + '.pkl')

			if epoch% saveInter == 0:
				torch.save(rnn_1.state_dict(), model_path + '/' + 'params.pkl')