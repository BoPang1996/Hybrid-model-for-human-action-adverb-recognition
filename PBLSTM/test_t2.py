import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from model import *
from dataLoader import *
from metrics import *

batch_size = 256
featureN = 2
withexpression = True
model_path = '/Disk8/poli/models/ADHA/pose/task2_feature2'




feature_path = '/Disk8/HMDB/pose_feature'
train_test_split_path = '/Disk8/HMDB/train_test_split'
label_path = '/Disk8/HMDB/labels/result'
expression_path = None
result_save_path = None
shuffle = False

torch.cuda.set_device(0)

rnn_1 = LSTM_T2_1(featureN)
rnn_2 = LSTM_T2_2(withexpression)





rnn_1.cuda()
rnn_2.cuda()

rnn = nn.Sequential(
	rnn_1,
	rnn_2
        )
rnn_1.load_state_dict(torch.load(model_path + '/' + 'params.pkl'))

ADHA_singleloaders = [torch.utils.data.DataLoader(
         ADHA_singleClass(feature_path=feature_path , label_path=label_path, train_test_split_path=train_test_split_path, expression_path=expression_path, train=False,action=i),
         batch_size= batch_size, shuffle=shuffle, num_workers= 2) for i in embedding_action]

# training and testing
map_adverb = []
hit_1_adverb = []
hit_5_adverb = []
map_class = np.zeros(32)
map_N = np.zeros(32)
hit_1_class = np.zeros(32)
hit_1_N = np.zeros(32)
hit_5_class = np.zeros(32)
hit_5_N = np.zeros(32)
N_step = 0
for loader in ADHA_singleloaders:
	if len(loader) > N_step:
		N_step = len(loader)

out = []
label = []
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


		out_adverb = rnn([b_x.float(), b_expression.float()])               # rnn output

		for i in out_adverb.data.cpu().numpy():
			out.append(i)
		for i in b_y_adverb.data.cpu().numpy():
			label.append(i)
		#print(b_y_adverb.data.cpu().numpy()[0])
		map_adverb.append(mAP(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 51))
		map_class[loaderID] += map_adverb[-1]
		map_N[loaderID] += 1
		hit_1_adverb.append(hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 1))
		hit_1_class[loaderID] += hit_1_adverb[-1]
		hit_1_N[loaderID] += 1
		hit_5_adverb.append(hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 5))
		hit_5_class[loaderID] += hit_5_adverb[-1]
		hit_5_N[loaderID] += 1
		print(" step: " + str(step) + " loaderID: " + str(loaderID) + " | map_adverb: " + str(map_adverb[-1]) + ' | hit_1_adverb: ' + str(hit_1_adverb[-1]) + ' | hit_5_adverb: ' + str(hit_5_adverb[-1]) )


print(" | map_adverb: " + str(float(sum(map_adverb))/float(len(map_adverb))) + ' | hit_1_adverb: ' + str(float(sum(hit_1_adverb))/float(len(hit_1_adverb))) + ' | hit_5_adverb: ' + str(float(sum(hit_5_adverb))/float(len(hit_5_adverb))) )
#print(" | map_adverb: " + str(mAP(pred=np.array(out), gt=np.array(label),n_class=51)) + ' | hit_1_adverb: ' + str(hit_k(predict=np.array(out), label=np.array(label), k=1)) + ' | hit_5_adverb: ' + str(hit_k(predict=np.array(out), label=np.array(label), k=5)) )
for ID, action in enumerate(embedding_action):
	print("Action: " + str(action) + "map: " + str(float(map_class[ID])/float(map_N[ID])) + '|' +  "hit@1: " + str(float(hit_1_class[ID])/float(hit_1_N[ID])) + '|' +  "hit@5: " + str(float(hit_5_class[ID])/float(hit_5_N[ID])) )

if result_save_path != None:
    if not shuffle:
	    pickle.dump(out, open(result_save_path + '/' + 'adverb_result.pickle','wb'))
	    pickle.dump(label, open(result_save_path + '/' + 'adverb_label.pickle','wb'))
        print("Save out the result")
    else:
        print("Using shuffle can not save out the result")


