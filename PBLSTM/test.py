import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from model import *
from dataLoader import *
from metrics import *
import pickle

batch_size = 128
task = 1  # 1 for action+adverb; 2 for adverb
featureN = 2
withexpression = True
model_path = '/Disk8/poli/models/ADHA/pose/task1_feature2'


feature_path = '/Disk8/HMDB/pose_feature'
train_test_split_path = '/Disk8/HMDB/train_test_split'
label_path = '/Disk8/HMDB/labels/result'
expression_path = None
result_save_path = None
shuffle = False

torch.cuda.set_device(1)

rnn = LSTM_T1(featureN, withexpression)



rnn.load_state_dict(torch.load(model_path + '/' + 'params.pkl'))

rnn.cuda()

ADHALoader = torch.utils.data.DataLoader(
         ADHA(feature_path=feature_path , label_path=label_path, train_test_split_path=train_test_split_path, expression_path=expression_path, train=False),
         batch_size= batch_size, shuffle= shuffle, num_workers= 2)

# training and testing
map_action = []
map_adverb = []
hit_1_action = []
hit_1_adverb = []
hit_5_action = []
hit_5_adverb = []

outs_adverb = []
labels_adverb = []
outs_action = []
labels_action = []
for step, (x, expression, y) in enumerate(ADHALoader):   # gives batch data
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


    if task == 1:
        out_action, out_adverb = rnn(b_x.float(), b_expression.float())               # rnn output

        for i in out_adverb.data.cpu().numpy():
            outs_adverb.append(i)
        for i in b_y_adverb.data.cpu().numpy():
            labels_adverb.append(i)

        for i in out_adverb.data.cpu().numpy():
            outs_action.append(i)
        for i in b_y_adverb.data.cpu().numpy():
            labels_action.append(i)

        #print(b_y_adverb.data.cpu().numpy()[0])
        map_action.append(mAP(out_action.data.cpu().numpy(), b_y_action.data.cpu().numpy(),32))
        map_adverb.append(mAP(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 51))
        hit_1_action.append(hit_k(out_action.data.cpu().numpy(), b_y_action.data.cpu().numpy(),1))
        hit_1_adverb.append(hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 1))
        hit_5_action.append(hit_k(out_action.data.cpu().numpy(), b_y_action.data.cpu().numpy(), 5))
        hit_5_adverb.append(hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 5))
        print("step: " + str(step) + " mAP_action: " + str(map_action[-1]) + " | map_adverb: " + str(map_adverb[-1]) + " | hit_1_action: " + str(hit_1_action[-1]) + ' | hit_1_adverb: ' + str(hit_1_adverb[-1]) + " | hit_5_action: " + str(hit_5_action[-1]) + ' | hit_5_adverb: ' + str(hit_5_adverb[-1]) )



#print("Total mAP_action: " + str(float(sum(map_action))/float(len(map_action))) + " | map_adverb: " + str(float(sum(map_adverb))/float(len(map_adverb))) + " | hit_1_action: " + str(float(sum(hit_1_action))/float(len(hit_1_action))) + ' | hit_1_adverb: ' + str(float(sum(hit_1_adverb))/float(len(hit_1_adverb))) + " | hit_5_action: " + str(float(sum(hit_5_action))/float(len(hit_5_action))) + ' | hit_5_adverb: ' + str(float(sum(hit_5_adverb))/float(len(hit_5_adverb))) )
print("Total mAP_action: " + str(mAP(np.array(outs_action),np.array(labels_action),32)) + " | map_adverb: " + str(mAP(np.array(outs_adverb),np.array(labels_adverb),51)) + " | hit_1_action: " + str(float(sum(hit_1_action))/float(len(hit_1_action))) + ' | hit_1_adverb: ' + str(float(sum(hit_1_adverb))/float(len(hit_1_adverb))) + " | hit_5_action: " + str(float(sum(hit_5_action))/float(len(hit_5_action))) + ' | hit_5_adverb: ' + str(float(sum(hit_5_adverb))/float(len(hit_5_adverb))) )

if result_save_path != None:
    if not shuffle:
        pickle.dump(outs_action, open(result_save_path + '/' + 'action_result.pickle','wb'))
	    pickle.dump(outs_adverb, open(result_save_path + '/' + 'adverb_result.pickle','wb'))
        pickle.dump(labels_action, open(result_save_path + '/' + 'action_label.pickle','wb'))
	    pickle.dump(labels_adverb, open(result_save_path + '/' + 'adverb_label.pickle','wb'))
        print("Save out the result")
    else:
        print("Using shuffle can not save out the result")