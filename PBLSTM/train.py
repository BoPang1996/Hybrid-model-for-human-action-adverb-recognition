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
batch_size = 256
featureN = 1
withexpression = True
model_path = '/Disk8/poli/models/ADHA/pose/task1_feature2'
resume = True


saveInter = 2
feature_path = '/Disk8/HMDB/pose_feature'
train_test_split_path = '/Disk8/HMDB/train_test_split'
label_path = '/Disk8/HMDB/labels/result'
expression_path = None

torch.cuda.set_device(1)

rnn = LSTM_T1(featureN, withexpression)


if resume:
    print("loading model")
    rnn.load_state_dict(torch.load(model_path + '/' + 'params.pkl'))
rnn.cuda()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
#loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
loss_func = nn.MSELoss()

ADHALoader = torch.utils.data.DataLoader(
         ADHA(feature_path=feature_path , label_path=label_path, train_test_split_path=train_test_split_path,expression_path=expression_path, train=True),
         batch_size= batch_size, shuffle= True, num_workers= 2)

# training and testing
for epoch in range(EPOCH):
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
		

        out_action, out_adverb = rnn(b_x.float(), b_expression.float())               # rnn output
        loss_action = loss_func(out_action, b_y_action.float())   # cross entropy loss
        loss_adverb = loss_func(out_adverb, b_y_adverb.float())
        loss= loss_adverb + loss_action
        #print(b_y_adverb.data.cpu().numpy()[0])
        map_action = mAP(out_action.data.cpu().numpy(), b_y_action.data.cpu().numpy(),32)
        map_adverb = mAP(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 51)
        hit_1_action = hit_k(out_action.data.cpu().numpy(), b_y_action.data.cpu().numpy(),1)
        hit_1_adverb = hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 1)
        hit_5_action = hit_k(out_action.data.cpu().numpy(), b_y_action.data.cpu().numpy(), 5)
        hit_5_adverb = hit_k(out_adverb.data.cpu().numpy(), b_y_adverb.data.cpu().numpy(), 5)
        print("Epoch: " + str(epoch) + " step: " + str(step) + " mAP_action: " + str(map_action) + " | map_adverb: " + str(map_adverb) + " | hit_1_action: " + str(hit_1_action) + ' | hit_1_adverb: ' + str(hit_1_adverb) + " | hit_5_action: " + str(hit_5_action) + ' | hit_5_adverb: ' + str(hit_5_adverb) )


        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if epoch% saveInter == 0:
            torch.save(rnn.state_dict(), model_path + '/' + 'params.pkl')