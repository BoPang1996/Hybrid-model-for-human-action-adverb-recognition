import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
	                 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
		                       padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, nb_classes, withexpression, channel=20):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7)
		if not withexpression:
			self.fc_custom_act = nn.Linear(512 * block.expansion, 32)
			self.fc_custom_adv = nn.Linear(512 * block.expansion, 51)
		else:
			self.fc_custom_act = nn.Linear(512 * block.expansion + 1, 32)
			self.fc_custom_adv = nn.Linear(512 * block.expansion + 1, 51)
		self.softmax_act = nn.Softmax()
		self.softmax_adv = nn.Softmax()

		self.withexpression = withexpression

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x, expression):
		x = self.conv1_custom(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		if not self.withexpression:
			out_act = self.fc_custom_act(x)
			out_adv = self.fc_custom_adv(x)
		else:
			out_act = self.fc_custom_act(torch.cat((x, expression),1))
			out_adv = self.fc_custom_adv(torch.cat((x, expression),1))

		out_act = self.softmax_act(out_act)
		out_adv = self.softmax_act(out_adv)
		return out_act, out_adv

class ResNet_Notop(nn.Module):
	def __init__(self, block, layers, nb_classes, channel=20):
		self.inplanes = 64
		super(ResNet_Notop, self).__init__()
		self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		expression = x[1]
		x = x[0]
		x = self.conv1_custom(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		return [x, expression]

class Classifier(nn.Module):
	def __init__(self, class_N, block, withexpression):
		super(Classifier, self).__init__()
		# the hidden_size is 51
		if not withexpression:
			self.fc_custom_adv = nn.Linear(512 * block.expansion, class_N)
		else:
			self.fc_custom_adv = nn.Linear(512 * block.expansion + 1, class_N)
		self.softmax = nn.Softmax()
		self.withexpression = withexpression

	def forward(self, input):
		expression = input[1]
		input = input[0]
		if not self.withexpression:
			out_adverb = self.fc_custom_adv(input)
			out_adverb = self.softmax(out_adverb)
		else:
			out_adverb = self.fc_custom_adv(torch.cat((input,expression),1))
			out_adverb = self.softmax(out_adverb)
		return out_adverb

def resnet18(pretrained=False, nb_classes=1000, **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [2, 2, 2, 2], nb_classes, **kwargs)
	if pretrained:
		pretrain_dict = model_zoo.load_url(model_urls['resnet18'])

		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	return model


def resnet34(pretrained=False, nb_classes=1000, **kwargs):
	"""Constructs a ResNet-34 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [3, 4, 6, 3], nb_classes ** kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
	return model


def resnet50(pretrained=False, nb_classes=1000, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], nb_classes, **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def resnet101(withexpression, pretrained=False, nb_classes=1000, channel =20, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], nb_classes,withexpression, channel, **kwargs)
	if pretrained:
		pretrain_dict = model_zoo.load_url(model_urls['resnet101'])  # modify pretrain code
		model_dict = model.state_dict()
		model_dict = weight_transform(model_dict, pretrain_dict, channel)
		model.load_state_dict(model_dict)

	return model

def resnet101_t2(pretrained=False, nb_classes=1000, channel =20, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet_Notop(Bottleneck, [3, 4, 23, 3],  nb_classes, **kwargs)
	if pretrained:
		pretrain_dict = model_zoo.load_url(model_urls['resnet101'])  # modify pretrain code
		model_dict = model.state_dict()
		model_dict = weight_transform(model_dict, pretrain_dict, channel)
		model.load_state_dict(model_dict)

	return model

def classifier(withexpression, nb_classes = 51):
	model = Classifier(nb_classes, Bottleneck, withexpression)
	return model


def resnet152(pretrained=False, **kwargs):
	"""Constructs a ResNet-152 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
	return model

def cross_modality_pretrain(conv1_weight, channel):
	# transform the original 3 channel weight to "channel" channel
	S=0
	for i in range(3):
		S += conv1_weight[:,i,:,:]
	avg = S/3.
	new_conv1_weight = torch.FloatTensor(64,channel,7,7)
	#print type(avg),type(new_conv1_weight)
	for i in range(channel):
		new_conv1_weight[:,i,:,:] = avg.data
	return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
	weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
	#print pretrain_dict.keys()
	w3 = pretrain_dict['conv1.weight']
	#print type(w3)
	wt = cross_modality_pretrain(w3,channel)
	weight_dict['conv1_custom.weight'] = wt
	model_dict.update(weight_dict)
	return model_dict