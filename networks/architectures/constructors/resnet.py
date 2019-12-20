import torch
import torch.nn as nn
import torchvision.models as model
from utils.network_utils import *
from networks.architectures.base_modules import *

class BasicBlock(nn.Module):
	def __init__(self, ic, oc, s, d):
		super(BasicBlock, self).__init__()
		self.expansion = 1
		self.conv1 = nn.Conv2d(ic, oc, 3, s, d, d, bias = False)
		self.conv2 = nn.Conv2d(oc, oc * self.expansion, 3, 1, 1, 1, bias = False)
		self.bn1 = nn.BatchNorm2d(oc)
		self.bn2 = nn.BatchNorm2d(oc * self.expansion)
		self.relu = nn.ReLU()

		self.learned_skip = (ic != oc * self.expansion or s != 1)
		if(self.learned_skip):
			self.downsample = nn.Sequential(
				nn.Conv2d(ic, oc * self.expansion, 1, s, 0, 1, bias = False),
				nn.BatchNorm2d(oc * self.expansion)
			)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		if(self.learned_skip):
			out_skip = self.downsample(x)
		else:
			out_skip = x
		out = self.relu(out + out_skip)
		return out

class BottleNeck(nn.Module):
	def __init__(self, ic, oc, s, d):
		super(BottleNeck, self).__init__()
		self.expansion = 4
		self.conv1 = nn.Conv2d(ic, oc, 1, 1, 0, 1, bias = False)
		self.conv2 = nn.Conv2d(oc, oc, 3, s, d, d, bias = False)
		self.conv3 = nn.Conv2d(oc, oc * self.expansion, 1, 1, 0, 1, bias = False)
		self.bn1 = nn.BatchNorm2d(oc)
		self.bn2 = nn.BatchNorm2d(oc)
		self.bn3 = nn.BatchNorm2d(oc * self.expansion)
		self.relu = nn.ReLU()

		self.learned_skip = (ic != oc * self.expansion or s != 1)
		if(self.learned_skip):
			self.downsample = nn.Sequential(
				nn.Conv2d(ic, oc * self.expansion, 1, s, 0, 1, bias = False),
				nn.BatchNorm2d(oc * self.expansion)
			)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		if(self.learned_skip):
			out_skip = self.downsample(x)
		else:
			out_skip = x
		out = self.relu(out + out_skip)
		return out

class ResNet(nn.Module):
	def __init__(self, ic, n_repeats, block, s, d):
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(ic, 64, 7, s[0], 3, d[0], bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(3, 2, 1)

		self.layer1, self.expansion = self.make_layer(block, n_repeats[0], 64, 64, s[1], d[1])
		self.layer2, _ = self.make_layer(block, n_repeats[1], 64*self.expansion, 128, s[2], d[2])
		self.layer3, _ = self.make_layer(block, n_repeats[2], 128*self.expansion, 256, s[3], d[3])
		self.layer4, _ = self.make_layer(block, n_repeats[3], 256*self.expansion, 512, s[4], d[4])

	def make_layer(self, block, n_repeat, ic, oc, s, d):
		l = [block(ic, oc, s, d)]
		expansion = l[0].expansion
		prev = oc * expansion
		for i in range(n_repeat - 1):
			l.append(block(prev, oc, 1, 1))
		return nn.Sequential(*l), expansion

	def forward(self, x):
		o1 = self.relu(self.bn1(self.conv1(x)))
		o2 = self.maxpool(o1)
		o3 = self.layer1(o2)
		o4 = self.layer2(o3)
		o5 = self.layer3(o4)
		o6 = self.layer4(o5)

		return out

def weight_copier(src, trg, verify = False):
	src_state_dict, trg_state_dict = src.state_dict(), trg.state_dict()
	for c in src_state_dict:
		if(trg_state_dict.get(c) is not None):
			trg_state_dict[c] = src_state_dict[c].clone()
	trg.load_state_dict(trg_state_dict)

	if(verify):
		with torch.no_grad():
			x = torch.randn(16, 3, 256, 256)
			src_y, trg_y = src(x), trg(x)
			is_equal = torch.equal(src_y, trg_y)
		if(is_equal):
			print('Weight Copying Success')
		else:
			print('Weight Copying Fail')

	return trg

def constructor(name, ic, pretrained = False, s = [2, 1, 2, 2, 2], d = [1, 1, 1, 1, 1]):
	if(name == 'resnet18'):
		net = ResNet(ic, [2, 2, 2, 2], BasicBlock, s, d)
	elif(name == 'resnet34'):
		net = ResNet(ic, [3, 4, 6, 3], BasicBlock, s, d)
	elif(name == 'resnet50'):
		net = ResNet(ic, [3, 4, 6, 3], BottleNeck, s, d)
	elif(name == 'resnet101'):
		net = ResNet(ic, [3, 4, 23, 3], BottleNeck, s, d)
	elif(name == 'resnet152'):
		net = ResNet(ic, [3, 8, 36, 3], BottleNeck, s, d)

	if(pretrained):
		pretrained = getattr(model, name)(pretrained = pretrained)
		net = weight_copier(pretrained, net)
	return net