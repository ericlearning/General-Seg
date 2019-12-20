import torch
import torch.nn as nn
from utils.network_utils import *
from networks.architectures.base_modules import *
from networks.architectures.constructors.resnet import constructor

class UNetEncoder(nn.Module):
	def __init__(self, opt, nf):
		super(UNetEncoder, self).__init__()
		ic, oc, norm_type, act_type = \
			opt.ic, opt.oc, opt.norm_type, opt.act_type
		self.blk1 = UNetBlk(ic, nf, norm_type, act_type, False)
		self.blk2 = nn.Sequential(
			DownSample('MP'),
			UNetBlk(nf, nf*2, norm_type, act_type, False)
		)
		self.blk3 = nn.Sequential(
			DownSample('MP'),
			UNetBlk(nf*2, nf*4, norm_type, act_type, False)
		)
		self.blk4 = nn.Sequential(
			DownSample('MP'),
			UNetBlk(nf*4, nf*8, norm_type, act_type, False)
		)
		self.blk5 = nn.Sequential(
			DownSample('MP'),
			UNetBlk(nf*8, nf*16, norm_type, act_type, False)
		)

	def forward(self, x):
		x1 = self.blk1(x)
		x2 = self.blk2(x1)
		x3 = self.blk3(x2)
		x4 = self.blk4(x3)
		x5 = self.blk5(x4)
		return [x1, x2, x3, x4, x5]

class ResNetEncoder(nn.Module):
	def __init__(self, opt, mode):
		super(ResNetEncoder, self).__init__()
		ic, oc, norm_type, act_type, constructor_mode = \
			opt.ic, opt.oc, opt.norm_type, opt.act_type, opt.enc_mode
		self.mode = mode
		if(self.mode == 'stride1'):
			s, d = [2, 1, 2, 2, 2], [1, 1, 1, 1, 1]
		elif(self.mode == 'stride2'):
			s, d = [1, 1, 2, 2, 2], [1, 1, 1, 1, 1]
		elif(self.mode == 'astrous1'):
			s, d = [2, 1, 2, 1, 1], [1, 1, 1, 2, 4]

		resnet = constructor(constructor_mode, ic, True, s, d)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(3, 2, 1)
		self.conv, self.bn = resnet.conv1, resnet.bn1
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4

	def forward(self, x):
		o1 = self.relu(self.bn(self.conv(x)))
		o2 = self.maxpool(o1)
		o3 = self.layer1(o2)
		o4 = self.layer2(o3)
		o5 = self.layer3(o4)
		o6 = self.layer4(o5)

		if(self.mode == 'stride1'):
			# (1, 2, 4, 8, 16)
			return [x, o1, o3, o4, o5]
		elif(self.mode == 'stride2'):
			# (1, 2, 4, 8, 16)
			return [o1, o3, o4, o5, o6]
		elif(self.mode == 'astrous1'):
			# (4, 4, 8, 8, 8)
			return [o2, o3, o4, o5, o6]
