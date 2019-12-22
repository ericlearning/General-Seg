import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model
from utils.network_utils import get_norm, get_activation

class UNetBlk(nn.Module):
	def __init__(self, ic, oc, norm_type, act_type, bias = False):
		super(UNetBlk, self).__init__()
		self.blk1 = nn.Sequential(
			nn.Conv2d(ic, oc, 3, 1, 1, bias = bias),
			get_norm(norm_type, oc),
			get_activation(act_type)
		)
		self.blk2 = nn.Sequential(
			nn.Conv2d(oc, oc, 3, 1, 1, bias = bias),
			get_norm(norm_type, oc),
			get_activation(act_type)
		)

	def forward(self, x):
		return self.blk2(self.blk1(x))

class UNetBlk_Concat(nn.Module):
	def __init__(self, ic, oc, ie, norm_type, act_type, bias = False, mode = 'concat'):
		super(UNetBlk_Concat, self).__init__()
		self.mode = mode
		if(self.mode == 'concat'):
			self.conv = nn.Conv2d(ic + ie, oc, 3, 1, 1, bias = bias)
		elif(self.mode == 'add'):
			self.conv = nn.Conv2d(ic, oc, 3, 1, 1, bias = bias)

		self.blk1 = nn.Sequential(
			self.conv,
			get_norm(norm_type, oc),
			get_activation(act_type)
		)
		self.blk2 = nn.Sequential(
			nn.Conv2d(oc, oc, 3, 1, 1, bias = bias),
			get_norm(norm_type, oc),
			get_activation(act_type)
		)

	def forward(self, x, e):
		if(self.mode == 'concat'):
			return self.blk2(self.blk1(torch.cat([x, e], dim = 1)))
		elif(self.mode == 'add'):
			return self.blk2(self.blk1(x + e))

class DownSample(nn.Module):
	def __init__(self, ds_type = 'MP'):
		super(DownSample, self).__init__()
		self.ds_type = ds_type
		if(self.ds_type == 'MP'):
			self.blk = nn.MaxPool2d(2)
		elif(self.ds_type == 'AP'):
			self.blk = nn.AvgPool2d(2)

	def forward(self, x):
		return self.blk(x)

class UpSample(nn.Module):
	def __init__(self, ds_type = 'TC', ic = None, m = 2):
		super(UpSample, self).__init__()
		self.ds_type, self.m = ds_type, m
		if(ic is not None):
			self.blk = nn.ConvTranspose2d(ic, ic, 2, 2, 0)

	def forward(self, x):
		if(self.ds_type == 'BI'):
			return F.interpolate(x, None, self.m, 'bilinear', True)
		elif(self.ds_type == 'NN'):
			return F.interpolate(x, None, self.m, 'nearest')
		elif(self.ds_type == 'TC'):
			return self.blk(x)