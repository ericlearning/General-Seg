import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return nn.BatchNorm2d(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm2d(size)
	elif(norm_type == None):
		return Nothing()

def get_activation(activation_type):
	if(activation_type == 'relu'):
		return nn.ReLU()
	elif(activation_type == 'leakyrelu'):
		return nn.LeakyReLU(0.2)
	elif(activation_type == 'elu'):
		return nn.ELU()
	elif(activation_type == 'selu'):
		return nn.SELU()
	elif(activation_type == 'prelu'):
		return nn.PReLU()
	elif(activation_type == 'tanh'):
		return nn.Tanh()
	elif(activation_type == None):
		return Nothing()

def get_transformations(opt):
	h, w, ic, oc = opt.height, opt.width, opt.ic, opt.oc
	dt_train = {
		'input' : transforms.Compose([
			transforms.Resize((h, w)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'target' : transforms.Compose([
			transforms.Resize((h, w), interpolation = Image.NEAREST),
			LabelToTensor()
		])
	}
	dt_val = {
		'input' : transforms.Compose([
			transforms.Resize((h, w)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'target' : transforms.Compose([
			transforms.Resize((h, w), interpolation = Image.NEAREST),
			LabelToTensor()
		])
	}
	return dt_train, dt_val

class LabelToTensor(object):
	def __init__(self):
		pass

	def __call__(self, x):
		return torch.LongTensor(np.array(x))

class Denormalize(object):
	def __init__(self, mean, var):
		self.mean = np.array(mean).reshape(3, 1, 1)
		self.var = np.array(var).reshape(3, 1, 1)

	def __call__(self, x):
		return x * self.var + self.mean

def get_nf(net_type):
	nf = {
		'unet': 1024,
		'resnet18': 512,
		'resnet34': 512,
		'resnet50': 2048,
		'resnet101': 2048,
		'resnet152': 2048
	}
	return nf[net_type]

def resize(x, scale):
	out = x
	if(scale > 1):
		out = F.adaptive_avg_pool2d(x, (x.shape[2] // scale, x.shape[3] // scale))
	return out

def expand_and_concat(x1, x2):
	return torch.cat([x1, x2.expand(-1, -1, x1.shape[2], x1.shape[3])], 1)

class Nothing(nn.Module):
	def __init__(self):
		super(Nothing, self).__init__()
		
	def forward(self, x):
		return x

def xavier_init(m):
	if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
		nn.init.xavier_normal_(m.weight)

def normal_init(m, v = 0.02):
	if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
		m.weight.data.normal_(0.0, v)
		if(m.bias is not None):
			m.bias.data.zero_()

def normalize(x, norm):
	mean, std = norm
	norm_x = (x + 1) / 2.0
	norm_x = (norm_x - mean) / std
	return norm_x

class CustomDataParallel(nn.Module):
	def __init__(self, m):
		super(CustomDataParallel, self).__init__()
		self.m = nn.DataParallel(m)

	def forward(self, *x):
		return self.m(*x)

	def __getattr__(self, attr):
		try:
			return super().__getattr__(attr)
		except:
			return getattr(self.m.module, attr)

def split(x, num):
	if(num > 1):
		bs = x.shape[0]
		return x.split(bs // num, dim = 0)
	else:
		return [x]

