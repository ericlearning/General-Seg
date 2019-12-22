import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from PIL import Image
from albumentations import *
from utils.transformations import *

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

def get_transformations_old(opt):
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

def get_transformations(opt):
	h, w, ic, oc = opt.height, opt.width, opt.ic, opt.oc
	dt_train = Compose([
		HorizontalFlip(),
		OneOf([
			GridDistortion(distort_limit=0.2),
			ElasticTransform(),
			OpticalDistortion(distort_limit=0.5, shift_limit=0.4)
		], p=0.3),
		CLAHE(p=0.3),
		RandomBrightnessContrast(0.1, 0.1),
		RandomGamma(),
		HueSaturationValue(),
		ShiftScaleRotate(rotate_limit=15),
		Resize(256, 256),
	])
	dt_val = Compose([
		Resize(256, 256),
	])
	return dt_train, dt_val

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

def IOU(y, y_fake, n_classes, ignore_index):
	per_class_iou = []
	cnt = 0
	y_fake = y_fake.argmax(dim = 1)
	for i in range(n_classes):
		if(i == ignore_index):
			continue
		cnt += 1
		class_in_y_fake = (y_fake == i)
		class_in_y = (y == i)
		intersection = (class_in_y_fake & class_in_y).sum()
		union = (class_in_y_fake | class_in_y).sum()
		iou = 1 if(union == 0) else float(intersection) / float(union)
		per_class_iou.append(iou)
	mean_iou = sum(per_class_iou) / cnt
	return mean_iou, per_class_iou

def freeze(parameters):
	for param in parameters:
		param.requires_grad = False

def str2list(s):
	return list(map(float, s.split(',')))

def str2list_2(s):
	s = s.split(',')
	len_s = len(s)
	if(len_s == 1):
		epoch_num = int(s[0])
		return (epoch_num, 'normal')
	elif(len_s == 2):
		epoch_num, decay_start_epoch = int(s[0]), int(s[1])
		return (epoch_num, decay_start_epoch, 'decay')
	elif(len_s == 3):
		if('.' not in s[2]):
			cycle_num, cycle_len, cycle_mult = int(s[0]), int(s[1]), int(s[2])
			return (cycle_num, cycle_len, cycle_mult, 'cosine_annealing')
		else:
			cycle_num, epoch_num, div = int(s[0]), int(s[1]), float(s[2])
			return (cycle_num, epoch_num, div, 'clr')

def get_total_epoch(s):
	schedule_type = s[-1]
	if(schedule_type == 'normal' or schedule_type == 'decay'):
		return s[0]
	elif(schedule_type == 'cosine_annealing'):
		return s[1] * (s[2] ** s[0] - 1) // (s[2] - 1)
	elif(schedule_type == 'clr'):
		return s[0] * s[1]

