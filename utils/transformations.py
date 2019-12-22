import os, cv2
import random
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms
from PIL import Image

class LabelToTensor(object):
	def __init__(self):
		pass

	def __call__(self, x):
		return torch.LongTensor(x)

class Denormalize(object):
	def __init__(self, mean, std):
		self.mean = np.array(mean).reshape(3, 1, 1)
		self.std = np.array(std).reshape(3, 1, 1)

	def __call__(self, x):
		return x * self.std + self.mean

class CustomToTensor(object):
	def __init__(self, mean, std):
		self.img_part = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		])
		self.label_part = transforms.Compose([
			LabelToTensor()
		])

	def __call__(self, x):
		x['image'] = self.img_part(x['image'])
		x['mask'] = self.label_part(x['mask'])
		return x