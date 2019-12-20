import os
import torch
import random
import pickle
import numpy as np
from utils.network_utils import get_transformations
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Dataset():
	def __init__(self, opt):
		self.train_dir = [opt.trn_src_pth, opt.trn_trg_pth]
		self.val_dir = [opt.val_src_pth, opt.val_trg_pth]
		self.n_classes = opt.oc
		self.dt_trn, self.dt_val = get_transformations(opt)
		self.num_workers = opt.num_workers

	def get_loader(self, bs):
		input_transform_trn = self.dt_trn['input']
		target_transform_trn = self.dt_trn['target']
		input_transform_val = self.dt_val['input']
		target_transform_val = self.dt_val['target']

		trn_dataset = Segmentation(self.train_dir[0], self.train_dir[1], input_transform_trn, target_transform_trn, self.n_classes)
		val_dataset = Segmentation(self.val_dir[0], self.val_dir[1], input_transform_val, target_transform_val, self.n_classes)

		trn_loader = DataLoader(trn_dataset, batch_size = bs, shuffle = True, num_workers = self.num_workers)
		val_loader = DataLoader(val_dataset, batch_size = 3, shuffle = False, num_workers = self.num_workers)

		returns = (trn_loader, val_loader)
		return returns

class Segmentation():
	def __init__(self, input_dir, target_dir, input_transform, target_transform, n_classes):
		self.input_dir = input_dir
		self.target_dir = target_dir
		self.input_transform = input_transform
		self.target_transform = target_transform
		self.n_classes = n_classes

		self.image_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.image_name_list.append(file)

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		input_img = Image.open(os.path.join(self.input_dir, self.image_name_list[idx]))
		target_img = Image.open(os.path.join(self.target_dir, self.image_name_list[idx]))

		input_img = self.input_transform(input_img)
		target_img = self.target_transform(target_img)

		sample = (input_img, target_img)
		return sample
