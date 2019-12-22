import os, cv2
import torch
import random
import pickle
import numpy as np
from utils.network_utils import get_transformations
from utils.transformations import CustomToTensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

class Dataset():
	def __init__(self, opt):
		self.train_dir = [opt.trn_src_pth, opt.trn_trg_pth]
		self.val_dir = [opt.val_src_pth, opt.val_trg_pth]
		self.dt_trn, self.dt_val = get_transformations(opt)
		self.num_workers = opt.num_workers

	def get_loader(self, bs):
		trn_dataset = Segmentation(self.train_dir[0], self.train_dir[1], self.dt_trn)
		val_dataset = Segmentation(self.val_dir[0], self.val_dir[1], self.dt_val)

		trn_loader = DataLoader(trn_dataset, batch_size = bs, shuffle = True, num_workers = self.num_workers)
		val_loader = DataLoader(val_dataset, batch_size = 3, shuffle = False, num_workers = self.num_workers)

		returns = (trn_loader, val_loader)
		return returns

class Segmentation():
	def __init__(self, input_dir, target_dir, transforms):
		self.input_dir = input_dir
		self.target_dir = target_dir
		self.transforms = transforms
		self.to_tensor = CustomToTensor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		self.image_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.image_name_list.append(file)

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		input_img = cv2.imread(os.path.join(self.input_dir, self.image_name_list[idx]))
		target_img = cv2.imread(os.path.join(self.target_dir, self.image_name_list[idx]), 0)
		input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

		out = self.to_tensor(self.transforms(image = input_img, mask = target_img))
		input_img = out['image']
		target_img = out['mask']

		sample = (input_img, target_img)
		return sample
