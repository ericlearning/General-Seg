import torch
import torch.nn as nn
from utils.network_utils import *
from networks.network import Segmentation_Network

class Segmentation_Trainer():
	def __init__(self, opt, iter_num):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.network = Segmentation_Network(opt, self.device).to(self.device)
		if(opt.multigpu):
			self.network.M = CustomDataParallel(self.network.M)
		self.opt = self.network.initialize_optimizers(iter_num)
		self.grad_acc = opt.grad_acc

	def preprocess_input(self, inputs):
		x, y = inputs[0].to(self.device), inputs[1].to(self.device)
		x, y = split(x, self.grad_acc), split(y, self.grad_acc)
		return x, y

	def step(self, inputs):
		x, y = self.preprocess_input(inputs)
		err = self.M_one_step((x, y))
		return err

	def M_one_step(self, inputs):
		x, y = inputs
		self.network.M.zero_grad()
		for x_, y_ in zip(x, y):
			err = self.network((x_, y_)) / self.grad_acc
			err.backward()
		self.opt.step()
		return err

	def save(self, filename):
		state = {
			'net' : self.network.M.state_dict(),
			'opt' : self.opt.state_dict(),
		}
		torch.save(state, filename)

	def load(self, filename):
		state = torch.load(filename)
		self.network.M.load_state_dict(state['net'])
		self.opt.load_state_dict(state['opt'])