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
		self.n_classes = opt.oc
		self.ignore_index = opt.ignore_index
		self.grad_acc = opt.grad_acc

	def preprocess_input(self, inputs, split_num):
		x, y = inputs[0].to(self.device), inputs[1].to(self.device)
		x, y = split(x, split_num), split(y, split_num)
		return x, y

	def evaluate(self, val_dl):
		self.network.eval()
		err_total, metric_total = 0, 0
		for data in val_dl:
			x, y = self.preprocess_input(data, 1)
			with torch.no_grad():
				err, y_fake = self.network((x[0], y[0]))
				metric, _ = IOU(y[0], y_fake[0], self.n_classes, self.ignore_index)
				err_total += err
				metric_total += metric
		return err_total / len(val_dl), metric_total / len(val_dl)

	def step(self, inputs):
		self.network.train()
		x, y = self.preprocess_input(inputs, self.grad_acc)
		err = self.M_one_step((x, y))
		return err

	def M_one_step(self, inputs):
		x, y = inputs
		self.network.M.zero_grad()
		for x_, y_ in zip(x, y):
			err, _ = self.network((x_, y_))
			err /= self.grad_acc
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