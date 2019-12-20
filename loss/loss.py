import torch
import torch.nn as nn
from utils.network_utils import *

class CrossEntropyLoss(nn.Module):
	def __init__(self, opt, ws):
		super(CrossEntropyLoss, self).__init__()
		ignore_index = opt.ignore_index
		self.loss = nn.CrossEntropyLoss(ignore_index = ignore_index)
		self.ws = ws

	def forward(self, ys, t):
		loss = 0
		for y, w in zip(ys, self.ws):
			loss += self.loss(y, t) * w
		return loss