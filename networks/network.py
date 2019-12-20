import torch
import torch.nn as nn
import torch.optim as optim
from utils.network_utils import *
from scheduler.scheduler import LinearDecay
from networks.architectures.models import *
from loss.loss import *

class Segmentation_Network(nn.Module):
	def __init__(self, opt, device):
		super(Segmentation_Network, self).__init__()
		self.opt = opt
		self.dec_type = opt.dec_type
		if(self.dec_type == 'unet'):
			self.M, ws = UNet(opt), [1.0]
		elif(self.dec_type == 'pspnet'):
			self.M, ws = PSPNet(opt), [1.0, 0.4]
		elif(self.dec_type == 'danet'):
			self.M, ws = DANet(opt), [1.0, 1.0, 1.0]
		self.CELoss = CrossEntropyLoss(opt, ws)

	def initialize_optimizers(self, iter_num):
		lr, beta1, beta2 = self.opt.lr, self.opt.beta1, self.opt.beta2
		model_params = list(self.M.parameters())

		opt = optim.Adam(model_params, lr = lr, betas = (beta1, beta2))
		opt = LinearDecay(self.opt, opt, iter_num)

		return opt

	def forward(self, inputs):
		x, y = inputs
		y_fake = self.generate(x)
		err = self.CELoss(y_fake, y)
		return err
		
	def generate(self, x):
		return self.M(x)