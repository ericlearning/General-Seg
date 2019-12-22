import torch
import torch.nn as nn
import torch.optim as optim
from utils.network_utils import *
from scheduler.scheduler import *
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
		backbone_mode, lr_division = self.opt.backbone_mode, self.opt.lr_division
		model_params = self.M.get_params()

		if(backbone_mode == 'freeze'):
			freeze(sum(model_params[1:], []))
			opt = optim.Adam(model_params[0], lr = lr, betas = (beta1, beta2))
		elif(backbone_mode == 'equal'):
			opt = optim.Adam(sum(model_params, []), lr = lr, betas = (beta1, beta2))
		elif(backbone_mode == 'discriminative'):
			opt = optim.Adam([
				{'params': model_params[0], 'lr': lr / lr_division[0]},
				{'params': model_params[1], 'lr': lr / lr_division[1]},
				{'params': model_params[2], 'lr': lr / lr_division[2]},
				{'params': model_params[3], 'lr': lr / lr_division[3]}
			], betas = (beta1, beta2))

		scheduler_type = self.opt.epoch[-1]
		scheduler_list = {
			'normal': Normal, 'decay': LinearDecay, 
			'cosine_annealing': CosineAnnealingLR, 
			'clr': CyclicLR
		}
		opt = scheduler_list[scheduler_type](self.opt, opt, iter_num)
		return opt

	def forward(self, inputs):
		x, y = inputs
		y_fake = self.generate(x)
		err = self.CELoss(y_fake, y)
		return err, y_fake
		
	def generate(self, x):
		return self.M(x)