import math

class LinearDecay():
	def __init__(self, opt, optimizer, iter_num):
		self.optimizer = optimizer
		self.init = opt.lr
		self.tot = iter_num * opt.epoch
		self.st = iter_num * opt.decay_start_epoch
		if(self.st < 0): self.st = self.tot
		self.cnt = 0
		self.state_dict = self.optimizer.state_dict()

	def step(self):
		for p in self.optimizer.param_groups:
			if(self.cnt < self.st):
				p['lr'] = self.init
			else:
				p['lr'] = self.init * (1.0 - (self.cnt - self.st) / (self.tot - 1 - self.st))
			self.cnt += 1
		self.optimizer.step()

	def zero_grad(self):
		self.optimizer.zero_grad()

	def state_dict(self):
		return self.state_dict