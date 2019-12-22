import math

class Normal():
	def __init__(self, opt, optimizer, iter_num):
		self.optimizer = optimizer

	def step(self):
		self.optimizer.step()

	def zero_grad(self):
		self.optimizer.zero_grad()

	def state_dict(self):
		return self.optimizer.state_dict()

class LinearDecay():
	def __init__(self, opt, optimizer, iter_num):
		epoch, decay_start_epoch, _ = opt.epoch
		self.optimizer = optimizer
		self.inits = [p['lr'] for p in self.optimizer.param_groups]
		self.tot, self.st = iter_num * epoch, iter_num * decay_start_epoch
		self.cnt = 0

	def step(self):
		for p, lr in zip(self.optimizer.param_groups, self.inits):
			if(self.cnt < self.st):
				p['lr'] = lr
			else:
				p['lr'] = lr * (1.0 - (self.cnt - self.st) / (self.tot - 1 - self.st))
		self.cnt += 1
		self.optimizer.step()

	def zero_grad(self):
		self.optimizer.zero_grad()

	def state_dict(self):
		return self.optimizer.state_dict()

class CosineAnnealingLR():
	def __init__(self, opt, optimizer, iter_num):
		cycle_num, cycle_len, cycle_mult, _ = opt.epoch
		self.optimizer = optimizer
		self.min_lr = 0.0
		self.inits = [p['lr'] for p in self.optimizer.param_groups]
		
		self.tots = [cycle_len * (cycle_mult ** i) * iter_num for i in range(cycle_num)]
		self.cnt = 0

	def step(self):
		for p, lr in zip(self.optimizer.param_groups, self.inits):
			p['lr'] = self.get_value(self.cnt, lr, self.tots[0])
		self.cnt += 1
		self.conditions()
		self.optimizer.step()

	def conditions(self):
		if(self.cnt >= self.tots[0]):
			self.cnt = 0
			self.tots.pop(0)

	def get_value(self, cnt, base_lr, total):
		dif = base_lr - self.min_lr
		c = math.cos(math.pi * cnt / total)
		return self.min_lr + dif * (1 + c) / 2.0

	def zero_grad(self):
		self.optimizer.zero_grad()

	def state_dict(self):
		return self.optimizer.state_dict()

class CyclicLR():
	def __init__(self, opt, optimizer, iter_num):
		cycle_num, cycle_len, div, _ = opt.epoch
		self.div = div
		self.optimizer = optimizer
		self.min_lr = [p['lr'] / 10.0 for p in self.optimizer.param_groups]
		self.max_lr = [p['lr'] for p in self.optimizer.param_groups]
		
		self.tots = [cycle_len * iter_num] * cycle_num
		self.cnt = 0

	def step(self):
		for p, min_, max_ in zip(self.optimizer.param_groups, self.min_lr, self.max_lr):
			total = self.tots[0]
			div_total = int(total * self.div)
			if(self.cnt < div_total):
				p['lr'] = self.get_value(self.cnt, min_, max_, div_total)
			else:
				p['lr'] = self.get_value(self.cnt - div_total, max_, 0, total - div_total)
		self.cnt += 1
		self.conditions()
		self.optimizer.step()

	def conditions(self):
		if(self.cnt >= self.tots[0]):
			self.cnt = 0
			self.tots.pop(0)

	def get_value(self, cnt, p1, p2, total):
		dif = p1 - p2
		c = math.cos(math.pi * cnt / total)
		return p2 + dif * (1 + c) / 2.0

	def zero_grad(self):
		self.optimizer.zero_grad()

	def state_dict(self):
		return self.optimizer.state_dict()