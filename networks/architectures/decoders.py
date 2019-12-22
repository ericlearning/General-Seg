import torch
import torch.nn as nn
from utils.network_utils import *
from networks.architectures.base_modules import *

class UNetDecoder(nn.Module):
	def __init__(self, opt, nf):
		super(UNetDecoder, self).__init__()
		ic, oc, norm_type, act_type, mode = \
			opt.ic, opt.oc, opt.norm_type, opt.act_type, opt.dec_mode

		self.u1 = UpSample('TC', nf)
		self.u2 = UpSample('TC', nf//2)
		self.u3 = UpSample('TC', nf//4)
		self.u4 = UpSample('TC', nf//8)
		
		self.blk1 = UNetBlk_Concat(nf, nf//2, nf//2, norm_type, act_type, False, mode)
		self.blk2 = UNetBlk_Concat(nf//2, nf//4, nf//4, norm_type, act_type, False, mode)
		self.blk3 = UNetBlk_Concat(nf//4, nf//8, nf//8, norm_type, act_type, False, mode)
		self.blk4 = UNetBlk_Concat(nf//8, nf//16, nf//16, norm_type, act_type, False, mode)
		self.blk5 = nn.Conv2d(nf//16, oc, 1, 1, 0, True)

	def forward(self, e):
		out = self.blk1(self.u1(e[4]), e[3])
		out = self.blk2(self.u2(out), e[2])
		out = self.blk3(self.u3(out), e[1])
		out = self.blk4(self.u4(out), e[0])
		out = self.blk5(out)
		return out

class PSPNetDecoder(nn.Module):
	def __init__(self, opt, nf):
		super(PSPNetDecoder, self).__init__()
		ic, oc, norm_type, act_type, mode = \
			opt.ic, opt.oc, opt.norm_type, opt.act_type, opt.dec_mode
		self.branch1 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(nf, nf, 1, 1, 0)
		)
		self.branch2 = nn.Sequential(
			nn.AdaptiveAvgPool2d(2),
			nn.Conv2d(nf, nf, 1, 1, 0)
		)
		self.branch3 = nn.Sequential(
			nn.AdaptiveAvgPool2d(3),
			nn.Conv2d(nf, nf, 1, 1, 0)
		)
		self.branch4 = nn.Sequential(
			nn.AdaptiveAvgPool2d(6),
			nn.Conv2d(nf, nf, 1, 1, 0)
		)
		self.branch5 = nn.Conv2d(nf, nf, 1, 1, 0)

		self.aux = nn.Sequential(
			nn.Conv2d(nf//2, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, oc, 1, 1, 0)
		)
		self.final = nn.Sequential(
			nn.Conv2d(nf*5, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, oc, 1, 1, 0)
		)

	def forward(self, e):
		aux, main = e[3], e[4]
		size = (main.shape[-2], main.shape[-1])
		b1 = F.interpolate(self.branch1(main), size=size)
		b2 = F.interpolate(self.branch2(main), size=size)
		b3 = F.interpolate(self.branch3(main), size=size)
		b4 = F.interpolate(self.branch4(main), size=size)
		out = torch.cat([b1, b2, b3, b4, main], dim = 1)
		out = self.final(out)
		aux = self.aux(aux)
		return aux, out

class PAM(nn.Module):
	def __init__(self, nf):
		super(PAM, self).__init__()
		self.query_conv = nn.Conv2d(nf, nf//8, 1, 1, 0)
		self.key_conv = nn.Conv2d(nf, nf//8, 1, 1, 0)
		self.value_conv = nn.Conv2d(nf, nf, 1, 1, 0)
		self.alpha = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		bs, nf, h, w = x.shape
		query = self.query_conv(x).view(bs, nf//8, h*w)
		key = self.key_conv(x).view(bs, nf//8, h*w).permute(0, 2, 1)
		value = self.value_conv(x).view(bs, nf, h*w)

		out = torch.softmax(torch.bmm(key, query), dim = -1).permute(0, 2, 1)
		out = torch.bmm(value, out).reshape(bs, nf, h, w)
		out = x + self.alpha * out
		return out

class CAM(nn.Module):
	def __init__(self, nf):
		super(CAM, self).__init__()
		self.query_conv = nn.Conv2d(nf, nf//8, 1, 1, 0)
		self.key_conv = nn.Conv2d(nf, nf//8, 1, 1, 0)
		self.value_conv = nn.Conv2d(nf, nf, 1, 1, 0)
		self.alpha = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		bs, nf, h, w = x.shape
		query = x.view(bs, nf, h*w)
		key = x.view(bs, nf, h*w).permute(0, 2, 1)
		value = x.view(bs, nf, h*w)

		out = torch.softmax(torch.bmm(query, key), dim = -1)
		out = torch.bmm(out, value).reshape(bs, nf, h, w)
		out = x + self.alpha * out
		return out

class DANetDecoder(nn.Module):
	def __init__(self, opt, nf):
		super(DANetDecoder, self).__init__()
		ic, oc, norm_type, act_type, mode  = \
			opt.ic, opt.oc, opt.norm_type, opt.act_type, opt.dec_mode
		self.PAM_branch = nn.Sequential(
			nn.Conv2d(nf, nf//4, 3, 1, 1),
			nn.BatchNorm2d(nf//4), nn.ReLU(),
			PAM(nf//4),
			nn.Conv2d(nf//4, nf//4, 3, 1, 1),
			nn.BatchNorm2d(nf//4), nn.ReLU()
		)
		self.CAM_branch = nn.Sequential(
			nn.Conv2d(nf, nf//4, 3, 1, 1),
			nn.BatchNorm2d(nf//4), nn.ReLU(),
			CAM(nf//4),
			nn.Conv2d(nf//4, nf//4, 3, 1, 1),
			nn.BatchNorm2d(nf//4), nn.ReLU()
		)

		self.PAM_aux = nn.Conv2d(nf//4, oc, 1, 1, 0)
		self.CAM_aux = nn.Conv2d(nf//4, oc, 1, 1, 0)
		self.up_p_aux = UpSample('BI', m = 8)
		self.up_c_aux = UpSample('BI', m = 8)

		if(mode == 'fullres'):
			self.final = nn.Conv2d(nf//4, nf//4, 3, 1, 1)
			self.upsample = nn.Sequential(
				UpSample('BI'), UNetBlk(nf//4, nf//4, norm_type, act_type, False),
				UpSample('BI'), UNetBlk(nf//4, nf//8, norm_type, act_type, False),
				UpSample('BI'), UNetBlk(nf//8, nf//16, norm_type, act_type, False),
				nn.Conv2d(nf//16, oc, 1, 1, 0)
			)

		else:
			self.final = nn.Conv2d(nf//4, oc, 1, 1, 0)
			self.upsample = UpSample('BI', m=8)


	def forward(self, x):
		p = self.PAM_branch(x[-1])
		p_aux = self.PAM_aux(p)
		c = self.CAM_branch(x[-1])
		c_aux = self.CAM_aux(c)
		out = self.final(p + c)

		p_aux, c_aux, out = self.up_p_aux(p_aux), self.up_c_aux(c_aux), self.upsample(out)
		return p_aux, c_aux, out
