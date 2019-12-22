import torch
import torch.nn as nn
from utils.network_utils import *
from networks.architectures.base_modules import *
from networks.architectures.constructors.resnet import *
from networks.architectures.encoders import *
from networks.architectures.decoders import *

class UNet(nn.Module):
	def __init__(self, opt):
		super(UNet, self).__init__()
		enc_type = opt.enc_type
		dec_type = opt.dec_type
		if(enc_type == 'unet'):
			self.encoder, nf = UNetEncoder(opt, 64), get_nf('unet')
		elif(enc_type == 'res1'): # Doesn't work, needs a new design
			self.encoder, nf = ResNetEncoder(opt, 'stride1'), get_nf(opt.enc_mode)
		elif(enc_type == 'res2'): # Doesn't work, needs a new design
			self.encoder, nf = ResNetEncoder(opt, 'stride2'), get_nf(opt.enc_mode)

		if(dec_type == 'unet'):
			self.decoder = UNetDecoder(opt, nf)

	def get_params(self):
		p1 = list(self.decoder.parameters())
		p2 = list(self.encoder.blk5.parameters())
		p3 = list(self.encoder.blk4.parameters()) + \
			 list(self.encoder.blk3.parameters())
		p4 = list(self.encoder.blk2.parameters()) + \
			 list(self.encoder.blk1.parameters())
		return [p1, p2, p3, p4]

	def forward(self, x):
		out = self.encoder(x)
		out = self.decoder(out)
		return (out,)

class PSPNet(nn.Module):
	def __init__(self, opt):
		super(PSPNet, self).__init__()
		enc_type = opt.enc_type
		dec_type = opt.dec_type
		self.sz = (opt.height, opt.width)
		if(enc_type == 'res1'):
			self.encoder, nf = ResNetEncoder(opt, 'astrous1'), get_nf(opt.enc_mode)

		if(dec_type == 'pspnet'):
			self.decoder = PSPNetDecoder(opt, nf)

	def get_params(self):
		p1 = list(self.decoder.parameters())
		p2 = list(self.encoder.layer4.parameters())
		p3 = list(self.encoder.layer2.parameters()) + \
			 list(self.encoder.layer3.parameters())
		p4 = list(self.encoder.conv.parameters()) + \
			 list(self.encoder.bn.parameters()) + \
			 list(self.encoder.layer1.parameters())
		return [p1, p2, p3, p4]

	def forward(self, x):
		out = self.encoder(x)
		aux, out = self.decoder(out)

		out = F.interpolate(out, size = (self.sz), mode = 'bilinear', align_corners = True)
		aux = F.interpolate(aux, size = (self.sz), mode = 'bilinear', align_corners = True)
		
		return (out, aux)

class DANet(nn.Module):
	def __init__(self, opt):
		super(DANet, self).__init__()
		enc_type = opt.enc_type
		dec_type = opt.dec_type
		self.sz = (opt.height, opt.width)
		if(enc_type == 'res1'):
			self.encoder, nf = ResNetEncoder(opt, 'astrous1'), get_nf(opt.enc_mode)

		if(dec_type == 'danet'):
			self.decoder = DANetDecoder(opt, nf)

	def get_params(self):
		p1 = list(self.decoder.parameters())
		p2 = list(self.encoder.layer4.parameters())
		p3 = list(self.encoder.layer2.parameters()) + \
			 list(self.encoder.layer3.parameters())
		p4 = list(self.encoder.conv.parameters()) + \
			 list(self.encoder.bn.parameters()) + \
			 list(self.encoder.layer1.parameters())
		return [p1, p2, p3, p4]

	def forward(self, x):
		out = self.encoder(x)
		p_aux, c_aux, out = self.decoder(out)

		out = F.interpolate(out, size = (self.sz), mode = 'bilinear', align_corners = True)
		p_aux = F.interpolate(p_aux, size = (self.sz), mode = 'bilinear', align_corners = True)
		c_aux = F.interpolate(c_aux, size = (self.sz), mode = 'bilinear', align_corners = True)

		return (out, p_aux, c_aux)