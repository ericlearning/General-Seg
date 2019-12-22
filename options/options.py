import argparse
from utils.network_utils import str2list, str2list_2

def options():
	p = argparse.ArgumentParser(description = 'Arguments for image2image-translation training.')
	p.add_argument('--epoch', type = str2list_2, \
							  help = 'epoch num. writing a single number will give you a normal schedule. ' \
									 'int: normal schedule, (int/int): linear decay, ' \
									 'int, int, int: CosineAnnealingLR, each cycle_num, cycle_len, cycle_mult. ' \
									 'int, int, float: CyclicLR, each cycle_num, epoch_num, div', \
									 default = '200')
	p.add_argument('--bs', type = int, help = 'batch size', default = 4)
	p.add_argument('--lr', type = float, help = 'learning rate', default = 0.003)
	p.add_argument('--beta1', type = float, help = 'beta1 parameter for the Adam optimizer', default = 0.9)
	p.add_argument('--beta2', type = float, help = 'beta2 parameter for the Adam optimizer', default = 0.99)

	p.add_argument('--ic', type = int, help = 'input channel num', default = 3)
	p.add_argument('--oc', type = int, help = 'output channel num', default = 20)
	p.add_argument('--height', type = int, help = 'image height (2^n)', default = 256)
	p.add_argument('--width', type = int, help = 'image width (2^n)', default = 256)

	p.add_argument('--backbone-mode', help = '(freeze/equal/discriminative)', default = 'freeze')
	p.add_argument('--lr-division', type = str2list, help = 'lr division value, only in discriminative mode. requires 4 values.', default = '1, 4, 16, 64')

	p.add_argument('--norm-type', help = 'normalization type', default = 'batchnorm')
	p.add_argument('--act-type', help = 'activation type', default = 'relu')

	p.add_argument('--enc-type', help = 'encoder type (unet/res1/res2)', default = 'unet')
	p.add_argument('--enc-mode', help = 'settings in the encoder based on enc_type, only in res1/2 currently', default = 'unet')

	p.add_argument('--dec-type', help = 'decoder type (unet/pspnet/danet)', default = 'unet')
	p.add_argument('--dec-mode', help = 'settings in the decoder based on dec_type, only in UNet currently', default = 'concat')

	p.add_argument('--print-freq', type = int, help = 'prints the loss value every few iterations', default = 100)
	p.add_argument('--vis-freq', type = int, help = 'saves the visualization every few iterations', default = 100)
	p.add_argument('--vis-pth', help = 'path to save the visualizations', default = 'visualizations/')
	p.add_argument('--model-pth', help = 'path to save the final model', default = 'models/model.pth')

	p.add_argument('--trn-src-pth', help = 'train src dataset path', default = 'data/train/src')
	p.add_argument('--trn-trg-pth', help = 'train trg dataset path', default = 'data/train/trg')
	p.add_argument('--val-src-pth', help = 'val src dataset path', default = 'data/val/src')
	p.add_argument('--val-trg-pth', help = 'val trg dataset path', default = 'data/val/trg')
	p.add_argument('--ignore-index', type = int, help = 'ignore index when training', default = -100)

	p.add_argument('--num-workers', type = int, help = 'num workers for the dataloader', default = 10)
	p.add_argument('--grad-acc', type = int, help = 'split the batch into n steps', default = 1)
	p.add_argument('--multigpu', action = 'store_true', help = 'use multiple gpus')

	args = p.parse_args()
	return args