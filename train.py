import os
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from options.options import options
from dataloaders.seg_dataloader import Dataset
from trainer.trainer import Segmentation_Trainer
from utils.network_utils import *
from utils.visualization_utils import *

opt = options()
ds = Dataset(opt)
trn_dl, val_dl = ds.get_loader(opt.bs)
iter_num, n_classes = len(trn_dl), opt.oc
trainer = Segmentation_Trainer(opt, iter_num)

save_cnt = 0
for epoch in range(opt.epoch):
	for i, data in enumerate(tqdm(trn_dl)):
		err = trainer.step(data)

		if(i % opt.print_freq == 0):
			print('[%d/%d] [%d/%d] err : %.4f'
				  %(epoch+1, opt.epoch, i+1, len(trn_dl), float(err)))

		if(i % opt.vis_freq == 0):
			sample_images_list = get_sample_images_list(trainer, val_dl, n_classes)
			plot_img = get_display_samples(sample_images_list, 3, 3)

			img_fn = str(save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg'
			img_pth = os.path.join(opt.vis_pth, img_fn)
			save_cnt += 1
			cv2.imwrite(img_pth, plot_img)

trainer.save(opt.model_pth)