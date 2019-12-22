import os, cv2
import pickle
import seaborn
import torch
import torch.nn as nn
import numpy as np
from utils.network_utils import *
from utils.transformations import *

def get_display_samples(samples, n_x, n_y):
	h = samples[0].shape[0]
	w = samples[0].shape[1]
	nc = samples[0].shape[2]
	display = np.zeros((h*n_y, w*n_x, nc))
	for i in range(n_y):
		for j in range(n_x):
			cur_sample = cv2.cvtColor((samples[i*n_x+j]*255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
			display[i*h:(i+1)*h, j*w:(j+1)*w, :] = cur_sample
	return display

def display_label(label, palette):
	h, w = label.shape
	rgb_label = np.zeros((h, w, 3))
	for i, color in enumerate(palette):
		rgb_label[label == i] = color
	return rgb_label

def generate_palette(n_classes):
	palette = seaborn.color_palette('hls', n_colors = n_classes)
	return palette

def get_sample_images_list(trainer, val_data, n_classes):
	device = trainer.device
	dec_type = trainer.network.dec_type
	val_data = list(val_data)[0]
	denormalize = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	palette = generate_palette(n_classes)

	sample_input_images = val_data[0]
	sample_output_images = val_data[1]

	sample_fake_images = trainer.network.generate(sample_input_images.to(device))[0]
	sample_fake_images = sample_fake_images.detach().cpu().numpy() # (bs, n_classes, h, w)
	sample_input_images = sample_input_images.numpy() # (bs, c, h, w)
	sample_output_images = sample_output_images.numpy()  # (bs, h, w)
	
	sample_input_images_list = []
	sample_output_images_list = []
	sample_fake_images_list = []
	sample_images_list = []

	for j in range(3):
		cur_img_fake = display_label(sample_fake_images[j].argmax(axis=0), palette)
		cur_img_output = display_label(sample_output_images[j], palette)
		cur_img_input = denormalize(sample_input_images[j])
		sample_fake_images_list.append(cur_img_fake)
		sample_input_images_list.append(cur_img_input.transpose(1, 2, 0))
		sample_output_images_list.append(cur_img_output)

	sample_images_list.extend(sample_input_images_list)
	sample_images_list.extend(sample_fake_images_list)
	sample_images_list.extend(sample_output_images_list)

	return sample_images_list
