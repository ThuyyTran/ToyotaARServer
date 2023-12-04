import os
import shutil
import numpy as np
from tqdm import tqdm
import csv
import time
import pickle
import cv2
import torch
from u2net.data_loader import RescaleT, ToTensor, ToTensorLab,SalObjDataset
from u2net.model import U2NET
from helpers import init_cascade_model, find_roi_update
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = U2NET(3,1)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'u2net/saved_models/u2net/u2net.pth')
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint)
net.to(device)
net.eval()

def normPRED(d):
		ma = torch.max(d)
		mi = torch.min(d)

		dn = (d-mi)/(ma-mi)

		return dn


def find_box(img, mask):
	h, w = img.shape[:2]
	full_area = h*w
	_, mask_thresold = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(mask_thresold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	boxs = []
	area_contours = []
	for contour in contours:
		area = cv2.contourArea(contour)
		area_contours.append(area)
	max_area = max(area_contours)
	for  i,contour in enumerate(contours):
		area = area_contours[i]
		if area/ max_area >=0.5 or area / full_area >=0.05:
			x,y,w,h = cv2.boundingRect(contour)
			boxs.append([x,y,x+w, y+h])
			# continue

	return boxs



def images2bbxs(images, base_source_folder = '', base_save_folder = '', block_size = 5000, batch_size=4): #paths
	# select image file
	image_endswith = ['.jpg', '.png', '.jpeg', '.gif']
	select_images = []
	for image_file in images:
		image_lower = image_file.lower()
		for endswith in image_endswith:
			if endswith in image_lower:
				select_images.append(image_file)
				break

	if len(images) % block_size == 0:
		num_block = len(select_images) // block_size
	else:
		num_block = len(select_images) // block_size + 1
	boxs = []
	new_images = []
	errors_img = []
	for block_id in tqdm(range(num_block)):
		current_paths = select_images[block_id * block_size: (block_id + 1) * block_size]
		current_paths = [os.path.join(base_source_folder, path) for path in current_paths]

		img_datasets = SalObjDataset(
			img_name_list = current_paths,
			lbl_name_list = [],
			transform = transforms.Compose([RescaleT(320),
				ToTensorLab(flag=0)
				])
			)
		img_loader = DataLoader(img_datasets, batch_size = batch_size, shuffle=False, num_workers = 6)
		pred_list = []
		start = time.time()
		for i, inputs in tqdm(enumerate(img_loader)):
			imgs = inputs['image']

			imgs = imgs.type(torch.FloatTensor)
			imgs = Variable(imgs.to(device))
			d1,d2,d3,d4,d5,d6,d7 = net(imgs)
			preds = d1[:, 0,:,:]
			del d1, d2,d3,d4,d5,d6,d7
			preds = normPRED(preds)
			if len(preds) > (batch_size - 1):
				preds = preds.data.squeeze()
			pred_list.extend(preds)
		for j, pred in tqdm(enumerate(pred_list)):
			try:
				pred = np.array(pred.data.cpu())
				path = current_paths[j]
				im = Image.fromarray(pred*255).convert('L')

				image = cv2.imread(path,0)
				mask = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
				mask = np.array(mask)

				# make save folder
				raw_image_path = path.replace(f'{base_source_folder}/', '')
				raw_image_name = os.path.basename(raw_image_path)
				raw_image_folder = os.path.dirname(raw_image_path)
				save_folder = os.path.join(base_save_folder, raw_image_folder)
				if not os.path.exists(save_folder):
					os.makedirs(save_folder)
				return_boxs = find_box(image, mask)

				if len(return_boxs) == 0:
					positions = np.where(mask > 127)
					if len(positions[0] > 0):
						ymin = min(positions[0])
						xmin = min(positions[1])
						ymax = max(positions[0])
						xmax = max(positions[1])
					if xmin == xmax or ymin == ymax:
						xmin = 0
						ymin = 0
						xmax = image.shape[1]
						ymax = image.shape[0]
					return_boxs = [[xmin, ymin, xmax, ymax]]

				boxs.append(return_boxs)
			except:
				img_pil = Image.open(path)
				w,h = img_pil.size
				return_boxs = [[0,0, w, h]]
				boxs.append(return_boxs)
				errors_img.append(path)
			img = cv2.imread(path)
			for k, (xmin, ymin, xmax, ymax) in enumerate(return_boxs):
				img_crop = img[ymin: ymax, xmin: xmax]
				new_name = os.path.join(save_folder, f'{k}_{raw_image_name}')
				try:
					cv2.imwrite(os.path.join(save_folder, f'{k}_{raw_image_name}'), img_crop)
					new_images.append(new_name)
				except:
					errors_img.append(path)

	return new_images, errors_img