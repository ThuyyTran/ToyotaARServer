import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import pathlib
from pathlib import Path
def get_histogram_color( image, imsize=320, usingRGB=False, query_check=False, threshold_min1=25, threshold_min2=40, threshold_color=80):
		image = resize_image(image, max_size=imsize)
		h ,w = image.shape[:2]
		img_cv = image.copy()
		# est_x = int(0.05*w)
		# est_y = int(0.05*h)
		# img_cv = image[est_y: h - est_y ,est_x:w-est_x ]
		# if not query_check:
		img_cv = cv2.GaussianBlur(img_cv,(3,3),cv2.BORDER_DEFAULT)
		# mean = cv2.mean(img_cv)
		mean,sdev = cv2.meanStdDev(img_cv)
		mean_color = (mean[0][0] + mean[1][0] + mean[2][0])/3
		min_color = min(mean[0][0], mean[1][0])
		min_color = min(min_color, mean[2][0])
		max_color = max(mean[0][0], mean[1][0])
		max_color = max(max_color, mean[2][0])

		min_std_color = min(sdev[0][0], sdev[1][0])
		min_std_color = min(min_std_color, sdev[2][0])
		max_std_color = max(sdev[0][0], sdev[1][0])
		max_std_color = max(max_std_color, sdev[2][0])
		m_check = (max_color - min_color  + max_std_color - min_std_color)

		mean =   (mean[0] - min_color,mean[1] - min_color,mean[2] - min_color)
		img_transf = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
		hist = cv2.calcHist([img_transf], [0 ], None, [12], [0, 255])

		hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
		return hist, mean , mean_color

def resize_image( image,max_size=320):
	height,width= image.shape[:2]
	if(height>width):
		scale= max_size/height
	else:
		scale= max_size/width
	if scale > 1:
		return image
	width= int(width*scale)
	height=int(height*scale)
	image= cv2.resize(image,(width,height))
	return image

def genHistImages(path_image, img_folder, save_folder):
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	with open(path_image,'r') as f:
		paths_labels_2ndstreet = [line.strip('\n') for line in f.readlines()]
	paths = []
	num = 0
	list_errors = []
	for line in tqdm(paths_labels_2ndstreet):
		path= line.split(',')[0]
		paths.append(path)
	folders = {os.path.dirname(f) for f in paths}
	folders = {os.path.join(save_folder, f) for f in folders}

	for f in folders:
		pathlib.Path(f).mkdir(parents=True, exist_ok=True)

	for j, path in tqdm(enumerate(paths)):
		sub_name = os.path.splitext(path)[0]

		path = os.path.join(img_folder,path)
		output_file_hist = os.path.join(save_folder, sub_name + ".pkl")

		genHist(path, output_file_hist)

def genHist(image, output_file):
  img_cv = cv2.imread(image)
  hist_value, mean, mean_color = get_histogram_color(img_cv)
  with open(output_file, 'wb') as file:
    pickle.dump((hist_value, mean,mean_color), file)

if __name__ == '__main__':

	path_image = "/media/anlab/data-1tb1/revamp_20220413/path.txt"
	img_folder = '/media/anlab/data-1tb1/revamp_crop'
	save_folder= "/media/anlab/data-1tb1/revamp_20220413/histogram/"
	genHist(path_image ,img_folder ,save_folder  )
