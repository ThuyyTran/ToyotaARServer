from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import math
import cv2
import copy
from model_superglue.matching import Matching
from model_superglue.utils import (convert_img_to_tensor, make_draw_matches)
from scipy.spatial import distance
from scipy.spatial.distance import euclidean

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
RANSAC_THRESH = 10

class SuperglueMatching:
	def __init__(self, first_img, second_img = None, threshold_area=0.2):
		self.img_first = self.preprocess_image_first(first_img)
		self.gray_first = self.resize_prop_rect(self.img_first, img_size=480)
		self.inp0 = convert_img_to_tensor(self.gray_first, device)
		self.threshold_area = threshold_area
		self.corner_points_img = np.array(
					[[(0, 0), (self.img_first.shape[1], 0), (self.img_first.shape[1], self.img_first.shape[0]), (0, first_img.shape[0])]],
					np.float32)
		config = {
			'superpoint': {
				'nms_radius': 4,
				'keypoint_threshold': 0.005,
				'max_keypoints': 1024
			},
			'superglue': {
				'weights': 'indoor',
				'sinkhorn_iterations': 20,
				'match_threshold': 0.2,
			}
		}
		print('Running inference on device \"{}\"'.format(device))
		if second_img is not None:
			self.gray_second = self.resize_prop_rect(second_img,  img_size=480)
			self.inp1 = convert_img_to_tensor(self.gray_second, device)
		else:
			self.gray_second = None
			self.inp1 = None
		self.matching = Matching(config).eval().to(device)
		self.vis = None

	def preprocess_image_first(self,image0, scale=0.1): 
		#crop to squares
		w= int(image0.shape[1])
		h=int(image0.shape[0])
		if(w>h):
			x= int(w/2 - h/2)
			y=0
			w=h
		else:
			x= 0
			y=int(h/2 - w/2)
			h=w
		image0 = image0[y:y+h, x:x+w]

		#crop 10%
		
		x_new= int(w*scale)
		w_new= int(w*(1-scale*2))
		y_new= int(h*scale)
		h_new= int(h*(1-scale*2))
		image0 = image0[y_new:y_new+h_new, x_new:x_new+w_new]
		return image0

	#find match 
	def find_matches_superglue(self, second_img, debug=True):
		if second_img is not None:
			self.gray_second = self.resize_prop_rect(second_img,  img_size=480)
			self.inp1 = convert_img_to_tensor(self.gray_second, device)
		ret_matches = False
		
		if self.inp1 is None:
			return ret_matches, None, None, None, None
		# print("souce shape" , self.gray_first.shape,self.gray_second .shape )
		pred =self.matching({'image0': self.inp0, 'image1': self.inp1})
		pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
		self.kpts0, self.kpts1 = pred['keypoints0'], pred['keypoints1']
		self.matches, self.conf = pred['matches0'], pred['matching_scores0']
		# Keep the matching keypoints.
		valid = self.matches > -1
		mkpts0 = self.kpts0[valid]
		ret_matches, score_matches = self.is_relevant_superglue()
		if debug:
			self.vis = self.draw_matches()
			cv2.imwrite("output.jpg" , self.vis )
		return ret_matches, score_matches

	@staticmethod
	def resize_prop_rect(src, img_size=720):
		MAX_SIZE = (img_size, img_size)

		xscale = MAX_SIZE[0] / src.shape[0]
		yscale = MAX_SIZE[1] / src.shape[1]
		scale = min(xscale, yscale)
		if scale > 1:
			return src
		dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_LINEAR)
		return dst

	@staticmethod
	def intersect(i_a, i_b, i_c, i_d):
		def ccw(c_a, c_b, c_c):
			return (c_c[1] - c_a[1]) * (c_b[0] - c_a[0]) > (c_b[1] - c_a[1]) * (c_c[0] - c_a[0])

		return ccw(i_a, i_c, i_d) != ccw(i_b, i_c, i_d) and ccw(i_a, i_b, i_c) != ccw(i_a, i_b, i_d)

	def is_convex(self):
			points = self.transformed_corner_points[0]
			for i in range(-4, 0):
				if not self.intersect(points[i], points[i+2], points[i+1], points[i+3]):
					return False
			return True

	@staticmethod
	def angle_of_3_points(a, b, c):
		"""
		Calculate angle abc of 3 points
		:param a:
		:param b:
		:param c:
		:return:
		"""
		ba = a - b
		bc = c - b
		cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
		angle = np.arccos(cosine_angle)
		return np.degrees(angle)

	def angle_conditions(self):
		points = self.transformed_corner_points[0]
		for i in range(0, 4):
			a = points[i % 4]
			b = points[(i+1) % 4]
			c = points[(i+2) % 4]
			angle = self.angle_of_3_points(a, b, c)
			# print("Angle: ", angle)
			if angle > 150 or angle < 30:
				return False
		return True
	
	@staticmethod
	def order_points(pts):
		x_sorted = pts[np.argsort(pts[:, 0]), :]
		left_most = x_sorted[:2, :]
		right_most = x_sorted[2:, :]
		left_most = left_most[np.argsort(left_most[:, 1]), :]
		(tl, bl) = left_most
		D = distance.cdist(tl[np.newaxis], right_most, "euclidean")[0]
		(br, tr) = right_most[np.argsort(D)[::-1], :]
		return np.array([tl, tr, br, bl], dtype="float32")

	def is_relevant_superglue(self):
		score_matches = 0
		valid = self.matches > -1
		mkpts0 = self.kpts0[valid]
		mkpts1 = self.kpts1[self.matches[valid]]
		if len(mkpts0) < 10:
			# print("matches size " , len(mkpts0))
			return False, score_matches
		src_pts = []
		dst_pts = []
		h, w  = self.gray_first.shape[:2]
		# mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
		box_max = [w, h , 0 , 0]
		for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
			# pt_1 = (int(x0), int(y0))
			# pt_2 = (int(x1) , int(y1))
			pt_1 = (x0, y0)
			pt_2 = (x1 , y1)
			src_pts.append(pt_1)
			dst_pts.append(pt_2)
			box_max[0] = min(box_max[0] , x0)
			box_max[1] = min(box_max[1] , y0)
			box_max[2] = max(box_max[2] , x0)
			box_max[3] = max(box_max[3] , y0)
		score_matches = (box_max[2] - box_max[0])*(box_max[3] - box_max[1])/(w*h)
		###########add code here##########################^M
		h2,w2= self.gray_second.shape
		# src_pts2 = []
		# dst_pts2 = []
		box_max2 = [w2, h2 , 0 , 0]

		for (x0, y0), (x1, y1) in zip(mkpts1, mkpts0):
			# pt_1 = (int(x0), int(y0))^M
		   # pt_2 = (int(x1) , int(y1))^M
			pt_1 = (x0, y0)
			pt_2 = (x1 , y1)
			#src_pts2.append(pt_1)
			#dst_pts2.append(pt_2)
			box_max2[0] = min(box_max2[0] , x0)
			#dst_pts2.append(pt_2)
			box_max2[0] = min(box_max2[0] , x0)
			box_max2[1] = min(box_max2[1] , y0)
			box_max2[2] = max(box_max2[2] , x0)
			box_max2[3] = max(box_max2[3] , y0)
		score_matches2 = (box_max2[2] - box_max2[0])*(box_max2[3] - box_max2[1])/(w2*h2)


		if(score_matches2 > score_matches):
			score_matches= score_matches2
			# src_pts=src_pts2
			# dst_pts=dst_pts2
			# box_max=box_max2
			# h=h2
			# w=w2
		##################################################
		if score_matches < self.threshold_area:
			# print("area_matches" , score_matches)
			return False, score_matches
		h_matrix, status = cv2.findHomography( np.float32(src_pts), np.float32(dst_pts), cv2.RANSAC, RANSAC_THRESH)
		self.corner_points_img = np.array(
					[[(box_max[0], box_max[1]), (box_max[2], box_max[1]), (box_max[2], box_max[3]), (box_max[0], box_max[3])]],
					np.float32)
		self.transformed_corner_points = cv2.perspectiveTransform(self.corner_points_img, h_matrix)

		rect = cv2.minAreaRect(self.transformed_corner_points)
		rotated_box = self.order_points(cv2.boxPoints(rect))

		side_length = [euclidean(rotated_box[0], rotated_box[1]), euclidean(rotated_box[0], rotated_box[-1])]
		side_length_ratio = max(side_length) / min(side_length)
		# print("side_length_ratio" ,side_length_ratio)
		# points_4 = self.transformed_corner_points[0]
		# drawing = cv2.cvtColor(self.gray_second, cv2.COLOR_GRAY2RGB)
		# for i in range(-1, 3):
		# 	cv2.line(drawing, (points_4[i][0], points_4[i][1]), (points_4[i + 1][0], points_4[i + 1][1]),
		# 			 (0, 255, 0), 1, cv2.LINE_AA)
		# cv2.imwrite("out_boxes.jpg" ,drawing )
		if  not self.is_convex():
			print("is_ordered or is_convex")
			return False, score_matches

		if not self.angle_conditions():
			print("angle")
			return False, score_matches
		
		return True	, score_matches

	def draw_matches(self):
		image0 = cv2.cvtColor(self.gray_first, cv2.COLOR_GRAY2RGB)
		image1 = cv2.cvtColor(self.gray_second, cv2.COLOR_GRAY2RGB)
		rot0, rot1 = 0, 0
		# Visualize the matches.
		valid = self.matches > -1
		
		mkpts0 = self.kpts0[valid]
		mkpts1 = self.kpts1[self.matches[valid]]
		mconf = self.conf[valid]
		color = cm.jet(mconf)
		text = [
			'SuperGlue',
			'Keypoints: {}:{}'.format(len(self.kpts0), len(self.kpts1)),
			'Matches: {}'.format(len(mkpts0)),
		]
		if rot0 != 0 or rot1 != 0:
			text.append('Rotation: {}:{}'.format(rot0, rot1))

		# Display extra parameter info.
		k_thresh = self.matching.superpoint.config['keypoint_threshold']
		m_thresh = self.matching.superglue.config['match_threshold']
		small_text = [
			'Keypoint Threshold: {:.4f}'.format(k_thresh),
			'Match Threshold: {:.2f}'.format(m_thresh),
			'Image Pair: {}:{}'.format("image1", 'image2'),
		]
		img_match = make_draw_matches(image0, image1, self.kpts0, self.kpts1, mkpts0, mkpts1, color,
			text, small_text=small_text)

		return img_match

def resize_rect(src, size_image=720):
	max_size = (size_image, size_image)

	xscale = max_size[0] / src.shape[0]
	yscale = max_size[1] / src.shape[1]
	scale = min(xscale, yscale)
	if scale > 1:
		return src
	dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_CUBIC )
	return dst
if __name__ == '__main__':
	
	output_dir = Path('dump_match_pairs')
	image0 = cv2.imread('/media/anlab/800gb/kbook_clone/query_images/1d4ca625-3bc7-419d-9c0c-b5799f2d9f1c_cropped.jpg')
	image1 = cv2.imread('/home/anlab/Desktop/input_crop_debug/32c8ac6447e963f8650b.jpg')

	# image0 = cv2.imread('/home/anlab/Desktop/input_crop_debug/92_64c01461-4031-4019-8ac7-c8e70868d078_NG.jpg')
	# image1 = cv2.imread('/home/anlab/Desktop/input_crop_debug/70_6769f355-9e42-440f-bc9a-d4f17d9103c4_NG.jpg')
	gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
	gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	gray0 = resize_rect(gray0)
	gray1 = resize_rect(gray1)

	# gray0 = resize_rect(image0)
	# gray1 = resize_rect(image1)
	M = SuperglueMatching(gray0)
	ret_matches, score = M.find_matches_superglue( gray1, debug=True)

	print("ret_matches", ret_matches, score )
		
	cv2.imwrite("output.jpg" , M.vis)
