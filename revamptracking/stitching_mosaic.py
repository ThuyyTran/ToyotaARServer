import cv2
from pathlib import Path
import numpy as np
import os
from numpy.linalg import inv
from shapely.geometry import Polygon

class VideMosaic:
	def __init__(self, first_image ,  output_height_times=2, output_width_times=4, detector_type="sift"):
		"""This class processes every frame and generates the panorama

		Args:
			first_image (image for the first frame): first image to initialize the output size
			output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
			output_width_times (int, optional): determines the output width based on input image width. Defaults to 4.
			detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".
		"""
		self.detector_type = detector_type
		if detector_type == "sift":
			self.detector = cv2.SIFT_create(500)
			self.bf = cv2.BFMatcher()
		elif detector_type == "orb":
			self.detector = cv2.ORB_create(1000)
			self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		self.visualize = False

		self.process_first_frame(first_image)

		self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(
			output_width_times*first_image.shape[1]), first_image.shape[2]))
		self.output_mask =  np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(
			output_width_times*first_image.shape[1])))
		# offset
		self.w_offset = int(self.output_img.shape[0]/2 - first_image.shape[0]/2)
		self.h_offset = int(self.output_img.shape[1]/6 - first_image.shape[1]/2)
		print("  ---------------- ", self.w_offset, self.h_offset, first_image.shape,self.output_img.shape )

		self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
						self.h_offset:self.h_offset+first_image.shape[1], :] = first_image
		self.output_bounding_box  = [self.h_offset ,self.w_offset , self.h_offset+first_image.shape[1] , self.w_offset+first_image.shape[0]]
		self.output_mask = cv2.rectangle(self.output_mask , (self.output_bounding_box[0], self.output_bounding_box[1]) ,
			(self.output_bounding_box[2] , self.output_bounding_box[3]) , (255), -1 )
		self.H_old = np.eye(3)
		self.H_old[0, 2] = self.h_offset
		self.H_old[1, 2] = self.w_offset

	def updateBoxesItemInit(self, item_boxes=None):
		result_corner = {}
		if item_boxes is not None:
			for key in item_boxes.keys():
				box  = item_boxes[key]
				box = [int(box[0] + self.h_offset ), int(box[1] + self.w_offset), int(box[2] + self.h_offset ), int(box[3] + self.w_offset)]
				item_boxes[key] = box
				corners = np.float32([ [box[0], box[1] ], [box[2], box[1]], [ box[2],box[3]], [box[0],box[3]]])
				result_corner[key ] = corners
		else:
			item_boxes = {}
		return item_boxes, result_corner


	def process_first_frame(self, first_image):
		"""processes the first frame for feature detection and description

		Args:
			first_image (cv2 image/np array): first image for feature detection
		"""
		self.frame_prev = first_image
		frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
		self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

	def match(self, des_cur, des_prev):
		"""matches the descriptors

		Args:
			des_cur (np array): current frame descriptor
			des_prev (np arrau): previous frame descriptor

		Returns:
			array: and array of matches between descriptors
		"""
		# matching
		if self.detector_type == "sift":
			pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
			matches = []
			for m, n in pair_matches:
				if m.distance < 0.9*n.distance:
					matches.append(m)

		elif self.detector_type == "orb":
			matches = self.bf.match(des_cur, des_prev)

		# Sort them in the order of their distance.
		matches = sorted(matches, key=lambda x: x.distance)

		# get the maximum of 20  best matches
		matches = matches[:min(len(matches), 200)]
		# Draw first 10 matches.
		if self.visualize:
			match_img = cv2.drawMatches(self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, matches, None,
										flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
			# cv2.imshow('matches', match_img)
		return matches

	def process_frame(self, frame_cur , item_boxes=None , H=None, threshold_angle=10):
		"""gets an image and processes that image for mosaicing

		Args:
			frame_cur (np array): input of current frame for the mosaicing
		"""
		result_corner = {}
		ret = True
		self.frame_cur = frame_cur
		if H is  None:
			frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
			self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

			self.matches = self.match(self.des_cur, self.des_prev)

			if len(self.matches) < 4:
				return

			self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
			# print("homography " , inv(H) , self.H )
			self.H = np.matmul(self.H_old, self.H)
			# TODO: check for bad Homography
			
			ret, vis, transformed_corners = self.warp(self.frame_cur, self.H , threshold_angle=threshold_angle)

			# loop preparation
			self.H_old = self.H
			self.kp_prev = self.kp_cur
			self.des_prev = self.des_cur
			self.frame_prev = self.frame_cur
		else:
			self.H = np.matmul(self.H_old, inv(H))
			# TODO: check for bad Homography

			ret, vis, transformed_corners = self.warp(self.frame_cur, self.H, threshold_angle=threshold_angle)

			# loop preparation
			self.H_old = self.H
			# self.kp_prev = self.kp_cur
			# self.des_prev = self.des_cur
			self.frame_prev = self.frame_cur
		if ret :
			if item_boxes is not None:
				for key in item_boxes.keys():
					box  = item_boxes[key]
					box, corners = self.cvtBoxesframMatrix(box ,self.H_old )
					result_corner[key] = corners
					item_boxes[key] = box
			else:
				item_boxes = {}

		return ret, item_boxes, result_corner, transformed_corners

	@ staticmethod
	def bounding_box_naive(points):
		bot_left_x = min(point[0] for point in points)
		bot_left_y = min(point[1] for point in points)
		top_right_x = max(point[0] for point in points)
		top_right_y = max(point[1] for point in points)
		box = [bot_left_x, bot_left_y, top_right_x, top_right_y]
		return box

	def cvtBoxesframMatrix(self , box, H ):
		corners = np.float32([ [box[0], box[1] ], [box[2], box[1]], [ box[2],box[3]], [box[0],box[3]]])
		corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) )
		box = self.bounding_box_naive(corners)
		return box, corners

	@ staticmethod
	def findHomography(image_1_kp, image_2_kp, matches):
		"""gets two matches and calculate the homography between two images

		Args:
			image_1_kp (np array): keypoints of image 1
			image_2_kp (np_array): keypoints of image 2
			matches (np array): matches between keypoints in image 1 and image 2

		Returns:
			np arrat of shape [3,3]: Homography matrix
		"""
		# taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

		image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
		image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
		for i in range(0, len(matches)):
			image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
			image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

		homography, mask = cv2.findHomography(
			image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

		return homography
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

	def angle_conditions(self , transformed_corner_points, threshold_angle=10):
		points = transformed_corner_points[0]
		for i in range(0, 4):
			a = points[i % 4]
			b = points[(i+1) % 4]
			c = points[(i+2) % 4]
			angle = self.angle_of_3_points(a, b, c)
			# print("Angle: ", angle)
			if abs(90 - angle) > threshold_angle:
				return False
		return True
	def warp(self, frame_cur, H, threshold_angle=10):
		""" warps the current frame based of calculated homography H

		Args:
			frame_cur (np array): current frame
			H (np array of shape [3,3]): homography matrix

		Returns:
			np array: image output of mosaicing
		"""
		transformed_corners = self.get_transformed_corners(frame_cur, H)
		bounding_box = self.bounding_box_naive(np.int32(transformed_corners[0]))
		transformed_corners = np.array(transformed_corners, dtype=np.int32)
		ret = self.angle_conditions(transformed_corners , threshold_angle = threshold_angle)
		if ret:
			
			self.output_bounding_box[0] = min(bounding_box[0] , self.output_bounding_box[0] )
			self.output_bounding_box[1] = min(bounding_box[1] , self.output_bounding_box[1] )
			self.output_bounding_box[2] = max(bounding_box[2] , self.output_bounding_box[2] )
			self.output_bounding_box[3] = max(bounding_box[3] , self.output_bounding_box[3] )
			warped_img = cv2.warpPerspective(
				frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)

			
			warped_img = self.draw_border(warped_img, transformed_corners)

			self.output_img[warped_img > 0] = warped_img[warped_img > 0]
			# output_temp = np.copy(self.output_img)
		
		# self.output_img = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))

		# cv2.imshow('output',  output_temp/255.)
		# cv2.imshow('warped_img',  warped_img)

		return ret, self.output_img, transformed_corners[0]

	@ staticmethod
	def get_transformed_corners(frame_cur, H):
		"""finds the corner of the current frame after warp

		Args:
			frame_cur (np array): current frame
			H (np array of shape [3,3]): Homography matrix 

		Returns:
			[np array]: a list of 4 corner points after warping
		"""
		corner_0 = np.array([0, 0])
		corner_1 = np.array([frame_cur.shape[1], 0])
		corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
		corner_3 = np.array([0, frame_cur.shape[0]])

		corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
		transformed_corners = cv2.perspectiveTransform(corners, H)

		# mask = np.zeros(shape=(output.shape[0], output.shape[1], 1))
		# cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
		# cv2.imshow('mask', mask)

		return transformed_corners

	def draw_border(self, image, corners, color=(0, 0, 0)):
		"""This functions draw rectancle border

		Args:
			image ([type]): current mosaiced output
			corners (np array): list of corner points
			color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).

		Returns:
			np array: the output image with border
		"""
		for i in range(corners.shape[1]-1, -1, -1):
			cv2.line(image, tuple(corners[0, i, :]), tuple(
				corners[0, i-1, :]), thickness=5, color=color)
		return image
def resize_image( image,max_size=720):
		height,width= image.shape[:2]
		if(height>width):
			scale= max_size/height
		else:
			scale= max_size/width

		width= int(width*scale)
		height=int(height*scale)
		image= cv2.resize(image,(width,height) ,  interpolation=cv2.INTER_AREA)
		return image

def get_key(my_dict , val):
	for key, value in my_dict.items():
		if val == value:
			return key
	return None

def cvtBoxestoSize(arr_dict_boxes, scale) :
	# print("boxes" , arr_dict_boxes[0])
	results = arr_dict_boxes.copy()
	for item_image in results:
		for key in item_image.keys():
			box  = item_image[key]
			box = [int(box[0]*scale), int(box[1]*scale), int(box[2]*scale), int(box[3]*scale)]
			item_image[key] = box
	# print("boxes" , arr_dict_boxes[0])
	return results

def IOU(pol1_xy, pol2_xy):
	# Define each polygon
		polygon1_shape = Polygon(pol1_xy)
		polygon2_shape = Polygon(pol2_xy)
		# print("polygon1_shape " , polygon1_shape)
		# print("polygon2_shape " , polygon2_shape)
		# Calculate intersection and union, and tne IOU
		polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
		polygon_union2 = polygon1_shape.area + polygon2_shape.area - polygon_intersection
		polygon_union =  polygon1_shape.area
		return polygon_intersection / polygon_union,  polygon_intersection/polygon_union2 ,polygon1_shape.area,polygon2_shape.area

def cvtCorner(corner, x, y ):
	result1 = corner.copy()
	for i , p in enumerate(corner):
		r = [int(p[0]-x) , int(p[1] - y)]
		result1[i] = r
	if len(result1) != 4:
		return None
	result  = (result1[0][0] , result1[0][1], result1[1][0] , result1[1][1],
				result1[2][0] , result1[2][1], result1[3][0] , result1[3][1] )
	return result

def compare_angle(p ,p1, p2 ):
	v1 = p1-p
	v2 =  np.array([0, 10])
	v3 = p2 -p
	cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	angle = np.arccos(cosine_angle)
	angle = np.degrees(angle)
	

	cosine_angle2 = np.dot(v3, v2) / (np.linalg.norm(v3) * np.linalg.norm(v2))
	angle2 = np.arccos(cosine_angle2)
	angle2= np.degrees(angle2)
	
	# print("angle ", angle, angle2)
	if abs(angle) > 90 :
		angle = 180 - abs(angle)
	if abs(angle2) > 90 :
		angle2 = 180 - abs(angle2)
	if angle < angle2:
		return 1
	else:
		return 2

def getIndexCorner(hull, w, h ):
	num_key = len(hull)
	idx_min = min((val[0][0], idx) for (idx, val) in enumerate(hull))
	idx_max = max((val[0][0], idx) for (idx, val) in enumerate(hull))
	idx_min = idx_min[1]
	idx_max = idx_max [1]
	idx_min2= -1
	idx_max2 = -1
	p0 = np.array(hull[idx_min][0])
	p1 = np.array(hull[(idx_min+1)%num_key][0])
	p2 =  np.array(hull[(idx_min+ num_key -1 )%num_key][0])
	idx = compare_angle(p0, p1 , p2 )
	if idx ==1 :
		
		idx_min2 = (idx_min+1)%num_key
	else:
		idx_min2 = (idx_min+ num_key -1)%num_key
	v1_min = hull[idx_min][0] - hull[idx_min2][0]
	if abs(v1_min[1]) < 0.2*h:
		idx_min2 = -1
	if num_key == 4:
		for i in range(num_key):
			if not i in [idx_min, idx_min2, idx_max]:
				idx_max2 = i
		
	else:
		p0 = np.array(hull[idx_max][0])
		p1 = np.array(hull[(idx_max+1)%num_key][0])
		p2 =  np.array(hull[(idx_max+ num_key -1 )%num_key][0])
		idx = compare_angle(p0, p1 , p2 )
		if idx ==1 :
			idx_max2 = (idx_max+1)%num_key
		else:
			idx_max2 = (idx_max+ num_key -1)%num_key
	v1_min = hull[idx_max][0] - hull[idx_max2][0]
	if abs(v1_min[1]) < 0.2*h:
		idx_max2 = -1
	

	print("hull " , hull, idx_min , idx_min2, idx_max, idx_max2 , v1_min, h)
	return idx_min , idx_min2, idx_max, idx_max2 

def getHomographyImageStit(image , boxes, mask):
	# cv2.imwrite("drawing3.jpg" , mask)
	kernel = np.ones((9,9),np.uint8)
	mask = np.array(mask * 255, dtype = np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imwrite("test/output/mask.jpg", mask)
	hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
	# For each contour, find the bounding rectangle and draw it
	area_max = 0 
	index = -1
	for component in zip(contours, hierarchy , range(len(contours))):
		currentContour = component[0]
		currentHierarchy = component[1]
		area  = cv2.contourArea(currentContour)
		if area > area_max :
			area_max = area
			index = component[2]
	h, w = image.shape[:2]
	if index >= 0:
		epsilon = 0.015*cv2.arcLength(contours[index],True)
		approx = cv2.approxPolyDP(contours[index],epsilon,True)
		hull = cv2.convexHull(approx)
		
		if len(hull) < 4 :
			return image , boxes
		idx_min , idx_min2, idx_max, idx_max2 = getIndexCorner(hull, w, h)
		if idx_max < 0 or idx_min < 0 or idx_max2 < 0 or idx_min2 < 0:
			return image , boxes
		pts1 = [] 
		pts2 = []
		for i, p in enumerate(hull):
			if not i in [idx_min , idx_min2, idx_max, idx_max2]:
				continue
			p1 = p[0]
			pts1.append(np.array(p1))
			if  i == idx_max2 :
				p2 = hull[idx_max][0]
				x =  int(0.5*p2[0] + 0.5*p1[0])
				pts2.append(np.array([x ,p1[1] ]))
			elif i == idx_max:
				p2 = hull[idx_max2][0]
				x =  int(0.5*p2[0] + 0.5*p1[0])
				pts2.append(np.array([x ,p1[1] ]))
			elif i == idx_min2 :
				p2 = hull[idx_min][0]
				x =  int(0.5*p2[0] + 0.5*p1[0])
				pts2.append(np.array([x ,p1[1] ]))
			elif  i == idx_min:
				p2 = hull[idx_min2][0]
				x =  int(0.5*p2[0] + 0.5*p1[0])
				pts2.append(np.array([x ,p1[1] ]))
			else:
				pts2.append(np.array(p1))
		pts1 = np.float32(pts1)
		pts2 = np.float32(pts2)
		
		
		matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10)
		if matrix is None: 
			return image , boxes
		# print("matrix" , matrix , pts1 , pts2)
		for key in boxes.keys():
			keypoint =  boxes[key]
			corners = np.float32([ [keypoint[0], keypoint[1]], [ keypoint[2], keypoint[3]],
				[ keypoint[4],  keypoint[5]], [keypoint[6] , keypoint[7]]])
			corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2) )
			keypoint  = (corners[0][0] , corners[0][1], corners[1][0] , corners[1][1],
				corners[2][0] , corners[2][1], corners[3][0] , corners[3][1] )
			boxes[key] = keypoint
		# image = cv2.polylines(image, [hull], True, (0, 0 , 255), 3)
		# cv2.imwrite("drawing2.jpg" , image)
		image = cv2.warpPerspective(image, matrix, (int(w),int(h) ) )
		# cv2.imwrite("drawing.jpg" , image)
	return image, boxes

def mergeBoxesItems(box1 , box2):
	xmin = min(box1[0],box2[0] )
	ymin = min(box1[1],box2[1] )
	xmax = max(box1[0] + box1[2],box2[0] ,box2[2] )
	ymax = max(box1[1] + box1[3],box2[1] ,box2[3] )
	return [xmin, ymin, xmax, ymax]

def processStitchingVideo(imgs ,arr_dict_boxes,  matrix, threshold_angle=20,
	threshold_box_overlap=0.2 , threshold_area_overlap=0.1, min_overlap_object=0.5):
	num_image_stitching = 0
	img_sumary = None
	bounding_box = []
	arr_dict_boxes_result = []
	arr_dict_corner_result = []
	ret_stitch = False
	if len(imgs) <= 0  :
		return ret_stitch, None, {}, {} , 0
	index = 0 
	# print("index " , index , len(imgs))
	im = imgs[index]
	# print("max size " , matrix["scale"])
	max_size = int(matrix["scale"])
	h_ori, w_ori = im.shape[:2]
	frame_cur = resize_image(im , max_size= max_size)
	ret = True
	arr_dict_boxes_result = cvtBoxestoSize (arr_dict_boxes, max_size/max(h_ori, w_ori))
	for i in range(len(arr_dict_boxes_result)):
		arr_dict_corner_result.append({})
	item_boxes = arr_dict_boxes_result[index]
	
	num_image_stitching +=1 
	list_warper = []
	mask_image = None
	while(ret):
		
		video_mosaic = VideMosaic(frame_cur, output_width_times=20, output_height_times=6, detector_type="sift")
		item_boxes, result_corner = video_mosaic.updateBoxesItemInit(item_boxes=item_boxes)
		arr_dict_boxes_result[index] = item_boxes
		arr_dict_corner_result[index] = result_corner
		img_frame = imgs[index +1:]
		count = 0
		
		homography =  matrix["matrix"][index : ]
		image_key = matrix["key_image"][index : ]
		if len(img_frame) > 0 :
			for key , im , H in zip(image_key, img_frame,homography):
				
				count += 1
				
				# print("image count: " , count)
				frame_cur = resize_image(im , max_size= max_size)
				item_boxes = arr_dict_boxes_result[index +count ]
				# process each frame
				ret, item_boxes, result_corner, transformed_corners = video_mosaic.process_frame(frame_cur, item_boxes= item_boxes,H=H, threshold_angle=threshold_angle)
				
				arr_dict_boxes_result[index +count ] = item_boxes
				arr_dict_corner_result[index +count] = result_corner
				# print("index ===================" , count , ret)
				if not ret:
					index = index + count -1
					break
				else:
					list_warper.append(transformed_corners)
					num_image_stitching +=1 
		
				# cv2.waitKey(0)
				# if cv2.waitKey(30) & 0xFF == ord('q'):
				# if count > 3:
				# 	break
			img_sumary = np.copy(video_mosaic.output_img)
			bounding_box = video_mosaic.output_bounding_box
			mask_image = video_mosaic.output_mask
			
			if ret :
				break
			ret = True
			break
		else:
			break
	list_id = {}
	list_frame = {}
	list_corners = {}
	
	

	if num_image_stitching > 0 :
		h, w  = img_sumary.shape[:2]
		if bounding_box[0] < 0 :
			bounding_box[0] = 0
		if bounding_box[1] < 0 :
			bounding_box[1] = 0
		if bounding_box[2] > w :
			bounding_box[2] = w
		if bounding_box[3] > h :
			bounding_box[3] = h
		for pts in list_warper:
			mask_image = cv2.fillPoly(mask_image, [pts], 255)
		
		mask_image = mask_image[bounding_box[1]:bounding_box[3] , bounding_box[0]:bounding_box[2]]
		img_sumary = img_sumary[bounding_box[1]:bounding_box[3] , bounding_box[0]:bounding_box[2]]
		
		h, w  = img_sumary.shape[:2]
		for k in range(num_image_stitching):
			# print("result item" , item_image)
			item_image = arr_dict_boxes_result[k]
			item_corner = arr_dict_corner_result[k]
			for key in item_image.keys():
				
				box  = item_image[key]
				box[0] -= bounding_box[0]
				box[1] -= bounding_box[1]
				box[2] -= bounding_box[0]
				box[3] -= bounding_box[1]
				corner = item_corner[key]
				corner = cvtCorner(corner , bounding_box[0] , bounding_box[1] )
				if box[0] < 0 :
					box[0] = 0
				if box[1] < 0 :
					box[1] = 0
				if box[2] > w :
					box[2] = w
				if box[3] > h :
					box[3] = h
				size = [box[2] - box[0] , box[3]- box[1]]
				x = box [0] + bounding_box[0]
				y =	box[1] + bounding_box[1]
				area_overlap = 0
				if size[0] < 3 or size[1] < 3:
					continue
				pt = np.array([[x, y] , [x , y +size[1] ] , [x+ size[0] , y +size[1] ] , [x+size[0], y]])
				if k > 0 :
					overlap, _ , _, _= IOU(pt , np.array(list_warper[k -1 ]))
					# print("overlap " , overlap )
				update_corner = True
				if key in list_id.keys():
					b_current  = list_id[key]
					x2 = b_current [0] + bounding_box[0]
					y2 = b_current[1] + bounding_box[1]
					w2 = b_current [2]
					h2 =	b_current[3] 
					pt2 = np.array([[x2, y2] , [x2 , y2 +h2 ] , [x2+ w2 , y2 +h2 ] , [x2+w2, y2]])
					overlap_boxes, area_overlap, area_box1 , area_box2 = IOU(pt , pt2)
					if area_box1 < area_box2:
						update_corner = False
					# if area_overlap > 0.3 :
					# 	list_frame[key] = [k, area_overlap]
					if overlap_boxes < threshold_box_overlap  or area_overlap < threshold_area_overlap or area_box1 < min_overlap_object*area_box2:
					# list_id[key] = [box[0], box[1],size[0] , size[1]]
					# list_frame[key] = k
						continue
				# if key in list_id.keys():
				# 	box = mergeBoxesItems(list_id[key] , [box[0], box[1],size[0] , size[1]]) 
				# 	size = [box[2] - box[0] , box[3]- box[1]]

				list_id[key] = [box[0], box[1],size[0] , size[1]]
				list_frame[key] = [k, area_overlap]
				if update_corner:
					list_corners[key] = corner
		list_key_remove = []
		for key in list_id.keys():
			box = list_id[key]
			x = box [0] + bounding_box[0]
			y =	box[1] + bounding_box[1]
			w = box[2]
			h = box[3]
			pt = np.array([[x, y] , [x , y +h ] , [x+ w , y +h ] , [x+w, y]])
			index = list_frame[key][0]
			for i in range(num_image_stitching-1):
				if i <= index +1 :
					continue
				overlap, area_overlap, _, _ = IOU(pt , np.array(list_warper[i]))
				
				if overlap > 0.5 :
					list_key_remove.append(key)
					break
				# print("overlap " , overlap )
			
		for x in list_key_remove:
			if x in list_id:
				del list_id[x]
				del list_corners[x]
		list_frame.clear()
	img_sumary, list_corners = getHomographyImageStit(img_sumary , list_corners , mask_image)
		
	arr_dict_corner_result.clear()
	arr_dict_boxes_result.clear()
	ret_stitch = True
	return ret_stitch, img_sumary, list_id , list_corners, num_image_stitching
		
def listdir_fullpath(d):
	return [os.path.join(d, f) for f in os.listdir(d)]

if __name__ == "__main__":
	output_folder = 'revamptracking/results20220303_merge'
	list_full_folder = listdir_fullpath("/media/anlab/ssd_samsung_256/Lashinbang-test/test_video")
	
