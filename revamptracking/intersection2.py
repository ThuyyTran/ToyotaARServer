import numpy as np
from numpy.lib.histograms import histogram
from numpy.matrixlib.defmatrix import matrix
import cv2
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import os
import math
def delete_key(dict_img2,dict_age_correspond,boxes_img,index_bf2):
	key_delete=[]
	for key in dict_age_correspond.keys():
		if(dict_age_correspond[key]>=3):
			key_delete.append(key)
			# del dict_age_correspond[key]
			# del dict_img2[key]
	for key in key_delete:
		boxes_img[index_bf2].remove(dict_img2[key])
		del dict_age_correspond[key]
		del dict_img2[key]
	return dict_age_correspond,dict_img2,boxes_img
def caculate_degree_2_vector(vector1,vector2):
	unit_vector_1 = vector1 / np.linalg.norm(vector1)
	unit_vector_2 = vector2 / np.linalg.norm(vector2)
	dot_product = np.dot(unit_vector_1, unit_vector_2)
	angle = np.arccos(dot_product)
	angle= math.degrees(angle)
	return angle
def get_key_dst_min(vector,box,dict_boxes_reverse,index_before1):
	if(index_before1==28 and "1" in list(dict_boxes_reverse.keys())):
		print("dict_boxes_reverse",dict_boxes_reverse)
	save_key=None
	save_kc=9*(10**7)
	# print("")
	for i,key in enumerate(dict_boxes_reverse.keys()):
		# if(index_before1==28 and key=="2"):
		# 	print("yes")
		box_temp=dict_boxes_reverse[key]
		if(box==box_temp or box is None):
			continue
		vector2=[box_temp[0]-box[0],box_temp[1]-box[1]]

		# center1=[(box[0]+box[2])/2,(box[1]+box[3])/2]
		# center2=[(box_temp[0]+box_temp[2])/2,(box_temp[1]+box_temp[3])/2]
		# vector2= [center2[0]-center1[0],center2[1]-center1[1]]


		angle= caculate_degree_2_vector(vector,vector2)
		kc= (box_temp[0]-box[0])**2+(box_temp[1]-box[1])**2
		# kc= (center2[0]-center1[0])**2+(center2[1]-center1[1])**2
		# if(key=="1" and index_before1==28):
		# 	print("check",box)
		# 	print("angle,kc at key ==1", angle,kc)
		# if(key=="20" and index_before1==28):
		# 	print("angle,kc at key ==20", angle,kc)
		if(abs(angle)<=20  and kc < save_kc):
			# print("angle",angle)
			save_key=key
			save_kc=kc
			# print("save_key,save_kc",save_key,save_kc)
	return save_key
def get_boxNear_vector(box_,boxes):
	import math
	# print("box_",box_)
	# print("boxes",boxes)
	save_index=0
	min_distance=9*(10**7)
	vector=None
	for i,box in enumerate(boxes):
		
		if(box==box_ or box is None):
			continue
		# center1=[(box[0]+box[2])/2,(box[1]+box[3])/2]
		# center2=[(box_[0]+box_[2])/2,(box_[1]+box_[3])/2]

		dst= math.sqrt((box[0]-box_[0])**2 + (box[1]-box_[1])**2)
		# dst= math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
		if(dst<min_distance):
			save_index =i
			min_distance=dst
			
			vector= [box_[0]-box[0],box_[1]-box[1]]

			# center1=[(box[0]+box[2])/2,(box[1]+box[3])/2]
			# center2=[(box_[0]+box_[2])/2,(box_[1]+box_[3])/2]
			# vector= [center2[0]-center1[0],center2[1]-center1[1]]
	
	return boxes[save_index],vector
def merge_boxes(dict_boxes,boxes,maxkey):
	
	arr_boxes_new=[]
	for box in boxes:
		key_save= None
		check_add=True
		if(box is None):
			continue
		p=Polygon([(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])])
		for i,key in enumerate(dict_boxes.keys()):
			box_temp=dict_boxes[key]
			q=Polygon([(box_temp[0],box_temp[1]),(box_temp[2],box_temp[1]),(box_temp[2],box_temp[3]),(box_temp[0],box_temp[3])])
			intersec= p.intersection(q).area
			intersec_percent1= intersec/p.area
			intersec_percent2= intersec/q.area
			if(intersec<0.00000001):
				continue
			if(max(intersec_percent1,intersec_percent2)>0.7):
				key_save=key
				if(q.area>p.area):
					box=box_temp
				break
		
		if(key_save is not None):
			arr_boxes_new.append(box)
	return arr_boxes_new
def fix_box_merge(box1,box2):
	# print("box1,box2,box fix",box1,box2)
	x1_save= min(box1[0],box2[0])
	y1_save= min(box1[1],box2[1])
	x2_save= x1_save+ abs(x1_save-max(box2[2],box1[2]))
	y2_save= y1_save+ abs(y1_save-max(box2[3],box1[3]))
	box_save=[x1_save,y1_save,x2_save,y2_save]
	return box_save

def get_intersec_max(box1,box2,width,height):
	if(box1[0]>width or box2[0]>width or box1[2]<0 or box2[2]<0):
		return 0
	# print("width, height",width,height)
	p = Polygon([(box1[0],box1[1]),(box1[2],box1[1]),(box1[2],box1[3]),(box1[0],box1[3])])
	q = Polygon([(box2[0],box2[1]),(box2[2],box2[1]),(box2[2],box2[3]),(box2[0],box2[3])])
	r = Polygon([(0,0),(width,0),(width,height),(0,height)])

	intersec1= p.intersection(q).area
	max1=max((intersec1/p.area),(intersec1/q.area))

	box1_temp=[]
	for i,value in enumerate(box1):
		if(i>3):
			continue
		if(i%2==0):
			if(value>width):
				value=width
			if(value<0):
				value=0
		if(i%2!=0):
			if(value>height):
				value=height
			if(value<0):
				value=0
		box1_temp.append(value)
	if(box1_temp[0]==box1_temp[2] or box1_temp[1]==box1_temp[3]):
		return max1
	
	box1_temp=Polygon([(box1_temp[0],box1_temp[1]),(box1_temp[2],box1_temp[1]),(box1_temp[2],box1_temp[3]),(box1_temp[0],box1_temp[3])])
	box2_temp=[]
	for i,value in enumerate(box2):
		if(i>3):
			continue
		if(i%2==0):
			if(value>width):
				value=width
			if(value<0):
				value=0
		if(i%2!=0):
			if(value>height):
				value=height
			if(value<0):
				value=0
		box2_temp.append(value)
	if(box2_temp[0]==box2_temp[2] or box2_temp[1]==box2_temp[3]):
		return max1


	box2_temp=Polygon([(box2_temp[0],box2_temp[1]),(box2_temp[2],box2_temp[1]),(box2_temp[2],box2_temp[3]),(box2_temp[0],box2_temp[3])])
	intersec2= box1_temp.intersection(box2_temp).area
	max2= max((intersec2/box1_temp.area),(intersec2/box2_temp.area))
	# if(max(max1,max2)>0.65):
	# 	print("box1,box2", box1,box2)
	# print("max1,max2)",(max1,max2))
	
	return max(max1,max2)
	# return max2

def get_pair_index(arr):
	arrs=[]
	for i in range(len(arr)-1):
		for j in range(i+1,len(arr)):
			arrs.append([i,j])
	return arrs


def get_iou(box1,box2):
    # print("box1,box2",box1,box2)
    p=Polygon([(box1[0],box1[1]),(box1[2],box1[1]),(box1[2],box1[3]),(box1[0],box1[3])])
    q=Polygon([(box2[0],box2[1]),(box2[2],box2[1]),(box2[2],box2[3]),(box2[0],box2[3])])
    intersec= p.intersection(q).area
    # print("p.area ,q.area, intersec",p.area,q.area,intersec)
    iou= intersec/(p.area+q.area-intersec)
    return iou

def filter_iou(boxes):
	list_box_new=[]
	for i in range(len(boxes)-1):
		box1=boxes[i]
		check_box1=True
		for j in range(i+1,len(boxes)):
			box2=boxes[j]
			if(get_iou(box1,box2)>0.7):
				check_box1=False
		if(check_box1==True):
			list_box_new.append(box1)
	print("len list_box_new",len(list_box_new))
	return list_box_new
	
def get_index_same(id,list_id):
    indexs=[]
    for i,value in enumerate(list_id):
        if(value==id):
            indexs.append(i)
    return indexs
def lk_tracker(img1 , img2 ):

	lkParameters = dict(winSize=(13, 13), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
	stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
					10, 0.03)
	h, w = img1.shape[:2]
	# mask = np.zeros_like(img1)
	# est_x = int(0.05*w)
	# est_y = int(0.05*h)
	# mask = cv2.rectangle(mask , (est_x, est_y) , (w- est_x, h - est_y ) , (255), -1 , 8)
	cornerA = cv2.goodFeaturesToTrack(img1, mask=None, maxCorners=1000, qualityLevel=0.01, minDistance=15, blockSize=5)
	# print(len(cornerA), img1.shape)
	cv2.cornerSubPix(img1, cornerA, (3, 3), (-1, -1),stop_criteria)
	previousCorners = cornerA.reshape(-1, 1, 2)
	#cornerB = copy.copy(cornerA)
	cornersB, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, previousCorners, None, **lkParameters)
	cornersB = cornersB[st == 1]
	previousCorners = previousCorners[st == 1]
	return previousCorners, cornersB


def find_homography(img1,img2):
	kps1,kps2=lk_tracker(img1 , img2)
	# print("kps1",kps2.shape)
	if(kps1.shape[0] >=8):
		homography, mask = cv2.findHomography(kps1, kps2, cv2.RANSAC)
		return homography
	return None

def intersec_area_2(img2,homography):
	height, width = img2.shape
	corner_points_img=np.array([[(0,0),(width,0),(width,height),(0,height)]],np.float32)
	transformed_corner_points=cv2.perspectiveTransform(corner_points_img, homography)
	p = Polygon(corner_points_img[0])
	q = Polygon(transformed_corner_points[0]).buffer(0)
	intersec= p.intersection(q).area
	max_area = max (p.area , q.area)
	intersec_percent= intersec/max_area
	return intersec_percent

def intersec_area_3(width, height,homography):
	corner_points_img=np.array([[(0,0),(width,0),(width,height),(0,height)]],np.float32)
	transformed_corner_points=cv2.perspectiveTransform(corner_points_img, homography)
	p = Polygon(corner_points_img[0])
	q = Polygon(transformed_corner_points[0]).buffer(0)
	intersec= p.intersection(q).area
	max_area = max (p.area , q.area)
	intersec_percent= intersec/max_area
	return intersec_percent


def intersec_area_origin(img1,img2,index_img1,count):
	height, width = img1.shape
	kps1,kps2=lk_tracker(img1 , img2)
	if(kps1.shape[0] >=4):
		homography, mask = cv2.findHomography(kps1, kps2, cv2.RANSAC)
		if(homography is None):
			if((count-index_img1) <10):
				return 1
			return 0
		corner_points_img=np.array([[(0,0),(width,0),(width,height),(0,height)]],np.float32)
		transformed_corner_points=cv2.perspectiveTransform(corner_points_img, homography)
		p = Polygon(corner_points_img[0])
		q = Polygon(transformed_corner_points[0]).buffer(0)
		intersec= p.intersection(q).area
		intersec_percent= intersec/q.area
	else:
		# print("keypoint very low")
		if((count-index_img1) <10):
			return 1
		return 0
	return intersec_percent,homography
if __name__ == '__main__' :

	img0 = cv2.imread('frame/IMG_6632/1/IMG_6632_1.jpg',0)          # queryImage
	img1 = cv2.imread('frame/IMG_6632/1/IMG_6632_2.jpg',0) # trainImage
	homography=find_homography(img0,img1)
	