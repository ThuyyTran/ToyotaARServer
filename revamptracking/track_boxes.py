import os
from itertools import combinations
import logging
import time
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import math
import copy
import time
from numpy.linalg import inv
from mergeNearbyBoxes import MergeOverlapBoxesContours
from intersection2 import intersec_area_2,intersec_area_3

def draw_tracking_obj(pair_imgs, pair_boxes, matrix_bf2af , index_bf, size_compare , matrix_all_before):
	# print(pair_imgs[0][0].shape)
	# print("pair_imgs",len(pair_imgs))
	# print("pair_boxes",len(pair_boxes))
	# print("pair_boxes",pair_boxes)
	# # print("matrix_bf2af",matrix_bf2af)
	# print("matrix_bf2af",len(matrix_bf2af))
	# print("index_bf",index_bf)
	# print("size_compare",size_compare)
	# print("matrix_all_before_key_image",(matrix_all_before["key_image"][0]))
	# print("matrix_all_before_key_image",len(matrix_all_before["key_image"]))
	# print("matrix_all_before_key_image",(matrix_all_before["key_image"][0]))
	# print("matrix_all_before_matrix",len(matrix_all_before["matrix"]))
	# print("matrix_all_before_scale",(matrix_all_before["scale"]))
	# # exit()

	list_f1=index_bf
	box_f1_sorted=[]
	box_f2_sorted=[]
	for value in pair_boxes:
		box_f1_sorted.append(value[0])
		box_f2_sorted.append(value[1])
	
	if(len(pair_imgs)>0):
		save_f1=[]
		box_save_f1=[]
		file_pair=[]

		height,width,_=pair_imgs[0][0].shape
		# print("width,height",width,height)

		if(height>width):
			scale= matrix_all_before["scale"]/height
		else:
			scale= matrix_all_before["scale"]/width
		if(matrix_all_before["scale"]>=720):
			box_f1_sorted_process=convert_box_scale(box_f1_sorted,scale)
		else:
			box_f1_sorted_process=box_f1_sorted
		# print("box_f1_sorted_process",box_f1_sorted_process)
		#get pair and check intersec
		list_of_pairs = get_pair_list(list_f1)
		
		for pair in list_of_pairs:
			# print("pair",pair)
			check_pair_name_before_forward= pair[0]+"_"+pair[1]
			check_pair_name_before_backward=pair[1]+"_"+pair[0]
			if check_pair_name_before_forward not in matrix_all_before["key_image"] and check_pair_name_before_backward not in matrix_all_before["key_image"]:
				# print("continue")
				continue
			if(check_pair_name_before_forward in matrix_all_before["key_image"]):
				index_pair= matrix_all_before["key_image"].index(check_pair_name_before_forward)
				homography= (matrix_all_before["matrix"][index_pair])
			else:
				index_pair= matrix_all_before["key_image"].index(check_pair_name_before_backward)
				homography= inv(matrix_all_before["matrix"][index_pair])

			area = intersec_area_3(width*scale,height*scale,homography)
			# print("area......",area)
			# exit()
			if(area>0.4):
				# print("list_f1.index(pair[0])",list_f1.index(pair[0]))
				box_before1=box_f1_sorted_process[list_f1.index(pair[0])]
				# print("box_before1",box_before1) #box process
				box_before2=convet_coordinate2rect(box_before1,homography)
				# print("box_before2",box_before2)
				#convert box compare size
				if(size_compare>=720):
					box_before2=convert_box_scale([box_before2],size_compare/(scale*max(width,height)))[0]

				# print("box_before2_process",box_before2)

				
				matrix_homo= matrix_bf2af[list_f1.index(pair[1])]
				#caculate coordinate imagebf2 to imageaf2

				box_af2=convet_coordinate2rect(box_before2,matrix_homo)
				# print("box_af2",box_af2)

				box_af2=convert_box_scale([box_af2],size_compare/max(width,height))[0]
				
				index_bf2= list_f1.index(pair[1]) #index img before 2
				# print("index_bf2",index_bf2)

				box_af2= box_af2+box_f2_sorted[index_bf2]
				
				######################################################
				box_save=[]
				for box in box_af2:
					# point=[box[0],box[1],box[2],box[3]]
					if(box[0]<0 and box[2]>=50):
						# start_point=(0,box[1])
						# end_point=(box[2],box[3])
						point=[0,box[1],box[2],box[3]]

					elif(box[0]>=0 and box[2]<=width):
						# start_point=(box[0],box[1])
						# end_point=(box[2],box[3])
						point=[box[0],box[1],box[2],box[3]]

					elif(box[0]<=width-50 and box[2]>width):
						# start_point=(box[0],box[1])
						# end_point=(0,box[3])
						point=[box[0],box[1],width,box[3]]
					else:
						continue
					box_save.append(point)
				if(pair[1] not in save_f1):
					save_f1.append(pair[1])
					box_save_f1.append(box_save)
					# file_pair.append(list_f2[index_bf2])
				else:
					box_save_f1[save_f1.index(pair[1])]+=box_save
		# print("box_save_f1",box_save_f1)
		# print("len box_save_f1",len(box_save_f1))
		# print("save_f1",save_f1)
		# print("len save_f1",len(save_f1))
		
		
		box_new=[]
		for i in range(len(box_save_f1)):
			_,box_process = MergeOverlapBoxesContours(box_save_f1[i], width, height)
			box_new.append(box_process)
		box_new=box_save_f1
		for i,value in enumerate(save_f1):
			index_before= list_f1.index(value)
			pair_boxes[index_before][1]=box_new[i]
		
		return pair_imgs, pair_boxes
		#################################################
		# for i in range (len((box_save_f1))):
		# 	_,box_process = MergeOverlapBoxesContours(box_save_f1[i], width, height)
		# 	# name_pair= save_f1[i]+"_"+file_pair[i]+"_compare.jpg"
		# 	# print("name_pair",name_pair)
		# 	# for value in list_result:
		# 	# 	if(save_f1[i]+"_"+file_pair[i] in value):
		# 	# 		name_pair=value.split("/")[-1]
		# 	# 		print("name_pair",name_pair)
		# 	# 		break
		# 	print("file ", f"{settings.PATH_DATABASE}/{room_video}/images/{save_f1[i]}.jpg")
		# 	img1=cv2.imread(f"{settings.PATH_DATABASE}/{room_video}/images/{save_f1[i]}.jpg")
		# 	print("img1 ",f"{settings.PATH_DATABASE}/{room_video}/images/{save_f1[i]}.jpg")
		# 	img2=cv2.imread(f"{settings.PATH_QUERY}/{room_video}/images/{file_pair[i]}.jpg")
		# 	print("img2 ",f"{settings.PATH_QUERY}/{room_video}/images/{file_pair[i]}.jpg")
		# 	im_h = cv2.hconcat([img1, img2])
		# 	for box in box_process:
		# 		start_point=(box[0]+width,box[1])
		# 		end_point=(box[2]+width,box[3])
		# 		im_h = cv2.rectangle(im_h, start_point, end_point, (0 , 0 , 255), 5)
		# 	im_h= cv2.putText(im_h,"Reference image",(0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5,cv2.LINE_AA)
		# 	im_h= cv2.putText(im_h,"Actual image",(width,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5,cv2.LINE_AA)
			# cv2.imwrite(f"output/{room_video}/"+name_pair,im_h)
			# cv2.imwrite(name_pair,im_h)
				######################################################

				# for box in box_af2:
				# 	if(box[0]+width>2*width-50 or box[2]+width<width+50):
				# 			continue
				# 	if(box[0]+width>width):
				# 		start_point=(box[0]+width,box[1])
				# 	else:
				# 		start_point=(width,box[1])
				# 	if(box[2]+width>width):
				# 		end_point=(box[2]+width,box[3])
				# 	else:
				# 		end_point=(width,box[3])
					
				# 	img = cv2.rectangle(img, start_point, end_point, (255,0,0), 5)
				# 	img = cv2.putText(img, str(name_frame1), start_point, cv2.FONT_HERSHEY_SIMPLEX, 
				# 	   2, (255,0,0), 5, cv2.LINE_AA)
				
				
				# cv2.imwrite(f"output/{room_video}/"+name_pair,img)
				# cv2.imwrite(name_pair,img)
def convet_coordinate2rect(box1,homography):
	box2=[]
	for box in box1:
		x1=box[0]
		y1=box[1]
		x2=box[2]
		y2=box[3]
		corner_points_img=np.array([[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]],np.float32)
		# print("corner_points_img",corner_points_img)
		# print("homography",homography)
		# print("*********************************")
		transformed_corner_points=cv2.perspectiveTransform(corner_points_img, homography)
		box2.append(transformed_corner_points)
	box_convert=[]
	for i,box in enumerate(box2):
		content_box=box2[i][0]
		x1min=int(min(content_box[0][0],content_box[1][0],content_box[2][0],content_box[3][0]))
		x2max=int(max(content_box[0][0],content_box[1][0],content_box[2][0],content_box[3][0]))
		y1min=int(min(content_box[0][1],content_box[1][1],content_box[2][1],content_box[3][1]))
		y2max=int(max(content_box[0][1],content_box[1][1],content_box[2][1],content_box[3][1]))
		box_convert.append([x1min,y1min,x2max,y2max])
	return box_convert
def resize_image_gray(image,max_size=720):
	height,width= image.shape
	if(height>width):
		scale= max_size/height
	else:
		scale= max_size/width

	width= int(width*scale)
	height=int(height*scale)
	image= cv2.resize(image,(width,height))
	return image
def get_pair_list_foward(list_arr):
	save=[]
	for i in range(len(list_arr)-1):
		save.append((list_arr[i],list_arr[i+1]))
	return save
def get_pair_list(list_arr):
	save=[]
	for i in range(len(list_arr)-1):
		save.append((list_arr[i],list_arr[i+1]))
		save.append((list_arr[i+1],list_arr[i]))
	return save
def convert_box_scale(box_f1_sorted,scale):
	box_f1_sorted_process=[]
	for box_ in box_f1_sorted:
		if(box_ is None):
			continue
		box_process=[]
		for value in box_:
			if(value is None):
				continue
			x1=int(value[0]*scale)
			y1=int(value[1]*scale)
			x2=int(value[2]*scale)
			y2=int(value[3]*scale)
			box_process.append([x1,y1,x2,y2])
		box_f1_sorted_process.append(box_process)
	return box_f1_sorted_process
