
from operator import index
import os
import time
import pickle
import json
# from sqlalchemy import false
import cv2
import numpy as np
import math
import uuid
from .detect import detect_images
import copy
from . import config_revamp
from intersection2 import  find_homography,intersec_area_2,intersec_area_3,delete_key,get_key_dst_min,get_boxNear_vector,merge_boxes,fix_box_merge,get_intersec_max,get_pair_index,filter_iou
from numpy.linalg import inv
import shutil
from track_boxes import convert_box_scale,get_pair_list,convet_coordinate2rect,get_pair_list_foward
from mergeImage import get_ImageMerge_and_infoBoxes
from shapely.geometry import Polygon
from logger import AppLogger
from stitching_mosaic import processStitchingVideo
import datetime
from getMovingCamera import camera_correct_trajectory
logger = AppLogger()




def listdir_fullpath(d):
	return [os.path.join(d, f) for f in os.listdir(d)]

class ERRORS_CODE:
	STATUS_WAITING = 0
	STATUS_UPLOAD_DONE = 1
	STATUS_GET_DONE = 2
	STATUS_IN_PROGRESS = 3
	STATUS_COMPLETED = 4
	STATUS_IMAGE_STITCH_LOOSE = 5 #wranning ảnh ghép xấu
	STATUS_VIDEO_PROCESS70 = 6 # wramning chi xu ly dc 70 % video
	STATUS_GET_FAILED = 7
	STATUS_RETAKE = 8
	STATUS_VIDEO_LENGTH_OVER = 9 #errors video nhỏ hơn 10s hoặc lớn hơn 90s
	STATUS_VIDEO_PROCESS_ERROR = 10 #errors không đọc được video hoặc là không cắt được ảnh ra từ video
	STATUS_VIDEO_ERRORS_LOOP = 11 #errors video người quay di chuyển bị lặp lại
	

class RevampTracking:
	def __init__(self):
		logger.info("init track revamp")
		self.folder_data= config_revamp.PATH_DATABASE
		self.query_folder = config_revamp.PATH_QUERY
		self.weight_path = config_revamp.MODEL_PATH
		if not os.path.isdir(self.query_folder):
			os.mkdir(self.query_folder)
		if not os.path.isdir(self.folder_data):
			os.mkdir(self.folder_data)

	def cutObjectFrame(self ,imgs, arr_dict_boxes, threshold_blur=60 ):
		num_data = len(arr_dict_boxes)
		print("num image procees  " ,num_data)
		result_id_search = []
		image_objects = []
		ratio_mean = {}
		ratio_list = []
		for i in range(num_data):

			img=imgs[i]
			# print("i",i)
			# print("img",img.shape)
			value = arr_dict_boxes[i]
			for key in value.keys():
				time_out = time.time()
				box=value[key]
				# print("box",box)

				check_box=True #check box in image
				if(box is None):
					# print("continue")
					continue
				if(len(box)>4):
					check_box=False

				x=box[0]
				y=box[1]
				if(x<0):
					x=0
				if(y<0):
					y=0
				w=box[2]-box[0]
				h=box[3]-box[1]

				# print("x,y,w,h",x,y,w,h)
				crop_img = img[y:y+h, x:x+w]
				gray_im = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
				f = self.variance_of_laplacian(gray_im)
				if f < threshold_blur:
					continue
				# crop_img = cv2.GaussianBlur(crop_img,(3,3),cv2.BORDER_DEFAULT)
				# Create search request
				img_id = f'{key}_{i}'
				result_id_search.append(key)
				image_objects.append(crop_img)
				ratio_list.append(float(w)/h)
				if key in ratio_mean.keys():
					val = ratio_mean[key]
					ratio = val[0] + float(w)/h
					ratio_mean[key] = [ratio, val[1] +1, max(val[2], float(w)/h) , min(val[3], float(w)/h), max(w*h, val[4])]
				else:
					ratio_mean[key] = [float(w)/h, 1, float(w)/h, float(w)/h, w*h]
		# check list ratio failed
		for key in ratio_mean.keys():
			val = ratio_mean[key]
			m =max( 0.1, 0.3*abs(val[2]-val[3]))
			m = min(m , 0.5)
			ratio_mean[key] = [val[0]/val[1] , val[1], m, val[3], val[4]]
			# ratio_mean[key] = [val[0], val[1], m, val[3]]

		list_id_ratio_failed = []
		for i, r in enumerate(ratio_list):
			val = ratio_mean[result_id_search[i]]
			im = image_objects[i]
			h, w = im.shape[:2]
			
			if val[1] >= 3 :
				ret = False
				if abs(r - val[0]) - 0.05*(w*h)/val[4] > val[2] and (w*h)/val[4] < 0.8:
					list_id_ratio_failed.append(i)
					ret = True

				# if result_id_search[i] == "63":
				# 	gray_current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				# 	f = self.variance_of_laplacian(gray_current)
				# 	print("ratio " , i , ret,abs(r - val[0]) - 0.05*(w*h)/val[4] , val[2] , f )
				# 	cv2.imwrite(f"revamptracking/output/0_{i}.jpg" , im)
		# exit()
		image_objects_new = [j for i, j in enumerate(image_objects) if i not in list_id_ratio_failed]
		result_id_search_new = [j for i, j in enumerate(result_id_search) if i not in list_id_ratio_failed]
		
		for im in image_objects:
			del im
		# 	im = image_objects.pop(x)
		# 	im = None
		# 	result_id_search.pop(x)
		list_id_ratio_failed.clear()
		ratio_mean.clear()
		ratio_list.clear()
		image_objects.clear()
		result_id_search.clear()
		
		return image_objects_new, result_id_search_new
				# crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
				# im_pil = Image.fromarray(crop_img)

	def checkDimensionImage(self, image):
		if image is None:
			return image
		h,w = image.shape[:2]
		if w > h :
			image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
		return image

	def getThumbnail(self, video_path, max_size=720):
		vid = cv2.VideoCapture(video_path)
		if not(vid.isOpened()):
			return None
		max_val = 0
		ret = True
		imageThumbnail = None
		count = 0
		while(ret):
			ret, frame = vid.read()
			if frame is None:
				break
			frame = self.checkDimensionImage(frame)
			count += 1
			frame=self.resize_image(frame, max_size=max_size)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			fm=self.variance_of_laplacian(gray)
			if max_val < fm:
				max_val = fm
				imageThumbnail = frame
			if count > 300:
				break
		return imageThumbnail


	def getStitChingVideo(self, video_path, threshold_max=0.8, threshold_min=0.65, start_frame=10 , max_size=720, threshold_angle=30 , threshold_time_min = 10, threshold_time_max = 90, threshold_time_process =0.7 ):
		threshold_time_pr_max = 0.85
		id_process = str(uuid.uuid4())
		# id_process = os.path.basename(video_path)
		# id_process = os.path.splitext(id_process)[0]

	
		logger.info(f'video path: {video_path}, {id_process} ' )
		
		errors_video = ERRORS_CODE.STATUS_COMPLETED
		ret_process ,list_img_save,  images_list ,label_detections,  matrix_save, video_time, time_process = self.convert_video2data(video_path , id_process  , base_save_paths=self.folder_data, threshold_max=threshold_max, threshold_min=threshold_min, start_frame= start_frame , max_size=max_size )

		if not ret_process:
			errors_video = ERRORS_CODE.STATUS_VIDEO_PROCESS_ERROR
		elif video_time < threshold_time_min or video_time > threshold_time_max:
			errors_video = ERRORS_CODE.STATUS_VIDEO_LENGTH_OVER
			ret_process = False
		

		id_objects = []
		img_list = []
		image_sumary = None
		list_objects_info = {}
		if ret_process:
			try:
				imgs, arr_dict_boxes, matrix = self.run_tracking(list_img_save,  images_list ,label_detections,  matrix_save )

				if len(imgs) > 0 :
				
					img_list, id_objects = self.cutObjectFrame(imgs, arr_dict_boxes)
					ret_stitch , image_sumary,_ , list_objects_info , num_image_stitching = processStitchingVideo(imgs,arr_dict_boxes ,  matrix, threshold_angle= threshold_angle  )
					
					# print("num_image_stitching " , num_image_stitching,0.8* len(imgs) , ret_stitch)
					if errors_video <= ERRORS_CODE.STATUS_COMPLETED and (not ret_stitch or num_image_stitching < 0.8* len(imgs)) :
						errors_video = ERRORS_CODE.STATUS_IMAGE_STITCH_LOOSE
					camera_correct = camera_correct_trajectory(imgs,matrix)

					if  not camera_correct :
						errors_video = ERRORS_CODE.STATUS_VIDEO_ERRORS_LOOP
			except Exception as e:
				errors_video = ERRORS_CODE.STATUS_VIDEO_PROCESS_ERROR
			if errors_video <= ERRORS_CODE.STATUS_COMPLETED and time_process >= threshold_time_process * video_time and time_process <= threshold_time_pr_max*video_time:
				errors_video = ERRORS_CODE.STATUS_VIDEO_PROCESS70

		return errors_video, image_sumary, img_list, id_objects,  list_objects_info


	def getBoxeFromLabel(self, pred , w, h) :
		boxes = []
		labels = []
		for b in pred :
			if len(b) < 6:
				continue
			size_x = int(w*b[2])
			size_y = int(h*b[3])
			xmin = int(b[0]*w) - size_x//2
			ymin = int(b[1]*h) - size_y//2
			xmax = int(b[0]*w) + size_x//2
			ymax = int(b[1]*h) + size_y//2
			boxes.append([xmin, ymin, xmax, ymax])
			labels.append(int(b[4]))
		return boxes, labels

	def getBoxes(self, list_labels , list_labels_key , base_name, w, h) :
		boxes = []
		labels = []
		if not base_name in list_labels_key:
			return boxes, labels
		count =  list_labels_key.index(base_name)
		f = open (list_labels[count] , 'r')
		l = []
		l = [line.split() for line in f]
		l=np.array(l)

		l=l.astype(np.float)

		for b in l :
			if len(b) < 5:
				continue
			size_x = int(w*b[3])
			size_y = int(h*b[4])
			xmin = int(b[1]*w) - size_x//2
			ymin = int(b[2]*h) - size_y//2
			xmax = int(b[1]*w) + size_x//2
			ymax = int(b[2]*h) + size_y//2
			boxes.append([xmin, ymin, xmax, ymax])
			labels.append(int(b[0]))
		return boxes, labels

	def run_tracking(self, list_img_save,  images_list ,label_detections,  matrix_save):

		boxes_img = []
		for image, pred in  zip(list_img_save, label_detections):
			h_resize , w_resize = image.shape[:2]
			boxes_idx, classids = self.getBoxeFromLabel(pred, w_resize, h_resize)
			boxes_img.append(boxes_idx)

		# track object danpm
		list_img_save, arr_dict_boxes =self.tracking_obj(list_img_save, boxes_img,  images_list, matrix_save)

		# folder_result = os.path.join(data_folder, "yolo")
		# if not os.path.isdir(folder_result):
		# 	os.mkdir(folder_result)
		# self.draw_boxes2(imgs,arr_dict_boxes,indexs , folder_result  )
		return list_img_save, arr_dict_boxes, matrix_save



	def create_DB_test(self,imgs,arr_dict_boxes, folder_save):
		if not os.path.isdir(folder_save):
			os.mkdir(folder_save)
		check_id=[]
		list_key=[]
		list_value=[]
		for i,value in enumerate(arr_dict_boxes):
			img=imgs[i]
			# print("i",i)
			# print("img",img.shape)
			for key in value.keys():
				box=value[key]
				# print("box",box)

				check_box=True #check box in image
				if(box is None):
					# print("continue")
					continue
				if(len(box)>4):
					check_box=False
				x=box[0]
				y=box[1]
				if(x<0):
					x=0
				if(y<0):
					y=0
				w=box[2]-box[0]
				h=box[3]-box[1]
				# print("x,y,w,h",x,y,w,h)
				crop_img = img[y:y+h, x:x+w]
				name_image= f"{key}_img{i}.jpg"
				path_key = os.path.join(folder_save ,key )
				# if not os.path.isdir(path_key):
				# 	os.mkdir(path_key)
				path_out = os.path.join(folder_save ,name_image )
				# print("path_out ", path_out)
				cv2.imwrite(path_out,crop_img)


	def draw_boxes2(self,imgs,arr_dict_boxes,indexs, base_save_folder):

		for i,value in enumerate(arr_dict_boxes):
			# if(i==1):
			# 	print("****************************")
			# 	print("arr_dict_boxes",arr_dict_boxes[1])
			img=copy.copy(imgs[i])

			for key in value.keys():
				# print("key....",key)
				box=value[key]
				check_frame_other=False
				if(box is None):
					continue
				if(len(box)>4):
					# print("box >4 ",key, len(box))
					check_frame_other=True
				start_point=(box[0],box[1])
				end_point=(box[2],box[3])
				color=(255,0,0)
				img = cv2.putText(img, key, start_point, cv2.FONT_HERSHEY_SIMPLEX,
							2, (0,0,255), 5, cv2.LINE_AA)
				if(check_frame_other == False):
					img=cv2.rectangle( img,start_point ,end_point ,color , 5)
				else:
					img=cv2.rectangle( img,start_point ,end_point ,(0,255,0), 5)

			path_out= os.path.join(base_save_folder  ,  indexs[i]+".jpg")
			# print("path_out ", path_out)

			cv2.imwrite(path_out,img)

	def draw_boxes(self,imgs,box_new,indexs):
		for i,value in enumerate(box_new):
			img=imgs[i]
			for key in value.keys():
				for j,box in enumerate(value[key]):
					start_point=(box[0],box[1])
					end_point=(box[2],box[3])
					color=(255,0,0)
					if(key!="origin"):
						color=(0,255,0)
						img = cv2.putText(img, key.split("_")[-1], start_point, cv2.FONT_HERSHEY_SIMPLEX,
								2, color, 5, cv2.LINE_AA)
					img=cv2.rectangle( img,start_point ,end_point ,color , 5)
			cv2.imwrite("results/IMG_5335/"+indexs[i]+".jpg",img)

	def tracking_obj(self,imgs,boxes_img,indexs,matrix):
		# boxes_img[0]=filter_iou(boxes_img[0])
		# print("*******************start backward*******************")
		boxes_img=self.tracking_obj_backward(imgs,boxes_img,indexs,matrix)
		# print("*******************end backward*******************")
		for i in range(len(boxes_img[0])):
			if(None in boxes_img[0]):
				boxes_img[0].remove(None)

		arr_dict_box=[]
		arr_dict_age=[]
		#create id box image 0
		dict_box_0={}
		age_box_0={}
		maxkey=-1
		for i,box in enumerate(boxes_img[0]):
			dict_box_0[f"{i}"]= box
			age_box_0[f"{i}"]=0
			maxkey=i
		arr_dict_box.append(dict_box_0)
		arr_dict_age.append(age_box_0)
		height,width,_=imgs[0].shape
		scale=matrix["scale"]/max(width,height)

		list_of_pairs = get_pair_list_foward(indexs)
		for pair in list_of_pairs:
			# print("pair",pair)
			
			box_f1_sorted_process=convert_box_scale(boxes_img,scale)

			check_pair_name_before_forward= pair[0]+"_"+pair[1]
			index_pair= matrix["key_image"].index(check_pair_name_before_forward)
			homography= (matrix["matrix"][index_pair])
			
			area = intersec_area_3(width*scale,height*scale,homography)
			# print("area......",area)
			# if(area>0.4): #check code cut image correct ?
			
			box_before1=box_f1_sorted_process[indexs.index(pair[0])]
			box_before2=convet_coordinate2rect(box_before1,homography)
			box_before2=convert_box_scale([box_before2],max(width,height)/matrix["scale"])[0]
			index_bf2= indexs.index(pair[1])
			###add code####
			#get index frame other
			index_bf1= indexs.index(pair[0])
			dict_img2_temp=self.get_key_correspond(arr_dict_box[index_bf1],box_before2)
			##add code here########
			box_before2_new=self.add_boxs(boxes_img[index_bf2],box_before2)
			box_before2_new=self.filter_box(box_before2_new,width,height)
			####add###
			pair_index_attensions=[]
			pair_indexs=get_pair_index(box_before2_new)
			
			for i_pair_index,pair_index in enumerate(pair_indexs):
				intersec_percent1=get_intersec_max(box_before2_new[pair_index[0]],box_before2_new[pair_index[1]],width,height)
				if(intersec_percent1>0.85):
					pair_index_attensions.append(pair_index)
			# print("pair_index_attensions",pair_index_attensions)
			for pair_index in pair_index_attensions:
				if(box_before2_new[pair_index[0]] is None or box_before2_new[pair_index[1]] is None):
					continue
				box_save=fix_box_merge(box_before2_new[pair_index[0]],box_before2_new[pair_index[1]])
				box_before2_new[pair_index[0]]=None
				box_before2_new[pair_index[1]]=None
				box_before2_new.append(box_save)
			##########
			boxes_img[index_bf2]=box_before2_new
			
			#######################
			
			dict_img2,maxkey,dict_age_correspond=self.labeling_list_img2(dict_img2_temp,boxes_img[index_bf2],maxkey,arr_dict_age,index_bf1,arr_dict_box)
			dict_age_correspond,dict_img2,boxes_img=delete_key(dict_img2,dict_age_correspond,boxes_img,index_bf2)
			
			arr_dict_age.append(dict_age_correspond)
			arr_dict_box.append(dict_img2)
		# print("box image 39 : ",arr_dict_box[3])
		# print("box image 90 : ",arr_dict_box[11])
		#### fix box #############
		for i,dict_box in enumerate(arr_dict_box):
			for key in dict_box.keys():
				for k,value in enumerate(dict_box[key]):

					if(value =="other frame"):
						continue
					if(value<0):
						arr_dict_box[i][key][k]=0
					elif(k%2==0):
						if(value>width):
							arr_dict_box[i][key][k]=width
					elif(k%2!=0):
						if(value>height):
							arr_dict_box[i][key][k]=height
						
		##########################
		return imgs,arr_dict_box

	def tracking_obj_backward(self,imgs,boxes_img,indexs,matrix):
		# boxes_img[-1]=filter_iou(boxes_img[-1])

		# print("matrix",matrix.keys())
		###reverse#########
		boxes_img.reverse()
		indexs.reverse()
		###############################

		# exit()
		arr_dict_box=[]
		arr_dict_age=[]
		#create id box image 0
		dict_box_0={}
		age_box_0={}
		maxkey=-1
		for i,box in enumerate(boxes_img[0]):
			dict_box_0[f"{i}"]= box
			age_box_0[f"{i}"]=0
			maxkey=i

		arr_dict_box.append(dict_box_0)
		arr_dict_age.append(age_box_0)
		height,width,_=imgs[0].shape
		scale=matrix["scale"]/max(width,height)

		list_of_pairs = get_pair_list_foward(indexs)
		for pair in list_of_pairs:
			# print("pair",pair)
			box_f1_sorted_process=convert_box_scale(boxes_img,scale)
			check_pair_name_backward= pair[1]+"_"+pair[0]
			if(check_pair_name_backward in matrix["key_image"]):
				index_pair= matrix["key_image"].index(check_pair_name_backward)
			homography= inv(matrix["matrix"][index_pair])
			area = intersec_area_3(width*scale,height*scale,homography)
			# print("area......",area)
			# if(area>0.4): #check code cut image correct ?
			box_before1=box_f1_sorted_process[indexs.index(pair[0])]

			# if(len(box_before1)>0):
			box_before2=convet_coordinate2rect(box_before1,homography)
			box_before2=convert_box_scale([box_before2],max(width,height)/matrix["scale"])[0]
			index_bf2= indexs.index(pair[1])
			###add code####
			#get index frame other
			index_bf1= indexs.index(pair[0])
			
			# print("arr_dict_box[index_bf1]",len(arr_dict_box[index_bf1]))

			dict_img2_temp=self.get_key_correspond(arr_dict_box[index_bf1],box_before2)
			
			##add code here########
			# print("boxes_img[index_bf2] before",len(boxes_img[index_bf2]))
			box_before2_new=self.add_boxs(boxes_img[index_bf2],box_before2,"backward")
			box_before2_new=self.filter_box(box_before2_new,width,height)
	
			####add###
			pair_index_attensions=[]
			pair_indexs=get_pair_index(box_before2_new)
			for pair_index in pair_indexs:
				intersec_percent1=get_intersec_max(box_before2_new[pair_index[0]],box_before2_new[pair_index[1]],width,height)
				if(intersec_percent1>0.85):
					pair_index_attensions.append(pair_index)
			for pair_index in pair_index_attensions:
				if(box_before2_new[pair_index[0]] is None or box_before2_new[pair_index[1]] is None):
					continue
				box_save=fix_box_merge(box_before2_new[pair_index[0]],box_before2_new[pair_index[1]])
				# box_save.append("other frame")
				box_before2_new[pair_index[0]]=None
				box_before2_new[pair_index[1]]=None
				box_before2_new.append(box_save)
			##########
			boxes_img[index_bf2]=box_before2_new
			#######################

			dict_img2,maxkey,dict_age_correspond=self.labeling_list_img2(dict_img2_temp,boxes_img[index_bf2],maxkey,arr_dict_age,index_bf1,arr_dict_box)
	
			dict_age_correspond,dict_img2,boxes_img=delete_key(dict_img2,dict_age_correspond,boxes_img,index_bf2)
			arr_dict_age.append(dict_age_correspond)
			arr_dict_box.append(dict_img2)
			# else:
			# 	arr_dict_age.append({})
			# 	arr_dict_box.append({})

		boxes_img.reverse()
		indexs.reverse()
		arr_dict_box.reverse()
		arr_dict_age.reverse()
		# print("check age", arr_dict_age[0])
		return boxes_img
		# return imgs,arr_dict_box


	def filter_box(self,list_box,width,height):
		list_box_new=[]

		p = Polygon([(0,0),(width,0),(width,height),(0,height)])
		for i,box in enumerate(list_box):
			if(box is None):
				continue
			q = Polygon([(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])])
			intersec= p.intersection(q).area
			if(intersec<0.00000000000000000001 or intersec/q.area <0.2):
				continue
			# for k,value in enumerate(box):
			# 	if(value<0):
			# 		box[k]=0
			# 	elif(k%2==0):
			# 		if(value>width):
			# 			box[k]=width
			# 	elif(k%2!=0):
			# 		if(value>height):
			# 			box[k]=height
			list_box_new.append(box)
		return list_box_new
	def add_boxs(self,list_box,list_box_add,mode="forward"):
		for box_check in list_box_add:
			check_box=self.check_add_box(box_check,list_box)
			if(check_box==False):
				if(mode=="backward"):
					list_box.append(box_check+["other frame"])
				else:
					list_box.append(box_check+["other frame"])
		return list_box



	def check_add_box(self,box_check,arr_box):
		# print("len arr box",len(arr_box))
		p = Polygon([(box_check[0],box_check[1]),(box_check[2],box_check[1]),(box_check[2],box_check[3]),(box_check[0],box_check[3])])
		# print("p ", p.area)
		for box in arr_box:
			if(box is None):
				continue
			q = Polygon([(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])])
			# print("q ", q.area)
			intersec= p.intersection(q).area
			if(intersec<0.00000000000000000001):
				continue
			intersec_percent1= intersec/p.area
			intersec_percent2= intersec/q.area
			# print("intersec_percent",intersec_percent1 , intersec_percent2)
			if(max(intersec_percent1,intersec_percent2)>0.7):
				return True
		return False

	def get_key_correspond(self,dict_box_img1,box_img2_correspond):
		#note  box_img2_correspond= dict_box_img1@homo
		dict_box_img2={}
		# print("dict_box_img1",dict_box_img1)
		for i,key in enumerate(dict_box_img1.keys()):
			dict_box_img2[key]=box_img2_correspond[i]
		return dict_box_img2

	def labeling_list_img2(self,dict_boxes, boxes,maxkey,arr_dict_age,index_bf1,arr_dict_box):
		# print("start labeling_list_img2")
		list_dict_correspond={}
		list_age_correspond={}
		arr_boxes_new=merge_boxes(dict_boxes,boxes,maxkey)
		#caculate avg height arr_boxes_new
		sum_height_boxes_new=0
		for i,box in enumerate(arr_boxes_new):
			h= box[3]-box[1]
			sum_height_boxes_new=sum_height_boxes_new+h
		if(len(arr_boxes_new)>0):
			avg_height_boxes_new=sum_height_boxes_new/len(arr_boxes_new)
		#########################################

		check_key_used=[]
		for box in boxes:

			if(box is None):
				continue
			key_save,box_,maxkey,check_first,check_box_new=self.labeling_box_img2(dict_boxes,box,maxkey,check_key_used)
			check_key_used.append(key_save)
			# print("box_,key_save, len(arr_boxes_new),check_box_new",box_,key_save,len(arr_boxes_new),check_box_new)
			##### relation box##############
			if(check_box_new==True and len(arr_boxes_new)>0):

				
				box_near,vector= get_boxNear_vector(box_, arr_boxes_new)
				
				if(vector is not None):
					
					key_box_near,_,_,_,_=self.labeling_box_img2(dict_boxes,box_near,maxkey,[])
					
					###
					# center_box_=[(box_[0]+box_[2])/2,(box_[1]+box_[3])/2]
					# center_box_near=[(box_near[0]+box_near[2])/2,(box_near[1]+box_near[3])/2]
					# dst_box_new=math.sqrt((center_box_[0]-center_box_near[0])**2+ (center_box_[1]-center_box_near[1])**2)
					###
					dst_box_new=math.sqrt((box_[0]-box_near[0])**2+ (box_[1]-box_near[1])**2)
					# dst_box_new=abs(box_[0]-box_near[0])
					key=None
					for i_,dict_boxes_reverse in enumerate(reversed(arr_dict_box)):
						if(key_box_near in dict_boxes_reverse.keys()):
							key= get_key_dst_min(vector,dict_boxes_reverse[key_box_near],dict_boxes_reverse,index_bf1)
							if(key is not None and key not in list(arr_dict_box[-1].keys())):
															
								###caculate dst box old
								box1= dict_boxes_reverse[key_box_near]
								box2= dict_boxes_reverse[key]
								#####
								# center_box1=[(box1[0]+box1[2])/2,(box1[1]+box1[3])/2]
								# center_box2=[(box2[0]+box2[2])/2,(box2[1]+box2[3])/2]
								# dst_box_old= math.sqrt((center_box1[0]-center_box2[0])**2+(center_box1[1]-center_box2[1])**2)
								#####
								dst_box_old= math.sqrt((box1[0]-box2[0])**2+(box1[1]-box2[1])**2)
								# dst_box_old=abs(box1[0]-box2[0])
								list_boxes_old= list(dict_boxes_reverse.values())
								#caculate avg height arr_boxes_new
								sum_height_boxes_old=0
								for i,box_old in enumerate(list_boxes_old):
									h= box_old[3]-box_old[1]
									sum_height_boxes_old=sum_height_boxes_old+h
								avg_height_boxes_old=sum_height_boxes_new/(len(list_boxes_old))
								#########################################
				
								if(abs((dst_box_new/(box_near[3]-box_near[1]))-(dst_box_old/(box1[3]-box1[1])))<0.23):
									key_save= key
									maxkey=maxkey-1
								break
			###############################
			list_dict_correspond[key_save]=box_#+["other frame"]
			if(key_save in list(arr_dict_age[index_bf1].keys()) and "other frame" in box_):
				list_age_correspond[key_save]=arr_dict_age[index_bf1][key_save]+1
			else:
				list_age_correspond[key_save]=0
			
		return list_dict_correspond,maxkey,list_age_correspond

	def labeling_box_img2(self,dict_boxes, box,maxkey,check_key_used):
		# dict_correspond={}
		# print("dict_boxes_test",dict_boxes)

		key_save=None
		check_first=False
		check_box_new=False
		p=Polygon([(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])])
		# maxkey=None
		for i,key in enumerate(dict_boxes.keys()):
			if(key in check_key_used):
				continue
			# print("key",key)
			# print("box",box)
			box_temp=dict_boxes[key]
			q=Polygon([(box_temp[0],box_temp[1]),(box_temp[2],box_temp[1]),(box_temp[2],box_temp[3]),(box_temp[0],box_temp[3])])
			intersec= p.intersection(q).area
			intersec_percent1= intersec/p.area
			intersec_percent2= intersec/q.area
			# print("intersec_percent",max(intersec_percent1,intersec_percent2))
			if(intersec<0.00000001):
				continue
			if(max(intersec_percent1,intersec_percent2)>0.7):
				key_save= key
				check_first=True
				if(q.area>p.area):
					box=box_temp
				break
			
		if(key_save is None):

			maxkey+=1
			check_box_new=True
			key_save=str(maxkey)
			# print("key_save ,check_first",key_save, check_first)
			
		# dict_correspond[key_save]=box
		return key_save,box,maxkey,check_first,check_box_new



	def convert_video2data(self, path_video , id_video , base_save_paths='output' ,  threshold_max=0.6, threshold_min=0.44,start_frame = 30,  max_size=720):

		images_list = []
		# cut room.
		# print("images_path " , images_path , folder_save_matrix , folder_save_labels)
		# exit()
		ret_process ,list_img_save,  images_list , matrix_save, video_time, time_process = self.cvtVideo2Image([path_video] , threshold_max=threshold_max , threshold_min= threshold_min,start_frame = start_frame,  max_size=max_size)
		print("size image cut: " , len(images_list))
		print("end cut image !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		if not ret_process:
			return ret_process, [] ,[], [],None, video_time, 0
		# run detect
		label_detections = self.run_detect_images(list_img_save )

		print("end detect !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		return ret_process ,list_img_save,  images_list ,label_detections,  matrix_save, video_time, time_process


	@staticmethod
	def variance_of_laplacian(image):
		# compute the Laplacian of the image and then return the focus
		# measure, which is simply the variance of the Laplacian
		return cv2.Laplacian(image, cv2.CV_64F).var()

	def checkBlurry(self, image, threshold=80):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		fm = self.variance_of_laplacian(gray)
		return fm

	def resize_image(self, image,max_size=720):
		height,width= image.shape[:2]
		if(height>width):
			scale= max_size/height
		else:
			scale= max_size/width

		width= int(width*scale)
		height=int(height*scale)
		image= cv2.resize(image,(width,height))
		return image

	def getHomography(self, gray1, gray2, matrix, area_current, estimate=0.9):
		homography=find_homography(gray1,gray2)
		#save matrix
		area = 0
		if(matrix is None):
			if (homography is not None):
				matrix= homography
				area= intersec_area_2(gray2,matrix)
			else:
				area= area_current*estimate

		else:
			if (homography is not None):
				matrix=   np.matmul(matrix,homography) #matrix@homography
				area= intersec_area_2(gray2,matrix)
			else:
				area= area_current*estimate
		return matrix, homography,  area

	def updateHomography(self , homographys , index_start , index_end):
		matrix = None
		if index_start >= index_end:
			return matrix

		h_matrix = homographys[index_start : index_end]
		matrix = h_matrix[0]
		for i , h in enumerate(h_matrix):
			if i > 0 and h is not None :
				matrix=  np.matmul(matrix,h)#matrix@h

		return matrix



	def cvtVideo2Image_update(self, list_src , base_save_images ,  base_save_matrix, threshold_max=0.6,threshold_min=0.44, start_frame = 30 , max_size=720):
		
		if(os.path.exists(base_save_matrix)==False):
			# print("check")
			os.mkdir(base_save_matrix)
		indexs_matrix_old = []
		matrix_save_forward_old = []
		path_matrix = os.path.join(base_save_matrix , "matrix_forward.pickle" )
		if(os.path.exists(path_matrix)==True):
			with open(path_matrix,"rb") as f:
				matrix_all_before= pickle.load(f)
				indexs_matrix_old = matrix_all_before["key_image"]
				matrix_save_forward_old = matrix_all_before["matrix"]
		ret_cutFrame = True
		count_image = 0
		images_list = []
		matrix_save_forward = []
		indexs_matrix = []
		count_f = 0
		video_time = 0
		time_process = 0
		for path in list_src:

			vid = cv2.VideoCapture(path)
			name_base = os.path.basename(path).split(".")[0]
			if not(vid.isOpened()):
				continue

			max_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
			fps = int(vid.get(cv2.CAP_PROP_FPS))

			# calculate dusration of the video
			seconds = max_frame / fps
			video_time = seconds
			time_process = seconds
			if count_f %3 != 0:
				continue
			ret = True
			current_frame = 0
			max_val = 0
			image_save = None
			gray_prew = None
			gray_current = None
			matrix= None
			area_current = 1
			matrix_update = None
			index_start  = 0
			index_start_back = 0
			index_end = 0
			homography_save_forward = []
			scale_x = 1.0
			scale_y = 1.0
			while(ret):
				ret, frame = vid.read()
				if frame is None:
					break
				frame = self.checkDimensionImage(frame)
				current_frame += 1
				if gray_prew is None:
					gray_prew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					gray_prew=self.resize_image(gray_prew, max_size=max_size)
				if current_frame <= start_frame:
					gray_prew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					gray_prew=self.resize_image(gray_prew, max_size=max_size)
					if current_frame == start_frame:
						index_start = 0
						path_out = os.path.join(base_save_images, f'{name_base}_{current_frame}.jpg' )
						images_list.append(path_out)
						cv2.imwrite(path_out,frame)
						count_image +=1
					continue
				gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				gray_current=self.resize_image(gray_current, max_size=max_size)
				matrix , homography, area_current = self.getHomography(gray_prew,gray_current , matrix, area_current)
				homography_save_forward.append(homography)
				if area_current >= threshold_min and area_current <= threshold_max:
					# print("area_current >= threshold_min and <= threshold_max")
					fm=self.variance_of_laplacian(gray_current)
					if max_val < fm:
						# print("change fm")
						max_val = fm
						image_save = frame
						index_end = current_frame
					# print("index_end " ,index_start ,  index_end , current_frame )
				elif area_current < threshold_min and image_save is not None:

					if index_start >= index_end:
						index_end = current_frame
						image_save = frame
					# print("area_current <= threshold_min and image_save is not none")
					max_val = 0
					path_out = os.path.join(base_save_images, f'{name_base}_{index_end}.jpg' )
					# print("start_frame",start_frame)
					index_end -= (start_frame )

					matrix_update = self.updateHomography(homography_save_forward ,index_start, index_end )
					matrix =  self.updateHomography(homography_save_forward ,index_end, current_frame - start_frame )
					if matrix_update is None:
						if current_frame < 0.7*max_frame:
							ret_cutFrame = False
						image_save= None
						time_process = current_frame / fps
						break
					scale_x *= abs(matrix_update[0][0])
					
					scale_y *= abs(matrix_update[1][1])
					scale_1 = scale_x
					scale_2 = scale_y
					if scale_1 < 1:
						scale_1 = 1/scale_1
					if scale_2 < 1:
						scale_2 = 1/scale_2
					# print("index_end " ,index_start ,  index_end , current_frame , 0.8*max_frame , scale_1 , scale_2  )
					if scale_2 > 4 or scale_2 > 4:
						if current_frame < 0.7*max_frame:
							ret_cutFrame = False
						image_save= None
						time_process = current_frame / fps
						break
					# area= intersec_ar	ea_2(gray_prew,matrix)
					# print("area " , area , area_current )
					
					index_start_back = index_start
					index_start = index_end
					matrix_save_forward.append(matrix_update)
					size_im = len(images_list)
					name1 = os.path.basename( images_list[size_im-1])
					name2 = os.path.basename( path_out)
					name1 = os.path.splitext(name1)[0]
					name2 = os.path.splitext(name2)[0]
					indexs_matrix.append( name1+ "_"+ name2 )
					images_list.append(path_out)
					cv2.imwrite(path_out,image_save)


					count_image +=1
					area_current = 1

				gray_prew = gray_current
			if area_current > threshold_min  and area_current < threshold_max and image_save is not None:
				path_out = os.path.join(base_save_images, f'{name_base}_{index_end}.jpg' )
				index_end -= (start_frame  )
				matrix_update = self.updateHomography(homography_save_forward ,index_start, index_end )

				index_start = index_end
				matrix_save_forward.append(matrix_update)

				size_im = len(images_list)
				name1 = os.path.basename( images_list[size_im-1])
				name2 = os.path.basename( path_out)
				name1 = os.path.splitext(name1)[0]
				name2 = os.path.splitext(name2)[0]
				indexs_matrix.append( name1+ "_"+ name2 )
				images_list.append(path_out)
				cv2.imwrite(path_out,image_save)
		indexs_matrix = indexs_matrix + indexs_matrix_old
		matrix_save_forward = matrix_save_forward + matrix_save_forward_old
		# print("matrix_save_forward",matrix_save_forward.shape)
		with open(path_matrix,"wb") as  f:
			pickle.dump({'key_image':indexs_matrix,'matrix':matrix_save_forward, 'scale':max_size},f)

		with open(path_matrix,"rb") as f:
			matrix_all_before= pickle.load(f)
		print(f'count image before: {count_image}')
		if  count_image < 2 :
			ret_cutFrame = False
		return ret_cutFrame, images_list, video_time, time_process
	
	def cvtVideo2Image(self, list_src , threshold_max=0.6,threshold_min=0.44, start_frame = 30 , max_size=720):
		indexs_matrix_old = []
		
		ret_cutFrame = True
		count_image = 0
		images_list = []
		list_img_save = []
		matrix_save_forward = []
		indexs_matrix = []
		count_f = 0
		video_time = 0
		time_process = 0
		
		for path in list_src:

			vid = cv2.VideoCapture(path)
			name_base = os.path.basename(path).split(".")[0]
			if not(vid.isOpened()):
				continue

			max_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
			fps = int(vid.get(cv2.CAP_PROP_FPS))

			# calculate dusration of the video
			seconds = max_frame / fps
			video_time = seconds
			time_process = seconds
			if count_f %3 != 0:
				continue
			ret = True
			current_frame = 0
			max_val = 0
			image_save = None
			gray_prew = None
			gray_current = None
			matrix= None
			area_current = 1
			matrix_update = None
			index_start  = 0
			index_start_back = 0
			index_end = 0
			homography_save_forward = []
			scale_x = 1.0
			scale_y = 1.0
			while(ret):
				ret, frame = vid.read()
				if frame is None:
					break
				frame = self.checkDimensionImage(frame)
				current_frame += 1
				if gray_prew is None:
					gray_prew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					gray_prew=self.resize_image(gray_prew, max_size=max_size)
				if current_frame <= start_frame:
					gray_prew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					gray_prew=self.resize_image(gray_prew, max_size=max_size)
					if current_frame == start_frame:
						index_start = 0
						path_out =  f'{name_base}_{current_frame}' 
						images_list.append(path_out)
						list_img_save.append(frame)
						# cv2.imwrite(path_out,frame)
						
						count_image +=1
					continue
				gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				gray_current=self.resize_image(gray_current, max_size=max_size)
				matrix , homography, area_current = self.getHomography(gray_prew,gray_current , matrix, area_current)
				homography_save_forward.append(homography)
				if area_current >= threshold_min and area_current <= threshold_max:
					# print("area_current >= threshold_min and <= threshold_max")
					fm=self.variance_of_laplacian(gray_current)
					if max_val < fm:
						# print("change fm")
						max_val = fm
						image_save = frame
						index_end = current_frame
					# print("index_end " ,index_start ,  index_end , current_frame )
				elif area_current < threshold_min and image_save is not None:

					if index_start >= index_end:
						index_end = current_frame
						image_save = frame
					# print("area_current <= threshold_min and image_save is not none")
					max_val = 0
					path_out = f'{name_base}_{index_end}' 
					# print("start_frame",start_frame)
					index_end -= (start_frame )

					matrix_update = self.updateHomography(homography_save_forward ,index_start, index_end )
					matrix =  self.updateHomography(homography_save_forward ,index_end, current_frame - start_frame )
					if matrix_update is None:
						if current_frame < 0.7*max_frame:
							ret_cutFrame = False
						image_save= None
						time_process = current_frame / fps
						break
					scale_x *= abs(matrix_update[0][0])
					
					scale_y *= abs(matrix_update[1][1])
					scale_1 = scale_x
					scale_2 = scale_y
					if scale_1 < 1:
						scale_1 = 1/scale_1
					if scale_2 < 1:
						scale_2 = 1/scale_2
					# print("index_end " ,index_start ,  index_end , current_frame , 0.8*max_frame , scale_1 , scale_2  )
					if scale_2 > 4 or scale_2 > 4:
						if current_frame < 0.7*max_frame:
							ret_cutFrame = False
						image_save= None
						time_process = current_frame / fps
						break
					# area= intersec_ar	ea_2(gray_prew,matrix)
					# print("area " , area , area_current )
					
					index_start_back = index_start
					index_start = index_end
					matrix_save_forward.append(matrix_update)
					size_im = len(images_list)
					name1 =images_list[size_im-1]
					name2 =  path_out
					
					indexs_matrix.append( name1+ "_"+ name2 )
					images_list.append(path_out)
					list_img_save.append(image_save)
					# cv2.imwrite(path_out,image_save)


					count_image +=1
					area_current = 1

				gray_prew = gray_current
			if area_current > threshold_min  and area_current < threshold_max and image_save is not None:
				path_out =f'{name_base}_{index_end}' 
				index_end -= (start_frame  )
				matrix_update = self.updateHomography(homography_save_forward ,index_start, index_end )

				index_start = index_end
				matrix_save_forward.append(matrix_update)

				size_im = len(images_list)
				name1 =images_list[size_im-1]
				name2 =  path_out
				indexs_matrix.append( name1+ "_"+ name2 )
				images_list.append(path_out)
				list_img_save.append(image_save)
				# cv2.imwrite(path_out,image_save)
		indexs_matrix = indexs_matrix 
		matrix_save_forward = matrix_save_forward 
		# print("matrix_save_forward",matrix_save_forward.shape)
		matrix_save = {}
		matrix_save['key_image'] = indexs_matrix
		matrix_save['matrix'] = matrix_save_forward
		matrix_save['scale'] = max_size
		
		print(f'count image before: {count_image}')
		if  count_image < 2 :
			ret_cutFrame = False
		return ret_cutFrame ,list_img_save,  images_list , matrix_save, video_time, time_process

	def run_detect(self, images_path,folder_save ):
		weights = [self.weight_path]
		imgsz = [640, 640]
		conf_thres = 0.6
		iou_thres=0.8  # NMS IOU threshold
		max_det=1000  # maximum detections per image
		device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
		view_img=False # show results
		save_txt=True  # save results to *.txt
		save_conf=False  # save confidences in --save-txt labels
		save_crop=False  # save cropped prediction boxes
		nosave=False  # do not save images/videos
		classes=None  # filter by class: --class 0, or --class 0 2 3
		agnostic_nms=False  # class-agnostic NMS
		augment=False  # augmented inference
		visualize=False  # visualize features
		update=False  # update all models
		project=folder_save  # save results to project/name
		exist_ok=False  # existing project/name ok, do not increment
		line_thickness=3  # bounding box thickness (pixels)
		hide_labels=False  # hide labels
		hide_conf=False  # hide confidences
		half=False  # use FP16 half-precision inference
		dnn=False  # use OpenCV DNN for ONNX inference

		detect_images(weights=weights,  # model.pt path(s)
			source=images_path,  # file/dir/URL/glob, 0 for webcam
			imgsz=imgsz,  # inference size (pixels)
			conf_thres=conf_thres,  # confidence threshold
			iou_thres=iou_thres,  # NMS IOU threshold
			max_det=max_det,  # maximum detections per image
			device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
			view_img=view_img,  # show results
			save_txt=save_txt,  # save results to *.txt
			save_conf=save_conf,  # save confidences in --save-txt labels
			save_crop=save_crop,  # save cropped prediction boxes
			nosave=nosave,  # do not save images/videos
			classes=classes,  # filter by class: --class 0, or --class 0 2 3
			agnostic_nms=agnostic_nms,  # class-agnostic NMS
			augment=augment,  # augmented inference
			visualize=visualize,  # visualize features
			update=update,  # update all models
			project=project,  # save results to project/name
			exist_ok=exist_ok,  # existing project/name ok, do not increment
			line_thickness=line_thickness,  # bounding box thickness (pixels)
			hide_labels=hide_labels,  # hide labels
			hide_conf=hide_conf,  # hide confidences
			half=half,  # use FP16 half-precision inference
			dnn=dnn  # use OpenCV DNN for ONNX inference
			)

	def run_detect_images(self, images , im_size=640 ):
		weights = [self.weight_path]
		imgsz = [im_size, im_size]
		conf_thres = 0.6
		iou_thres=0.8  # NMS IOU threshold
		max_det=1000  # maximum detections per image
		device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
		classes=None  # filter by class: --class 0, or --class 0 2 3
		agnostic_nms=False  # class-agnostic NMS
		augment=False  # augmented inference
		visualize=False  # visualize features
		update=False  # update all models
		exist_ok=False  # existing project/name ok, do not increment
		line_thickness=3  # bounding box thickness (pixels)
		hide_labels=False  # hide labels
		hide_conf=False  # hide confidences
		half=False  # use FP16 half-precision inference
		dnn=False  # use OpenCV DNN for ONNX inference

		label_detections = detect_images(weights=weights,  # model.pt path(s)
									img_list=images,  # file/dir/URL/glob, 0 for webcam
									imgsz=imgsz,  # inference size (pixels)
									conf_thres=conf_thres,  # confidence threshold
									iou_thres=iou_thres,  # NMS IOU threshold
									max_det=max_det,  # maximum detections per image
									device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
									classes=classes,  # filter by class: --class 0, or --class 0 2 3
									agnostic_nms=agnostic_nms,  # class-agnostic NMS
									augment=augment,  # augmented inference
									visualize=visualize,  # visualize features
									update=update,  # update all models
									exist_ok=exist_ok,  # existing project/name ok, do not increment
									line_thickness=line_thickness,  # bounding box thickness (pixels)
									hide_labels=hide_labels,  # hide labels
									hide_conf=hide_conf,  # hide confidences
									half=half,  # use FP16 half-precision inference
									dnn=dnn  # use OpenCV DNN for ONNX inference
								)
		return label_detections


if __name__ == '__main__':

	output_folder = 'results'

	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)
	jp_process = RevampTracking()
	path_video = "IMG_5337.MOV"
	key = os.path.basename(path_video)
	key = os.path.splitext(key)[0]
	result = os.path.join(output_folder, key)
	if not os.path.isdir(result):
		os.mkdir(result)
	ret_process , image_list, arr_dict_boxes = jp_process.processTracking(path_video  )
