from PIL import Image
import os
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import json
from scipy.spatial import distance
import settings
import time 


class Rerank():
	def __init__(self):
		print("init Rerank")
		# model_rerank = DISTS()
		#self.net = model_rerank
		#self.device = torch.device('cuda')
		#self.net.to(self.device)

	def transformRerank(self,image_query, imsize=256,size_output=None):
		# image_query = self.change_contrast(image_query)
		if size_output is not None:
			image_query = image_query.resize(size_output)
		else:
			image_query = transforms.functional.resize(image_query, imsize)
		size_out = image_query.size
		image_query = transforms.ToTensor()(image_query)

		return image_query, size_out

	def change_contrast(self, img_pil):
		img = np.asarray(img_pil)
		img_transf = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		img_transf[:,:,2] = cv2.equalizeHist(img_transf[:,:,2])
		img_out = cv2.cvtColor(img_transf, cv2.COLOR_HSV2RGB)
		# You may need to convert the color.
		im_pil = Image.fromarray(img_out)
		return im_pil

	def nomal_data(self, data):
		if len(data) <=2 :
			return data
		
		max_data = 1
		min_data = -1
		max_data = max_data - min_data
		# print("data nomal " , min_data , max_data)
		for i, e in enumerate(data):
			data[i] =  2 -  (e - min_data)/max_data 
			# data[i] =    (e - min_data)/max_data 
		return data 
		
	def get_histogram_color(self, image, imsize=360, usingRGB=False, query_check=False, threshold_min1=25, threshold_min2=40, threshold_color=80):
		image = self.resize_image(image, max_size=imsize)
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
		ret_query = False
		
		# print(max_color - min_color , mean, mean_color)
		if query_check and (( m_check > threshold_min1 and mean_color > threshold_color )or m_check > threshold_min2 )  :
			ret_query = True
		mean =   (mean[0] - min_color,mean[1] - min_color,mean[2] - min_color)
		# if query_check:
		# 	print("mean_color " ,ret_query, mean, m_check , mean_color , max_color - min_color , max_std_color - min_std_color)
		img_transf = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
		if usingRGB or ret_query:
			hist = cv2.calcHist([img_transf], [0 ], None, [12], [0, 255])
		else:
			hist = cv2.calcHist([img_transf], [0 , 2 ], None, [8 , 16], [0 , 255 , 0 , 255 ])
		
		# hist = cv2.calcHist([img_transf], [0 ], None, [16], [0, 255])
		
		hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
		return hist, ret_query, mean , mean_color

	def resize_image2(self, image,max_size=320):
		height,width= image.shape[:2]
		
		scale= max_size/width
		width= int(width*scale)
		height=int(height*scale)
		if scale > 1:
			return image
		image= cv2.resize(image,(width,height))
		return image

	def resize_image(self, image,max_size=320):
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

	def rerank_hist(self, image_query, list_top, scores_search, path_db, path_hist_folder, using_mtc_hist=True , weight=0.6, threshold_color=50):
		m_query, ret_query, q_dst_color, q_color  = self.get_histogram_color(image_query, usingRGB=False, query_check=True)
		list_color = []
		list_errors = []
		# print("ret_query : " , ret_query)
		if not ret_query:
			return list_top, scores_search , []
		for j, path in enumerate(list_top):
			
			path = path.replace(" ", "")
			file = os.path.join(path_db , path)
			sub_name = os.path.splitext(path)[0]
			hist_path = os.path.join(path_hist_folder , f'{sub_name}.pkl' )
				# img = Image.open(path).convert('RGB')
			need_to_read_image = True
			try:
				file2 = open(hist_path, 'rb')
				m_train, t_dst_color ,t_color = pickle.load(file2)
				diff_color = cv2.compareHist(m_query, m_train, cv2.HISTCMP_CORREL) 
				need_to_read_image = False
			except (OSError, ValueError):
				logger.error("Error opening descs file '%s'. Will try to use the corresponding image file." % hist_path)
			if need_to_read_image:
				try:
					img = cv2.imread(file)
					if img is None:
						list_errors.append(path)
						diff_color = 0
						continue
					# print("time read image ", path , time.time() -ts)
					m_train, _ , t_dst_color, t_color = self.get_histogram_color(img  , usingRGB=ret_query, query_check=False)
					diff_color = cv2.compareHist(m_query, m_train, cv2.HISTCMP_CORREL) 
				except (OSError, ValueError):
					list_errors.append(path)
					diff_color = 0
					logger.error("Error opening descs file '%s'. Will try to use the corresponding image file." % file)
				# diff_color = max(0, diff_color)
				
		
			# print("path_score" ,path , diff_color )
		
			list_color.append(diff_color)
		list_color = self.nomal_data(list_color)
		list_color = np.array(list_color)
		# scores_search = self.nomal_data(scores_search, init =1 )
		scores_search = np.array(scores_search)
		
		# print("scores_search1 ",scores_search)
		# scores_search = list_color
		scores_search = (1- weight)* list_color  + weight*scores_search
		# print("scores_search ",scores_search)
		paths_scores = zip(list_top, np.array(scores_search))
		paths_scores = sorted(paths_scores, key=lambda x:x[1], reverse=False)
		
		# print("m_query " , m_query )
		# print("end rerank  " , paths_scores )
		re_paths = []
		re_scores = []
		
		for path, score in paths_scores:
			# print("path_score" ,path , score )
			score = float(score.item())
			
			re_paths.append(path)
			re_scores.append(score)
		return re_paths, re_scores, list_errors


def get_nth_key(dictionary, n=0):
	if n < 0:
		n += len(dictionary)
	for i, key in enumerate(dictionary.keys()):
		if i == n:
			return key
	raise IndexError("dictionary index out of range")

if __name__ == '__main__':

	with open('total_images_info_update_replate_2.pkl','rb') as read:
		dict_images_info = pickle.load(read)
	print(get_nth_key(dict_images_info), 0)
	model_rerank = Rerank()
	base_query_folder = '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari'
	# query_path = os.path.join(base_query_folder,'Women/Jacket/m25591075493/m25591075493_4.jpg')
	query_path = 'test_query/aoni/aoni_2/b51919ca-8983-436d-94c2-035a53714cf7_1627964790.5400193.jpg'
	mask_path = os.path.join(base_query_folder,'Women/Jacket/m25591075493/mask_m25591075493_4.jpg')
	img_mask = cv2.imread(mask_path, 0)
	path_strip = query_path.replace(" ", "")

	query_image = Image.open(path_strip).convert('RGB')
	full_rt_paths = ['/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m52862128949/m52862128949_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/2da155bb1b9ec8c92f042a3b7f8144a7/1167798797.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m48006849034/m48006849034_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m39894836373/m39894836373_3.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/2849847566ec7e2761d4cb632389af86/1169643670.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m95749770975/m95749770975_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m50014173166/m50014173166_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m70407194989/m70407194989_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/01c18f115dcd7bf878f7232cfd3db79d/959282273.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m13078277322/m13078277322_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m13092748553/m13092748553_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m40297694247/m40297694247_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m96657283992/m96657283992_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m37645230145/m37645230145_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m16054894305/m16054894305_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m79483526810/m79483526810_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m43684560552/m43684560552_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/3a1fe69d6bf227df9658e0b9394a2861/1211272010.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m18302095953/m18302095953_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m16484249472/m16484249472_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Women/Tops/beadecdbc59e273e9b24d2430e5deef1/1099820333.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m11938100718/m11938100718_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m71643772420/m71643772420_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_2ndstreet/Men/Tops/2320240597039/1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m93933481861/m93933481861_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/d7eb09387a49faca92a003da18711b24/1168767312.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m47456051349/m47456051349_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m19194562268/m19194562268_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m90081290030/m90081290030_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m54410652289/m54410652289_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/21fa2ce7f6ab9a88fecd75ac9189a36a/984494205.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/6342350a7b9d24aad5575780277c6527/984495066.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/c7943524ebc0c46be8b9236771843be9/1145779910.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m65490424602/m65490424602_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/cb876b13890b26578d99514eebac02dd/968128778.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m29783360363/m29783360363_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/e8940476313e76e42013a72aa1db82b8/1154074436.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Men/Tops/2044d771dbd2ad8ea02e8bfc01e778ff/1141691698.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m24917926039/m24917926039_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m60131606660/m60131606660_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m60969980734/m60969980734_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m44970842089/m44970842089_2.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Women/Tops/m23224173788/m23224173788_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m15003492499/m15003492499_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m23240555488/m23240555488_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Women/Tops/c1a518f419e34dca58c5a33d78e72920/90475129.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Men/Tops/m58307246796/m58307246796_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Women/Tops/m70705382499/m70705382499_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_mercari/Women/Tops/m68127527858/m68127527858_1.jpg', '/media/anlabadmin/OS_Window/db_shuppingrobo/db_rakuma/Women/Tops/4d7715a815acc3665f18582a43c5bd27/1013510914.jpg']
	scores =  [0.8158507943153381, 0.6802607774734497, 0.6678892374038696, 0.6418381929397583, 0.624300479888916, 0.6229931712150574, 0.6100004315376282, 0.6029602885246277, 0.6008658409118652, 0.5993368029594421, 0.5979506373405457, 0.59695965051651, 0.5825807452201843, 0.5793135166168213, 0.5768839120864868, 0.5690630078315735, 0.5543713569641113, 0.5520092844963074, 0.5506555438041687, 0.5474848747253418, 0.5471555590629578, 0.5423081517219543, 0.5395427942276001, 0.5317323803901672, 0.5302866697311401, 0.5290616154670715, 0.525022566318512, 0.517804741859436, 0.5061128735542297, 0.49989044666290283, 0.4995759129524231, 0.4995759129524231, 0.49794062972068787, 0.4925263822078705, 0.48145100474357605, 0.46557462215423584, 0.4447961747646332, 0.44034942984580994, 0.429781436920166, 0.3597295582294464, 0.3485390543937683, 0.34845679998397827, 0.34111401438713074, 0.33512216806411743, 0.3296177387237549, 0.3287408649921417, 0.3268633186817169, 0.326429158449173, 0.32601702213287354, 0.3225018382072449]

	# scores = [1]
	# full_rt_paths = []
	# full_rt_paths.append(path_strip)
	# img_mask = cv2.imread(mask_path, 0)
	re_paths,re_scores, list_errors = model_rerank.rerank_hist(query_image, None, dict_images_info,full_rt_paths,scores)
	exit() 
	base_query_folder = '/media/anlabadmin/data_Window/shuppingRobo/shuppingRobo_2ndstreet_crop'
	folder_1 = '/media/anlabadmin/data_Window/shuppingRobo/shuppingRobo_2ndstreet_crop'
	folder_2 = '/media/anlabadmin/data_Window/shuppingRobo/shuppingRobo_friljp_crop'
	folder_3 = '/media/anlabadmin/OS_Window/mercari_crop_20210724'
	dict_result_rerank = {}
	model_rerank = Rerank()
	with open('dict_top50_of_100img_2ndstreet.pkl','rb') as read:
		dict_top100 = pickle.load(read)
	with open('dict_2ndstreet_in_Rakuma_mercari_new.json','r') as read:
		dict_results = json.load(read)
	correct = 0

	for file in tqdm(dict_top100):
		query_path = os.path.join(base_query_folder,file,'1.jpg')
		query_image = Image.open(query_path).convert('RGB')
		top100 = dict_top100[file][0]
		scores = dict_top100[file][1]
		full_rt_paths = []
		true_label = dict_results[file]
		for rt_path in top100:
			full_rt_path = os.path.join(folder_1,rt_path)
			if not os.path.exists(full_rt_path):
				full_rt_path = os.path.join(folder_2,rt_path)
			if not os.path.exists(full_rt_path):
				full_rt_path = os.path.join(folder_3,rt_path)
			full_rt_paths.append(full_rt_path)

		re_paths,re_scores = model_rerank.rerank_hist(query_image,full_rt_paths,scores)
		re_paths = re_paths[:5]
		search = False
		for path in re_paths:
			rt_folder = path.split('/')[-2]
			if rt_folder in true_label:
				search = True
		if search == True:
			correct +=1
	print(correct)

	exit()


	# img_folder = '/media/anlabadmin/data_Window/cnn_cirtorch/search_top10_20210519/test'

	# source_folder = '/media/anlabadmin/data_Window/cnn_cirtorch/search_top10_20210519/test'
	# image_query = '/media/anlabadmin/data_Window/cnn_cirtorch/search_top10_20210519/test/069b86d4-6977-44be-a861-32bdca08add0.jpg'
	# image_query = Image.open(image_query).convert('RGB')
	# list_files = sorted(os.listdir(source_folder))
	# print(list_files)
	# list_top = []
	# for file in list_files:
	# 	# file = file.split('_')[1]
	# 	# file = file.split('.jpg')[0]
	# 	path = os.path.join(img_folder, file)
	# 	list_top.append(path)
	# model = Rerank()
	# re_paths, re_scores = model.rerank(image_query, list_top,bs=1)
	# # model_rerank = DISTS()
	# # re_paths, re_scores = model_rerank.Reranking(image_query,list_top)
	# for path, score in zip(re_paths, re_scores):
	# 	print(path, score)
