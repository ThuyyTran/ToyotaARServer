import cv2
import copy
import os
import numpy as np

def lk_tracker1(img1 , img2 ):
	m = cv2.mean(img1)
	m2 = cv2.mean(img2)
	res = False
	x_direction = 0 
	y_direction = 0
	M = []
	if(m[0] > 180 or m2[0] > 180  ):
		return res, [x_direction, y_direction] , M
	lkParameters = dict(winSize=(31, 31), maxLevel=5, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
	stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
					20, 0.03)
	cornerA = cv2.goodFeaturesToTrack(img1, mask=None, maxCorners=350, qualityLevel=0.01, minDistance=20, blockSize=5)
	# if(cornerA.size() <= 3 ):
	#     return res, [x_direction, y_direction] , M

	cornerA = np.array(cornerA)
	cv2.cornerSubPix(img1, cornerA, (5, 5), (-1, -1),stop_criteria)
	previousCorners = cornerA.reshape(-1, 1, 2)
	#cornerB = copy.copy(cornerA)
	cornersB, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, previousCorners, None, **lkParameters)
	cornersB = cornersB[st == 1]
	previousCorners = previousCorners[st == 1 ]
	#cornersB = cornersB[err <= 50  ]
	#previousCorners = previousCorners[err <= 50 ]
	if cornersB.shape[0] > 10:
		num_track = 0
		cornersX = []
		cornersY = []
		for i in range(cornersB.shape[0]):
			x, y = cornersB[i]
			xPrev, yPrev = previousCorners[i]

			
			if(err[i] < 50 ):
				cornersX.append(previousCorners[i])
				cornersY.append(cornersB[i])
				#print("errors ",err[i] )
				num_track += 1
				x_direction += (x-xPrev)
				y_direction += (y- yPrev)
		x_direction /= num_track
		y_direction /= num_track
		print("lk tracker size ", len(cornersX))
		cornersX = np.array(cornersX)
		cornersY = np.array(cornersY)
		M, mask = cv2.findHomography(cornersX, cornersY, cv2.RANSAC, 5.0)
		res = True
	return res, [x_direction, y_direction] , M

def lk_tracker_new(img1 , img2 ):
	m = cv2.mean(img1)
	m2 = cv2.mean(img2)
	res = False
	x_direction = 0
	y_direction = 0
	M = []
	if(m[0] > 180 or m2[0] > 180  ):
		return res, [x_direction, y_direction] , M
	lkParameters = dict(winSize=(31, 31), maxLevel=5, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.001))
	stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
					20, 0.03)
	cornerA = cv2.goodFeaturesToTrack(img1, mask=None, maxCorners=350, qualityLevel=0.01, minDistance=20, blockSize=5)
	# if(cornerA.size() <= 3 ):
	#     return res, [x_direction, y_direction] , M

	cornerA = np.array(cornerA)
	cv2.cornerSubPix(img1, cornerA, (5, 5), (-1, -1),stop_criteria)
	previousCorners = cornerA.reshape(-1, 1, 2)
	#cornerB = copy.copy(cornerA)
	cornersB, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, previousCorners, None, **lkParameters)
	cornersC, st2, err2= cv2.calcOpticalFlowPyrLK( img2 , img1, cornersB, None, **lkParameters)
	#cornersB = cornersB[st == 1]
	#previousCorners = previousCorners[st == 1]
	#cornersC = cornersC[st2 == 1]
	# cornersB = cornersB[err <= 50  ]
	# previousCorners = previousCorners[err <= 50 ]
	if cornersB.shape[0] > 10:
		num_track = 0
		cornersX = []
		cornersY = []
		for i in range(cornersB.shape[0]):
			if(st[i] == False or st2[i] == False ):
				continue
			x, y = cornersB[i][0]
			xPrev, yPrev = previousCorners[i][0]
			x_back, y_back =cornersC[i][0]
			if (err[i] < 50):
				distance_ = distance.euclidean((xPrev, yPrev), (x_back , y_back) )
				# print("distance ", distance_)
				if(distance_ < 5 ):
					cornersX.append(previousCorners[i])
					cornersY.append(cornersB[i])
					# print("errors ",err[i] )
					num_track += 1
					x_direction += (x - xPrev)
					y_direction += (y - yPrev)


		print("lk tracker size ", len(cornersX))
		if(num_track >= 50 ):
			x_direction /= num_track
			y_direction /= num_track
			cornersX = np.array(cornersX)
			cornersY = np.array(cornersY)
			M, mask = cv2.findHomography(cornersX, cornersY, cv2.RANSAC, 5.0)
			res = True
		del cornersX
		del cornersY
	del cornersB
	del cornerA
	return res, [x_direction, y_direction], M

def predict_motion(model_tracker, maxframe, pre):
	result = [0,0]
	res = False
	if( len(model_tracker) < maxframe + pre):
		print("not predict" )
	else :
		maxframe = maxframe + 2
		A1 = 0.25
		A2 = 0.5
		P = []
		V = []
		num = len(model_tracker)
		max_detect = max(0,int( num - maxframe))
		for i in range(max_detect,num - pre):
			V.append(model_tracker[i])
		for i in range(3, len(V)):
			d1 = A1*(V[i][0] - V[i - 2][0]) + A2*(V[i - 1][0] - V[i - 2][0])
			d2 = A1*(V[i][1] - V[i - 2][1]) + A2*(V[i - 1][1] - V[i - 2][1])
			P.append([d1,d2])
		for i in range(len(P)):
			result = [result[0] + P[i][0] , result[1] + P[i][1] ]
		if(len(P) > 0 ):
			result=[result[0]/len(P), result[1]/len(P)]
			res = True
	return res, result

def getPose(M):
	x_detect = 0 
	y_detect = 0
	pts = np.float32([[150, 150 ], [150, 200 ], [200, 200 ], [200, 150]]).reshape(-1, 1, 2)
	pts2 = cv2.perspectiveTransform(pts, M)
	pts2 -= pts

#box_tracker2 = box_tracker2 + pts2
	n = pts2.shape[0]


	x_up = (pts2[3][0][0] + pts2[2][0][0] - pts2[1][0][0] - pts2[0][0][0])/2.0
	y_up = (pts2[2][0][1] + pts2[1][0][1] - pts2[3][0][1] - pts2[0][0][1])/2.0
	scale_ = (x_up +y_up)/2.0
	for i in range(n):
		x_detect += pts2[i][0][0]
		y_detect += pts2[i][0][1]
	x_detect /= n
	y_detect /= n
	return x_detect, y_detect

def tracker(vcapture):
	width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = vcapture.get(cv2.CAP_PROP_FPS)
	success, prev_image = vcapture.read()
	prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

	tracker_positons = []
	box_draw = []
	backup_ = [0, 0]
	while success:
		success, image = vcapture.read()
		if not success:
			continue

		

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if(prev_gray.shape[-1] <= 0 ):
			prev_gray = copy.copy(gray)
		res, directions_,M  =  lk_tracker_new(prev_gray ,gray )
		
		if(res):
			x_detect ,y_detect =  getPose(M)
		else:
			if(len(tracker_positons) > 10 ):
				res1 , predict_mot  = predict_motion(tracker_positons , 7 , 0)
				x_detect ,y_detect = predict_mot[0], predict_mot[1]
			else:
				x_detect ,y_detect = 0, 0
		backup_ = [backup_[0] + x_detect, backup_[1] + y_detect]
		tracker_positons.append([backup_[0], backup_[1]])
			
		
		if(len(tracker_positons) > 10 ):
			res1 , predict_mot  = predict_motion(tracker_positons , 7 , 0)
			print ("predict " ,predict_mot)
		# 	if(res1):
		# 		box_tracker1= [box_tracker1[0] + predict_mot[0] , box_tracker1[1] + predict_mot[1] , box_tracker1[2], box_tracker1[3]]
				
		# 	else:
		# 		box_tracker1= [box_tracker1[0] +directions_[0] , box_tracker1[1] +directions_[1] , box_tracker1[2], box_tracker1[3]]

		# 		#tracker_positons.append(box_tracker)
		# else:
			
		# 	box_tracker1= [box_tracker1[0] +directions_[0] , box_tracker1[1] +directions_[1] , box_tracker1[2], box_tracker1[3]]
			#tracker_positons.append(box_tracker)
		backup_ = [backup_[0]+directions_[0], backup_[1]+directions_[1]]
		tracker_positons.append([backup_[0], backup_[1]])
		prev_gray = copy.copy(gray)
		if len(tracker_positons) >2 :
			n = len(tracker_positons)
			for i  in range(n):
				cv2.line(image, (tracker_positons[i][0], tracker_positons[i][1]), (tracker_positons[(i+1)%n][0], tracker_positons[(i+1)%n][1]), (255,255,0) , 1)
		cv2.imshow("image" , image)
		cv2.waitKey(30)

if __name__ == '__main__':
	video_path = 'Daiso_iPhone/IMG_5335.MOV'
	vcapture = cv2.VideoCapture(video_path)
	tracker(vcapture)
	# vwriter = cv2.VideoWriter("output/nucleis.avi",
	#                         cv2.VideoWriter_fourcc(*'MJPG'),
	#                         fps, (width, height))
   