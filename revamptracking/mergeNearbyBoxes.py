import cv2
import numpy as np

# tuplify
def tup(point):
	return (point[0], point[1])

# returns true if the two boxes overlap
def overlap(source, target):
	# unpack points
	tl1, br1 = source
	tl2, br2 = target

	
	# checks
	if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
		return False
	if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
		return False
	return True

def cal_iou(box_dt, box_gt):
	
	startX_dt, startY_dt, endX_dt, endY_dt = box_dt
	startX_gt, startY_gt, endX_gt, endY_gt = box_gt

	startX_union_box = max(startX_dt, startX_gt)
	startY_union_box = max(startY_dt, startY_gt)
	endX_union_box = min(endX_dt, endX_gt)
	endY_union_box = min(endY_dt, endY_gt)

	area_boxdt = (endX_dt - startX_dt) * (endY_dt - startY_dt)
	area_boxgt = (endX_gt - startX_gt) * (endY_gt - startY_gt)
	areaBoxIou = (endX_union_box - startX_union_box) * (endY_union_box - startY_union_box)

	if endX_dt < startX_gt or startX_dt > endX_gt:
		iou = 0
	elif endY_dt < startY_gt or startY_dt > endY_gt:
		iou = 0
	else:
		iou = areaBoxIou / (area_boxgt + area_boxdt - areaBoxIou)
	# print("iou ", box_dt, box_gt ,iou )
	return iou

# returns all overlapping boxes
def getAllOverlaps(boxes, bounds, index):
	overlaps = []
	for a in range(len(boxes)):
		if a != index:
			# if overlap(bounds, boxes[a]):
			if cal_iou(bounds, boxes[a]) > 0.2:
				overlaps.append(a)
	return overlaps

def MergeOverlapBoxes(boxes, w, h,merge_margin = 0.):

	# this is gonna take a long time
	finished = False
	while not finished:
		# set end con
		finished = True
		# loop through boxes
		index = 0
		while index < len(boxes):
			# grab current box
			curr = boxes[index]
			est_margin = int(min(curr[3] - curr[1] , curr[2] - curr[0]) *merge_margin)
			curr[0] = max(0, int(curr[0] - est_margin))
			curr[1] = max(0, int(curr[1] - est_margin))
			curr[2] = min(w, int(curr[2] + est_margin))
			curr[3] = min(h, int(curr[3] + est_margin))

			# get matching boxes
			overlaps = getAllOverlaps(boxes, curr, index)
			
			# check if empty
			if len(overlaps) > 0:
				# combine boxes
				# convert to a contour
				con = []
				overlaps.append(index)
				xmax = 0
				ymax = 0
				xmin = 1000
				ymin = 1000
				for ind in overlaps:
					x1, y1, x2, y2 = boxes[ind]
					xmin = min(x1, xmin)
					ymin = min(y1, ymin)
					xmax = max(x2, xmax)
					ymax = max(y2, ymax)
					

				merged = [xmin, ymin, xmax, ymax]

				# remove boxes from list
				overlaps.sort(reverse = True)
				for ind in overlaps:
					del boxes[ind]
				boxes.append(merged)

				# set flag
				finished = False
				break

			# increment
			index += 1
	return boxes


def MergeOverlapBoxesContours(boxes, w, h, threshold_area=0.6):
	ret = True
	if len(boxes) < 2 :
		return ret, boxes
	results = []
	mask = np.zeros((h,w), dtype="uint8")
	for b in boxes:
		mask = cv2.rectangle(mask, (b[0]  , b[1] ) , ( b[2], b[3]), (255), 20)
		mask = cv2.rectangle(mask, (b[0]  , b[1] ) , ( b[2], b[3]), (255), -1)
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imwrite("test/output/mask.jpg", mask)
	hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
	# For each contour, find the bounding rectangle and draw it
	for component in zip(contours, hierarchy , range(len(contours))):
		currentContour = component[0]
		currentHierarchy = component[1]
		x,y,w1,h1 = cv2.boundingRect(currentContour)
		area = cv2.contourArea(currentContour)
		ratio = w1/h1
		if ratio <1 :
			ratio = 1/ ratio
		if ratio > 3.5 :
			continue
		
		results.append([x,y, x+w1, y+h1])
	area = 0
	results_correct = []
	for b in results:
		area +=(b[2]  - b[0])* (b[3]  - b[1])
	if area > threshold_area*(w*h) :
		ret = False
	del hierarchy
		
	return ret, results
	



# # go through the contours and save the box edges
# boxes = [] # each element is [[top-left], [bottom-right]]
# boxes.append([0 , 0 , 30, 30 ])
# boxes.append([5 , 19 , 50, 25 ])
# # go through the boxes and start merging
# print("int boxes " , boxes)
# MergeOverlapBoxes(boxes)
# print("result boxes " , boxes)