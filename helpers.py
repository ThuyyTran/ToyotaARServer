# import the necessary packages
import numpy as np
import base64
import sys
import os
import time
import redis
import settings
import cv2
from datetime import datetime
from PIL import Image
from logger import AppLogger

from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
#
from pathlib import Path
from u2net.data_loader import RescaleT
from u2net.data_loader import ToTensor
from u2net.data_loader import ToTensorLab
from u2net.data_loader import SalObjDatasetFromList

from u2net.model import U2NET  # full size version 173.6 MB
from u2net.model import U2NETP  # small version u2net 4.7 MB

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

logger = AppLogger()

# encode raw data of image


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")

# decode image to raw data


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a


def resize(image, width, height, keep_ratio):
    w, h = image.width, image.height
    if keep_ratio:
        # if w < width and h < height:
        #     return image, w, h
        if w > h:
            w, h = width, int(float(h) * width / w)
        else:
            w, h = int(float(w) * height / h), height
        image = image.resize((w, h), Image.ANTIALIAS)
    else:
        image = image.resize((width, height), Image.ANTIALIAS)
    return image, image.width, image.height


def multi_pop(r, q, n):
    arr = []
    count = 0
    while True:
        try:
            p = r.pipeline()
            p.multi()
            for i in range(n):
                p.lpop(q)
            arr = p.execute()
            return arr
        except redis.ConnectionError:
            count += 1
            logger.error("Connection failed in %s times" % count)
            if count > 3:
                raise
            backoff = count * 5
            logger.info('Retrying in {} seconds'.format(backoff))
            time.sleep(backoff)
            r = redis.StrictRedis(host=settings.REDIS_HOST,
                                  port=settings.REDIS_PORT, db=settings.REDIS_DB)


def try_int(value):
    try:
        return int(value)
    except:
        return -1


def segmentWatershed(image):
    h_orig, w_orig = image.shape[:2]
    mask = np.zeros_like(image[:, :, 0]).astype(np.int32)
    x_est = int(w_orig/2)
    y_est = int(h_orig/2)
    mask[0:5, 0:w_orig] = 1
    mask[0:h_orig, 0:5] = 1
    mask[0:h_orig, w_orig - 5:w_orig] = 1
    mask[h_orig - 5:h_orig, 0:w_orig] = 1
    mask[int(h_orig/2 - y_est/2): int(h_orig/2 + y_est/2),
         int(w_orig/2 - x_est/2): int(w_orig/2 + x_est/2)] = 2
    mask = cv2.watershed(image, mask)
    #cv2.rectangle(masker, (x , y), (x+ w , y + h), (0, 255, 0), 2, cv2.LINE_AA)
    #bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.convertScaleAbs(mask, alpha=1.0, beta=0)
    ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2*2 + 1, 2*2 + 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    #cv2.imshow("mask" , mask)
    return mask


def edgedetect(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    # Some values seem to go above 255. However RGB channels has to be within 0-255
    sobel[sobel > 255] = 255
    return sobel


def getBoundingBox(image, edgeImg):
    contours, heirarchy = cv2.findContours(
        edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h_orig, w_orig = image.shape[:2]
    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)
    #significant = []
    # If contour isn't covering 5% of total area of image then it probably is too small
    tooSmall = h_orig * w_orig / 20
    bounding_box = [0, 0, w_orig, h_orig]
    threshold_center = 0.2
    threshold_area = 0.3
    area_max = 0
    for tupl in level1:
        contour = contours[tupl[0]]
        x, y, w, h = cv2.boundingRect(contour)
        area = w*h
        if area > tooSmall:
            #significant.append([contour, area])
            #significant.append([x ,y , x + w , y+ h ])
            x_center = x + w/2
            y_center = y + h/2
            if x_center < threshold_center * w_orig or x_center > (1 - threshold_center) * w_orig or y_center < threshold_center * h_orig or y_center > (1-threshold_center) * h_orig:
                continue
            if(area < threshold_area * w_orig*h_orig or area_max > area):
                continue
            area_max = area
            bounding_box = [x, y, x+w, y+h]
    if area_max > 0.9 * w_orig*h_orig:
        bounding_box = [0, 0, w_orig, h_orig]
    #print ([x[1] for x in significant]);
    return bounding_box


def findSignificantContours(image, edgeImg):
    contours, heirarchy = cv2.findContours(
        edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)
    # From among them, find the contours with large surface area.
    significant = []
    # If contour isn't covering 5% of total area of image then it probably is too small
    tooSmall = edgeImg.size * 5 / 100
    for tupl in level1:
        contour = contours[tupl[0]]
        area = cv2.contourArea(contour)
        if area > tooSmall:
            significant.append([contour, area])

            # Draw the contour on the original image
            #cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)

    significant.sort(key=lambda x: x[1])
    #print ([x[1] for x in significant]);
    return [x[0] for x in significant]


def find_objects(image):
    #cv2.imshow("input" , image)
    edgeImg = np.max(np.array([edgedetect(image[:, :, 0]), edgedetect(
        image[:, :, 1]), edgedetect(image[:, :, 2])]), axis=0)

    mean = np.mean(edgeImg)
    # # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg <= mean] = 0
    #cv2.imshow("edgeImg" , edgeImg)
    edgeImg_8u = np.asarray(edgeImg, np.uint8)

    # Find contours
    significant = findSignificantContours(image, edgeImg_8u)
    # Mask
    mask = edgeImg.copy()

    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    kernel = np.ones((2*2 + 1, 2*2 + 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = np.array(mask)
    #cv2.imshow("mask" , mask)
    # cv2.imwrite(output_name ,mask
    mask = np.asarray(mask, np.uint8)
    # Invert mask
    #mask = np.logical_not(mask)
    return mask


def com_iou(box1, box2):
    iou_area = 0
    x_min = max(box1[0], box2[0])
    x_max = min(box1[2], box2[2])
    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    if x_max > x_min and y_max > y_min:
        iou_area = (x_max - x_min) * (y_max - y_min)
    max_area = (box1[2] - box1[0])*(box1[3] - box1[1])
    if max_area <= 1:
        max_area = 1
    iou_area /= max_area

    return iou_area


def faceDetect(image, mask, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30))
    res = True
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        img_crop = mask[y:y+h, x:x+w]
        count = cv2.countNonZero(img_crop)
        ratio = float(count) / (w*h)
        if ratio > 0.3:
            res = False
    return res

#crop with mask

def find_roi_update(image, mask, cascade):
    h_orig, w_orig = image.shape[:2]
    bounding_box = [0, 0, w_orig, h_orig]
    res = False
    min_threshold = 5
    ret, mask_sod = cv2.threshold(mask, min_threshold, 255, cv2.THRESH_BINARY)
    m = cv2.mean(mask, mask_sod)

    if m[0] < 150:
        return res, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

    kernel = np.ones((2*2 + 1, 2*2 + 1), np.uint8)
    mask_sod = cv2.dilate(mask_sod, kernel, iterations=1)
    mask_sod = cv2.erode(mask_sod, kernel, iterations=1)
    mask_sod = cv2.dilate(mask_sod, kernel, iterations=1)
    moments = cv2.moments(mask_sod)
    huMoments = cv2.HuMoments(moments)
    boxes_center = [image.shape[1]//2 - image.shape[1]//5, image.shape[0]//2 - image.shape[0] //
                    5, image.shape[1]//2 + image.shape[1]//5, image.shape[0]//2 + image.shape[0]//5]
    #print(huMoments)
    contours, hierarchy = cv2.findContours(
        mask_sod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    max_index = -1
    is_segment = False
    area_full = float(mask.shape[0] * mask.shape[1])
    needCrop = False

    for i in range(len(contours)):
        area_ = cv2.contourArea(contours[i])
        if(area_ < 0.1*area_full):
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        img_crop = mask_sod[int(y):int(y+h), int(x): int(x+w)]
        count = cv2.countNonZero(img_crop)
        rect = cv2.minAreaRect(contours[i])

        ratio_size = rect[1][0]/rect[1][1]
        if ratio_size > 1:
            ratio_size = 1/ratio_size

        ratio_area = count/float(rect[1][0]*rect[1][1])

        bounding_box = [x, y, x + w, y + h]
        iou_area = com_iou(boxes_center, bounding_box)

        if(ratio_size > 0.65 and ratio_area > 0.9 and iou_area > 0.9):
            needCrop = True

        if ratio_size < 0.5 or ratio_area < 0.65 or iou_area < 0.4:
            continue

        if(area_ > max_area):
            max_area = area_
            max_index = i

    if(max_index >= 0):
        x, y, w, h = cv2.boundingRect(contours[max_index])
        bounding_box = [x, y, x + w, y + h]

        for i in range(len(contours)):
            if(i != max_index):
                cv2.drawContours(mask_sod, contours, i,
                                 (0, 0, 0), -1, cv2.LINE_AA, maxLevel=1)

        m, dev = cv2.meanStdDev(mask, mask_sod)
        if(m[0] > 30):
            res = True

    w = bounding_box[2] - bounding_box[0]
    h = bounding_box[3] - bounding_box[1]

    if(h > w):
        estimaze_hor = int((h-w))
        if(bounding_box[0] >= 0.5 * estimaze_hor):
            bounding_box[0] -= int(0.5 * estimaze_hor)
        else:
            bounding_box[0] = 0
        if(bounding_box[2] < image.shape[1] - 0.5 * estimaze_hor):
            bounding_box[2] += int(0.5 * estimaze_hor)
        else:
            bounding_box[2] = image.shape[1]
        if bounding_box[2] - bounding_box[0] < h:
            est = h - bounding_box[2] + bounding_box[0]
            est = bounding_box[0] - est
            bounding_box[0] = max(0, est)
            est = h - bounding_box[2] + bounding_box[0]
            est = bounding_box[2] + est
            bounding_box[2] = min(image.shape[1], est)
    else:
        estimaze_col = int((w-h))
        if(bounding_box[1] >= 0.5 * estimaze_col):
            bounding_box[1] -= int(0.5 * estimaze_col)
        else:
            bounding_box[1] = 0
        if(bounding_box[3] < image.shape[0] - 0.5 * estimaze_col):
            bounding_box[3] += int(0.5 * estimaze_col)
        else:
            bounding_box[3] = image.shape[0]

        if bounding_box[3] - bounding_box[1] < w:
            est = w - bounding_box[3] + bounding_box[1]
            est = bounding_box[1] - est
            bounding_box[1] = max(0, est)
            est = w - bounding_box[3] + bounding_box[1]
            est = bounding_box[3] + est
            bounding_box[3] = min(image.shape[0], est)

    if res == True and needCrop == False:
        res = faceDetect(image, mask, cascade)

    return res, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

def crop_image(img_orig):
    h_orig, w_orig = img_orig.shape[:2]
    bounding_box = [0 , 0 ,w_orig ,h_orig]

    w = bounding_box[2] - bounding_box[0]
    h = bounding_box[3] - bounding_box[1]

    if(w < h):
        estimaze_hor = int((h - w))
        bounding_box[1] += int(0.5 * estimaze_hor)
        bounding_box[3] -= int(0.5 * estimaze_col)
    else:
        estimaze_col = int((w - h))
        bounding_box[0] += int(0.5 * estimaze_col)
        bounding_box[2] -= int(0.5 * estimaze_col)

    return bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

def find_roi(img_orig):
    size_compare = 320
    scale = img_orig.shape[1] / float(size_compare)
    img = cv2.resize(img_orig, (size_compare, int(
        size_compare*img_orig.shape[0]/img_orig.shape[1])), 0, 0, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Remove noise
    mask_seg = segmentWatershed(img)
    mask_edge = find_objects(img)
    mask_edge = cv2.bitwise_and(mask_edge, mask_seg)
    kernel = np.ones((2*2 + 1, 2*2 + 1), np.uint8)
    mask_edge = cv2.dilate(mask_edge, kernel, iterations=1)
    bounding_box = getBoundingBox(img, mask_edge)
    # cv2.imshow("mask_seg" , mask_seg)
    # cv2.imshow("mask_edge" , mask_edge)
    # cv2.imshow("mask out" , mask_edge)
    # print("box " , bounding_box , img.shape)
    for i in range(4):
        bounding_box[i] = int(bounding_box[i]*scale)

    w = bounding_box[2] - bounding_box[0]
    h = bounding_box[3] - bounding_box[1]
    if(w < h):
        estimaze_hor = int((h - w))
        if(bounding_box[0] >= 0.5 * estimaze_hor):
            bounding_box[0] -= int(0.5 * estimaze_hor)
        else:
            bounding_box[0] = 0
        if(bounding_box[2] < img_orig.shape[1] - 0.5 * estimaze_hor):
            bounding_box[2] += int(0.5 * estimaze_hor)
        else:
            bounding_box[2] = img_orig.shape[1]
    else:
        estimaze_col = int((w - h))
        if(bounding_box[1] >= 0.5 * estimaze_col):
            bounding_box[1] -= int(0.5 * estimaze_col)
        else:
            bounding_box[1] = 0
        if(bounding_box[3] < img_orig.shape[0] - 0.5 * estimaze_col):
            bounding_box[3] += int(0.5 * estimaze_col)
        else:
            bounding_box[3] = img_orig.shape[0]

    return bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]


def find_center(size):
    width, height = size
    new_width = min(width, height)
    new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    return left, top, right, bottom

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def init_sod_model(base_path=os.getcwd()):
    print("Initializing the sod model ...")
    model_name = 'u2netp'
    model_dir = os.path.join(
        base_path, 'u2net/saved_models', model_name, model_name + '.pth')

    # --------- 3. model define ---------
    if(model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif(model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(model_dir))
    else:
        net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.eval()

    return net
def init_ClipSegModel():
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to('cuda:0')
    return processor,model

def init_cascade_model(path_model):
    cascade = cv2.CascadeClassifier(path_model)
    print("cascase.empty = ", cascade.empty())

    return cascade
def extract_mask_ClipSeg_model(processor,model,images):
    prompt = Image.open('/media/anlab/data-2tb/ANLAB_THUY/ToyotaAR/Dataset/support1_black.jpg')
    encoded_image = processor(images=images, return_tensors="pt").to('cuda:0')
    encoded_prompt = processor(images=[prompt], return_tensors="pt").to('cuda:0')
    with torch.no_grad():
        outputs = model(**encoded_image, conditional_pixel_values=encoded_prompt.pixel_values).logits
    # for i in range(len(outputs)):
    current_time = datetime.now()
    cv2.imwrite('/media/anlab/data-2tb/ANLAB_THUY/lashinbang-server-dzungdk/SaveQuery/'+current_time.strftime("%Y%m%d_%H%M%S")+'.png',cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
    preds = outputs.unsqueeze(1)
    preds = torch.transpose(preds, 0, 1).squeeze(0).cpu().numpy()
    THRESHOLD = -1.5
    preds[preds > THRESHOLD] = 255
    preds[preds <= THRESHOLD] = 0
    mask = Image.fromarray(preds).convert("L")
    mask = mask.resize(Image.fromarray(images[0]).size)
    foreground = Image.new("RGBA", Image.fromarray(images[0]).size)
    foreground.paste(Image.fromarray(images[0]), mask=mask)
    foreground = foreground.convert('RGB')
    mask_cv = np.array(mask)
    _, mask_binary = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    (x,y,w,h) = cv2.boundingRect(cnts[0])  
    foreground = foreground.crop((x,y,w+x,h+y))
    # foreground.save('testCrop.jpg')
    return [foreground]
def extract_sod_batch_by_array(model, images):
    test_salobj_dataset = SalObjDatasetFromList(img_list=images,
                                                transform=transforms.Compose([RescaleT(320),
                                                                              ToTensorLab(flag=0)])
                                                )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    sod_images = []

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # create sod mask
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np*255).convert('RGB')
#        imo = im.resize((w, h),resample=Image.BILINEAR)
        sod_images.append(im)

        del d1, d2, d3, d4, d5, d6, d7

    return sod_images

def find_bounding_crop(image, mask, threshold_area= 0.25, color_threshold=150, threshold_ratio=1.5):
	height, width = image.shape[:2]
	bounding_box = [0, 0, width, height]
	ret_detect = False
	min_threshold = 5
	ret, mask_sod = cv2.threshold(mask, min_threshold, 255, cv2.THRESH_BINARY)
	m = cv2.mean(mask, mask_sod)
	if m[0] < color_threshold:
		return ret_detect, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

	kernel = np.ones((2*2 + 1, 2*2 + 1), np.uint8)
	mask_sod = cv2.dilate(mask_sod, kernel, iterations=1)
	mask_sod = cv2.erode(mask_sod, kernel, iterations=1)
	mask_sod = cv2.dilate(mask_sod, kernel, iterations=1)
	moments = cv2.moments(mask_sod)
	huMoments = cv2.HuMoments(moments)
	boxes_center = [image.shape[1]//2 - image.shape[1]//5, image.shape[0]//2 - image.shape[0] //
					5, image.shape[1]//2 + image.shape[1]//5, image.shape[0]//2 + image.shape[0]//5]
	#print(huMoments)
	contours, hierarchy = cv2.findContours(
		mask_sod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(contours) <= 0 :
		return ret_detect
	hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
	# For each contour, find the bounding rectangle and draw it
	xmin = int(width)
	xmax = int(0)
	ymin = int(height)
	ymax = int(0)
	size_min = int(0.2*width)
	max_area = 0
	for component in zip(contours, hierarchy , range(len(contours))):
		currentContour = component[0]
		area_contour = cv2.contourArea(currentContour)
		
		x,y,w,h = cv2.boundingRect(currentContour)
		if area_contour < 50 or (h < size_min and w < size_min):
			continue
		# print( x,y,w,h)
		if(area_contour > max_area):
			xmin = int(x)
			xmax = int(x+w)
			ymin =  int(y)
			ymax = int(y+h)
	if xmax -xmin > size_min and ymax - ymin > size_min :
		ratio_area_char =  (xmax -xmin )*(ymax - ymin)/float(width*height)
		
		ratio_detect = (xmax -xmin )/(ymax - ymin)
		if ratio_detect < 1:
			ratio_detect = 1/ratio_detect
		# print("ratio_area_char" ,ratio_area_char, ratio_detect )
		if ratio_area_char > threshold_area and ratio_detect > threshold_ratio:
			ret_detect = True
			bounding_box = [xmin, ymin,xmax, ymax]
	
	return ret_detect, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

def detect_key_image(image, sod_image, threshold_area= 0.8, threshold_area_size=0.2, color_threshold=100 ):
	ret_detect = False
	area_original = (image.shape[0]*image.shape[1])
	res_crop, left, top, right, bottom = find_bounding_crop(image, sod_image )
	if res_crop:
		sod_image = sod_image[top:bottom, left:right]
		image = image[top:bottom, left:right]
	else:
		return ret_detect, None, None, None, None
	height,width = sod_image.shape[0:2]
	
	# print(sod_image.shape)
	ret, mask_sod_ori = cv2.threshold(sod_image, 10, 255, cv2.THRESH_BINARY)
	bounding_box = [0, 0, width, height]
	m = cv2.mean(sod_image, mask_sod_ori)
	if m[0] < color_threshold:
		return ret_detect,  None, None, None, None
	kernel = np.ones((2*3 + 1, 2*3 + 1), np.uint8)
	mask_sod = cv2.dilate(mask_sod_ori, kernel, iterations=1)
	size = min(height,width)
	size_min = int(0.5*size)
	size2 = max(height,width)
	size_kernel = max(3, int(0.03*size2))
	kernel = np.ones((2*size_kernel + 1, 2*size_kernel + 1), np.uint8)
	mask_sod = cv2.erode(mask_sod, kernel, iterations=1)
	mask_sod = cv2.dilate(mask_sod, kernel, iterations=1)
	
	contours, hierarchy = cv2.findContours(mask_sod, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) <= 0 :
		return ret_detect,  None, None, None, None
	hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
	# For each contour, find the bounding rectangle and draw it
	xmin = int(width)
	xmax = int(0)
	ymin = int(height)
	ymax = int(0)
	area_objecs = float(width*height)
	for component in zip(contours, hierarchy , range(len(contours))):
		currentContour = component[0]
		area_contour = cv2.contourArea(currentContour)
		
		x,y,w,h = cv2.boundingRect(currentContour)
		if area_contour < 0.1*area_objecs or (h < size_min and w < size_min):
			continue
		# print( x,y,w,h)
		xmin = min(xmin,int(x))
		xmax = max(xmax,int(x+w))
		ymin = min(ymin , int(y))
		ymax = max(ymax, int(y+h))
	if xmax -xmin > size_min and ymax - ymin > size_min :
		ratio_area_char =  (xmax -xmin )*(ymax - ymin)/float(width*height)
		ratio_area_char2 =  (xmax -xmin )*(ymax - ymin)/float(area_original)
		is_check_top = True

		threshold_ratio= height/width
		if threshold_ratio < 1:
			threshold_ratio = 1/threshold_ratio
		else:
			r_botom = height - ymax
			if r_botom > 0.7*ymin:
				is_check_top = False

		ratio_detect = (ymax - ymin)/(xmax -xmin )
		if ratio_detect < 1:
			ratio_detect = 1/ratio_detect
			r_right = width - xmax
			if xmin > r_right:
				img_drop = mask_sod_ori[0:height, 0:xmin]
			else:
				img_drop = mask_sod_ori[0:height, xmax:width]
		else:
			if is_check_top:
				img_drop = mask_sod_ori[0:ymin, 0:width]
		
		if is_check_top:
			m_drop = cv2.mean(img_drop)
			if m_drop[0] > color_threshold:
				is_check_top = False
				
		if is_check_top and ratio_area_char < threshold_area and ratio_detect < threshold_ratio and ratio_area_char2 > threshold_area_size:
			ret_detect = True
			bounding_box = [xmin+left, ymin+top,xmax+left, ymax+top]
			# img_crop = image[ymin:ymax,xmin:xmax  ]
	# print(ret_detect , bounding_box)
	return ret_detect , bounding_box[0] , bounding_box[1] , bounding_box [2] , bounding_box[3]

def detect_key_list(sod_model, base_path, files, base_save_folder="crop_foreground"):
	# full_paths = [os.path.join(base_path, x) for x in files]
	# if not os.path.exists(base_save_folder):
	# 	os.mkdir(base_save_folder)
	images = []
	path_sod = []

	for f in files:
		
		image_path = os.path.join(base_path, f)
		if not os.path.isfile(image_path):
			print("error file " , image_path)
			continue
		base=os.path.basename(image_path)
		if not base.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
			print("not file image" , image_path)
			# print(image_path)
			continue
		image = io.imread(image_path)
		if image is None:
			continue
		images.append(image)
		path_sod.append(f)
	sod_images = extract_sod_batch_by_array(sod_model, images)
	list_key_result = []
	for i, sod_image in enumerate(sod_images):
		sod_image = sod_image.resize((images[i].shape[1], images[i].shape[0]), resample=Image.BILINEAR)
		sod_image = cv2.cvtColor(np.array(sod_image), cv2.COLOR_RGB2GRAY)
		try:
			ret_detect, left, top, right, bottom = detect_key_image(images[i], sod_image)
		except ValueError as err:
			ret_detect= False
		if ret_detect:
			sub_path = os.path.dirname(path_sod[i])
			sub_path_id = os.path.join(base_save_folder, sub_path)
			sub_path_save = os.path.join(base_path, sub_path_id)
			# print(f'sub_path_save: {sub_path_id}')
			Path(sub_path_save).mkdir(parents=True, exist_ok=True)
			base = os.path.basename(path_sod[i])
			found = base.rfind(".")
			base_name = base[0:found]
			_ext = os.path.splitext(base)[1]
			file_out = os.path.join(sub_path_save, f'{base_name}c{_ext}')
		   
			list_key_result.append(os.path.join(sub_path_id, f'{base_name}c{_ext}'))
			image_crop = images[i][top:bottom, left:right]
			image_crop = cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR)
			cv2.imwrite(file_out, image_crop)
	return list_key_result

def convertBRG2RGB(base_folder, files, save_folder="convert_crop_foreground"):
	save_folder = '/media/anlab/DATA/lashinbang/images-download/convert_crop_foreground'
	count = 0
	for f in files:
		count +=1 
		print(count, f)
		path = os.path.join(base_folder, f)
		if not os.path.isfile(path):
			print("error file " , path)
			continue
		image = cv2.imread(path)
		if image is None:
			continue
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


		sub_path = os.path.dirname(f)
		sub_path_save = os.path.join(save_folder, sub_path)
		# print(f'sub_path_save: {sub_path_id}')
		Path(sub_path_save).mkdir(parents=True, exist_ok=True)
		# print(sub_path_save)
		base = os.path.basename(f)
		file_out = os.path.join(sub_path_save, base)
		cv2.imwrite(file_out, image)
