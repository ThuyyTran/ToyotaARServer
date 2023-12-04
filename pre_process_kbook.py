
import numpy as np
import cv2
import os
import time
import imageio
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path

# def listdir_fullpath(d):
#     return [os.path.join(d, f) for f in os.listdir(d)]

class ImagePreProcess:
    def __init__(self, WIDTH_PRICE=500, HEIGHT_PRICE = 200):
        self.WIDTH_PRICE = WIDTH_PRICE
        self.HEIGHT_PRICE = HEIGHT_PRICE

    def findContourMax(self , mask , thresh_min=0.3):
        index_max = -1
        area_max = 0
        min_thresh = thresh_min
        area_image = mask.shape[0]*mask.shape[1]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
        # For each contour, find the bounding rectangle and draw it
        for component in zip(contours, hierarchy , range(len(contours))):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
            area = cv2.contourArea(currentContour)
            if area > min_thresh*area_image and area > area_max:
                area_max = area
                index_max = int(component[2])
        del hierarchy
        return contours , index_max

    def findContourPrice(self , mask , thresh_min=0.85 , ratio_thresh=2 , pos_h_low=0.5 , area_thresh=0.05):
        index_max = -1
        area_max = 0
        height , width  = mask.shape
        area_image =height *width
        contours, hierarchy  = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0 :
            return contours , index_max
        hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
        for component in zip(contours, hierarchy , range(len(contours))):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
            area = cv2.contourArea(currentContour)
            ratio_area = float(area) / (w*h)
            ratio_size = float(w) / (h)
            if y < pos_h_low*height:
                continue
            #print (w*h  , area_thresh*area_image , ratio_area , ratio_size  , y+h ,  2*height/3 )
            thresh1 = 0.5
            thresh2 = 4
            res1 = w*h > area_thresh*area_image and (ratio_area > thresh_min and ratio_size > ratio_thresh) # and
            res2 = w*h > area_thresh*area_image and ratio_area > thresh1 and ratio_size > thresh2 and y+h > 2*height/3
            res3 = w*h > area_thresh*area_image and ratio_area > (thresh1+thresh_min)/2.0 and ratio_size > thresh2-1 and y+h > 2*height/3
            if  (res1 or res2 or res3) and area > area_max:
                area_max = area
                index_max = int(component[2])
        del hierarchy
        return contours , index_max


    def fitBoxes(self,  contours , index, width , height ):
        x,y,w,h = cv2.boundingRect(contours[index])
        area = cv2.contourArea(contours[index])
        ratio = float(area) / (w*h)
        #print("ratio " , ratio )
        if ratio > 0.85:
            return  x, y , w , h
        mask = np.zeros((height,width ), np.uint8)
        mask = cv2.drawContours(mask, contours, index, (255), -1)
        size_morph = int(0.05*w)
        kernel = np.ones((2*size_morph + 1,2*size_morph + 1),np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("mask" , mask)
        contours2 , index_max = self.findContourMax(mask , thresh_min=0.4 )
        if index_max >= 0 :
            x,y,w,h = cv2.boundingRect(contours2[index_max])
        return  x, y , w , h

    def segmentPrice(self ,  image ):
        res = False
        if image is None:
            return False, res , None, None , None , None
        height , width , _ = image.shape
        y_min = int(2*height/3)
        img_clone = image[y_min:height , 0 :width ]
        height , width , _ = img_clone.shape
        img_clone = cv2.GaussianBlur(img_clone,(5,5),0)

        hsvImage = cv2.cvtColor(img_clone, cv2.COLOR_BGR2HSV)
        red_lower = np.array([100, 50, 100], np.uint8)
        red_upper = np.array([130, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvImage, red_lower, red_upper)
        #cv2.imshow("red_mask" , red_mask)

        blue_lower = np.array([0, 50, 100], np.uint8)
        blue_upper = np.array([50, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvImage, blue_lower, blue_upper)
        #cv2.imshow("blue_mask" , blue_mask)

        mask = cv2.bitwise_or(red_mask, blue_mask)
        #cv2.imshow("mask" , mask)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((1,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours , index_max = self.findContourPrice(mask )
    # print("index_max " , index_max)
        if index_max >= 0 :
            x,y,w,h = cv2.boundingRect(contours[index_max])
            #img_clone = cv2.drawContours(img_clone, contours, index_max, (0,255,0), 3)
            y += y_min
            res = True
            return True ,res , x , y , w , h
        m = cv2.mean(mask)

        has_data = m[0] > 20
        return has_data , res , None, None , None , None

    def fitCol(self, image):
        res = False
        if image is None:
            return res , None
        height , width , _ = image.shape
        image = cv2.GaussianBlur(image,(5,5),0)
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(imageHSV, cv2.MORPH_GRADIENT, kernel)
        #cv2.imshow("image" , image)
        gray = cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
       # cv2.imshow("gray" , gray)
        ret3,edges = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #edges = cv2.Canny(gray,20,60,apertureSize = 3)
        #cv2.imshow("edges" , edges)
        lines = cv2.HoughLines(edges,1,np.pi/180,height//3)
        isMatch = False
        theta_detect = 0
        x_detect = int(0)
        if lines is not None:
            for line in lines:
                for rho,theta in line:
                    angle = theta*180/3.1416
                    #print("theta: " , angle , rho)
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    if (angle <2 or angle > 178) and x0 > width/2 :
                        isMatch = True
                        if x_detect < x0:
                            x_detect = int(x0)
                            theta_detect = angle
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))

                        cv2.line(image,(x0,y0),(x2,y2),(0,0,255),2)
                        cv2.line(image,(x0,y0),(x1,y1),(255,0,0),2)
            if isMatch:
                thresh = 60
                if x_detect > 9*width/10:
                    mean, sdev  = cv2.meanStdDev(image)
                    m = (mean[0] + mean[1] + mean[2])/3
                    if m < thresh:
                        isMatch = False
                else:
                    img_crop = image[0:height , x_detect:width]
                    mean, sdev  = cv2.meanStdDev(img_crop)
                    m = (mean[0] + mean[1] + mean[2])/3
                    if m > thresh:
                        isMatch = False
            #     print("mean " , mean, sdev )
            # cv2.imshow("line col" , image)
        if not(isMatch):
            x_crop = int(width//3)
            for i in range(3):
                img_crop = image[0:height , width - x_crop:width]
                mask_crop = edges[0:height , width - x_crop:width]
                mean, sdev  = cv2.meanStdDev(img_crop)
                mean2, sdev2  = cv2.meanStdDev(mask_crop)
                thresh = 60
                thresh2 = 6
                m = (mean[0] + mean[1] + mean[2])/3
               # print("mean2 " , mean, mean2  ,x_crop )
                #cv2.imshow("crop"  , img_crop)
                if (m < thresh and mean2[0] <thresh2) or (m < thresh/2 and mean2[0] <thresh2*2):
                    isMatch = True
                    x_detect = width - x_crop
                    break
                x_crop = x_crop//2

       # cv2.waitKey()
        return isMatch , x_detect

    def fitRow(self, image , w_detect):
        res = False
        if image is None:
            return res , None
        height , width , _ = image.shape
        image = cv2.GaussianBlur(image,(5,5),0)
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(imageHSV, cv2.MORPH_GRADIENT, kernel)
        #cv2.imshow("image" , image)
        gray = cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray" , gray)
        ret3,edges = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow("edges" , edges)
        lines = cv2.HoughLines(edges,1,np.pi/180,width//2)
        isMatch = False
        theta_detect = 0
        y_detect = int(0)
        if lines is not None:
            for line in lines:
                for rho,theta in line:
                    angle = theta*180/3.1416
                    #print("theta: " , angle , rho)
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    if (angle > 88 and angle < 92) and y0 > height/2 :
                        isMatch = True
                        if y_detect < y0:
                            y_detect = int(y0)
                            theta_detect = angle
                        # x1 = int(x0 + 1000*(-b))
                        # y1 = int(y0 + 1000*(a))
                        # x2 = int(x0 - 1000*(-b))
                        # y2 = int(y0 - 1000*(a))

                        # cv2.line(image,(x0,y0),(x2,y2),(0,0,255),2)
                        # cv2.line(image,(x0,y0),(x1,y1),(255,0,0),2)
            if isMatch:
                if height - y_detect < 5 :
                    y_detect = height
                else:
                    img_crop = image[y_detect:height , 0:width]
                    mean, sdev  = cv2.meanStdDev(img_crop)
                    thresh = 60
                    m = (mean[0] + mean[1] + mean[2])/3
                    if m > thresh:
                        isMatch = False
                    #print("mean " , mean, sdev )
                    #cv2.line(image,(0,y_detect),(width,y_detect),(0,255,0),5)
        #cv2.imshow("line" , image)
        if not(isMatch):
            y_detect = min(int(w_detect*1.6), height)
            isMatch = True
        return isMatch , y_detect

    def isObject(self, image):
        res = False
        if image is None:
            return res , None , None
        height , width , _ = image.shape
        mean, sdev  = cv2.meanStdDev(image)
        m = (mean[0] + mean[1] + mean[2])/3
        d = (sdev[0] + sdev[1] + sdev[2])/3
        thresh = 40
        thresh2 = 20
        #print( "thresh" ,m, d )
        # if m > thresh and d > thresh2:
        #     res = True
        if m < thresh:
            return res , None , None
        image = cv2.GaussianBlur(image,(5,5),0)
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(imageHSV, cv2.MORPH_GRADIENT, kernel)
        #cv2.imshow("image" , image)
        gray = cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray" , gray)
        ret3,edges = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #edges = cv2.Canny(gray,20,60,apertureSize = 3)
        #cv2.imshow("edges" , edges)
        img_crop = edges[0:height , 0:width//2]
        x_crop = int(width//6)
        img_crop_step = edges[0:height , width - x_crop:width]
        mean, sdev  = cv2.meanStdDev(img_crop)
        mean2, sdev2  = cv2.meanStdDev(img_crop_step)
        #print("mean , " ,mean, sdev, mean2, sdev2  )
        res_color = m > thresh and d > thresh2
        res_edge_min = mean[0] > 20 and mean2[0] > 15 and m > thresh and d > 0.8*thresh2
        res = res_color or res_edge_min
        return res , width, height

    def segmentImageLabel(self , image ):
        xmin = 0
        ymin = 0
        width = 1
        height = 1
        res = False
        if image is None:
            return res , xmin, ymin , width , height
        res_col, x_detect = self.fitCol(image)
        if not(res_col):
            res , width, height = self.isObject(image)
            return res , xmin, ymin , width , height
        width = x_detect -  xmin

        res_row, y_detect = self.fitRow(image ,width )
        if not(res_row):
            return res , xmin, ymin , width , height
        res = True
        ratio_crop = float(x_detect)/y_detect
        ratio_h = float(y_detect)/image.shape[0]
        if  ratio_crop > 0.8 and ratio_h < 0.85:
            y_detect = min(int(width*1.6), image.shape[0])
        height = y_detect - ymin
        # cv2.line(image,(0,y_detect),(image.shape[1],y_detect),(0,0,255),2)
        # cv2.line(image,(x_detect,0),(x_detect,image.shape[0]),(0,0,255),2)
        # cv2.imshow("drawing"  , image)
        # cv2.waitKey(0)
        return res , xmin, ymin , width , height



    def detectPriceAndImage(self, image ):
        box_price = [0 , 0 , 1 , 1]
        box_image = [0 , 0 , 1 , 1]
        res_detect = False
        if image is None:
            return res_detect , box_price , box_image
        height , width , _ = image.shape
        size_input = 480.0
        scale = size_input/width
        img_size = cv2.resize(image , None , None , scale , scale , interpolation=cv2.INTER_AREA)
        height , width , _ = img_size.shape

        has_data, res_price , x_price , y_price , w_price , h_price = self.segmentPrice(img_size)
        if not has_data:
            return res_detect , box_price , box_image
        #img_size = cv2.rectangle(img_size, (x_price , y_price), (x_price+ w_price , y_price + h_price), (0 , 255 , 0 ),2 )
        #cv2.imshow("img_clone" , img_size)
        estimate = int(30)
        x = 0
        y = 0
        w = width
        h = height
        if res_price:
            h = y_price  - 5
        else:
            est_y = int(h/6)
            h = int(est_y*5)
            img_price = image[h-est_y : h , 0:w]
            x_price = 0
            w_price = w
            y_price = h-est_y
            h_price = est_y
        res_price = True

        img_clone = img_size[y +estimate : y+h  , x + estimate:x +w - estimate//2]

        res_detect , xmin , ymin , w_image , h_image = self.segmentImageLabel(img_clone)

        if res_detect:
            w_image += (xmin + estimate)
            h_image += (ymin + estimate)
            xmin = 0
            ymin = 0
            xmin = int(xmin/scale)
            ymin = int(ymin/scale)
            w_image = int(w_image/scale)
            h_image = int(h_image/scale)
            box_image = [xmin , ymin , w_image , h_image]
            box_price = [int(x_price/scale) , int(y_price/scale) , int(w_price/scale) , int(h_price/scale) ]

        return res_detect , box_price , box_image


    def cropImagePrice(self , image_path):

        image = imageio.imread(image_path)
        if image is None:
            print ("empty image ")
            return False , None , None
        h, w, _ = image.shape
        if w > h :
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w, _ = image.shape
        res , box_price , box_image = self.detectPriceAndImage(image)
        print("res crop " , res , box_price , box_image)
        #if not(res):
        #    cv2.waitKey(0)
        if res :
            img_crop = image[box_image[1] : box_image[1]+box_image[3] , box_image[0]: box_image[0]+ box_image[2]]
            img_price = image[box_price[1] : box_price[1]+box_price[3] , box_price[0]: box_price[0]+ box_price[2]]
        else:

            est_y = int(h/6)
            img_crop = image[0 : h-est_y , 0:w]
            img_price = image[h-est_y : h , 0:w]
        return  res , img_crop , img_price

    def cropBookList(self , base_folder , list_path ,  base_save_folder):
        if not os.path.isdir(base_save_folder):
            os.mkdir(base_save_folder)
        save_folder_image = []
        base_save_price = os.path.join(base_save_folder , "price")
        base_save_images = os.path.join(base_save_folder , "images")
        base_save_thumbnails = os.path.join(base_save_folder , "thumbnails")
        if not os.path.isdir(base_save_price):
            os.mkdir(base_save_price)
        if not os.path.isdir(base_save_images):
            os.mkdir(base_save_images)

        image_list = []
        id_list_images = []
        id_list_price = []
        id_list_errors = []
        for i, file in enumerate(list_path):
            print("===============\n")
            image_path = os.path.join(base_folder, file)

            sub_path = os.path.dirname(file)
            print(f'sub_path: {sub_path}')

            sub_path_images = os.path.join(base_save_images, sub_path)
            print(f'sub_path_images: {sub_path_images}')
            Path(sub_path_images).mkdir(parents=True, exist_ok=True)

            sub_path_prices = os.path.join(base_save_price, sub_path)
            print(f'sub_path_prices: {sub_path_prices}')
            Path(sub_path_prices).mkdir(parents=True, exist_ok=True)

            sub_path_thumbnails = os.path.join(base_save_thumbnails, sub_path)
            print(f'sub_path_thumbnails: {sub_path_thumbnails}')
            Path(sub_path_thumbnails).mkdir(parents=True, exist_ok=True)

            if os.path.isfile(image_path):
                print("image_path " , image_path)
                base=os.path.basename(image_path)
                if len(base)  == len(file) :
                    base=os.path.basename(image_path)
                    if base.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) == False:
                        id_list_errors.append(file)
                        continue
                    res , img_crop, img_price = self.cropImagePrice(image_path)
                    if res :
                        img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img)
                        #image_list.append(img)
                        id_list_images.append( file)
                        id_list_price.append(os.path.join('price' , file))
                        res_thumbnails, img_thumbnails = self.getThumbnailsFromImage(img_crop ,img_price )
                        imageio.imwrite(os.path.join(sub_path_images, base), img_crop)
                        imageio.imwrite(os.path.join(sub_path_prices, base), img_price)
                        if res_thumbnails:
                            imageio.imwrite(os.path.join(sub_path_thumbnails, base), img_thumbnails)
                    #cv2.waitKey(0)

                    else:
                        id_list_errors.append(file)
                    continue
                label = file[0:int(len(file) - len(base) -1 ) ]
                #print("base" , base)
                #continue
                if base.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) == False:
                    id_list_errors.append(file)
                    continue
                s = time.time()
                res , img_crop, img_price = self.cropImagePrice(image_path)

                # save_folder_image = os.path.join(base_save_images, label)
                # if not os.path.exists(save_folder_image):
                #     os.mkdir(save_folder_image)

                # save_folder_price = os.path.join(base_save_price, label)
                # if not os.path.exists(save_folder_price):
                #     os.mkdir(save_folder_price)
                print("time process : " , (time.time() - s) )
                if res:
                    #print(os.path.join(save_folder_price , base) ,  img_price.shape  )
                    #print(os.path.join(save_folder_image , base)   )
                    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    #image_list.append(img)
                    id_list_images.append(file )
                    id_list_price.append(os.path.join('price' , file))
                    res_thumbnails,img_thumbnails = self.getThumbnailsFromImage(img_crop ,img_price )
                    imageio.imwrite(os.path.join(sub_path_images, base), img_crop)
                    imageio.imwrite(os.path.join(sub_path_prices, base), img_price)
                    if res_thumbnails:
                        imageio.imwrite(os.path.join(sub_path_thumbnails, base), img_thumbnails)
                else:
                    id_list_errors.append(file)
            else:
                id_list_errors.append(file)
                # cv2.imshow("img_crop" , img_crop)
                # cv2.imshow("img_price" , img_price)
            #cv2.waitKey(0)
        return image_list, id_list_images , id_list_price , id_list_errors

    def cropBookFolder(self , base_folder, base_save_folder ):

        if not os.path.isdir(base_save_folder):
            os.mkdir(base_save_folder)
        base_save_price = os.path.join(base_save_folder , "price")
        base_save_images = os.path.join(base_save_folder , "images")
        if not os.path.isdir(base_save_price):
            os.mkdir(base_save_price)
        if not os.path.isdir(base_save_images):
            os.mkdir(base_save_images)
        list_folder = os.listdir(base_folder)
        for label in list_folder:
            folder = os.path.join(base_folder, label)
            if os.path.isfile(folder):
                base=os.path.basename(folder)
                if base.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    res , img_crop, img_price = self.cropImagePrice(folder)
                    if res :
                        imageio.imwrite(os.path.join(base_save_images , base) , img_crop)
                        imageio.imwrite(os.path.join(base_save_price , base) , img_price)
                        #cv2.imshow("img_crop" , img_crop)
                        #cv2.imshow("img_price" , img_price)
                    #cv2.waitKey(0)
                continue

            save_folder_image = os.path.join(base_save_images, label)
            if not os.path.exists(save_folder_image):
                os.mkdir(save_folder_image)

            save_folder_price = os.path.join(base_save_price, label)
            if not os.path.exists(save_folder_price):
                os.mkdir(save_folder_price)

            list_img = os.listdir(folder)
            #print(folder , base_save_images)

            for i, file in enumerate(list_img):
                print("===============\n")
                image_path = os.path.join(folder, file)
                print("image_path " , image_path)
                base=os.path.basename(image_path)
                if base.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    #print("base" , base)
                    #continue
                    s = time.time()
                    res , img_crop, img_price = self.cropImagePrice(image_path)

                    print("time process : " , (time.time() - s) )
                    if res:
                        #print(os.path.join(save_folder_price , base) ,  img_price.shape  )
                        #print(os.path.join(save_folder_image , base)   )
                        imageio.imwrite(os.path.join(save_folder_image , base) , img_crop)
                        imageio.imwrite(os.path.join(save_folder_price , base) , img_price)
                        # cv2.imshow("img_crop" , img_crop)
                        # cv2.imshow("img_price" , img_price)
                    #cv2.waitKey(0)

    def create_blank(self , width, height, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image


    def get_text_size(self ,text, image, font):
        im = Image.new('RGB', (image.width, image.height))
        draw = ImageDraw.Draw(im)
        return draw.textsize(text, font)

    def find_font_size(self , text , font, image, target_width_ratio):
        tested_font_size = 100
        tested_font = ImageFont.truetype(font, tested_font_size)
        observed_width, observed_height = self.get_text_size(text, image, tested_font)
        estimated_font_size = tested_font_size / (observed_width / image.width) * target_width_ratio
        return round(estimated_font_size)


    def drawPrice(self ,line1 , line2 , image , fontpath , font_size=10  , estimate_y=0):
        img_pil = Image.fromarray(image)
        estimate_x=0
        font = ImageFont.truetype(fontpath, int(font_size))
        observed_width, observed_height = self.get_text_size(line1, img_pil, font)
        estimate_x = (img_pil.width - observed_width)//2
        if estimate_x <0 :
            estimate_x = 0
        font2 = ImageFont.truetype(fontpath, int(font_size/2))
        draw = ImageDraw.Draw(img_pil)
        draw.text((estimate_x , estimate_y), line1, font=font)
        observed_width, observed_height = self.get_text_size(line2, img_pil, font2)
        estimate_x = (img_pil.width - observed_width)//2
        if estimate_x <0 :
            estimate_x = 0
        draw.text((estimate_x, estimate_y + font_size), line2 , font=font2)
        img = np.array(img_pil)
        return img
    # update price
    def updatePrice(self , sale_price , purchase_price , fontpath="HGrPrE.ttc"):
        text_price = '￥'
        text_purchase = '買値'
        text_sale = '売値'

        sale_price = text_price + sale_price
        purchase_price = text_price +  purchase_price
        # sale_price
        blue = (50, 50, 255)
        image_blue = self.create_blank(self.WIDTH_PRICE ,self.HEIGHT_PRICE , blue)

        # purchase_price
        red = (255, 50, 50)
        image_red = self.create_blank(self.WIDTH_PRICE ,self.HEIGHT_PRICE , red)

        line_detect_size = sale_price
        if len(purchase_price) > len(sale_price):
            line_detect_size = purchase_price

        img_pil = Image.fromarray(image_red)
        width_ratio = 0.8
        font_size = self.find_font_size(line_detect_size, fontpath, img_pil, width_ratio)
        #print("font_size ", font_size , self.HEIGHT_PRICE)
        estimate_y = 0
        if 1.5*font_size > 1.1*self.HEIGHT_PRICE :
            font_size = int(self.HEIGHT_PRICE/1.5)

        elif 1.5*font_size < 0.9*self.HEIGHT_PRICE :
            estimate_y = int((self.HEIGHT_PRICE - 1.5*font_size)/2.0)



        image_blue = self.drawPrice(sale_price ,text_sale , image_blue ,fontpath , font_size = font_size ,estimate_y=estimate_y )
        image_red = self.drawPrice(purchase_price ,text_purchase , image_red,fontpath , font_size = font_size ,estimate_y=estimate_y)
        h ,w , _ = image_blue.shape
        h1 ,w1 , _ = image_red.shape
        image_price = np.zeros((max(h, h1), w+w1 + 3, 3), np.uint8)
        image_price[0:h , 0 :w] =  image_blue
        image_price[0:h1, w+3 :w + 3 + w1] =  image_red
        # cv2.imshow("image_price" , image_price)
        # cv2.waitKey()
        return image_price
    def resizeImage(self , image , size_width ):
        height , width , _ = image.shape
        size_height = int(height*size_width/width)
        img_size = cv2.resize(image , (size_width , size_height), interpolation=cv2.INTER_AREA)
        return img_size

    # get Thumbnails image & price from path
    def getThumbnailsFromPath(self, image_path, price_path, output_size=128):
        image = imageio.imread(image_path)
        if image is None:
            return False ,None
        price = imageio.imread(price_path)
        if price is None:
            return False ,None

        if image.shape[1] > price.shape[1]:
            image = self.resizeImage(image , price.shape[1])
        else:
            price = self.resizeImage(price , image.shape[1])

        (hA, wA) = image.shape[:2]
        (hB, wB) = price.shape[:2]
        estimate = int(0.2*hB)
        output = np.zeros((hA + hB +estimate, max(wA , wB), 3), dtype="uint8")
        (h, w) = output.shape[:2]
        x_est_image = int(w - wA)//2
        output[0:hA, x_est_image:wA + x_est_image] = image
        x_est_price = int(w - wB)//2
        output[hA +estimate : hA + estimate + hB , x_est_price: wB + x_est_price] = price
        output = self.resizeImage(output , output_size)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return True, output

    # get Thumbnails image & price from image
    def getThumbnailsFromImage(self, image, price, output_size=128):
        if image is None:
            return False ,None
        if price is None:
            return False ,None
        if image.shape[1] > price.shape[1]:
            image = self.resizeImage(image , price.shape[1])
        else:
            price = self.resizeImage(price , image.shape[1])

        (hA, wA) = image.shape[:2]
        (hB, wB) = price.shape[:2]
        estimate = int(0.2*hB)
        output = np.zeros((hA + hB +estimate, max(wA , wB), 3), dtype="uint8")
        (h, w) = output.shape[:2]
        x_est_image = int(w - wA)//2
        output[0:hA, x_est_image:wA + x_est_image] = image
        x_est_price = int(w - wB)//2
        output[hA +estimate : hA + estimate + hB , x_est_price: wB + x_est_price] = price
        output = self.resizeImage(output , output_size)
        return True, output


if __name__ == '__main__':
    folder_path = "/media/anlabadmin/data_Window/dungtd/Kbook/1"
    out_path = "/media/anlabadmin/data_Window/output/images/images_20210404"
    pre_process = ImagePreProcess()
    # input_list = []
    # input_list.append('A4_20210323/202103231707_page-0001.jpg')
    # input_list.append('A4_20210323/202103231707_page-0020.jpg')
    # input_list.append('A4_20210323/202103231707_page-0015.jpg')

    # input_list.append('A5_20210323/202103231139_page-0001.jpg')
    # input_list.append('A5_20210323/202103231139_page-0002.jpg')
    # input_list.append('A5_20210323/202103231139_page-0003.jpg')

    # input_list.append('202103231707_page-0016.jpg')

    # image_list, id_list_images, id_list_price, errors_list = pre_process.cropBookList(folder_path , input_list , out_path )
    # print(id_list_images )
    # print("output image " , len(image_list))
    # res , image = pre_process.getThumbnailsForPath('/media/anlabadmin/DATA/suruga/images/A4_20210323/202103231707_page-0001.jpg' , '/media/anlabadmin/DATA/suruga/price/A4_20210323/202103231707_page-0001.jpg')
    # cv2.imshow("image" , image)
    # imageio.imwrite("test.jpg" , image)
    # cv2.waitKey()
    pre_process.cropBookFolder(folder_path ,out_path )
    #image_price = pre_process.updatePrice("123.456" , "200")
