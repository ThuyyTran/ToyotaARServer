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
from shapely.geometry import Polygon
import math
import settings

AKAZE_THRESH = 1e-3 # AKAZE detection threshold set to locate about 1000 keypoints
AKAZE_NOCTAVELAYERS = 4
RANSAC_THRESH = 10 # RANSAC inlier threshold
NN_MATCH_RATIO = 0.8 # Nearest-neighbour matching ratio

class PlanarMatching:
    def __init__(self, first_img, second_img = None, mtc_feature="akaze", is_need_homography=True):
        self.is_need_homography = is_need_homography

        #self.count_match = 0
        if mtc_feature == "akaze":
            self.detector = cv2.AKAZE_create()
            self.detector.setThreshold(AKAZE_THRESH)
            self.detector.setNOctaveLayers(AKAZE_NOCTAVELAYERS)
        else:
            self.detector = cv2.SIFT_create(2000)

        self.img_1 = first_img
        self.img_2 = second_img

        self.img_1 = self.resize_prop_rect(self.img_1)
        if len(self.img_1.shape) == 3:
            _, _, c = self.img_1.shape
            if c == 3:
                gray_1 = cv2.cvtColor(self.img_1, cv2.COLOR_BGR2GRAY)
            else:
                gray_1 = self.img_1
        else:
            gray_1 = self.img_1
        gray_1 = cv2.GaussianBlur(gray_1, (3, 3), 0)

        self.kps_1, self.features_1 = self.detector.detectAndCompute(gray_1, None)
        self.kps_1 = np.float32([kp.pt for kp in self.kps_1])
        self.img_1_w, self.img_1_h = self.img_1.shape[1], self.img_1.shape[0]

        if self.img_2 is not None:
            self.img_2 = self.resize_prop_rect(self.img_2)
            if len(self.img_2.shape) == 3:
                gray_2 = cv2.cvtColor(self.img_2, cv2.COLOR_BGR2GRAY)
            else:
                gray_2 = self.img_2
            self.kps_2, self.features_2 = self.detector.detectAndCompute(gray_2, None)
            self.kps_2 = np.float32([kp.pt for kp in self.kps_2])
            self.img_2_w, self.img_2_h = self.img_2.shape[1], self.img_2.shape[0]
        else:
            self.gray_2 = None

        self.corner_points_img_1 = np.array(
            [[(0, 0), (self.img_1.shape[1], 0), (self.img_1.shape[1], self.img_1.shape[0]), (0, self.img_1.shape[0])]],
            np.float32)
        self.transformed_corner_points = None
        self.status = None
        self.vis = None

    def re_match_use_mser_image(self, first_img, second_img ):
        is_match_region = False

        if len(first_img.shape) != len(second_img.shape) :
            if len(second_img.shape) == 2:
                _, _, c = first_img.shape
                if c == 3:
                    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
                first_img = first_img.reshape(first_img.shape[0] , first_img.shape[1])
            elif len(first_img.shape) == 2:
                _, _, c = second_img.shape
                if c == 3:
                    second_img = cv2.cvtColor(second_img, cv2.COLOR_BGR2GRAY)
                second_img = second_img.reshape(second_img.shape[0] , second_img.shape[1])
            else:
                return is_match_region , None
        if first_img is None or second_img is None:
            return  is_match_region , None

        img_clone_1 = self.resize_prop_rect(first_img)
        res_mser_1, mask_1 = self.mserDetector(img_clone_1)
        if res_mser_1:
            self.img_1 = mask_1
        else:
            return  is_match_region , None
            # if len(img_clone_1) == 3:
            #     _, _, c = img_clone_1.shape
            #     if c == 3:
            #         self.img_1 = cv2.cvtColor(img_clone_1, cv2.COLOR_BGR2GRAY)
            # else:
            #     self.img_1 = img_clone_1

        self.kps_1, self.features_1 = self.detector.detectAndCompute(self.img_1, None)
        self.kps_1 = np.float32([kp.pt for kp in self.kps_1])
        self.img_1_w, self.img_1_h = self.img_1.shape[1], self.img_1.shape[0]


        img_clone_2 = self.resize_prop_rect(second_img)
        res_mser_2, mask_2 = self.mserDetector(img_clone_2)
        if res_mser_2:
            self.img_2 = mask_2
        else:
            return  is_match_region , None
            # if len(img_clone_2) == 3:
            #     _, _, c = img_clone_2.shape
            #     if c == 3:
            #         self.img_2 = cv2.cvtColor(img_clone_2, cv2.COLOR_BGR2GRAY)
            # else:
            #     self.img_2 = img_clone_2
        self.kps_2, self.features_2 = self.detector.detectAndCompute(self.img_2, None)
        self.kps_2 = np.float32([kp.pt for kp in self.kps_2])
        self.img_2_w, self.img_2_h = self.img_2.shape[1], self.img_2.shape[0]

        self.corner_points_img_1 = np.array(
            [[(0, 0), (self.img_1.shape[1], 0), (self.img_1.shape[1], self.img_1.shape[0]), (0, self.img_1.shape[0])]],
            np.float32)
        self.transformed_corner_points = None
        self.status = None
        self.vis = None

        #match sift
        res_matches, area_match =self.check_condition_relevant(output_vis=False)
        return res_matches, area_match



    def detectBinaryRegions(self, image):
        th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,3)
        return th2

    def mserDetector(self , image):
        res_mser = False
        if image is None:
            return res_mser , None
        if len(image) == 3:
            _, _, c = image
            if c == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
        else:
            gray = image
        h, w = gray.shape
        mser = cv2.MSER_create()
        gray = self.detectBinaryRegions(gray)
        regions, _ = mser.detectRegions(gray)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        mask = np.zeros((h, w), np.uint8)
        #cv2.drawContours(vis, hulls, -1, (0, 255, 0), 1)
        area_image = w*h
        distance_center = []
        index_regions = []
        center = (w/2 , h/2)
        boxes_regions = []
        for i , p in enumerate(regions):
            xmax, ymax = np.amax(p, axis=0)
            xmin, ymin = np.amin(p, axis=0)
            if xmin >= xmax or ymin>=ymax:
                continue
            ratio = float(xmax - xmin)/(ymax - ymin)

            ratio_area = ((xmax - xmin)*(ymax - ymin))/float(area_image)
            if ratio > 4 or ratio < 0.25 or (xmax - xmin) > 0.7*w or (ymax - ymin) > 0.7*h:
                continue
            M = cv2.moments(p)
            if abs(M['m00']) < 0.1:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center_p = (cx, cy)
            distance = math.sqrt( (cx -center[0] )* (cx -center[0] ) + (cy -center[1] )*(cy -center[1] ))
            distance_center.append(distance)
            index_regions.append(i)
            boxes_regions.append([xmin, ymin,xmax, ymax])

        criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)

        K = 3
        if len(distance_center) < 2*K:
            return res_mser , None
        ret, label, center = cv2.kmeans(
            np.float32(distance_center), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center_sort = sorted(range(len(center)), key=lambda k: center[k])
        inliner_cluster = []
        for i in range(K):
            inliner_cluster.append(int(0))
        for i , l in enumerate(label ):
            inliner_cluster[int(l)] +=1

        good_inliner = inliner_cluster[center_sort[0]] + inliner_cluster[center_sort[1]]
        if inliner_cluster[center_sort[2]] > 0 :
            good_inliner /=float(inliner_cluster[center_sort[2]])
        if good_inliner > 1:
            x_min_crop = w
            y_min_crop = h
            x_max_crop = 0
            y_max_crop = 0
            for i , l in enumerate(label ):
                if l == center_sort[0] or l==center_sort[1]:
                    cv2.drawContours(mask, regions,index_regions[i], (255), 2)
                    xmin, ymin, xmax, ymax = boxes_regions[i]
                    #print( xmin, ymin, xmax, ymax)
                    x_min_crop= min(x_min_crop,xmin)
                    x_max_crop= max(x_max_crop,xmax)
                    y_min_crop= min(y_min_crop,ymin)
                    y_max_crop= max(y_max_crop,ymax)
            res_mser = True

            #cv2.rectangle(vis, (x_min_crop,y_min_crop ), (x_max_crop, y_max_crop), (0, 0, 255), 2)
        #print("inlier list " ,inliner_cluster , good_inliner )
        mask = cv2.resize(mask , (image.shape[1] , image.shape[0]), interpolation=cv2.INTER_AREA)
        return res_mser, mask

    @staticmethod
    def resize_prop_rect(src):
        MAX_SIZE = (settings.MTC_IMAGE_WIDTH, settings.MTC_IMAGE_HEIGHT)

        xscale = MAX_SIZE[0] / src.shape[0]
        yscale = MAX_SIZE[1] / src.shape[1]
        scale = min(xscale, yscale)
        if scale > 1:
            return src
        dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_LINEAR)
        return dst

    @staticmethod
    def draw_matches_ransac(image_1, image_2, keypoints_1, keypoints_2, m, sta, points_4):
        (hA, wA) = image_1.shape[:2]
        (hB, wB) = image_2.shape[:2]
        # color1  = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = image_1
        vis[0:hB, wA:] = image_2
        for pt in points_4:
            cv2.circle(vis[0:hB, wA:], (pt[0], pt[1]), 5, (0, 0, 255), 5, cv2.LINE_AA)
        for i in range(-1, 3):
            cv2.line(vis[0:hB, wA:], (points_4[i][0], points_4[i][1]), (points_4[i + 1][0], points_4[i + 1][1]),
                     (0, 255, 0), 1, cv2.LINE_AA)
        for ((trainIdx, queryIdx), s) in zip(m, sta):
            if s == 1:
                pt_1 = (int(keypoints_1[queryIdx][0]), int(keypoints_1[queryIdx][1]))
                pt_2 = (int(keypoints_2[trainIdx][0]) + wA, int(keypoints_2[trainIdx][1]))
                cv2.line(vis, pt_1, pt_2, (255, 0, 0), 1, cv2.LINE_AA)
        return vis

    @staticmethod
    def intersect(i_a, i_b, i_c, i_d):
        def ccw(c_a, c_b, c_c):
            return (c_c[1] - c_a[1]) * (c_b[0] - c_a[0]) > (c_b[1] - c_a[1]) * (c_c[0] - c_a[0])

        return ccw(i_a, i_c, i_d) != ccw(i_b, i_c, i_d) and ccw(i_a, i_b, i_c) != ccw(i_a, i_b, i_d)

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

    def find_matches(self, output_vis):
        t1 = time.time()

        matcher = cv2.DescriptorMatcher.create("BruteForce")
        #matcher = cv2.DescriptorMatcher.create("FlannBased")

        raw_matches = []
        try:
            raw_matches = matcher.knnMatch(np.asarray(self.features_1, np.float32), np.asarray(self.features_2, np.float32), 2)
        except Exception as e:
            print("raw_matches erro = ", e)

        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * NN_MATCH_RATIO:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            pts_1 = np.float32([self.kps_1[i] for (_, i) in matches])
            pts_2 = np.float32([self.kps_2[i] for (i, _) in matches])
            h_matrix, self.status = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, RANSAC_THRESH)
        else:
            return None, None
        if h_matrix is None:
            return None , None
        self.transformed_corner_points = cv2.perspectiveTransform(self.corner_points_img_1, h_matrix)
        if output_vis:
            self.vis = self.draw_matches_ransac(self.img_1, self.img_2, self.kps_1, self.kps_2, matches, self.status,
                                             self.transformed_corner_points[0])
            # cv2.imwrite("debug/result" + str(self.count_match) + ".jpg" , self.vis)
            # self.count_match += 1

        return cv2.countNonZero(self.status), matches

    def is_ordered(self):
        points = self.transformed_corner_points[0]
        for i in range(-4, 0):
            if self.intersect(points[i], points[i+1], points[i+2], points[i+3]):
                return False
        return True

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

    def is_image_relevant(self, img_2=None, output_vis=False):
        res_matches = False
        area_match = 0.0
        if img_2 is not None:
            self.img_2 = self.resize_prop_rect(img_2)
            if len(self.img_2.shape) == 3:
                gray_2 = cv2.cvtColor(self.img_2, cv2.COLOR_BGR2GRAY)
            else:
                gray_2 = self.img_2
            self.kps_2, self.features_2 = self.detector.detectAndCompute(gray_2, None)
            self.kps_2 = np.float32([kp.pt for kp in self.kps_2])
            self.img_2_w, self.img_2_h = self.img_2.shape[1], self.img_2.shape[0]

        res_matches, area_match =self.check_condition_relevant(output_vis=False)
        return res_matches , area_match

    def is_descs_relevant(self, kps, descs, w, h):
        self.kps_2, self.features_2 = kps, descs
        self.img_2_w, self.img_2_h = w, h
        res_matches, area_match =self.check_condition_relevant(output_vis=False)
        return res_matches, area_match

    def find_mask_keypoint(self , src_pt, dst_pt):
        w1, h1 = self.img_1_w, self.img_1_h
        w2, h2 = self.img_2_w, self.img_2_h
        # print("size ", w1 , h1, w2 ,h2 )
        mask_dst = np.zeros((h2 ,w2), np.uint8)
        mask_src = np.zeros((h1 ,w1), np.uint8)
        size_fill  = int(w1/50)
        if(w1 < 50 ):
            size_fill  = 5
        for idx_pt , pt in enumerate (src_pt):
            x_min = pt[0] - size_fill
            y_min = pt[1] - size_fill
            x_max = pt[0] + size_fill
            y_max = pt[1] + size_fill
            if(x_min <=0 ):
                x_min = 0
            if(y_min <=0 ):
                y_min = 0
            if(x_max >= w1):
                x_max = w1
            if(y_max >= h1):
                y_max = h1
            #print("image1 " , x_min, y_min ,x_max, y_max , w1, h1)
            cv2.rectangle(mask_src, (x_min, y_min), (x_max, y_max), (255),-1)
            
            x_min = dst_pt[idx_pt][0] - size_fill
            y_min = dst_pt[idx_pt][1] - size_fill
            x_max = dst_pt[idx_pt][0] + size_fill
            y_max = dst_pt[idx_pt][1] + size_fill
            if(x_min <=0 ):
                x_min = 0
            if(y_min <=0 ):
                y_min = 0
            if(x_max >= w2):
                x_max = w2
            if(y_max >= h2):
                y_max = h2
            #print("image2 ", x_min, y_min ,x_max, y_max , w2, h2)
            cv2.rectangle(mask_dst, (x_min, y_min), (x_max, y_max), (255),-1)

        kernel = np.ones((size_fill*2 +1 ,size_fill*2 +1),np.uint8)
        mask_dst = cv2.dilate(mask_dst,kernel,iterations = 1)
        mask_src = cv2.dilate(mask_src,kernel,iterations = 1)

        #mask_dst = cv2.erode(mask_dst,kernel,iterations = 1)
        # cv2.imshow("image 1 " ,mask_src )
        # cv2.imshow("image 2 " ,mask_dst )
        # mask_dst = cv2.bitwise_not(mask_dst)
        # copy_mask = vis_segmentation_transparent(img_color , mask_dst )
        # out_path2 = "sample/sample" +  str(index_target) + ".png"
        # cv2.imwrite(out_path2,copy_mask )
        return mask_src , mask_dst

    def area_conditions(self, keypoints_1, keypoints_2, m, sta , area_thresh_min=0.15 , area_thresh_med= 0.2 , area_thresh_max= 0.5):
        res_area = False
        src_point = []
        dst_point = []
        ratio_1 = 0.0
        ratio_2 = 0.0
        w1 ,h1 = self.img_1_w, self.img_1_h
        w2 ,h2 = self.img_2_w, self.img_2_h
        box1 = [w1  , h1 , 0 , 0 ]
        box2 =[ w2 , h2 , 0 , 0]
        count = 0
        pts_1 = np.float32([keypoints_1[i] for (_, i) in m])
        pts_2 = np.float32([keypoints_2[i] for (i, _) in m])
        h_matrix, _ = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, RANSAC_THRESH)
        pts_in = pts_1.reshape(1, -1, 2)
        pts_out = pts_2.reshape(1, -1, 2)
        out_pts = cv2.perspectiveTransform(pts_in, h_matrix)

        #print("out_pts" , len(out_pts[0]) , len(sta) , len(m))
        for (i , p1 , p2 , p3, s) in zip(range(len(sta)) ,pts_out[0] , out_pts[0] , pts_in[0], sta):
            if s == 1:
                distance_ = math.sqrt( (p1[0] -p2[0])* (p1[0] -p2[0]) + (p1[1] -p2[1])*(p1[1] -p2[1]) )
                if(distance_ < 5 ):
                    x_1 = int(p1[0])
                    y_1 = int(p1[1])
                    x_2 = int(p2[0])
                    y_2 = int(p2[1])

                    x_3 = int(p3[0])
                    y_3 = int(p3[1])

                    box1[0] = min(box1[0] , x_3)
                    box1[1] = min(box1[1] , y_3)
                    box1[2] = max(box1[2] , x_3)
                    box1[3] = max(box1[3] , y_3)

                    box2[0] = min(box2[0] , x_1)
                    box2[1] = min(box2[1] , y_1)
                    box2[2] = max(box2[2] , x_1)
                    box2[3] = max(box2[3] , y_1)
                    count += 1
                    src_point.append((x_3 , y_3))
                    dst_point.append((x_1 , y_1))
                else:
                    sta[i] = [0]
        # self.vis = self.draw_matches_ransac(self.img_1, self.img_2, keypoints_1, keypoints_2, m, sta,
        #                                     self.transformed_corner_points[0])
        # cv2.imwrite("debug/result" + str(self.count_match) + ".jpg" , self.vis)
        # self.count_match += 1
        if len(src_point) >5 :
            # vis = self.draw_matches_ransac(self.img_1, self.img_2, keypoints_1, keypoints_2, m, sta,
            #                                 self.transformed_corner_points[0])
            # vis = cv2.rectangle(vis, (box1[0] , box1[1]), (box1[2], box1[3]), (0 , 0 , 255), 2)
            # vis = cv2.rectangle(vis, (box2[0] + w1 , box2[1]), (box2[2] + w1, box2[3]), (0 , 0 , 255), 2)
            # cv2.imshow("vis" , vis)
            mask_src , mask_dst = self.find_mask_keypoint(src_point , dst_point)
            area_1 = cv2.countNonZero(mask_src)
            area_2 = cv2.countNonZero(mask_dst)

            area_box1 = (box1[2] - box1[0])*(box1[3] - box1[1])
            area_box2 = (box2[2] - box2[0])*(box2[3] - box2[1])
            ratio_1 = float(area_1)/(w1 * h1)
            ratio_2 = float(area_2)/(w2*h2)
            res_area = True
            print("ratio " , ratio_1 , ratio_2)


        return res_area , ratio_1 , ratio_2

    def check_condition_relevant(self, output_vis=False):
        area_match = 0
        res_matches = False
        if not self.is_need_homography:
            res_matches = self.is_relevant(output_vis=False)
        else:
            res_matches, area_match = self.is_relevant_homography(output_vis=False)
        return  res_matches, area_match

    def getAreaMatching(self,  keypoints_1, keypoints_2, m, sta):
        ratio_1 = 0.0
        ratio_2 = 0.0
        w1 ,h1 = self.img_1_w, self.img_1_h
        w2 ,h2 = self.img_2_w, self.img_2_h
        box1 = [w1  , h1 , 0 , 0 ]
        box2 =[ w2 , h2, 0 , 0]
        count = 0

        pts_1 = np.float32([keypoints_1[i] for (_, i) in m])
        pts_2 = np.float32([keypoints_2[i] for (i, _) in m])
        pts_in = pts_1.reshape(1, -1, 2)
        pts_out = pts_2.reshape(1, -1, 2)



        box_original_1 = [w1  , h1 , 0 , 0 ]
        box_original_2 =[ w2 , h2 , 0 , 0]
        #print("out_pts" , len(out_pts[0]) , len(sta) , len(m))
        for (i , p1 , p2, s) in zip(range(len(sta)) ,pts_in[0] , pts_out[0], sta):
            x_1 = int(p1[0])
            y_1 = int(p1[1])
            x_2 = int(p2[0])
            y_2 = int(p2[1])
            box_original_1[0] = min(box_original_1[0] , x_1)
            box_original_1[1] = min(box_original_1[1] , y_1)
            box_original_1[2] = max(box_original_1[2] , x_1)
            box_original_1[3] = max(box_original_1[3] , y_1)

            box_original_2[0] = min(box_original_2[0] , x_2)
            box_original_2[1] = min(box_original_2[1] , y_2)
            box_original_2[2] = max(box_original_2[2] , x_2)
            box_original_2[3] = max(box_original_2[3] , y_2)
            if s == 1:
                
                box1[0] = min(box1[0] , x_1)
                box1[1] = min(box1[1] , y_1)
                box1[2] = max(box1[2] , x_1)
                box1[3] = max(box1[3] , y_1)
                
                box2[0] = min(box2[0] , x_2)
                box2[1] = min(box2[1] , y_2)
                box2[2] = max(box2[2] , x_2)
                box2[3] = max(box2[3] , y_2)
                
                count += 1
        if count >5 :
            # (hA, wA) = self.img_1.shape[:2]
            # (hB, wB) = self.img_2.shape[:2]
            # vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
            # vis[0:hA, 0:wA] = self.img_1
            # vis[0:hB, wA:] = self.img_2
            # vis = cv2.rectangle(vis, (box1[0] , box1[1]), (box1[2], box1[3]), (0 , 0 , 255), 2)
            # vis = cv2.rectangle(vis, (box2[0] + w1 , box2[1]), (box2[2] + w1, box2[3]), (0 , 0 , 255), 2)

            # vis = self.draw_matches_ransac(self.img_1, self.img_2, keypoints_1, keypoints_2, m, sta,
            #                                 self.transformed_corner_points[0])
            # vis = cv2.rectangle(vis, (box1[0] , box1[1]), (box1[2], box1[3]), (0 , 0 , 255), 2)
            # vis = cv2.rectangle(vis, (box2[0] + w1 , box2[1]), (box2[2] + w1, box2[3]), (0 , 0 , 255), 2)
            # cv2.imshow("vis" , vis)
            area_max1 = (box_original_1[2] - box_original_1[0])*(box_original_1[3] - box_original_1[1])
            area_max2 = (box_original_2[2] - box_original_2[0])*(box_original_2[3] - box_original_2[1])
            area_box1 = (box1[2] - box1[0])*(box1[3] - box1[1])
            area_box2 = (box2[2] - box2[0])*(box2[3] - box2[1])
            if area_max1 > 10 and area_max2 > 10:
                ratio_1 = float(area_box1)/(w1 * h1)
                ratio_2 = float(area_box2)/(w2*h2)
            
        return max(ratio_1 , ratio_2)
        

    def is_relevant(self, output_vis=False):
        r , matches = self.find_matches(output_vis)

        if self.transformed_corner_points is None or r is None:
            print("None")
            return False
        if r < 10:
            #print("< 15, r = ", r)
            return False

        if not self.is_ordered() or not self.is_convex():
            #print("is_ordered or is_convex")
            return False

        if not self.angle_conditions():
            #print("angle")
            return False

        # if not self.area_conditions(self.kps_1 , self.kps_2, matches , self.status):
        #     #print("area smaller")
        #     return False

        # area = Polygon(self.transformed_corner_points[0]).area
        # area_ratio = area / (self.img_2_w * self.img_2_h)
        area_ratio =   self.getAreaMatching(self.kps_1 , self.kps_2, matches , self.status)
        rect = cv2.minAreaRect(self.transformed_corner_points)
        rotated_box = self.order_points(cv2.boxPoints(rect))

        side_length = [euclidean(rotated_box[0], rotated_box[1]), euclidean(rotated_box[0], rotated_box[-1])]
        side_length_ratio = max(side_length) / min(side_length)
        # print("Area ratio: ", area_ratio)
        # print("Is ordered: ", self.is_ordered())
        # print("Is convex: ", self.is_convex())
        # print("Side: ", side_length_ratio)
        # print("points: ", self.transformed_corner_points)
        # print("box: ", rotated_box)
        # print("")
        return area_ratio > 0.05 and side_length_ratio < 4


    def is_relevant_homography(self, output_vis=False):
        r , matches = self.find_matches(output_vis)

        if self.transformed_corner_points is None or r is None:
            print("None")
            return False, 0
        if r < 10:
            #print("< 15, r = ", r)
            return False, 0

        if not self.is_ordered() or not self.is_convex():
            #print("is_ordered or is_convex")
            return False, 0

        if not self.angle_conditions():
            #print("angle")
            return False, 0

        

        # area = Polygon(self.transformed_corner_points[0]).area
        # area_ratio = area / (self.img_2_w * self.img_2_h)
        area_ratio =   self.getAreaMatching(self.kps_1 , self.kps_2, matches , self.status)

        rect = cv2.minAreaRect(self.transformed_corner_points)
        rotated_box = self.order_points(cv2.boxPoints(rect))

        side_length = [euclidean(rotated_box[0], rotated_box[1]), euclidean(rotated_box[0], rotated_box[-1])]
        side_length_ratio = max(side_length) / min(side_length)
        res_area , area_box1 ,area_box2 = self.area_conditions(self.kps_1 , self.kps_2, matches , self.status)
        area_ratio = max(area_box1 ,area_box2)
        # print("area_ratio" , area_ratio)
        # print("Area ratio: ", area_ratio)
        # print("Is ordered: ", self.is_ordered())
        # print("Is convex: ", self.is_convex())
        # print("Side: ", side_length_ratio)
        # print("points: ", self.transformed_corner_points)
        # print("box: ", rotated_box)
        # print("")
        res = area_ratio > 0.05 and side_length_ratio < 4 and res_area
        return res , area_ratio


import time
if __name__ == '__main__':
    # img_a = cv2.imread("/media/anlabadmin/data_ubuntu/tund/lashinbang/image-customer-not-detect/2020-01-27 03-06-43.jpg")
    # img_b = cv2.imread("/media/anlabadmin/big_volume/Lashinbang_data/Images_300_399/355/203710227355_L.jpg")
    # img_a = cv2.imread("/media/anlabadmin/data_ubuntu/tund/lashinbang/image-customer-not-detect/2020-01-27 03-49-34.jpg")
    # img_b = cv2.imread("/media/anlabadmin/big_volume/Lashinbang_data/Images_800_899/816/203710453816_L.jpg")

    img_a = cv2.imread("/media/anlabadmin/data_Window/dungtd/KBOOK/0_6f05ace6-1a49-4645-8f6b-4239092393d2_OK.jpg")
    # img_b = cv2.imread("/media/anlabadmin/big_volume/Lashinbang_data/Images_200_300/243/203510199243_L.jpg")
    img_b = cv2.imread("/media/anlabadmin/data_Window/dungtd/KBOOK/202103301250_page-0009.jpg")


    t1 = time.time()
    M = PlanarMatching(img_a)

    print("is relevant: ", M.is_image_relevant(img_2=img_b,output_vis=False))
    print("-------------------------------")
    if M.vis is not None:
        cv2.imshow("1", M.vis)
    cv2.waitKey()
