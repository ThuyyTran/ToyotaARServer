from shapely.geometry import Polygon,Point,LineString
import numpy as np
import cv2
import math
from numpy.linalg import inv

def resize_ratio(img, imsize = 720):
    h, w = img.shape[:2]
    max_wh = max(h,w)
    if h < max_wh:
        new_w = imsize
        new_h = int(h * imsize/w)
    else:
        new_h = imsize
        new_w = int(w * imsize/h)
    img = cv2.resize(img, (new_w, new_h))
    return img

def get_points(list_size,list_homo):
    point_min=[[0,0]]
    for i,homo in enumerate(list_homo):
        point_check1=[list_size[0][0]//2,list_size[0][1]//2]
        point_check1 = np.array([[point_check1]], dtype='float32')
        point_check2 = cv2.perspectiveTransform(point_check1, inv(homo))
        size=[int(list_size[0][0]*(1/homo[0,0])),int(list_size[0][1]*(1/homo[1][1]))]
        
        size_=[size[0]//2,size[1]//2]
        transformed_corner_points=point_check2-size_
        transformed_corner_points=(transformed_corner_points*list_size[i+1]/size)[0][0]
        point_min.append([point_min[-1][0]+int(transformed_corner_points[0]),point_min[-1][1]+int(transformed_corner_points[1])])
    return point_min
        
def get_indexs_homo(matrixs):
    indexs=[]
    list_homo=[]
    for i,matrix in enumerate(matrixs["matrix"]):
        indexs.append(i)
        list_homo.append(matrix)
    indexs.append(i+1)
    return indexs,list_homo

def get_size_imges(height,width,list_homo):
    list_size=[[width,height]]
    for i,matrix in enumerate(list_homo):
        size=[int(list_size[-1][0]*(1/matrix[0,0])),int(list_size[-1][1]*(1/matrix[1][1]))]
        list_size.append(size)
    return list_size


def result_video(pts_new,list_size,angle_bad):
    list_size=np.array(list_size)
    max_size=list_size+pts_new
    max_size=np.max(max_size,axis=0)
    #draw lines
    pts_new_center=[]
    for i,pt in enumerate(pts_new):
        pt_center= pt+[list_size[i][0]//2,list_size[i][1]//2]
        pts_new_center.append(pt_center.tolist())
    pts_new_center=np.array(pts_new_center)
    # print("pts_new_center",pts_new_center)
    pts_new_center=(pts_new_center/4).astype(int)

    check_video=True
    # create pts_new_center with format linestring
    pts_new_center_linestring=[tuple(pts_new_center[0]),tuple(pts_new_center[1])]
    # print("pts_new_center_linestring",pts_new_center_linestring)
    for i in range(len(pts_new_center)-1):
        start=pts_new_center[i]
        end=pts_new_center[i+1]
        # check back angle
        if(i>0):
            vector1=[start[0]-pts_new_center[i-1][0],start[1]-pts_new_center[i-1][1]]
            vector2=[end[0]-start[0],end[1]-start[1]]
            direction_vector= vector1[0]*vector2[0]+vector1[1]*vector2[1]
            if(direction_vector<0):
                angle=180-caculate_degree_2_vector(vector1,vector2)
                # print("caculate_degree_2_vector,angle ,i",caculate_degree_2_vector(vector1,vector2),angle,i)
                if(angle<angle_bad):
                    check_video=False
        ###################
        #check intersec####################
        LineString1=LineString(pts_new_center_linestring)
        LineString2=LineString([tuple(pts_new_center[i]),tuple(pts_new_center[i+1])])
        if(check_intersec(LineString1,LineString2) ==True and i>2):
            check_video=False
        if(i>1):
            pts_new_center_linestring.append(tuple(pts_new_center[i]))
        #####################################
        
    return check_video
    

def camera_correct_trajectory(list_imgs,matrixs,angle_bad=30):
    indexs,list_homo=get_indexs_homo(matrixs)
    #get list size images
    img=cv2.cvtColor(list_imgs[0], cv2.COLOR_BGR2GRAY)
    height,width= resize_ratio(img,matrixs["scale"]).shape
    list_size=get_size_imges(height,width,list_homo)
    # print("list_size",list_size)
    #get list  points
    pts_new=get_points(list_size,list_homo)
    #draw summary
    scale_origin=max(img.shape[0],img.shape[1])/matrixs["scale"]
    # print("pts_new 720",pts_new)
    pts_new=(np.array(pts_new)*scale_origin).astype(int)
    # print("pts_new 720 convert",pts_new)
    list_size=(np.array(list_size)*scale_origin).astype(int).tolist()
    pts_new=np.array(pts_new)
    pts_new+=abs(np.min(pts_new,axis=0))
    
    check_video=result_video(pts_new,list_size,angle_bad)

    return check_video

def caculate_degree_2_vector(vector1,vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle= math.degrees(angle)
    return angle

def check_intersec(LineString1,LineString2):
    return LineString2.intersects(LineString1)

if __name__ == '__main__':
    
    check_video=camera_correct_trajectory(imgs,matrix,angle_bad=30)
    print("check_video*******",check_video)