import enum
# from importlib.resources import pathy
import os
import pickle
from tracemalloc import start
from tqdm import tqdm
from intersection2 import  find_homography,intersec_area_2,intersec_area_3
import cv2
import numpy as np
from shapely.geometry import Polygon
from numpy.linalg import inv
import math

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
def newsort(e):
	head= int(e.split("/")[-1].split(".")[0].split("_")[-2])
	tail=int(e.split("/")[-1].split(".")[0].split("_")[-1])
	return (head*10000 + tail)

def get_points_check(list_size,list_homo):
    point_min=[[0,0]]
    # print("***************")
    # print("len list homo",len(list_homo))
    # print("list_size",list_size)
    # with open("homo_check.pickle","wb") as f:
    #     pickle.dump(list_homo,f)
    for i,homo in enumerate(list_homo):

        # point_img2 = np.array([[(0,0),(list_size[0][0],0),(list_size[0][0],list_size[0][1]),(0,list_size[0][1])]],np.float32)
        # transformed_corner_points=cv2.perspectiveTransform(point_img2, inv(list_homo[i]))[0][0]
        # # print("transformed_corner_points",transformed_corner_points)

        #caculate homo intersec invert
        # intersec_inv=intersec_area_3(list_size[0][0],list_size[0][1],inv(homo))
        point_check1=[list_size[0][0]//2,list_size[0][1]//2]
        point_check1 = np.array([[point_check1]], dtype='float32')
        point_check2 = cv2.perspectiveTransform(point_check1, inv(homo))
        #add
        size=[int(list_size[0][0]*(1/homo[0,0])),int(list_size[0][1]*(1/homo[1][1]))]
        
        size_=[size[0]//2,size[1]//2]
        transformed_corner_points=point_check2-size_
        # print("size",size)
        transformed_corner_points=(transformed_corner_points*list_size[i+1]/size)[0][0]
        # print("transformed_corner_points",transformed_corner_points)
        # print("transformed_corner_points",transformed_corner_points)
        point_min.append([point_min[-1][0]+int(transformed_corner_points[0]),point_min[-1][1]+int(transformed_corner_points[1])])
    return point_min

# def get_points_check
def get_points_test(paths_new,list_size,list_homo):
    print("***************")
    print("len list homo",len(list_homo))
    print("list_size",list_size)
    points_min=[[0,0]]
    height,width=resize_ratio(cv2.imread(paths_new[0],0)).shape
    # pts=[[0,0]]
    for i,matrix in enumerate(list_homo):
        center1= list_size[i]
        center1 = np.array([[center1]], dtype='float32')/2
        center2 = cv2.perspectiveTransform(center1, matrix)
        print("center1 center2",center1 ,center2)
        # pts.append(center2[0][0].tolist())
        vector=(center1-center2)[0][0].tolist()
        print("vector",vector)
        # ptx=points_min[-1][0]+int(((vector[0]+center1[0][0][0]+list_size[i+1][0]/2)/max(list_size[i][0],list_size[i][1]))*list_size[i+1][0])
        # pty=points_min[-1][1]+int(((vector[1]+center1[0][0][1]-list_size[i+1][0]/2)/max(list_size[i][0],list_size[i][1]))*list_size[i+1][1])
        print("list_size[i+1][1] list_size[i][1]",list_size[i+1][1],list_size[i][1])
        ptx=((vector[0]+center1[0][0][0]+list_size[i+1][1]/2))#*list_size[i+1][0]/list_size[i][0])
        pty=((vector[1]+center1[0][0][1]-list_size[i+1][1]/2))#*list_size[i+1][1]/list_size[i][1])
        points_min.append([ptx,pty])
        # pts.append([int(ptx),int(pty)])
    print("top left",points_min)
    return points_min


def get_points(paths_new,list_size,list_homo):
    points_min=[[0,0]]
    print("len paths_new", len(paths_new))
    print("len list_size",len(list_size))
    print("len list homo",len(list_homo))
    height,width=resize_ratio(cv2.imread(paths_new[0],0)).shape

    pts=[[int(width/2),int(height/2)]]

    save_vector=[]
    for i,matrix in enumerate(list_homo):
        center1= pts[-1]
        # center1=list_size[i]
        center1 = np.array([[center1]], dtype='float32')
        # center1=center1/2
        center2 = cv2.perspectiveTransform(center1, matrix)
        vector=(center1-center2)[0][0].tolist()
        save_vector.append(vector)
        pt_y=0
        pt_x=0
        for j,vec in enumerate(save_vector):
            # print("vec0,1", vec)
            pt_x= pt_x  + vec[0] #list_size[j][0] #+vec[0]*list_homo[j][0][0]
            pt_y= pt_y +vec[1] 
            # print("points_min",pt_x ,pt_y, list_size[j][0] )
        points_min.append([pt_x,pt_y])
    return points_min
        
def get_indexs_homo(list_imgs,matrixs):
    indexs=[0]
    list_homo=[]
    homo=None
    # img=cv2.imread(paths[0],0)
    img=cv2.cvtColor(list_imgs[0], cv2.COLOR_BGR2GRAY)
    img=resize_ratio(img,720)
    ps_save=None
    # homo_check=None
    for i,_ in enumerate(list_imgs[:-1]):
        if(i<0):
            continue
        if(homo is None):
            homo=matrixs["matrix"][i]
            # continue
        else:
            homo= np.matmul(homo,matrixs["matrix"][i]  )#homo@matrixs["matrix"][i]  
        intersec=intersec_area_2(img,homo)
        #####check image after
        if(i== len(matrixs["matrix"])-1):
            homo_check=homo
        else:
            homo_check= np.matmul(homo,matrixs["matrix"][i +1 ]  ) #homo@matrixs["matrix"][i+1]
        intersec_check=intersec_area_2(img,homo_check)
        # print("intersec_check, intersec ",intersec_check,  intersec)
        if(intersec_check <0.1):
            # print("ps***",ps)

            ######################
            # #caculate homo intersec invert
            # intersec_inv=intersec_area_2(img,inv(homo))
            # point_check1=[202,360]
            # point_check1 = np.array([[point_check1]], dtype='float32')
            # point_check2 = cv2.perspectiveTransform(point_check1, inv(homo))
            # print(" intersec inv, point_check2",intersec_inv,point_check2)
            

            ###################################
            # print("intersec ",intersec_check,  intersec)
            # ps_save=ps
            # print("homo",homo)
            # print("******************************image",paths[i+1])
            # indexs.append(i+1)
            # list_homo.append(homo)
            # homo=None
            if(intersec_check<0.1 or i== len(matrixs["matrix"])-1):
                indexs.append(i+1)
                list_homo.append(homo)
                homo=None
    # print("list size  " , len(list_homo))
    return indexs,list_homo

def get_size_imges(height,width,list_homo):
    list_size=[[width,height]]
    scale=None
    for i,matrix in enumerate(list_homo):
        scale_matrix=   min(matrix[0,0],matrix[1,1])
        # size=[int(list_size[-1][0]*(1/scale_matrix)),int(list_size[-1][1]*(1/scale_matrix))]
        size=[int(list_size[-1][0]*(1/matrix[0,0])),int(list_size[-1][1]*(1/matrix[1][1]))]
        # size=[int(list_size[-1][0]*(matrix[0,0])),int(list_size[-1][1]*(matrix[1][1]))]
        # scale_matrix=(matrix[0,0]+matrix[1,1])/2
        # if(i==0):
        #     scale=scale_matrix
        # else:
        #     scale=scale*scale_matrix
        # size= [int(width*(1/scale)),int(height*(1/scale))]
        list_size.append(size)
    print("list size *****",list_size)
    return list_size

def get_dict_box(list_size,pts_new,arr_dict_box_new):
    
    arr_dict_box_new_=[]
    for i,value in enumerate(arr_dict_box_new):
        dict_box={}
        for key in value.keys():
            # print("checkkkk",pts_new[i][0])
            x1=pts_new[i][0]+int((value[key][0]/list_size[0][0]) * list_size[i][0])
            x2=pts_new[i][1]+int((value[key][1]/list_size[0][1]) * list_size[i][1])
            x3=pts_new[i][0]+int((value[key][2]/list_size[0][0]) * list_size[i][0])
            x4=pts_new[i][1]+int((value[key][3]/list_size[0][1]) * list_size[i][1])
            dict_box[key]=[x1,x2,x3,x4]
        arr_dict_box_new_.append(dict_box)
    return pts_new,arr_dict_box_new_

def draw(list_imgs,indexs,pts_new,list_size,arr_dict_box_new):
    list_size=np.array(list_size)
    # size_max=(np.max(pts_new,axis=0))+np.max(list_size,axis=0)
    # width_max=size_max[0]
    # height_max=size_max[1]
    ###add code####
    max_size=list_size+pts_new
    # print("max_size",max_size.shape)
    # print("max_size",np.max(max_size,axis=0))
    max_size=np.max(max_size,axis=0)
    width_max=max_size[0]
    # print("width_max",width_max)
    height_max=max_size[1]
    back_ground = np.zeros((int(height_max)+1,int(width_max)+1,3))

    print("back ground",back_ground.shape)
    for i,index in enumerate(indexs):
        # print("****************i****************",i)
        y1=round(pts_new[i][1])
        y2=(y1+round(list_size[i][1]))
        x1=round(pts_new[i][0])
        x2=(x1+round(list_size[i][0]))
        # image=(cv2.imread(paths[index]))
        # image=cv2.cvtColor(list_imgs[index], cv2.COLOR_BGR2GRAY)
        image=resize_ratio(list_imgs[index])
        image=cv2.resize(image,(list_size[i][0],list_size[i][1]))
        back_ground[y1:y2,x1:x2]=image
        back_ground=cv2.rectangle(back_ground, (x1,y1), (x2,y2), (0,0,255), 5)
        # for key in arr_dict_box_new[i].keys():
        #     start_point= (arr_dict_box_new[i][key][0],arr_dict_box_new[i][key][1])
        #     end_point=(arr_dict_box_new[i][key][2],arr_dict_box_new[i][key][3])
        #     back_ground=cv2.rectangle(back_ground, start_point, end_point, (0,255,0), 5)
    # cv2.imwrite("background.jpg",back_ground)
    return back_ground
    

def get_ImageMerge_and_infoBoxes(list_imgs,matrixs,arr_dict_box):
    #get index image intersec 10%
    # arr_dict_box_=arr_dict_box.copy()
    indexs,list_homo=get_indexs_homo(list_imgs,matrixs)

    # print("indexs",indexs)
    
    #get list size images
    # img=cv2.imread(paths[0],0)
    img=cv2.cvtColor(list_imgs[0], cv2.COLOR_BGR2GRAY)
    height,width= resize_ratio(img,matrixs["scale"]).shape
    list_size=get_size_imges(height,width,list_homo)
    #get list  points
    # paths_new=[paths[index] for index in indexs ]
    list_imgs_new=[list_imgs[index] for index in indexs ]
    pts_new=get_points_check(list_size,list_homo)

    
    #draw summary
    scale_origin=max(img.shape[0],img.shape[1])/matrixs["scale"]
    # print("pts_new 720",pts_new)
    pts_new=(np.array(pts_new)*scale_origin).astype(int)
    # print("pts_new 720 convert",pts_new)
    list_size=(np.array(list_size)*scale_origin).astype(int).tolist()
    # print("list_size new",list_size) 
    arr_dict_box_new=[arr_dict_box[index] for index in indexs ]
    
    # fix coordinates box
    pts_new_=np.copy(pts_new)
    # print("pts_new**",pts_new)
    # print("pts",(pts_new).shape)
    for i in range(len(pts_new)):
        if(i==0):
            continue
        vector1=[1,0]
        vector2=[pts_new_[i][0]-pts_new_[i-1][0],pts_new_[i][1]-pts_new_[i-1][1]]
        # print("i pts1 pts2 vector2",i,pts_new_[i],pts_new_[i-1],vector2)
        # vector2=pts_new
        angle=caculate_degree_2_vector(vector1,vector2)
        if(angle<45):
            # print("angle<45")
            # print("pts_new[:i+1,0]",pts_new[i:,0])
            pts_new[i][0]=pts_new[i-1][0]+list_size[i-1][0]
            pts_new[i][1]=pts_new[i-1][1]+(pts_new_[i][1]-pts_new_[i-1][1])
        elif(angle<90 ):
            # print("angle<90")
            pts_new[i][0]=pts_new[i-1][0]#+list_size[i-1][0]
            if(vector2[1]>0):
                pts_new[i][1]=pts_new[i-1][1]+list_size[i-1][1]
            else:
                pts_new[i][1]=pts_new[i-1][1]-list_size[i][1]
            
        elif(angle>135):
            # print("angle>135",angle)
            pts_new[i][1]=pts_new[i-1][1]+(pts_new_[i][1]-pts_new_[i-1][1])#+list_size[i-1][1]
            pts_new[i][0]=pts_new[i-1][0]-list_size[i][0]
        else:
            # print("90<angle<135",angle)
            pts_new[i][0]=pts_new[i-1][0]-list_size[i][0]
            pts_new[i][1]=pts_new[i-1][1]+list_size[i-1][1]
    # #############################################3
    pts_new=np.array(pts_new)
    pts_new+=abs(np.min(pts_new,axis=0))
    # print("arr_dict_box000000000",arr_dict_box[0])
    pts_new,arr_dict_box_new=get_dict_box(list_size,pts_new,arr_dict_box_new)
    # print("arr_dict_box000000000",arr_dict_box[0])
    # print("pts_new",pts_new)

    #get list_object_info
    list_objects_info=[]
    for i,index in enumerate(indexs):
        for key in arr_dict_box_new[i].keys():
            objects_info={}
            objects_info["index_frame"]=index
            objects_info["box"]=arr_dict_box_new[i][key]
            objects_info["index_object"]=key
            list_objects_info.append(objects_info)
    # list_objects_info=[]
    # for i,index in enumerate(indexs):
    #     objects_info={}
    #     objects_info["index_frame"]=index
    #     objects_info["box"]=arr_dict_box_new[i]
    #     objects_info["index_object"]=[]
    #     for key in arr_dict_box_new[i].keys():
    #         if(len(arr_dict_box_new[i][key])>4):
    #             objects_info["index_object"].append(key)
    #     list_objects_info.append(objects_info)

    # print("dict_full_image",dict_full_image.keys())
    back_ground=draw(list_imgs,indexs,pts_new,list_size,arr_dict_box_new)
    return back_ground,list_objects_info
    # draw2(paths,indexs,list_size)
def caculate_degree_2_vector(vector1,vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle= math.degrees(angle)
    return angle

def get_index_have_same_id(list_id):
    list_id_check=[]
    list_index=[]
    for i, value in enumerate(list_id):
        if(value not in list_id_check):
            list_id_check.append(value)
        else:
            list_index.append(i)
    return list_index

if __name__ == '__main__':
    video="5336"
    with open(f"data_revamp/IMG_{video}/matrix/matrix_forward.pickle","rb") as f:
        matrixs=pickle.load(f)
        print("matrixs", matrixs.keys())
        print("scale",matrixs["scale"])
    with open(f"arr_dict_box_{video}.pickle","rb") as f:
        arr_dict_box=pickle.load(f)
    paths_ = os.listdir(f"data_revamp/IMG_{video}/images")
    paths_= [f"data_revamp/IMG_{video}/images/"+value for value in paths_]
    paths=sorted(paths_)
    paths.sort(key=newsort)
    list_imgs=[cv2.imread(value) for value in paths]
    # exit()
    # paths=paths[:15]
    # print("index 96",paths.index("data_revamp/IMG_5335/images/IMG_5335_96.jpg"))


    back_ground,list_objects_info=get_ImageMerge_and_infoBoxes(list_imgs,matrixs,arr_dict_box)
    print("*********dict_full_iamge_index",list_objects_info[4]["index_frame"])
    print("*********dict_full_iamge_tracking",list_objects_info[4]["index_object"])
    print("*********dict_full_iamge_box",list_objects_info[4]["box"])
    
    
    # draw box
    for i in range(len(list_objects_info)):
        for key in list_objects_info[i]["box"].keys():
            start_point= (list_objects_info[i]["box"][key][0],list_objects_info[i]["box"][key][1])
            end_point=(list_objects_info[i]["box"][key][2],list_objects_info[i]["box"][key][3])
            back_ground=cv2.rectangle(back_ground, start_point, end_point, (0,255,0), 5)
            back_ground = cv2.putText(back_ground, key, start_point, cv2.FONT_HERSHEY_SIMPLEX, 
							2, (0,0,255), 5, cv2.LINE_AA)
    cv2.imwrite(f"backgroundcamera{video}.jpg",back_ground)





    # print("paths0",paths[0])
    # print("index",paths.index("data_revamp/IMG_5341/images/IMG_5341_541.jpg"))


    # print("matrixs",matrixs["key_image"])
    # print("matrixs",matrixs["key_image"][52:60])
    # print("matrixs",matrixs["key_image"].index("IMG_5341_568_IMG_5341_572"))
    # matrix_test=matrixs["matrix"][52]
    # for i in range(53,60):
    #     matrix_test=matrix_test@matrixs["matrix"][i]
    # print("matrix_test",matrix_test)
   

    
    
    
    
    

    

    

        
    
