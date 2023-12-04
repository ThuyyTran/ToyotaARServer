# Get report when send 200 images to server
# Input:
# 1) List_origin_image.txt: all images name
# 2) test_images_request: folder image request
# 3) Url server
# Output:
# 1) report_port_6001.txt: response data

import os, sys
from io import BytesIO
from six.moves import urllib

import numpy as np
from PIL import Image
import cv2
import time
from shutil import copyfile
from glob import glob
from datetime import datetime

import requests
import urllib.request
import shutil

k_server_url              = 'http://192.168.1.190:5000/predict'

def save_result_image(request_image_path, image_name, topn):
    # f = request.files["file"]
    img = cv2.imread(request_image_path)
    img_mat = np.array(img)
    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    unique_name = image_name + ".jpg"
    file_path = os.path.join(basepath, 'data/test_images_response/', unique_name)
    cv2.imwrite(file_path, img_mat.astype(np.uint8))

    index_response = 0
    for matched_file in topn:
      url_image = matched_file["image"]
      print(url_image)
      file_name_response = image_name + '-' + str(index_response) + '.jpg'
      file_path_response = os.path.join(basepath, 'data/test_images_response/', file_name_response)
      urllib.request.urlretrieve(url_image, file_path_response)
      index_response = index_response + 1

    return file_path

def image_id(string):
  img_path = string.split(',')[-1]
  img_dir = os.path.dirname(img_path)
  img_name = os.path.basename(img_path)
  img_id = img_name.split('_')[0]
  return os.path.join(img_dir, img_id)

def get_origin_image_id_array():
  with open(k_file_list_origin_images,"r") as f:
    origin_image_array = f.readlines()
    origin_image_name_array = [x.split(' ') for x in origin_image_array]
    origin_image_name_array = [[image_id(item) for item in nested] for nested in origin_image_name_array]

  return origin_image_name_array

def send_request_image(request_image_path, image_id_array, image_name):
  t1 = time.time()

  url_list = [k_server_url]
  response_list = []
  json_response = {}
  topn = []

  # one thread
  for url in url_list:
    files = {'file': open(os.path.join(request_image_path), 'rb')}
    payload_form = {'app_code': app_code}
    response = requests.post(url, files=files, data=payload_form)
    response_list.append(response)

  # top_match_files = 5
  for response in response_list:
    if 'status' not in response.json():
      continue

    if response.json()['status'] == 0:
      matched_files = response.json()['matched_files']
      for matched_file in matched_files:
        topn.append(matched_file)

  f = lambda x: os.path.join(os.path.dirname(x), os.path.basename(x).split('_')[0])
  topn_by_id = list(map(lambda x: f(x['image']), topn))

  true_response_index = -1
  score = 0
  try:
    print(topn_by_id)
    true_response_index = topn_by_id.index(image_id_array)
  except:
      print("No true response id")

  print("true_response_index = ", true_response_index)
  print('time = ', time.time() - t1)

  file1 = open(k_file_txt,"a+")
  L = ['"'+str(topn)+'",',str(true_response_index)+',',str(time.time() - t1)+'\n']
  file1.writelines(L)
  file1.close() #to change file access modes
  print("--------------------")
  return topn

def send_request_multi_images():
  origin_image_id_array = get_origin_image_id_array()
  image_request_path = k_image_request_path
  for i in range(1, samples_number + 1):
    image_name = str(i) + ".jpg"
    image_request_full_path = os.path.join(image_request_path, image_name)
    index_image_id = (i - 1) // 2
    image_id_array = origin_image_id_array[index_image_id]
    print(image_request_full_path + ": " + image_id_array[0])

    with open(k_file_txt, "a+") as f:
      tmp = [str(image_id_array[0])+',',str(image_name)+',']
      f.writelines(tmp)
      f.close() #to change file access modes

    send_request_image(image_request_full_path, image_id_array[0], str(i))

if __name__ == '__main__':
  if len(sys.argv) > 1:
      app_code = sys.argv[1]
      job_id = sys.argv[2]
  else:
      app_code = 'lashinbang'
      job_id = 0

  k_file_list_origin_images = f"data/txt/list_origin_images_{str(job_id)}.txt"
  k_image_request_path      = os.path.join("data/test_images_request_qa/", str(job_id))
  k_file_txt                = f"data/csv/{str(job_id)}_report_qa.csv"

  if not os.path.isdir(k_image_request_path):
    print(f"{k_image_request_path} not exists")
    sys.exit(0)

  if not os.path.isfile(k_file_list_origin_images):
    print(f"{k_file_list_origin_images} not exists")
    sys.exit(0)

  # send request
  samples_number = len(os.listdir(k_image_request_path))
  send_request_multi_images()

  # clear sample
  os.remove(k_file_list_origin_images)
  shutil.rmtree(k_image_request_path)

  print("DONE")
