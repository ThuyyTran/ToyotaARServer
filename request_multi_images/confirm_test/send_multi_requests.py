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

import requests
import urllib.request
import shutil

k_server_url              = 'http://192.168.1.190/predict'
k_file_txt                = "data/csv/report_190_5000_qa.csv"

k_file_list_origin_images = "data/txt/list_origin_images.txt"
k_image_request_path      = "data/test_images_request_qa/"

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
  img_name = string.split('/')[-1]
  img_id = img_name.split('_')[0]
  return img_id


def get_origin_image_id_array():
  file1 = open(k_file_list_origin_images,"r")
  # print(file1.readlines())
  origin_image_array = file1.readlines()
  origin_image_name_array = [x.split(' ') for x in origin_image_array]
  # print(origin_image_array)
  # print(origin_image_id_array)
  origin_image_name_array = [[image_id(item) for item in nested] for nested in origin_image_name_array]
  file1.close()
  return origin_image_name_array

def send_request_image(request_image_path, image_id_array, image_name):
  t1 = time.time()
  # save request image
  # request_image_path = get_file_path_and_save(request)

  url_list = [k_server_url]
  response_list = []
  json_response = {}
  topn = []

  # two threads
  # processes = []
  # with ThreadPoolExecutor(max_workers=1) as executor:
  #     for url in url_list:
  #         processes.append(executor.submit(request_server, url, request_image_path))

  # for task in as_completed(processes):
  #     print(task.result())
  #     response_list.append(task.result())

  # one thread
  for url in url_list:
    files = {'file': open(os.path.join(request_image_path), 'rb')}
    payload_form = {'app_code': app_code}
    response = requests.post(url, files=files, data=payload_form)
    response_list.append(response)

  # top_match_files = 5
  for response in response_list:
    print(response.json())
    if 'status' not in response.json():
      continue
    if response.json()['status'] == 0:
      matched_files = response.json()['matched_files']
      for matched_file in matched_files:
        topn.append(matched_file)


    # top_match_files = response.json()['top']

  # print("topn = ", topn)

  f = lambda x: os.path.splitext(os.path.basename(str(x)))[0].split('_')[0]
  response_has_image_id_array = list(filter(lambda item: f(item['image']) == str(image_id_array[0]), topn))

  topn_by_id = list(map(lambda x: f(x['image']), topn))

  true_response_index = -1
  score = 0
  for image_id in image_id_array:
    try:
      true_response_index = topn_by_id.index(str(image_id))
      break
    except:
      # save_result_image(request_image_path, image_name, topn)
      print("No true response id")

  # if true_response_index > -1:
  #   score = response_has_image_id_array[0]['Score']
  #   print("Score = ", response_has_image_id_array[0]['Score'])
  # print(response_has_image_id_array)
  print("true_response_index = ", true_response_index)

  # topn = sorted(topn, key = lambda i: i['Score'],reverse=True)
  # removeDuplicateID(topn)

  # if topn:
  #     json_response["matched_files"] = topn
  #     json_response["status"] = 0
  #     json_response["message"] = "successful"
  # else:
  #   json_response["status"] = 1
  #   json_response["message"] = "No matches found"

  # result = json.dumps(json_response, ensure_ascii=False).encode("utf8")
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
  for i in range(1,201):
    image_name = str(i) + ".jpg"
    image_request_full_path = image_request_path + image_name
    index_image_id = int((i-1)/2)
    image_id_array = origin_image_id_array[index_image_id]
    print(image_request_full_path + ": " + image_id_array[0])

    file1 = open(k_file_txt,"a+")
    L = [str(image_id_array[0])+',',str(image_name)+',']
    file1.writelines(L)
    file1.close() #to change file access modes

    send_request_image(image_request_full_path, image_id_array, str(i))

# clean csv folder
for filename in os.listdir("data/csv/"):
  file_path = os.path.join("data/csv/", filename)
  try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
      elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
  except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
  if len(sys.argv) > 1:
    app_code = sys.argv[-1]
  else:
    app_code = 'lashinbang'

  send_request_multi_images()
  print("DONE")
