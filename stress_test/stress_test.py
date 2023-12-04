# USAGE
# python stress_test.py
# import the necessary packages
import os
from threading import Thread
import requests
import time
import csv
import random
import uuid

import numpy as np
from PIL import Image

# initialize the Keras REST API endpoint URL along with the input
KERAS_REST_API_URL = "http://192.168.1.190:5001/predict"
#KERAS_REST_API_URL = "http://192.168.1.109:5000/predict"
#KERAS_REST_API_URL = "http://52.194.200.43/predict"

#PATH = '/media/anlab/ssd_samsung_256/Lashinbang-server/stress_test/query_images/'
#PATH = 'query_images_20210208/2021-01-22'
PATH = '/home/anlabadmin/Documents/Lashinbang/20210323/query_images'
if 1:
    IMAGE_PATH = os.listdir(PATH)
    IMAGE_PATH = [os.path.join(PATH, x) for x in IMAGE_PATH]
#    PATH = 'query_images_20210208/2021-01-25'
#    IMAGE_PATH1 = os.listdir(PATH)
#    IMAGE_PATH1 = [os.path.join(PATH, x) for x in IMAGE_PATH1]
#    IMAGE_PATH.extend(IMAGE_PATH1)
else:
    testFolders = ['000', '100']
    testFolders = ['200', '300', '400']
    listOfFiles = []
    for folder in testFolders:
        dirName = os.path.join('/home/anlab/projects/lashinbang/src/cnnimageretrieval-pytorch/lashinbang/data/train/', folder)
        for (dirpath, dirnames, filenames) in os.walk(dirName):
            found_expected_result = False
            for file in filenames:
                if len(file) < 20:
                    found_expected_result = True
                    break
            if found_expected_result:
                for file in filenames:
                    if len(file) < 20:
                        continue
                    listOfFiles.append(os.path.join(dirpath, file))

    IMAGE_PATH = listOfFiles

print("#files : ", len(IMAGE_PATH))
print(IMAGE_PATH[:20])

#PATH = '/media/anlab/DATA_Backup_XLA/Backup/Lashinbang_data/Images_20200720/'
#with open('/media/anlab/ssd_samsung_256/Lashinbang-server/script/update_image_20200721/updated_list.txt', 'r') as f:
#    IMAGE_PATH = f.readlines()
#IMAGE_PATH = list(map(lambda x: x[:-1], IMAGE_PATH))

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = len(IMAGE_PATH)
print(f'Total samples: {NUM_REQUESTS}')
REQ_PER_SEC = 0.75
SLEEP_COUNT = 1 / REQ_PER_SEC
APP_CODE = "kbook"

row_list = []
IS_CROP = False

def get_info(n):
    path_0 = IMAGE_PATH[n]

    req = requests.get(f'http://192.168.1.88/api/products?image={path_0.split("/")[-1]}')
    res = str(req.json()["data"]["images"])

    print(f'{path_0}, {res}, {path_0 in res}')

def call_predict_endpoint(n):
    t1 = time.time()
    # load the input image and construct the payload for the request
    path_0 = IMAGE_PATH[n]
    path = path_0
#    path = os.path.join(PATH, path_0)

    if IS_CROP:
        im = Image.open(path)
        width, height = im.size
        new_width = min(width, height)
        new_height = new_width
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        # Crop the center of the image
        im = im.crop((left, top, right, bottom))
        im.save(f"tmp/{path_0}", "JPEG")
        path = os.path.join("tmp", f"{path_0}")

    #image = np.array(im)
    image = open(path, "rb").read()
    payload_files = { "file": image }
    payload_form = { "shopid": "stress_test", "app_code": APP_CODE }

    response = requests.post(KERAS_REST_API_URL, files=payload_files, data=payload_form)
    req_time = response.elapsed.total_seconds()
    r = response.json()
    msg = str(r["message"])
    rst = ""
    if (msg == "successful"):
        arr = r["matched_files"]
        data = []
        for item in arr:
            data.append(item["image"])
        _str = '|'.join([str(elem) for elem in data])
        rst = f"{n},{APP_CODE},{path_0}, {_str}, {(time.time() - t1)}"
        if not path_0 in _str:
            arr_false.append(rst)
    else:
        rst = f"{n},{APP_CODE},{path_0}, {msg}, {(time.time() - t1)}"
    print(rst)

    with open('result.csv', 'a') as f:
        f.write(f'{rst}\n')

    # if r["message"] == "successful":
    #     print("--- %s - %s seconds --- %s --- path: %s -- result: %s" % (n, req_time, (time.time() - t1), path, 'success'))
    #     # print(req_time)
    # else:
    #     # print("--ERROR--")
    #     print("--- %s - %s seconds --- %s --- path: %s -- error: %s" % (n, req_time, (time.time() - t1), path, str(r["message"])))

if os.path.exists('result.csv'):
    os.remove('result.csv')

if os.path.exists('false.csv'):
    os.remove('false.csv')

arr_false = []

# loop over the number of threads
for i in range(NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    # t = Thread(target=get_info, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

with open('false.csv', 'a') as f:
    for item in arr_false:
        f.write(f'{item}\n')

time.sleep(600)
