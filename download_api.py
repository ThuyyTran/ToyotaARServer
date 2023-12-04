import json
import os
import requests
import time
import csv
import random
import numpy as np
from PIL import Image
import shutil
from pathlib import Path
import base64
import io
import sys
import re
from io import BytesIO
import time 
def getI420FromBase64(codec):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def download_api(images_list ,base_save_folder, app_code="lashinbang", url_name="https://test-la.anlab.info/api/get-raw-images"):
    Path(base_save_folder).mkdir(parents=True, exist_ok=True)
    _dict =  [('type', app_code)]
    for i , c in enumerate(images_list):
        _dict.append(('images[]', c))
    response = requests.post(url_name, data=tuple(_dict), timeout=10000)
    r = response.json()
    list_data = list(r["data"].keys())
    for i , l in enumerate(list_data):
        img_base64_str = str(r["data"][l])
        img = getI420FromBase64(img_base64_str)
        dir_name = os.path.dirname(l)
        sub_folder = os.path.join(base_save_folder , dir_name)
        Path(sub_folder).mkdir(parents=True, exist_ok=True)
        file_out =  os.path.join(base_save_folder , l)
        img.save(file_out)

    # response = requests.post(url_name, data=(('type', app_code), ('images[]', images_list[0]), ('images[]', images_list[1])), timeout=10000)
    # print("response " , response.json())
    # for i , c in enumerate(images_list):
    #     try:
    #         payload_form = { "type": app_code, "images[]":images_list }
    #         req_time = response.elapsed.total_seconds()
    #         r = response.json()
    #         list_data = list(r["data"].keys())
    #         # print("list_data " , list_data)
    #         img_base64_str = str(r["data"][list_data[0]])
    #         img = getI420FromBase64(img_base64_str)
    #     except:
    #         print("erros " ,i,  c)
    #         img = None
    

def download_images(paths ,url_name, app_code="lashinbang"):
    # Path(base_save_folder).mkdir(parents=True, exist_ok=True)
    _dict =  [('type', app_code), ('count', len(paths))]
#    for i , c in enumerate(paths):
#        _dict.append(('images[]', c))
    for i, c in enumerate(paths):
        key = f'image{i+1}'
        _dict.append((key, c))
    response = requests.post(url_name, data=tuple(_dict), timeout=10000)
    r = response.json()
    list_data = list(r["data"].keys())
    images = {}
    for i , l in enumerate(list_data):
        img_base64_str = str(r["data"][l])
        img = getI420FromBase64(img_base64_str)
        if img is not None:
            images[paths[i]] = img
        else:
            images[paths[i]] = None
        # dir_name = os.path.dirname(l)
        # sub_folder = os.path.join(base_save_folder , dir_name)
        # Path(sub_folder).mkdir(parents=True, exist_ok=True)
        # file_out =  os.path.join(base_save_folder , l)
        # img.save(file_out)

    return images
# app_code = "cosmetic-toppan"
# images_list = ["d03/DBE_BR-10_open.jpg", "d01/TEC_039_close.jpg"]
# # base_save_folder = "db_test"
# # download_images(images_list, base_save_folder ,app_code)
# download_api(images_list, app_code= app_code)
