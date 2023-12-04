import sys
sys.path.insert(1, '../../')

import os
import numpy as np
import time
import pathlib
import cv2

from datetime import datetime
from tqdm import tqdm
from extract_cnn import CNN
from pathlib import Path
from create_index.create_descs_db import save_features

ADD_LIST = "data/update_images.txt"
MASTER_LIST = "/home/ubuntu/efs/data_backup/data/INDEX/cluster_0_paths.txt"
NPY_DIR = "/home/ubuntu/efs/data_backup/npy/"
IMAGE_DIR = "/home/ubuntu/efs/lashinbang-images/"
DESCS_DIR = "/home/ubuntu/efs/descs/"

def update_paths(name, source):
    for f in source:
        with open(name, 'a') as fw:
            fw.write("%s\n" % f)

def create_desc_db(files):
    for i in tqdm(range(len(files))):
        f = files[i]
        f = f.replace("/media/anlabadmin/big_volume/Lashinbang_data/", '')

        # create sub path
        sub_path = os.path.dirname(f)
        sub_path = os.path.join(DESCS_DIR, sub_path)
        pathlib.Path(sub_path).mkdir(parents=True, exist_ok=True)

        input_file = os.path.join(IMAGE_DIR, f)
        output_file = os.path.join(DESCS_DIR, f + ".pkl")

        if os.path.exists(output_file):
            n = pathlib.Path(output_file).stat().st_size
            if n > 0:
                continue

        im = cv2.imread(input_file)
        if im is None:
            print('Cannot read file "%s"' % (input_file))
            continue
        save_features(im, output_file)

def update_features():
    # update index
    num_blocks = len(os.listdir(NPY_DIR))
    last_block_name = os.path.join(NPY_DIR, f'block_{num_blocks - 1}.npy')
    last_block = np.load(last_block_name)
    block_offset = block_size - len(last_block)

    # fill to last block
    append_list = add_paths[:block_offset]
    append_list = [os.path.join(IMAGE_DIR, item) for item in append_list]
    if len(append_list) > 0:
        features_offset = model.extract_feat_batch(append_list)
        data = np.concatenate((last_block, features_offset))
        np.save(last_block_name, data)

    # generate new block
    generate_list = add_paths[block_offset:]
    generate_list = [os.path.join(IMAGE_DIR, item) for item in generate_list]
    if len(generate_list) > 0:
        for i in range(len(generate_list) // block_size + 1):
            try:
                master_data_path_current = generate_list[i * block_size: (i + 1) * block_size]
                feature_current = model.extract_feat_batch(master_data_path_current)
                np.save(os.path.join(NPY_DIR, f'block_{num_blocks}.npy'), feature_current)
                num_blocks += 1
            except Exception as e:
                print(e)

if __name__ == '__main__':
    model = CNN(useRmac=True)
    block_size = 10000

    # if len(sys.argv) <= 1:
    #     print('error: python update_features.py <path_to_folder>')
    #     exit()

    # path_upload = sys.argv[1]
    # if not os.path.isdir(path_upload):
    #     print('path update folder does not exists!')
    #     exit()

    with open(MASTER_LIST, 'r') as f:
        master_paths = f.readlines()
    master_paths = list(map(lambda x: x[:-1], master_paths))

    with open(ADD_LIST, 'r') as f:
        add_paths = f.readlines()
    add_paths = list(map(lambda x: x[:-1], add_paths))

    print('Start update features')
    t1 = time.time()

    # update cluster_0_paths.txt
    print("update paths")
    update_paths(MASTER_LIST, add_paths)

    # # # generate desc db
    # print("create descs db")
    # create_desc_db(add_paths)

    # print("update features")
    # update_features()

    print('success in %s' % (time.time() - t1))
