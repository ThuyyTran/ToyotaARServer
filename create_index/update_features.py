import sys
sys.path.insert(1, '../')

import os
import numpy as np
import time
import settings
import pathlib
import cv2

from datetime import datetime
from tqdm import tqdm
from extract_cnn import CNN
from pathlib import Path
from create_index.create_descs_db import save_features

# path_upload = "/media/anlabadmin/big_volume/Lashinbang_data/Images_20200326"

def update_paths(name, source):
    for f in source:
        with open(name, 'a') as fw:
            f = f.replace('/media/anlabadmin/big_volume/', '')
            fw.write("%s\n" % f)

def create_desc_db(files):
    for i in tqdm(range(len(files))):
        f = files[i]
        f = f.replace("/media/anlabadmin/big_volume/Lashinbang_data/", '')

        # create sub path
        sub_path = os.path.dirname(f)
        sub_path = os.path.join(settings.MTC_DESCS_DB_FOLDER, sub_path)
        pathlib.Path(sub_path).mkdir(parents=True, exist_ok=True)

        input_file = os.path.join(settings.MTC_IMAGE_DB_FOLDER, f)
        output_file = os.path.join(settings.MTC_DESCS_DB_FOLDER, f + ".pkl")

        if os.path.exists(output_file):
            n = pathlib.Path(output_file).stat().st_size
            if n > 0:
                continue

        im = cv2.imread(input_file)
        if im is None:
            print('Cannot read file "%s"' % (input_file))
            break
        save_features(im, output_file)

def update_features():
    master_paths = []

    # write all_update_image.out
    files = [val for sublist in [[os.path.join(i[0], j) for j in i[2] if '_S' not in j] for i in os.walk(path_upload)] for val in sublist]
    master_paths.extend(files)

    # update cluster_0_paths.txt
    update_paths('../data/all_images.out', master_paths)

    # # generate desc db
    print("create descs db")
    create_desc_db(master_paths)

    # update index
    num_blocks = len(os.listdir('../data/npy'))
    last_block_name = '../data/npy/block_{0}.npy'.format(num_blocks - 1)
    last_block = np.load(last_block_name)
    block_offset = block_size - len(last_block)

    # fill to last block
    append_list = master_paths[:block_offset]
    features_offset = model.extract_feat_batch(append_list)
    data = np.concatenate((last_block, features_offset))
    np.save(last_block_name, data)

    # generate new block
    generate_list = master_paths[block_offset:]
    for i in range(len(generate_list) // block_size + 1):
        try:
            master_data_path_current = generate_list[i * block_size: (i + 1) * block_size]
            feature_current = model.extract_feat_batch(master_data_path_current)
            np.save('../data/npy/block_%d.npy' % num_blocks, feature_current)
            num_blocks += 1
        except Exception as e:
            print(e)

if __name__ == '__main__':
    model = CNN()
    block_size = 10000

    if len(sys.argv) <= 1:
        print('error: python update_features.py <path_to_folder>')
        exit()

    path_upload = sys.argv[1]
    if not os.path.isdir(path_upload):
        print('path update folder does not exists!')
        exit()

    print('Start update features')
    t1 = time.time()
    update_features()
    print('success in %s' % (time.time() - t1))
