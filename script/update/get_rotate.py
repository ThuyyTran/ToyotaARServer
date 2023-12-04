import os
import numpy as np
import json
from shutil import copyfile
from tqdm import tqdm

MASTER_IMG_LST = '/home/ubuntu/Lashinbang-server/data/INDEX/cluster_0_paths.txt'
NOTFOUND_IMG_LIST = 'data/replace_images.txt'

def reverse_lookup_search(a, b):
    reverse_lookup = {x:i for i, x in enumerate(b)}
    for i, x in enumerate(tqdm(a)):
        x_rotate_name = '{}r{}'.format(os.path.splitext(os.path.basename(x))[0], os.path.splitext(os.path.basename(x))[1])
        x_rotate_path = os.path.join('rotate/', os.path.dirname(x), x_rotate_name)
        file_id = reverse_lookup.get(x_rotate_path, -1)
        if file_id == -1:
            print('Error: i: %s - x: %s' % (i, x))
        else:
            if 'deleted' not in master_data_paths[file_id]:
                # prepare deleted list
                block_idx = file_id // block_size
                at_id = file_id - (block_idx * block_size)

                # set deleted path
                arr_deleted.append(master_data_paths[file_id])
                master_data_paths[file_id] = f'deleted({master_data_paths[file_id]})'

if __name__ == "__main__":
    block_size = 10000
    arr_deleted = []

    with open(MASTER_IMG_LST, 'r') as f:
        master_data_paths = f.readlines()
    master_data_paths = list(map(lambda x: x[:-1], master_data_paths))

    with open(NOTFOUND_IMG_LIST, 'r') as f:
        notfound_data_paths = f.readlines()
    notfound_data_paths = list(map(lambda x: x[:-1], notfound_data_paths))

    reverse_lookup_search(notfound_data_paths, master_data_paths)

    with open('data/rotate_delete.txt', 'w') as f:
        for delete in arr_deleted:
            f.write(f'{delete}\n')