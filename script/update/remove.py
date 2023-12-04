import os
import numpy as np
import json
from shutil import copyfile
from tqdm import tqdm

NPY_FOLDER = '/home/ubuntu/efs/data_backup/npy/'
MASTER_IMG_LST = '/home/ubuntu/Lashinbang-server/data/INDEX/cluster_0_paths.txt'
NOTFOUND_IMG_LIST = 'data/rotate_delete.txt'

def remove_file(files):
    # find in list
    for i in tqdm(range(len(files))):
        if ('_S' in files[i]):
            continue

        # find idx and block_idx
        file_id = master_data_paths.index(files[i])
        if file_id < 0 or file_id > len(master_data_paths):
            print(f'file {files[i]} (id {file_id}) not exists!')
            continue

        block_idx = file_id // block_size
        idx = file_id - (block_idx * block_size)

        # set path is null
        master_data_paths[file_id] = f'deleted({files[i]})'

        if str(block_idx) in arr_deleted:
            arr_deleted[str(block_idx)].append(idx)
        else:
            arr_deleted[str(block_idx)] = [idx]

    with open(f'cluster_0_paths.txt', 'w') as f:
        for item in master_data_paths:
            f.write("%s\n" % item)

    dump_dict()

def dump_dict():
    with open('app.json', 'w') as fp:
        json.dump(arr_deleted, fp)

def load_dict():
    global arr_deleted
    with open('app.json') as fp:
        arr_deleted = json.load(fp)

def reverse_lookup_search(a, b):
    reverse_lookup = {x:i for i, x in enumerate(b)}
    for i, x in enumerate(tqdm(a)):
        file_id = reverse_lookup.get(x, -1)
        if file_id == -1:
            print('Error: i: %s - x: %s' % (i, x))
        else:
            if 'deleted' not in master_data_paths[file_id]:
                # prepare deleted list
                block_idx = file_id // block_size
                at_id = file_id - (block_idx * block_size)

                # set deleted path
                master_data_paths[file_id] = f'deleted({master_data_paths[file_id]})'

                if str(block_idx) in arr_deleted:
                    arr_deleted[str(block_idx)].append(at_id)
                else:
                    arr_deleted[str(block_idx)] = [at_id]

def remove_feature():
    for block_idx in arr_deleted:
        block_name = os.path.join(NPY_FOLDER, 'block_{0}.npy'.format(block_idx))
        block = np.load(block_name)
        print(block_name)
        for i in tqdm(range(len(arr_deleted[block_idx]))):
            idx = arr_deleted[block_idx][i]
            block[idx] = np.zeros(2048)
        np.save(block_name, block)

if __name__ == "__main__":
    block_size = 10000
    arr_deleted = {}

    with open(MASTER_IMG_LST, 'r') as f:
        master_data_paths = f.readlines()
    master_data_paths = list(map(lambda x: x[:-1], master_data_paths))
    # short_master_paths = [item.split('/')[-1].split(')')[0] for item in master_data_paths]

    with open(NOTFOUND_IMG_LIST, 'r') as f:
        notfound_data_paths = f.readlines()
    notfound_data_paths = list(map(lambda x: x[:-1], notfound_data_paths))
    # short_notfound_data_paths = [item.split('/')[-1].split(')')[0] for item in notfound_data_paths]

    reverse_lookup_search(notfound_data_paths, master_data_paths)

    remove_feature()

    if os.path.isfile('cluster_0_paths.txt'):
        os.remove('cluster_0_paths.txt')
    with open(f'cluster_0_paths.txt', 'w') as f:
        for item in master_data_paths:
            f.write("%s\n" % item)
