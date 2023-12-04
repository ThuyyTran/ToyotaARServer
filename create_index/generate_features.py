import sys
sys.path.insert(1, '../')

import os
import numpy as np
from tqdm import tqdm

from extract_cnn import CNN


if __name__ == '__main__':
	model = CNN(useRmac=True)
	block_size = 10000

	with open('../data/all_images.out', 'r') as f:
		master_data_paths = f.readlines()
	master_data_paths = list(map(lambda x: '/media/anlabadmin/big_volume/' + x, master_data_paths))
	master_data_paths = list(map(lambda x: x[:-1], master_data_paths))
	print(f'Number of images: {len(master_data_paths)}')

	for i in tqdm(range(len(master_data_paths) // block_size + 1)):
		print((f'../data/npy/block_{i}.npy'))
		if os.path.exists(f'../data/npy/block_{i}.npy'):
			continue
		try:
			master_data_path_current = master_data_paths[i*block_size: (i+1)*block_size]
			feature_current = model.extract_feat_batch(master_data_path_current)
			np.save('../data/npy/block_%d.npy' % i, feature_current)
		except Exception as e:
			print(e)