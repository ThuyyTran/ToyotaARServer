import os, sys
sys.path.insert(1, '../')
import faiss
import numpy as np
import time
from tqdm import tqdm

import settings

nlist = 1000
d = 2048
num_cluster = 1
num_block = len(os.listdir('../data/npy'))
block_size = 10000

with open('../data/all_images.out', 'r') as f:
	master_data_paths = f.readlines()

num_blocks_per_cluster = num_block // num_cluster
for cluster_id in range(num_cluster):
	block_start = cluster_id * num_blocks_per_cluster
	block_end = (cluster_id+1) * num_blocks_per_cluster

	if cluster_id == num_cluster - 1:
		block_start = cluster_id * num_blocks_per_cluster
		block_end = num_block

	cluster_paths = []
	if cluster_id == 1:
		cluster_paths = master_data_paths
	else:
		cluster_paths = master_data_paths[block_start * block_size: block_end * block_size]

	with open(f'../data/INDEX/cluster_{cluster_id}_paths.txt', 'w') as f:
		f.writelines(cluster_paths)

	sub_index = faiss.read_index('../data/trained.index')
	if settings.CNN_IMAGE_FEATURE_USING_PCA:
		pca_matrix = faiss.read_VectorTransform("data/PCA.pca")
		index = faiss.IndexPreTransform(pca_matrix, sub_index)
	else:
		index = sub_index

	id_start = 0
	for block_id, i in tqdm(enumerate(range(block_start, block_end))):
		arr = np.ascontiguousarray(np.load(f'../data/npy/block_{i}.npy'))
		id_end = id_start + arr.shape[0]
		ids = np.arange(id_start, id_end)
		if settings.CNN_IMAGE_FEATURE_FULL_SIZE > settings.CNN_IMAGE_FEATURE_REDUCED_SIZE:
			index.add(arr)
		else:
			index.add_with_ids(arr, ids)
		id_start = id_end

	faiss.write_index(sub_index, f'../data/INDEX/cluster_{cluster_id}.index')

