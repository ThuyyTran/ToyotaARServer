import os, sys
sys.path.insert(1, '../')

import numpy as np
from tqdm import tqdm
from random import shuffle
import faiss
import time

import settings

block_size = 10000
x_train = []
num_block = len(os.listdir('../data/npy'))
num_block_training = 200
d = settings.CNN_IMAGE_FEATURE_FULL_SIZE
nlist = 1024

ls = list(range(num_block))
shuffle(ls)

x_train = np.zeros([num_block_training*block_size, d], dtype=np.float32)
c = 0
for i in tqdm(ls):
	block = np.load(f'../data/npy/block_{i}.npy')
	# x_train.append(block)
	if block.shape[0] == block_size:
		x_train[c * block_size: (c+1) * block_size] = block
		c += 1
	if c == num_block_training:
		break

x_train = np.ascontiguousarray(x_train)

if settings.CNN_IMAGE_FEATURE_USING_PCA:
    pca_n_components = settings.CNN_IMAGE_FEATURE_REDUCED_SIZE
    quantizer = faiss.IndexFlatIP(pca_n_components)
    sub_index = faiss.IndexIVFFlat(quantizer, pca_n_components, nlist, faiss.METRIC_L2)
    pca_matrix = faiss.PCAMatrix(d, pca_n_components, 0, True)
    index = faiss.IndexPreTransform(pca_matrix, sub_index)
else:
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

t1 = time.time()
print(f'Index is trained: {index.is_trained}')
index.train(x_train)
print(f'Index is trained: {index.is_trained}')
print(f'Training time: {time.time() - t1}')

if settings.CNN_IMAGE_FEATURE_USING_PCA:
    faiss.write_index(sub_index, 'data/trained.index')
    faiss.write_VectorTransform(pca_matrix, "data/PCA.pca")
else:
    faiss.write_index(index, '../data/trained.index')
