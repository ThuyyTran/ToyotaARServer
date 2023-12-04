import sys, os
sys.path.insert(1, '../')

import numpy as np
import settings
import faiss
import time
import pathlib
import cv2

from tqdm import tqdm
from extract_cnn import CNN
from random import shuffle
from configparser import ConfigParser
from create_descs_db import save_features

class GenerateIndex:
    def __init__(self,
        npy_folder_path=None,
        master_paths_file=None,
        index_folder_path=None,
        use_pca=False,
        use_matching=True,
        mtc_folder_images=None,
        use_pre_computed=False,
        mtc_folder_desc=None,
        use_rmac=True,
        use_solar=True,
        mtc_feature='akaze',
        mtc_use_homography=False,
        mtc_use_superglue=True
    ) -> None:
        self._npy_folder_path = npy_folder_path
        self._master_paths_file = master_paths_file
        self._index_folder_path = index_folder_path
        self._mtc_folder_images = mtc_folder_images
        self._use_matching = use_matching
        self._use_pca = use_pca
        self._use_pre_computed = use_pre_computed
        self._mtc_folder_desc = mtc_folder_desc
        self._mtc_feature = mtc_feature
        self._use_rmac = use_rmac
        self._use_solar = use_solar
        self._mtc_use_homography = mtc_use_homography
        self._mtc_use_superglue = mtc_use_superglue
        self.block_size = 10000

    #region Properties
    @property
    def npy_folder_path(self):
        return self._npy_folder_path

    @npy_folder_path.setter
    def npy_folder_path(self, new_path):
        if (type(new_path) == str):
            self._npy_folder_path = new_path
        else:
            raise TypeError('npy_folder_path need str.')

    @property
    def master_paths_file(self):
        return self._master_paths_file

    @master_paths_file.setter
    def master_paths_file(self, new_path_file):
        if (type(new_path_file) == str):
            self._master_paths_file = new_path_file
        else:
            raise TypeError('master_paths_file need str.')

    @property
    def index_folder_path(self):
        return self._index_folder_path

    @index_folder_path.setter
    def index_folder_path(self, new_path):
        if (type(new_path) == str):
            self._index_folder_path = new_path
        else:
            raise TypeError('index_folder_path need str.')

    @property
    def use_matching(self):
        return self._use_matching

    @use_matching.setter
    def use_matching(self, val):
        if (type(val) == bool):
            self._use_matching = val
        else:
            raise TypeError('use_matching need bool.')

    @property
    def use_pca(self):
        return self._use_pca

    @use_pca.setter
    def use_pca(self, val):
        if (type(val) == bool):
            self._use_pca = val
        else:
            raise TypeError('use_pca need bool.')

    @property
    def mtc_folder_images(self):
        return self._mtc_folder_images

    @mtc_folder_images.setter
    def mtc_folder_images(self, new_path):
        if (type(new_path) == str):
            self._mtc_folder_images = new_path
        else:
            raise TypeError('mtc_folder_images need string.')

    @property
    def use_pre_computed(self):
        return self._use_pre_computed

    @use_pre_computed.setter
    def use_pre_computed(self, val):
        if (type(val) == bool):
            self._use_pre_computed = val
        else:
            return TypeError('use_pre_computed need bool.')

    @property
    def mtc_folder_desc(self):
        return self._mtc_folder_desc

    @mtc_folder_desc.setter
    def mtc_folder_desc(self, new_path):
        if (type(new_path) == str):
            self._mtc_folder_desc = new_path
        else:
            return TypeError('mtc_folder_desc need str.')

    @property
    def mtc_feature(self):
        return self._mtc_feature

    @mtc_feature.setter
    def mtc_feature(self, val):
        if (type(val) == str):
            self._mtc_feature = val
        else:
            return TypeError('mtc_feature need str.')

    @property
    def use_rmac(self):
        return self._use_rmac

    @use_rmac.setter
    def use_rmac(self, val):
        if (type(val) == bool):
            self._use_rmac = val
        else:
            return TypeError('use_rmac need bool.')

    @property
    def use_solar(self):
        return self._use_solar

    @use_solar.setter
    def use_solar(self, val):
        if (type(val) == bool):
            self._use_solar = val
        else:
            return TypeError('use_solar need bool.')

    @property
        # mtc_use_superglue=True
    def mtc_use_homography(self):
        return self._mtc_use_homography

    @mtc_use_homography.setter
    def mtc_use_homography(self, val):
        if (type(val) == bool):
            self._mtc_use_homography = val
        else:
            return TypeError('mtc_use_homography need bool.')

    @property
    def mtc_use_superglue(self):
        return self._mtc_use_superglue

    @mtc_use_superglue.setter
    def mtc_use_superglue(self, val):
        if (type(val) == bool):
            self._mtc_use_superglue = val
        else:
            return TypeError('mtc_use_superglue need bool.')
    #endregion

    #region Methods
    """
    Create needed folder
    """
    def prepare_generate(self):
        if self._npy_folder_path: pathlib.Path(self._npy_folder_path).mkdir(parents=True, exist_ok=True)
        if self._mtc_folder_desc: pathlib.Path(self._mtc_folder_desc).mkdir(parents=True, exist_ok=True)
        if self._index_folder_path: pathlib.Path(os.path.join(self._index_folder_path, 'INDEX')).mkdir(parents=True, exist_ok=True)

    """
    Generate npy feature
    """
    def generate_feature(self):
        if not self._master_paths_file:
            raise ValueError('master path is not None')

        with open(self._master_paths_file, 'r') as f:
            raw_data = f.readlines()
        master_data = list(map(lambda x: os.path.join(self._mtc_folder_images, x), raw_data))
        master_data = list(map(lambda x: x[:-1], master_data))
        master_data = [item.split(',')[0] for item in master_data]
        print(f'Number of images: {len(master_data)}')

        for i in tqdm(range(len(master_data) // self.block_size + 1)):
            print((f'{self._npy_folder_path}/block_{i}.npy'))
            if os.path.exists(os.path.join(self._npy_folder_path, f'block_{i}.npy')):
                continue
            try:
                master_data_path_current = master_data[i * self.block_size: (i + 1) * self.block_size]
                feature_current = self.model.extract_feat_batch(master_data_path_current)
                np.save(f'{self._npy_folder_path}/block_{i}.npy', feature_current)
            except Exception as e:
                print(f'Generate feature error: {e}')

        return raw_data

    """
    Create pkl image if use pre computed
    """
    def create_desc_image(self, files):
        for i in tqdm(range(len(files))):
            f = files[i]

            # create sub path
            sub_path = os.path.dirname(f)
            sub_path = os.path.join(self._mtc_folder_desc, sub_path)
            pathlib.Path(sub_path).mkdir(parents=True, exist_ok=True)

            f = f.replace('Lashinbang_data/', '')
            input_file = os.path.join(self._mtc_folder_images, f)
            output_file = os.path.join(self._mtc_folder_desc, f + ".pkl")

            if os.path.exists(output_file):
                n = pathlib.Path(output_file).stat().st_size
                if n > 0:
                    continue

            im = cv2.imread(input_file)
            if im is None:
                print('Cannot read file "%s"' % (input_file))
                continue

            save_features(im, output_file)

    """
    Generate trained.index (and PCA)
    """
    def generate_training_data(self):
        x_train = []
        num_block = len(os.listdir(self._npy_folder_path))
        num_block_training = 200
        d = settings.CNN_IMAGE_FEATURE_FULL_SIZE
        nlist = 1024

        if num_block <= 0:
            raise ValueError('feature is null')

        ls = list(range(num_block))
        shuffle(ls)

        x_train = np.zeros([num_block_training * self.block_size, d], dtype=np.float32)
        c = 0
        for i in tqdm(ls):
            block = np.load(f'{self._npy_folder_path}/block_{i}.npy')
            if block.shape[0] == self.block_size:
                x_train[c * self.block_size: (c+1) * self.block_size] = block
                c += 1
            if c == num_block_training:
                break

        x_train = np.ascontiguousarray(x_train)

        if self.use_pca:
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

        if self.use_pca:
            faiss.write_index(sub_index, f'{self._index_folder_path}/trained.index')
            faiss.write_VectorTransform(pca_matrix, f"{self._index_folder_path}/PCA.pca")
        else:
            faiss.write_index(index, f'{self._index_folder_path}/trained.index')

    """
    Generate cluster_0.index & cluster_0_paths.txt
    """
    def generate_index(self):
        num_cluster = 1
        num_block = len(os.listdir(self._npy_folder_path))

        with open(self._master_paths_file, 'r') as f:
            master_data = f.readlines()

        num_blocks_per_cluster = num_block // num_cluster
        for cluster_id in range(num_cluster):
            block_start = cluster_id * num_blocks_per_cluster
            block_end = (cluster_id + 1) * num_blocks_per_cluster

        if cluster_id == num_cluster - 1:
            block_start = cluster_id * num_blocks_per_cluster
            block_end = num_block

        cluster_paths = []
        if cluster_id == 1:
            cluster_paths = master_data
        else:
            cluster_paths = master_data[block_start * self.block_size: block_end * self.block_size]

        with open(os.path.join(self._index_folder_path, 'INDEX', f'cluster_{cluster_id}_paths.txt'), 'w') as f:
            f.writelines(cluster_paths)

        sub_index = faiss.read_index(f'{self._index_folder_path}/trained.index')
        if self.use_pca:
            pca_matrix = faiss.read_VectorTransform(f"{self._index_folder_path}/PCA.pca")
            index = faiss.IndexPreTransform(pca_matrix, sub_index)
        else:
            index = sub_index

        id_start = 0
        for block_id, i in tqdm(enumerate(range(block_start, block_end))):
            arr = np.ascontiguousarray(np.load(f'{self._npy_folder_path}/block_{i}.npy'))
            id_end = id_start + arr.shape[0]
            ids = np.arange(id_start, id_end)
            if settings.CNN_IMAGE_FEATURE_FULL_SIZE > settings.CNN_IMAGE_FEATURE_REDUCED_SIZE:
                index.add(arr)
            else:
                index.add_with_ids(arr, ids)
            id_start = id_end

        faiss.write_index(sub_index, os.path.join(self._index_folder_path, 'INDEX', f'cluster_{cluster_id}.index'))

    """
    Generate config file
    """
    def generate_config(self):
        config = ConfigParser()
        config.optionxform = str
        filename = f'{self.index_folder_path}/config.ini'
        config.read(filename)
        config.add_section(settings.DATABASE_KEY)
        config.set(settings.DATABASE_KEY, settings.INDEX_FILE_KEY, 'INDEX/cluster_0.index')
        config.set(settings.DATABASE_KEY, settings.IMG_LIST_FILE_KEY, 'INDEX/cluster_0_paths.txt')
        config.set(settings.DATABASE_KEY, settings.PCA_MATRIX_FILE_KEY, 'PCA.pca')
        config.set(settings.DATABASE_KEY, settings.CNN_IMAGE_FEATURE_USING_PCA_KEY, str(self._use_pca))
        config.set(settings.DATABASE_KEY, settings.INDEX_NPROBE_KEY, str(64))
        config.set(settings.DATABASE_KEY, settings.MTC_IMAGE_DB_FOLDER_KEY, str(self._mtc_folder_images))
        config.set(settings.DATABASE_KEY, settings.MTC_DESCS_DB_FOLDER_KEY, str(self._mtc_folder_desc))
        config.set(settings.DATABASE_KEY, settings.MTC_PRE_COMPUTED_KEY, str(self._use_pre_computed))
        config.set(settings.DATABASE_KEY, settings.MTC_FEATURE_KEY, str(self._mtc_feature))
        config.set(settings.DATABASE_KEY, settings.MTC_IS_NEED_HOMOGRAPHY_KEY, str(False))
        config.set(settings.DATABASE_KEY, settings.MTC_USING_SUPERGLUE_KEY, str(True))
        config.set(settings.DATABASE_KEY, settings.MATHCHING_CONFIG, str(self._use_matching))
        config.set(settings.DATABASE_KEY, settings.DESC_MODE_CONFIG, 'solar' if self._use_solar else 'cirtorch')

        with open(filename, "w") as config_file:
            config.write(config_file)

    """
    Init CNN model
    """
    def init_model(self):
        self.model = CNN(useRmac=self.use_rmac, use_solar=self.use_solar)

    """
    Start generate
    """
    def generate(self):
        # prepare generate
        self.prepare_generate()

        # step 1 - generate features
        files = self.generate_feature()

        # step 2 - generate desc image
        if self._use_pre_computed:
            self.create_desc_image(files)

        # step 3 - generate training data
        self.generate_training_data()

        # step 4 - generate index
        self.generate_index()

        # step 5 - generate config
        self.generate_config()
    #endregion

def main():
    # Init generate object
    generate_index = GenerateIndex()

    # Input
    while not generate_index.npy_folder_path: generate_index.npy_folder_path = input('Feature folder (string): ')
    while not generate_index.master_paths_file: generate_index.master_paths_file = input('List images file (string): ')
    while not generate_index.index_folder_path: generate_index.index_folder_path = input('Data index folder (string): ')
    while not generate_index.mtc_folder_images: generate_index.mtc_folder_images = input('Input image folder (string): ')
    generate_index.use_pca = bool(int(input('Use PCA ? 1 for Yes, 0 for No (default False): ') or generate_index.use_pca))
    generate_index.use_matching = bool(int(input('Use matching ? 1 for Yes, 0 for No (default True): ')  or generate_index.use_matching))
    generate_index.use_pre_computed = bool(int(input('Use pre computed ? 1 for Yes, 0 for No (default False): ') or generate_index.use_pre_computed))
    if generate_index.use_pre_computed:
        while not generate_index.mtc_folder_desc: generate_index.mtc_folder_desc = input('Desc image folder (string): ')
    generate_index.use_rmac = bool(int(input('Use rmac ? 1 for Yes, 0 for No (Default True): ') or generate_index.use_rmac))
    generate_index.use_solar = bool(int(input('Use solar ? 1 for Yes, 0 for No (Default True): ') or generate_index.use_solar))

    if generate_index.use_matching:
        generate_index.mtc_use_homography = bool(int(input('Use homography ? 1 for Yes, 0 for No (Default True): ') or generate_index.mtc_use_homography))
        generate_index.mtc_use_superglue = bool(int(input('Use superglue ? 1 for Yes, 0 for No (Default True): ') or generate_index.mtc_use_superglue))
        generate_index.mtc_feature = input('mtc feature (default akaze): ') or generate_index.mtc_feature

    # Generate
    generate_index.init_model()
    generate_index.generate()

if __name__ == "__main__":
    main()