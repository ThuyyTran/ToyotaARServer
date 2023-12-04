import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

import torch
from torch.utils.model_zoo import load_url
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pickle
from solar_global.networks.imageretrievalnet import init_network, extract_vectors
from solar_global.datasets.testdataset import configdataset
from solar_global.utils.download import download_test
from solar_global.utils.evaluate import compute_map_and_print
from solar_global.utils.general import get_data_root, htime
from solar_global.utils.networks import load_network
from solar_global.utils.plots import plot_ranks, plot_embeddings
from solar_global.datasets.datahelpers import im_resize
from solar_global.layers.pooling import RMAC,Rpool,GeM,GeMmp,SPoC
import torchvision.transforms.functional as F


class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, max_wh - w -hp, max_wh - h - vp)
		return F.pad(image, padding, 255, 'constant')


class Resize_ratio():
    def __init__(self, imsize):
        self.imsize = imsize
    def __call__(self, image):
        image = im_resize(image, self.imsize)
        return image
# some conflicts between tensorflow and tensoboard 
# causing embeddings to not be saved properly in tb

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Example')
parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                    help="network to be evaluated. " )
# parser.add_argument('--datasets', '-d', metavar='DATASETS', default='roxford5k,rparis6k',
#                     help="comma separated list of test datasets: " +
#                         " | ".join(datasets_names) +
#                         " (default: 'roxford5k,rparis6k')")
parser.add_argument('--image-size', '-imsize', dest='image_size', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--soa', action='store_true',
                    help='use soa blocks')
parser.add_argument('--soa-layers', type=str, default='45',
                    help='config soa blocks for second-order attention')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")



def main():
    args = parser.parse_args()

    # check if there are unknown datasets

    # check if test dataset are downloaded
    # and download if they are not
    # download_test(get_data_root())

    # setting up the visible GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network
    net = load_network(network_name='model_best.pth.tar')
    net.mode = 'test'
    net.pool=RMAC(3)
    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))

    print(">>>> Evaluating scales: {}".format(ms))

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        SquarePad(),
        Resize_ratio(500),
        transforms.ToTensor(),
        normalize
    ])
    # with open('/media/congtc/Data/TriCong/Project/SOLAR/all_images.txt','r') as f:
    #     images = [line.strip('\n') for line in f.readlines()]
    # images = images[:102]
    # img_folder = '/media/congtc/DATA_Backup_XLA/Backup'
    # images = [os.path.join(img_folder,file) for file in images]
    # img_folder = 'add_341img'
    # images = [os.path.join(img_folder,file) for file in sorted(os.listdir(img_folder))]
    # with open('add_341img.txt','w') as f:
    #     for path in images:
    #         print(path,file=f)
    with open('687query_crop.txt','r') as f:
        images = [line.strip('\n') for line in f.readlines()]
    # with open('/media/congtc/Data/TriCong/Project/SOLAR/450_query_boxs.pkl','rb') as read:
    #     data = pickle.load(read)
    # bbxs = data['boxs']
    # block_size = 10000
    # save_folder = 'npy_data'
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    
    # num_block = len(images) // block_size + 1

    # evaluate on test datasets
    # datasets = args.datasets.split(',')
    # for dataset in datasets:
        # qvecs = extract_vectors(net, qimages, None, transform, bbxs=bbxs, ms=ms, mode='test')
    # for i in range(num_block):
        # if i <25:
        #     continue
        # current_images = images[i*block_size:(i+1)*block_size]
    vecs = extract_vectors(net, images, None, transform, bbxs=None, ms=ms, mode='test')
    vecs = vecs.numpy().T
        # print(vecs)
        # print("vecs ",vecs.shape)
    # with open('vecs_567kimg.pkl','wb') as write:
    np.save('687query_sod_resize_padding_resize_500.npy',vecs)

  


     

if __name__ == '__main__':
    main()
