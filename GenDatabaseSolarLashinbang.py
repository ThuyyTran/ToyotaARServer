import time
import os

import numpy as np
import logging
import settings
import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors, extract_vectors_by_arrays , extract_vectors_by_arrays2 , extract_db_array
from cirtorch.utils.general import get_data_root
from cirtorch.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC, Rpool
from cirtorch.datasets.datahelpers import im_resize
from solar_global.utils.networks import load_network
from PIL import Image
import torch.nn.functional as F
import cv2
import pickle
from pre_process_kbook import ImagePreProcess
from tqdm import tqdm

class Resize_ratio():
    def __init__(self, imsize):
        self.imsize = imsize
    def __call__(self, image):
        image = im_resize(image, self.imsize)
        return image

PRETRAINED = {
    'rSfM120k-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}


def squarepad(tensors):
    b, c, w, h = tensors.shape
    tensor = tensors.squeeze()

    c, w, h = tensor.shape
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (vp, max_wh - h -vp, hp,max_wh - w - hp)
    image = F.pad(tensor, pad=padding, mode='constant', value=0)
    image = image.unsqueeze(0)

    return image

to_tensor = transforms.ToTensor()
def square_images(images, image_size=500):
    h, w = images.shape[:2]
    max_wh = max(h,w)
    if max_wh != image_size:

        if h > w:
            images = cv2.resize(images, (int(w * image_size / h), image_size))
        else:
            images = cv2.resize(images, (image_size,int(h*image_size / w)))
    tensors = np.zeros(( image_size, image_size,3))
    h, w, c = images.shape
    pad_top = int((image_size - h)/2)
    pad_left = int((image_size - w)/2)
    tensors[ pad_top:pad_top + h, pad_left: pad_left + w,:] = images
    return tensors
def square_multyimages(images, image_size=500):
    b = len(images)
    results_image = []
    for i,img in enumerate(images):
        h, w = img.shape[:2]
        max_wh = max(h,w)
        if max_wh != image_size:

            if h > w:
                img = cv2.resize(img, (int(w * image_size / h), image_size))
            else:
                img = cv2.resize(img, (image_size,int(h*image_size / w)))
            # print(img.shape, images[i].shape)
            # images[i] = img
        results_image.append(img)
    tensors = torch.zeros((b, 3, image_size, image_size))
    images = None
    for i, img in enumerate(results_image):
        h, w, c = img.shape
        img = img.astype(np.float32)
        img = img / 255.
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)
        img = torch.from_numpy(img)

        pad_top = int((image_size - h)/2)
        pad_left = int((image_size - w)/2)
        tensors[i,:, pad_top:pad_top + h, pad_left: pad_left + w] = img

    return tensors
def calculate_resized_dimensions(image, length_ratio):
    """ Calculate the new dimensions of the image based on the length ratio. """
    height, width = image.shape[:2]
    if width > height:
        new_width = int(length_ratio)
        new_height = int((length_ratio / width) * height)
    else:
        new_height = int(length_ratio)
        new_width = int((length_ratio / height) * width)
    return new_width, new_height
class CNN:
    def __init__(self, useRmac=False, use_solar=False, transform_ratio=500):
        network = 'rSfM120k-tl-resnet101-gem-w'
        state = load_url(PRETRAINED[network], model_dir=os.path.join(get_data_root(), 'networks'))
        if use_solar:
            if torch.cuda.is_available():
              state = torch.load('data/networks/model_best.pth.tar')
            else:
              state = torch.load('data/networks/model_best.pth.tar', map_location=torch.device('cpu'))
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', True)

        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False
        if use_solar:
            net = load_network('model_best.pth.tar')
        else:
            net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        self.pre_process_fea = ImagePreProcess()

        if useRmac:
            net.pool = RMAC(3)

        # net.cuda()
        net.eval()

        self.model = net
        if useRmac:
            pass
        else:
            self.msp = net.pool.p.item()

        self.normalize = transforms.Normalize(
            mean=net.meta['mean'],
            std=net.meta['std']
        )
        self.transform = transforms.Compose([
            Resize_ratio(transform_ratio),
            transforms.ToTensor(),
            self.normalize
        ])

    def extract_feat_batch(self, img_paths, pad=0, bs=1):
        feats = extract_vectors(self.model, img_paths, None, self.transform, pad=pad,bs=bs)
        feats = feats.numpy()
        feats = feats.T
        return feats

    def extract_db_array_feat_batch(self, img_paths , base_folder, out_path='output', pad=0, bs=1):
        block_size = 10
        id_list_images = []
        id_list_price = []
        errors_list_id = []
        feature = None
        for i in tqdm(range(len(img_paths) // block_size + 1)):
            img_paths_current = img_paths[i*block_size: (i+1)*block_size]
            images, id_list_images_current, id_list_price_current, errors_list_id_current = self.pre_process_fea.cropBookList(base_folder , img_paths_current , out_path )
            #(self.model, imgs, self.normalize, pad=pad,ms=ms, msp=msp)
            if len(images) <= 0 :
                continue
            feats = extract_db_array(self.model, images, self.transform, pad=pad,bs=bs)
            feats = feats.numpy()
            feats = feats.T
            if feature is None:
                feature = feats
            else:
                feature = np.concatenate((feature, feats), 0)
            id_list_images.extend(id_list_images_current)
            id_list_price.extend(id_list_price_current)
            errors_list_id.extend(errors_list_id_current)
        print("size : " , feature.shape)
        return feature , id_list_images , errors_list_id
    def extract_feat_batch_by_arrays(self, images, image_size=500,pad=0,ms=[1], msp=1):
        # images has shape: N x H x W x C range [0, 255] uint8
        # input tensor has shape N x C x H x W range [0.0, 1.0] float32
        imglist = []
        for i in range(len(images)):
            # newsize = calculate_resized_dimensions(images[i],500)
            img = cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
            # x = cv2.resize(img, (newsize[0], newsize[1]))
            # tensor_img = torch.from_numpy(x).float()
            img = square_images(img,500)
            # cv2.imwrite('test.jpg',x)
            tensor_img = torch.from_numpy(img).float()
            tensor_img = tensor_img.permute(2, 0, 1)
            tensor_img = tensor_img/255
            imglist.append(tensor_img)
        feats, feats_atscale1 = extract_vectors_by_arrays2(self.model,imglist, self.normalize, pad=pad,ms=ms, msp=msp)
        feats = feats.numpy()
        feats = feats.T

        feats_atscale1 = feats_atscale1.numpy()
        feats_atscale1 = feats_atscale1.T

        return feats, feats_atscale1
model2 = CNN(useRmac=True, use_solar=True)
multiscale = '[1, 1/2**(1/2), 1/2]'
ms = list(eval(multiscale))

dictResult = []
id = 0
rootfolder = '/media/anlab/data-2tb/ANLAB_THUY/ToyotaAR/Dataset/NewData/SearchData/'
for foldername in tqdm(os.listdir(rootfolder)):
    for filename in os.listdir(rootfolder+foldername):
        tmp = {}
        img = cv2.imread(os.path.join(rootfolder,foldername,filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        ms_query_feats, query_feats = model2.extract_feat_batch_by_arrays([img], image_size=settings.CNN_IMAGE_WIDTH, ms=ms, pad=0)
        emb = ms_query_feats.squeeze()
        # emb = get_embedding(os.path.join(rootfolder,foldername,filename),test_model)
        tmp['id'] = id
        tmp['path'] = foldername+'/'+filename
        listEmb = []
        for val in emb:
            listEmb.append(str(val))
        tmp['vector'] = list(listEmb)
        dictResult.append(tmp)
        id+=1
import json
# Writing to sample.json
with open("/media/anlab/data-2tb/ANLAB_THUY/ToyotaAR/Dataset/NewData/Vector_Solar_0112_AddTransformBGR_AddNewDataV2_Augment_Square.json", "w") as outfile:
    json.dump(dictResult, outfile)