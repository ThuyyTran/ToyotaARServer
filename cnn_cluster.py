import sys
from NumpyEncoder import NumpyEncoder
import settings
import base64
import helpers
import redis
from extract_cnn import CNN
import os
import warnings
import pickle
import configparser
import gc
import ctypes

import numpy as np
import faiss
import cv2
import time
import json
import collections

from logger import AppLogger
from PIL import Image
from types import SimpleNamespace

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# initialize redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

# initialize app logger
logger = AppLogger()

# initialize config parser
config = configparser.ConfigParser(inline_comment_prefixes='#')


def crop_sod_images(payloads):
    """ crop sod images """
    image_array = {}
    pre_sod_image_array = {}
    crop_img_s = []
    result_boxes = []
    # decode payload data
    for payload in payloads:
        img_shape = (payload.height, payload.width, settings.CNN_IMAGE_CHANS)
        if payload.pre_computed_features == '':
            img_decoded = helpers.base64_decode_image(payload.image, np.uint8, img_shape)
            if len(image_array) > 0:
                image_array['app_code'].append(payload.app_code)
                image_array['id'].append(payload.id)
                image_array['shape'].append(img_shape)
                image_array['raw'].append(img_decoded)
                image_array['nr_retr'].append(payload.nr_retr)
                image_array['category'].append(payload.category)
            else:
                image_array['app_code'] = [payload.app_code]
                image_array['id'] = [payload.id]
                image_array['shape'] = [img_shape]
                image_array['raw'] = [img_decoded]
                image_array['nr_retr'] = [payload.nr_retr]
                image_array['category'] = [payload.category]
        else:
            img_decoded = helpers.base64_decode_image(payload.image, np.uint8, img_shape)
            image = Image.fromarray(img_decoded).convert('RGB')
            left, top, right, bottom = 0, 0, image.size[0], image.size[1]

            if len(pre_sod_image_array) > 0:
                pre_sod_image_array['app_code'].append(payload.app_code)
                pre_sod_image_array['id'].append(payload.id)
                pre_sod_image_array['shape'].append(img_shape)
                pre_sod_image_array['nr_retr'].append(payload.nr_retr)
                pre_sod_image_array['category'].append(payload.category)
                pre_sod_image_array['pre_computed_features'].append(payload.pre_computed_features)
                pre_sod_image_array[settings.IMAGE_BOXES].append([left, top, right, bottom])
            else:
                pre_sod_image_array['app_code'] = [payload.app_code]
                pre_sod_image_array['id'] = [payload.id]
                pre_sod_image_array['shape'] = [img_shape]
                pre_sod_image_array['nr_retr'] = [payload.nr_retr]
                pre_sod_image_array['category'] = [payload.category]
                pre_sod_image_array['pre_computed_features'] = [payload.pre_computed_features]
                pre_sod_image_array[settings.IMAGE_BOXES] = [[left, top, right, bottom]]
    if len(image_array) == 0:
        logger.info("No images to do SOD calculation")
        return image_array, pre_sod_image_array

    time_start_crop = time.time()
    # sod images
    # sod_images = helpers.extract_sod_batch_by_array(
    #     sod_model, image_array['raw'])
    time_1 = time.time() - time_start_crop
    sod_images = helpers.extract_mask_ClipSeg_model(clipSeg_processor,clipSeg_model,image_array['raw'],image_array['id'])
    # start crop
    logger.info("Start crop")
    crop_img_s_tmp = []
    for i in range(len(sod_images)):
        crop_img_s_tmp.append(np.array(sod_images[i]))
    for i in range(len(sod_images)):
        image = Image.fromarray(image_array['raw'][i]).convert('RGB')
        sod_image_id = image_array['id'][i]
        sod_image_shape = image_array['shape'][i]

        sod_image = sod_images[i].resize(
            (sod_image_shape[1], sod_image_shape[0]), resample=Image.BILINEAR)
        sod_image = cv2.cvtColor(np.array(sod_image), cv2.COLOR_RGB2GRAY)
        opencvImage = cv2.cvtColor(
            np.array(image_array['raw'][i]), cv2.COLOR_RGB2BGR)
        left, top, right, bottom = 0, 0, image.size[0], image.size[1]

        # crop image
        try:
            res_crop, left, top, right, bottom = helpers.find_roi_update(
                opencvImage, sod_image, cascade)

            if res_crop == True:
                roi_image = image.crop((left, top, right, bottom))
            else:
                roi_image = image
        except Exception as e:
            logger.error(
                f"Cannot crop image {sod_image_id}")
            roi_image = image
        
        debugging = False
        # Save crop image
        if settings.CBIR_SAVE_QUERY_IMAGES and debugging:
            path_to_query_image = str(sod_image_id) + "_cropped.jpg"
            path_to_query_image = os.path.join(
                settings.QUERY_IMAGE_FOLDER, path_to_query_image)
            logger.info("Save crop done")
            try:
                roi_image.save(path_to_query_image)
                sod_images[0].save(os.path.join(
                    settings.QUERY_IMAGE_FOLDER, str(sod_image_id) + "_sod.jpg"))
            except IOError:
                logger.error(
                    f"Cannot save the cropped image to file {path_to_query_image}")
        search_image, w, h = helpers.resize(
            roi_image, settings.CNN_IMAGE_WIDTH, settings.CNN_IMAGE_HEIGHT, True)
        search_image = np.array(search_image)
        crop_img_s.append(search_image)
        result_boxes.append([left, top, right, bottom])
        roi_image.crop([left, top, right, bottom])
        
    time_end_crop = time.time()
    logger.info("SOD processing consumes : %s - %s",
                time_1, (time_end_crop - time_start_crop))
    image_array.update({'sod': crop_img_s_tmp})
    # image_array.update({'sod': crop_img_s})
    image_array.update({settings.IMAGE_BOXES: result_boxes})
    del image_array['raw']
    return image_array, pre_sod_image_array


def group_dict_by_app_code(dict):
    """ split sod dict by app code """
    input_dict = {}
    if len(dict) == 0:
        return input_dict
    app_code_arr = dict['app_code']
    for i in range(len(app_code_arr)):
        app_code = app_code_arr[i]
        if app_code in input_dict:
            data_dict = input_dict[app_code]
            data_dict['id'].append(dict['id'][i])
            data_dict['sod'].append(dict['sod'][i])
            data_dict[settings.IMAGE_BOXES].append(
                dict[settings.IMAGE_BOXES][i])
            data_dict['nr_retr'].append(dict['nr_retr'][i])
            data_dict['category'].append(dict['category'][i])
        else:
            input_dict[app_code] = {
                'id': [dict['id'][i]],
                'sod': [dict['sod'][i]],
                'box': [dict[settings.IMAGE_BOXES][i]],
                'nr_retr': [dict['nr_retr'][i]],
                'category': [dict['category'][i]]
            }

    return input_dict


def extract_features(sod_dict, pre_computed_sod_dict):
    """ extract CNN features """
    feature_dict = {}
    input_dict = group_dict_by_app_code(sod_dict)
    if len(input_dict) > 0:
        t0 = time.time()
        for app_code in input_dict.keys():
            data_dict = input_dict[app_code]

            logger.info("Start extract %s images" % len(data_dict['id']))

            img_data = data_dict['sod']
            box = data_dict['box']
            nr_retr = data_dict['nr_retr']
            category = data_dict['category']
            multiscale = '[1, 1/2**(1/2), 1/2]'
            ms = list(eval(multiscale))
            db_in_use = multi_db_obj_arr[app_code]
            desc_model = db_in_use['desc_mode']
            if desc_model == 'solar':
                ms_query_feats, query_feats = model2.extract_feat_batch_by_arrays(
                    img_data, image_size=settings.CNN_IMAGE_WIDTH, ms=ms, pad=0)
            else:
                ms_query_feats, query_feats = model1.extract_feat_batch_by_arrays(
                    img_data, image_size=settings.CNN_IMAGE_WIDTH, ms=ms, pad=0)
            feature_dict[app_code] = {
                'id': data_dict['id'], 'nr_retr': nr_retr, 'ms_query_feats': ms_query_feats, 'query_feats': query_feats, 'box': box, 'category': category}

            logger.info("Extract done in: %s seconds" % (time.time() - t0))

    # Merge with pre-computed features
    if len(pre_computed_sod_dict) > 0:
        logger.info("Merging pre-computed features")
        app_code_arr = pre_computed_sod_dict['app_code']

        for i in range(len(app_code_arr)):
            app_code = app_code_arr[i]

            # decode pre-computed features
            pre_computed_features = helpers.base64_decode_image(pre_computed_sod_dict['pre_computed_features'][i], np.float32, (2048, ))

            if app_code in feature_dict:
                data_dict = feature_dict[app_code]
                data_dict['id'].append(pre_computed_sod_dict['id'][i])
                data_dict['box'].append(pre_computed_sod_dict['box'][i])
                data_dict['nr_retr'].append(pre_computed_sod_dict['nr_retr'][i])
                data_dict['category'].append(pre_computed_sod_dict['category'][i])
                data_dict['ms_query_feats'] = data_dict['ms_query_feats'].append(pre_computed_features)
                # FIXME
                data_dict['query_feats'] = data_dict['query_feats'].append(pre_computed_features)
            else:
                feature_dict[app_code] = {
                    'id': [pre_computed_sod_dict['id'][i]],
                    'box': [pre_computed_sod_dict[settings.IMAGE_BOXES][i]],
                    'nr_retr': [pre_computed_sod_dict['nr_retr'][i]],
                    'category': [pre_computed_sod_dict['category'][i]],
                    'ms_query_feats': np.array([pre_computed_features]),
                    # FIXME
                    'query_feats': np.array([pre_computed_features])
                }
    return feature_dict


def run():
    """ run process """
    print("The CNN cluster is ready ...")

    while True:
        if db.get('ProcessMode').decode("utf-8") == 'Search':
            requests = helpers.multi_pop(
            db, settings.IMAGE_QUEUE, settings.BATCH_SIZE)
            sod_arr = []
            # step 1: parsing requests
            for req in requests:
                if req is None:
                    continue
                req_payload = json.loads(
                    req, object_hook=lambda d: SimpleNamespace(**d))
                sod_arr.append(req_payload)
            # step 2: crop sod images
            if len(sod_arr) > 0:
                sod_dict, pre_computed_sod_dict = crop_sod_images(sod_arr)
            else:
                continue
            # step 3: Extract features
            if len(sod_dict) > 0 or len(pre_computed_sod_dict) > 0:
                feature_dict = extract_features(sod_dict, pre_computed_sod_dict)
            else:
                continue
            # step 4: prepare inputs for faiss search
            if len(feature_dict) > 0:
                searchPayload = json.dumps(feature_dict, cls=NumpyEncoder)
                db.rpush(settings.FAISS_QUEUE, searchPayload)
                logger.info('Push faiss payload to redis')
            else:
                continue
            # clear if done
            sod_arr.clear()
            # sleeppppp...
            # time.sleep(settings.SERVER_SLEEP)
        else:
            requests = helpers.single_pop(db,settings.IMAGE_QUEUE)
            if requests == None:
                continue
            else:
                my_json = requests.decode('utf8').replace("'", '"')
                data = json.loads(my_json)
                if data['process_status'] == 'AddIndex':
                    status = helpers.addIndex(data,model1,model2,clipSeg_processor,clipSeg_model,db)
                elif data['process_status'] == 'AddImages':
                    status = helpers.addImages(data,model1,model2,clipSeg_processor,clipSeg_model,db)
                elif data['process_status'] == 'RemoveIndex':
                    status = helpers.removeIndex(data,db)
                elif data['process_status'] == 'RemoveImages':
                    status = helpers.removeImage(data,db)
                db.set(data['uuid'], json.dumps(status))
                db.set('ProcessMode','Search')

def load_config(db_path):
    # read database config
    config_path = os.path.join(db_path, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    desc_mode = db_config[settings.DESC_MODE_CONFIG]

    return desc_mode


if __name__ == '__main__':
    print("Starting the CNN cluster")
    cluster_id = 0

    # for sod - model crop
    sod_model = helpers.init_sod_model()
    clipSeg_processor,clipSeg_model = helpers.init_ClipSegModel()
    cascade = helpers.init_cascade_model('lbpcascade_animeface.xml')

    # load config from config file
    _config_path = settings.DATABASE_LIST_PATH
    with open(_config_path, 'r') as json_db:
        db_paths = json.load(json_db)

    # for search
    model1 = CNN(useRmac=True, use_solar=False)
    model2 = CNN(useRmac=True, use_solar=True)

    # set config object
    multi_db_obj_arr = {}
    for k, v in db_paths.items():
        desc_mode = load_config(v)
        multi_db_obj_arr[k] = {
            'db_path': v,
            'desc_mode': desc_mode
        }

    # run process
    run()
