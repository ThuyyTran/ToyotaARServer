import os
import warnings
import pickle

import numpy as np
import cv2
import time
import json
import configparser

warnings.filterwarnings("ignore")

# test lib
import redis
import sys
import base64
from PIL import Image


import download_api


import helpers
import settings
from logger import AppLogger
from matching import PlanarMatching
from match_superglue import SuperglueMatching

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# initialize redis server
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)
# initialize app logger
logger = AppLogger()

# initialize config parser
config = configparser.ConfigParser(inline_comment_prefixes='#')

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

def download_images(search_path, app_code, local_img_folder):
    paths = []
    for path in search_path:
        candidate_id = path[settings.IMAGE_KEY]
        paths.append(candidate_id)

    time_s = time.time()
    if os.path.isdir(local_img_folder):
        # local img folder exists
        images = {}
        for p in paths:
            p = p.replace("imgs/", "")
            p1 = local_img_folder+ p
            print(p1)
            img = Image.open(p1)
            if img is not None:
                images[p] = img
            else:
                images[p] = None
    else:
        logger.info("Need to download images from %s" % settings.URL_IMAGE)
        images = download_api.download_images(paths, url_name=settings.URL_IMAGE , app_code=app_code)
    logger.info("time images download: %f" % float(time.time() - time_s))
    return images

def match_process():
    logger.info("Matching server is ready ...")

    while True:
        requests = helpers.multi_pop(db, settings.MTC_IMAGE_QUEUE, settings.MTC_BATCH_SIZE)
        if len(requests) <= 0:
            continue

        input_array = []
        input_ids = []
        matching_sources = []
        app_code_arr = []
        boxes = []
        rt_categories = []
        for req in requests:
            if req is None:
                continue
            req = json.loads(req)
            img_id = req[settings.ID_KEY]
            img_raw_data = req[settings.IMAGE_KEY]
            # opencv's Mats have a reversed shape
            img_shape = (req[settings.IMAGE_HEIGHT_KEY], req[settings.IMAGE_WIDTH_KEY], settings.MTC_IMAGE_CHANS)
            candidates = req[settings.SIMILAR_IMAGES_KEY]
            app_code = req[settings.APP_CODE_KEY]

            image = helpers.base64_decode_image(img_raw_data, np.uint8, img_shape)
            matching_sources.append(candidates)
            input_array.append(image)
            input_ids.append(img_id)
            app_code_arr.append(app_code)
            boxes.append(req[settings.IMAGE_BOXES])
            rt_categories.append(req['categories'])

        if len(input_ids) > 0:
            # geometric verification
            for img_id, source, box, search_path, app_code, categories in zip(input_ids, input_array, boxes, matching_sources, app_code_arr,rt_categories):
                
                t1 = time.time()
                logger.info('Start processing image_id: %s' % img_id)
                logger.debug("%s - candidates : {}".format(' '.join(map(str, search_path))) % img_id)

                verified = []
                json_response = {}
                index = 0

                # get current db
                db_in_use = multi_db_obj_arr[app_code]

                
                # load config
                _mtc_pre_computed = db_in_use['mtc_pre_computed']
                _mtc_descs_db_folder = db_in_use['mtc_descs_db_folder']
                _mtc_image_db_folder = db_in_use['mtc_image_db_folder']
                _mtc_feature = db_in_use['mtc_feature']
                _mtc_is_need_homography = db_in_use['mtc_is_need_homography']
                _mtc_using_superglue = db_in_use['mtc_using_superglue']

                # FIXME: force it to download images
                _mtc_pre_computed = False
                logger.info("Before downloading images")
                images = download_images(search_path, app_code, _mtc_image_db_folder)
                if _mtc_using_superglue:
                    source = source[int(box[1]): int (box[3]) , int(box[0]): int (box[2])]
                M = PlanarMatching(source, mtc_feature=_mtc_feature, is_need_homography=_mtc_is_need_homography)
                top1_path = ''
                for path in search_path:
                    is_relevant = False
                    t2 = time.time()
                    candidate_id = path[settings.IMAGE_KEY]
                    candidate_id = candidate_id.replace('imgs/', '')
                    need_to_read_image = True
                    score_matches = 0
                    if _mtc_pre_computed:
                        img_2_descs_fn = os.path.join(_mtc_descs_db_folder, candidate_id) + ".pkl"
                        try:
                            file = open(img_2_descs_fn, 'rb')
                            kps, descs, w, h = pickle.load(file)
                            is_relevant , score_matches = M.is_descs_relevant(kps, descs, w, h)
                            need_to_read_image = False
                        except (OSError, ValueError):
                            logger.error("Error opening descs file '%s'. Will try to use the corresponding image file." % img_2_descs_fn)
                    if need_to_read_image:
                        #img_2_fn = os.path.join(_mtc_image_db_folder, candidate_id)
                        img_pil = images[candidate_id]
                        if img_pil is not None :
                            img_2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            if index <= 0:
                                img_db = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                        else:
                            img_2 = None

                        #img_2 = cv2.imread(img_2_fn)
                        if img_2 is None:
                            logger.error("File not found : '%s'" % img_2_fn)
                            is_relevant = False
                        else:
                            is_relevant , score_matches = M.is_image_relevant(img_2)
                    if index <= 0:
                        top1_path = os.path.join(_mtc_image_db_folder, candidate_id)

                    logger.debug("Matching consumes : %s" % (time.time() - t2))
                    if _mtc_is_need_homography:
                        score_matches = path[settings.SCORE_KEY]
                    if True:
                        response_dict = {}
                        response_dict[settings.IMAGE_KEY] = path[settings.IMAGE_KEY]
                        response_dict[settings.SCORE_KEY] =  path[settings.SCORE_KEY]
                        response_dict[settings.SCORE_MATCHES] = score_matches
                        full_id = os.path.splitext(os.path.basename(path[settings.IMAGE_KEY]))[0]
                        if len(full_id.split('_')) > 0:
                            response_dict[settings.ID_KEY] = full_id.split('_')[0]
                        else:
                            response_dict[settings.ID_KEY] = full_id

                        verified.append(response_dict)
                    res_score = score_matches > 0.3 or _mtc_is_need_homography

                    # if there is a match image in top 5, break
                    if index == 4 and len(verified) > 0 and res_score:
                        break
                    index = index + 1
                # if not match found re-check match regions mser
                if len(verified) <= 0:
                    logger.info("No match found, trying to match by mser")
                    img_db = cv2.imread(top1_path, 0)
                    if img_db is not None:
                        try:
                            is_relevant , score_matches = M.re_match_use_mser_image(source , img_db)
                        except (OSError, ValueError):
                            logger.error("Error re-match top1")

                        if _mtc_using_superglue:
                            # matching superglue 
                            if not is_relevant:
                                try:
                                    if len(source.shape) != len(img_db.shape) and len(source.shape) == 3 :
                                        _, _, c = source.shape
                                        if c == 3:
                                            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
                                        source = source.reshape(source.shape[0] , source.shape[1])
                                    M_S = SuperglueMatching(source)
                                    is_relevant , score_matches = M_S.find_matches_superglue( img_db)
                                    print("is_relevant " , is_relevant , score_matches)
                                except (OSError, ValueError):
                                    logger.error("Error superglue matching")

                        if True:
                            path_update = search_path[0]
                            response_dict = {}
                            response_dict[settings.IMAGE_KEY] = path_update[settings.IMAGE_KEY]
                            response_dict[settings.SCORE_KEY] =  path_update[settings.SCORE_KEY]
                            response_dict[settings.SCORE_MATCHES] = score_matches
                            full_id = os.path.splitext(os.path.basename(path_update[settings.IMAGE_KEY]))[0]
                            if len(full_id.split('_')) > 0:
                                response_dict[settings.ID_KEY] = full_id.split('_')[0]
                            else:
                                response_dict[settings.ID_KEY] = full_id

                            verified.append(response_dict)

                verified = sorted(
                        verified, key=lambda x: x[settings.SCORE_MATCHES], reverse=True)

                for idx , v in enumerate(verified):
                    path_db = v[settings.IMAGE_KEY]
                    paths_update = path_db.split("/")
                    if len(paths_update) > 2 :
                        path_out = os.path.join(paths_update[-2] , paths_update[-1])
                        v[settings.IMAGE_KEY] = path_out
                        verified[idx] = v
                json_response["result"] = verified
                json_response["box"] = box
                json_response['categories'] = categories
                db.set(img_id, json.dumps(json_response, ensure_ascii=False).encode("utf8"))
                logger.info("Processing %s done in %s" % (img_id, time.time() - t1))

        time.sleep(settings.SERVER_SLEEP)

def load_config(db_path):
    config_path = os.path.join(db_path, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    _mtc_pre_computed = db_config.getboolean(settings.MTC_PRE_COMPUTED_KEY)
    _mtc_image_db_folder = db_config[settings.MTC_IMAGE_DB_FOLDER_KEY]
    _mtc_descs_db_folder = db_config[settings.MTC_DESCS_DB_FOLDER_KEY]
    _mtc_feature = db_config[settings.MTC_FEATURE_KEY]
    _mtc_is_need_homography = db_config.getboolean(settings.MTC_IS_NEED_HOMOGRAPHY_KEY)
    _mtc_using_superglue = db_config.getboolean(settings.MTC_USING_SUPERGLUE_KEY)

    return _mtc_pre_computed, _mtc_image_db_folder, _mtc_descs_db_folder, _mtc_feature, _mtc_is_need_homography, _mtc_using_superglue

if __name__ == '__main__':
    debug_mode = False
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == "debug":
    #         debug_mode = True
    # if debug_mode:
    #     print("Working in the debug mode")
    #     for i in range(1000):
    #         t1 = time.time()
    #         test_match_process()
    #         print("Consumed time : ", time.time() - t1)
    #     exit(0)

    # load config from config file
    _config_path = settings.DATABASE_LIST_PATH
    with open(_config_path, 'r') as json_db:
        db_paths = json.load(json_db)

    # read database config
    multi_db_obj_arr = {}
    for k, v in db_paths.items():
        _mtc_pre_computed, _mtc_image_db_folder, _mtc_descs_db_folder, _mtc_feature, _mtc_is_need_homography, _mtc_using_superglue = load_config(v)
        multi_db_obj_arr[k] = {
            'db_path': v,
            'mtc_pre_computed': _mtc_pre_computed,
            'mtc_image_db_folder': _mtc_image_db_folder,
            'mtc_descs_db_folder': _mtc_descs_db_folder,
            'mtc_feature': _mtc_feature,
            'mtc_is_need_homography': _mtc_is_need_homography,
            'mtc_using_superglue': _mtc_using_superglue
        }

    match_process()
