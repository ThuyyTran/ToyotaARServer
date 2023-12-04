import base64
from distutils.command.upload import upload
import settings
import helpers
import redis
from extract_cnn import CNN
import os
import warnings
import configparser
import requests

import numpy as np
import faiss
import cv2
import time
import json
from PIL import Image

from logger import AppLogger
from PIL import Image
from types import SimpleNamespace
from revamptracking.revamp_tracking import RevampTracking, ERRORS_CODE
from urllib.parse import urljoin
from Reranking import Rerank

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" Constants """
VIDEO_MASTER_PATH = '/media/anlab/data-1tb1/revamp_store_video'

""" initialize app logger """
logger = AppLogger()

""" initialize config parser """
config = configparser.ConfigParser(inline_comment_prefixes='#')

""" search image in faiss """
def faiss_search(img_list):
    search_result_dict = {}
    if len(img_list) > 0:
        t0 = time.time()
        logger.info("Start search %s images" % len(img_list))
        img_data = np.array(img_list)
        multiscale = '[1, 1/2**(1/2), 1/2]'
        ms = list(eval(multiscale))

        ms_query_feats, query_feats = model.extract_feat_batch_by_arrays(img_data, image_size=settings.CNN_IMAGE_WIDTH, ms=ms, pad=0)

        ms_query_feats = np.ascontiguousarray(ms_query_feats)
        query_feats = np.ascontiguousarray(query_feats)
        full_query_feats = np.concatenate((query_feats, ms_query_feats), axis=0)
        t1 = time.time()

        distances, indices = db_config['index'].search(full_query_feats, 10)
        distances = np.vsplit(distances, 2)
        indices = np.vsplit(indices, 2)
        agg_distances = np.concatenate(distances, axis=1)
        agg_indices = np.concatenate(indices, axis=1)
        search_result_dict = { 'distances': agg_distances, 'indices': agg_indices }
        logger.info("End search in %s seconds: extract in: %s seconds - faiss search in: %s seconds" % (time.time() - t0, t1 - t0, time.time() - t1))

    return search_result_dict

""" Search results to response object """
def response_search(search_result_dict):
    response_arr = []
    db_product_category = db_config['db_product_category']
    db_category_name = db_config['db_category_name']

    for distance, indice in zip(search_result_dict['distances'], search_result_dict['indices']):
        json_response = {}
        topn = []
        ids = []
        for i, d in zip(indice, distance):
            if i in ids:
                index_ = ids.index(i)
                score_update = min(float(d) ,topn[index_][settings.SCORE_KEY] )
                topn[index_][settings.SCORE_KEY] = float(score_update)
                continue
            ids.append(i)

            # get master path
            img_path = db_config['master_path'][i]
            prod_path = img_path.split(',')[0]

            # get product info
            prod_id = img_path.split(',')[-1]
            if prod_id in db_product_category:
                prod_category_id = str(db_product_category[prod_id])
            else:
                prod_category_id = '-2'
            prod_category_name = db_category_name[prod_category_id]

            response_dict = { settings.IMAGE_KEY: prod_path,
                            settings.SCORE_KEY: float(d),
                            settings.SKU_CD: prod_id,
                            settings.CATEGORIES_KEY: prod_category_name }

            topn.append(response_dict)

        if topn:
            json_response["topn"] = topn

            # create json result
            response_arr.append(json_response)

    return response_arr

""" Convert nnumpy array to base64 string """
def ndarray_to_base64(array):
    _, encoded_img = cv2.imencode('.jpg', array)
    base64_img = base64.b64encode(encoded_img).decode("utf-8")
    return base64_img

""" send report request """
def send_request(upload_id, type = 'report-status', msg = ''):
    if type == 'report-shelf-upload':
        REPORT_API = "/api/report-shelf-upload/{0}?type=revamp" . format(upload_id)
    elif type == 'update-thumbnail':
        REPORT_API = "/api/update-thumbnail/{0}?type=revamp" . format(upload_id)
    else:
        REPORT_API = "/api/report-status/{0}?type=revamp" . format(upload_id)

    host_name =  urljoin(settings.WEB_SERVER_HOST, REPORT_API)
    logger.info(f"Send request to {host_name}")
    header = { "x-api-key": db_config['api_key'] }
    print(f"API key: {header}")

    try:
        res = requests.post(host_name, json=msg, headers=header)
        r = res.json()
        logger.info(f"Response {r}")
        return r['status'] == 1
    except:
        return False

""" search images """
def run():
    print("The service is ready ...")

    while True:
        # TODO: pop job from redis
        queue_key = '%s_upload_revamp_shelf_video_jobs' % settings.APP_DB_NAME
        task_upload_id = web_redis.lpop(queue_key)
        if task_upload_id is not None:
            t1 = time.time()
            task_upload_id = task_upload_id.decode('utf-8')

            logger.info(f"Get content of task {task_upload_id}")
            task_upload_content_key = '%s_upload_revamp_shelf_video_task_%s_job' % (settings.APP_DB_NAME, task_upload_id)
            task_upload_content = web_redis.get(task_upload_content_key)
            if task_upload_content is not None:
                task_upload_content = json.loads(task_upload_content.decode("utf-8"), object_hook=lambda d: SimpleNamespace(**d))

                # Get task info
                upload_id = task_upload_content.id
                video_path = os.path.join(VIDEO_MASTER_PATH, task_upload_content.video_path)
                if not os.path.isfile(video_path):
                    logger.error(f"Video {video_path} not exist. Break!")
                    send_request(upload_id, 'report-status', { 'status': ERRORS_CODE.STATUS_GET_FAILED })
                    continue

                logger.info(f"Handle task {task_upload_content.id} with video {video_path}")

                # get thumbnail
                thumbnail_arr = revamp_tracking.getThumbnail(video_path)
                if not thumbnail_arr.size:
                    logger.error(f"Video {video_path} get thumbnail error. Break!")
                    send_request(upload_id, 'report-status', { 'status': ERRORS_CODE.STATUS_GET_FAILED })
                    continue

                thumbnail_str = ndarray_to_base64(thumbnail_arr)
                req = { 'image': thumbnail_str }
                logger.info(f"Update thumbnail for task {task_upload_id}")
                res = send_request(upload_id, 'update-thumbnail', req)
                if not res:
                    logger.error(f'Report thumbnail of upload {upload_id} error')
                    continue

                # begin processing video
                statusCode, im_sumary_arr, img_list, id_list, list_objects_info = revamp_tracking.getStitChingVideo(video_path)

                # check video status
                if statusCode != ERRORS_CODE.STATUS_COMPLETED and statusCode != ERRORS_CODE.STATUS_IMAGE_STITCH_LOOSE:
                    send_request(upload_id, 'report-status', { 'status': statusCode })
                    logger.error(f'Process video at {video_path} error')
                    continue
                else:
                    send_request(upload_id, 'report-status', { 'status': ERRORS_CODE.STATUS_IN_PROGRESS })

                # Search product
                search_results = search_product(img_list)

                # groud search result
                topn = group_result(id_list, search_results, img_list)

                # get box
                merge_topn_box = {}
                for key, value in list_objects_info.items():
                    if key in merge_topn_box:
                        continue

                    if key in topn:
                        prod = os.path.basename(os.path.dirname(topn[key].split(',')[0]))
                        merge_topn_box[key] = {'pos': ','.join([str(item) for item in value]), 'product': prod }

                # Create json response
                if len(merge_topn_box) > 0:
                    json_response = {}
                    json_response['status'] = int(statusCode)
                    json_response['data'] = merge_topn_box
                    json_response['panorama'] = ndarray_to_base64(im_sumary_arr)
                    send_request(upload_id, 'report-shelf-upload', json_response)
                else:
                    send_request(upload_id, 'report-status', { 'status': ERRORS_CODE.STATUS_RETAKE })

                # result = json.dumps(json_response, ensure_ascii=False).encode("utf8")
                # with open('json_response.txt', 'w') as f:
                #     json.dump(json_response, f)

                if res:
                    logger.info(f"Run task {task_upload_id} success in {time.time() - t1}")
                else:
                    logger.error(f"Send request report task {task_upload_id} failed !")

        # sleeppppp...
        time.sleep(settings.SERVER_SLEEP)

""" handle queries to searching """
def search_product(queries):
    search_result = []
    query_size = 100
    search_result_dict = {}
    for i in range(len(queries) // query_size + 1):
        queries_current = queries[i * query_size: (i + 1) * query_size]

        # Faiss search
        if len(queries_current) > 0:
            search_result_dict = faiss_search(queries_current)

        # Result
        if len(search_result_dict) > 0:
            res = response_search(search_result_dict)
            search_result.extend(res)

    return search_result

""" group search results by product id """
def group_result(id_list, results, images_search, using_rerank=True):
    # Group result
    result_id_search = {}
    # Filter duplicate result
    for i_im, idx, result in zip(range(len(id_list)), id_list, results):
        search_results = []
        top_search = {}
        search_results.extend(result['topn'])
        search_results = sorted(search_results, key=lambda x: x[settings.SCORE_KEY], reverse=False)
        if len(search_results) <= 0 :
            result_id_search[idx] = top_search
            break
        if using_rerank:
            list_path = []
            list_score = []
            for i, re in enumerate(search_results):
                p_im = re[settings.IMAGE_KEY]
                s = re[settings.SCORE_KEY]
                list_path.append(p_im)
                list_score.append(s)
            path_db = db_config['mtc_image_db_folder']	
            path_hist_folder = db_config['mtc_hist_db_folder']	
            try:
                image_cv = images_search[i_im].copy()
                re_paths, re_scores, list_errors = reRanking.rerank_hist(image_cv,list_path , list_score, path_db, path_hist_folder)
                for i, re in enumerate(search_results):
                    p_im = re[settings.IMAGE_KEY]
                    if p_im in re_paths:
                        ix = re_paths.index(p_im)
                        search_results[i][settings.SCORE_KEY] = re_scores[ix]
                    else:
                        search_results[i][settings.SCORE_KEY] = re[settings.SCORE_KEY]*1.5
            except Exception as e:
                logger.info(f"[Rerank error")
        search_results = sorted(search_results, key=lambda x: x[settings.SCORE_KEY], reverse=False)
        search_results = search_results[:5]
        list_sku_cd = []
        for item in search_results:
            image_path = item[settings.IMAGE_KEY]
            sku_cd = item[settings.SKU_CD]
            score = item[settings.SCORE_KEY]
            if sku_cd in list_sku_cd:
                continue
            list_sku_cd.append(sku_cd)
            if idx in result_id_search.keys() and len(search_results) > 0:
                top_search = result_id_search[idx]
                if sku_cd in top_search.keys():
                    data = top_search[sku_cd]
                    # data[settings.SCORE_KEY] = (0.5 * score + 0.5 * data[settings.SCORE_KEY]) * 0.95
                    data[settings.SCORE_KEY] = ( score +  data[settings.SCORE_KEY]) 
                    data["count"] = int(data["count"]) + 1
                    top_search[sku_cd] = data
                else:
                    im_dict = {}
                    im_dict[settings.IMAGE_KEY] = image_path
                    im_dict[settings.SKU_CD] = sku_cd
                    im_dict[settings.SCORE_KEY] = score
                    im_dict["count"] = 1
                    top_search[sku_cd] = im_dict
                result_id_search[idx] = top_search
            else:
                im_dict = {}
                im_dict[settings.IMAGE_KEY] = image_path
                im_dict[settings.SKU_CD] = sku_cd
                im_dict["score"] = score
                im_dict["count"] = 1
                top_search[sku_cd] = im_dict
                result_id_search[idx] = top_search

    # Get top 5
    group_result = {}
    for idx in result_id_search:
        top_search = result_id_search[idx]
        list_result = []
        sum = 0
        for sku_cd in top_search.keys():
            val = top_search[sku_cd]
            val[settings.SCORE_KEY] = (1- 0.03*val["count"])*val[settings.SCORE_KEY]/val["count"]
            list_result.append(val)
            # list_result.append(top_search[sku_cd])
            sum += top_search[sku_cd]["score"]
        list_result = sorted(list_result, key=lambda x: x[settings.SCORE_KEY], reverse=False)
        result_id_search[idx] = list_result[0]
        n_max = min(5, len(list_result))
        re_top5 = ""
        score_top = ""
        for i in range(n_max):
            re_top5 = re_top5 + f'{list_result[i][settings.IMAGE_KEY]},'
            score_top = score_top + f'{list_result[i]["score"]},'
        group_result[idx] = re_top5

        rst = f'{idx},{re_top5},{score_top},{sum}'
        # with open('test_revamp2022/revamp_tracking_5336.csv', 'a') as f:
        #     f.write(f'{rst}\n')

    return group_result

""" init config """
def load_config(db_path):
    # read database config
    config_path = os.path.join(db_path, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    _index_path = os.path.join(db_path, db_config[settings.INDEX_FILE_KEY])
    _pca_path = os.path.join(db_path, db_config[settings.PCA_MATRIX_FILE_KEY])
    _img_list_path = os.path.join(
        db_path, db_config[settings.IMG_LIST_FILE_KEY])
    _cnn_image_feature_using_pca_key = db_config.getboolean(
        settings.CNN_IMAGE_FEATURE_USING_PCA_KEY)
    desc_mode = db_config[settings.DESC_MODE_CONFIG]
    _mtc_image_db_folder = db_config[settings.MTC_IMAGE_DB_FOLDER_KEY]
    _mtc_hist_db_folder = db_config[settings.MTC_HIST_DB_FOLDER_KEY]
    # load index
    sub_index = faiss.read_index(_index_path)
    sub_index.nprobe = int(db_config[settings.INDEX_NPROBE_KEY])
    if _cnn_image_feature_using_pca_key:
        pca_matrix = faiss.read_VectorTransform(_pca_path)
        idx = faiss.IndexPreTransform(pca_matrix, sub_index)
    else:
        idx = sub_index

    with open(_img_list_path, 'r') as f:
        master_data_paths = f.readlines()
    master_data_paths = list(map(lambda x: x[:-1], master_data_paths))

    # load api-key
    api_key = '' if settings.API_KEY not in db_config else db_config[settings.API_KEY]

    # load dict label to category
    db_category_name = None
    dict_label_path = os.path.join(db_path, db_config[settings.DICT_LABEL_CATEGORY]) if settings.DICT_LABEL_CATEGORY in db_config else None
    if dict_label_path and os.path.exists(dict_label_path):
        with open(dict_label_path,'r') as read:
            db_category_name = json.load(read)

    # load category
    db_product_category = None
    path_dict_item_category = os.path.join(db_path, db_config[settings.DICT_ITEM_CATEGORY]) if settings.DICT_ITEM_CATEGORY in db_config else None
    if path_dict_item_category and os.path.exists(path_dict_item_category):
        with open(path_dict_item_category,'r') as read:
            db_product_category = json.load(read)

    return idx, master_data_paths, desc_mode, api_key, db_category_name, db_product_category, _mtc_image_db_folder, _mtc_hist_db_folder

""" main """
if __name__ == '__main__':
    print("Starting REVAMP shelves detect service")

    # for sod - model crop
    sod_model = helpers.init_sod_model()
    cascade = helpers.init_cascade_model('lbpcascade_animeface.xml')
    print(cascade.empty())

    # load config from config file
    _config_path = settings.DATABASE_LIST_PATH
    with open(_config_path, 'r') as json_db:
        db_paths = json.load(json_db)

    # set config object
    multi_db_obj_arr = {}
    if 'revamp' in db_paths:
        idx, master_paths, desc_mode, api_key, db_category_name, db_product_category, mtc_image_db_folder, mtc_hist_db_folder = load_config(db_paths['revamp'])
        db_config = {
            'index': idx,
            'master_path': master_paths,
            'desc_mode': desc_mode,
            'db_category_name': db_category_name,
            'db_product_category': db_product_category,
            'api_key': api_key,
            'mtc_image_db_folder': mtc_image_db_folder,
            'mtc_hist_db_folder': mtc_hist_db_folder,
        }
    else:
        print('Revamp config is not exist. Exit!')
        exit(0)

	# Load model
    model = CNN(useRmac=True, use_solar=(db_config['desc_mode'] == 'solar'))
    revamp_tracking = RevampTracking()
    reRanking = Rerank()
    # initialize web redis
    web_redis = redis.StrictRedis(host=settings.WEB_REDIS_HOST, port=settings.WEB_REDIS_PORT, db=settings.WEB_REDIS_DB)

    # Run script
    run()
