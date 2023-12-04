import configparser
import os
from re import I
import time
import faiss
import numpy as np
import settings
import json
import helpers
import redis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from types import SimpleNamespace
from extract_cnn import CNN
from logger import AppLogger

def load_config(config, db_path):
    # read database config
    config_path = os.path.join(db_path, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    _index_path = os.path.join(db_path, db_config[settings.INDEX_FILE_KEY])
    _pca_path = os.path.join(db_path, db_config[settings.PCA_MATRIX_FILE_KEY])
    _img_list_path = os.path.join(db_path, db_config[settings.IMG_LIST_FILE_KEY])
    _cnn_image_feature_using_pca_key = db_config.getboolean(settings.CNN_IMAGE_FEATURE_USING_PCA_KEY)
    desc_mode = db_config[settings.DESC_MODE_CONFIG]
    use_category = db_config.getboolean(settings.USE_CATEGORY) if settings.USE_CATEGORY in db_config else False

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

    return idx, master_data_paths, desc_mode, use_category, db_category_name, db_product_category

def group_request():
    requests = helpers.multi_pop(db, settings.FAISS_QUEUE, settings.BATCH_SIZE)
    feature_dict = {}

    for req in requests:
        if req is None:
            continue
        payload = json.loads(req)

        for app_code, feature in payload.items():
            if app_code in feature_dict:
                feature_dict[app_code]['id'].extend(feature['id'])
                feature_dict[app_code]['ms_query_feats'].extend(feature['ms_query_feats'])
                feature_dict[app_code]['query_feats'].extend(feature['query_feats'])
                feature_dict[app_code]['box'].extend(feature['box'])
                feature_dict[app_code]['nr_retr'].extend(feature['nr_retr'])
                feature_dict[app_code]['category'].extend(feature['category'])
            else:
                feature_dict[app_code] = feature

        logger.info("Group request done")

    return feature_dict

def group_dict_by_nr_retr(dict):
    nr_dict = {}
    nr_retr_arr = dict['nr_retr']
    for i in range(len(nr_retr_arr)):
        nr_retr = nr_retr_arr[i]
        if nr_retr in nr_dict:
            data_dict = nr_dict[nr_retr]
            data_dict['id'].append(dict['id'][i])
            if 'query_feats' in dict.keys():
                data_dict['query_feats'].append(dict['query_feats'][i])
            data_dict['ms_query_feats'].append(dict['ms_query_feats'][i])
            data_dict['box'].append(dict['box'][i])
            data_dict['category'].append(dict['category'][i])
        else:
            if 'query_feats' in dict.keys():
                nr_dict[nr_retr] = {
                    'id': [dict['id'][i]],
                    'query_feats': [dict['query_feats'][i]],
                    'ms_query_feats': [dict['ms_query_feats'][i]],
                    'box': [dict['box'][i]],
                    'category':[dict['category'][i]]
                }
            else:
                nr_dict[nr_retr] = {
                    'id': [dict['id'][i]],
                    #'query_feats': [dict['query_feats'][i]],
                    'ms_query_feats': [dict['ms_query_feats'][i]],
                    'box': [dict['box'][i]],
                    'category':[dict['category'][i]]
                }
    return nr_dict

def faiss_search(dbs, logger, feature_dict):
    search_result_dict = {}
    if len(feature_dict) > 0:
        t0 = time.time()
        for app_code in feature_dict.keys():
            data_dict = feature_dict[app_code]

            nr_dict = group_dict_by_nr_retr(data_dict)
            t1 = time.time()
            for nr_retr in nr_dict.keys():
                #logger.info("Start search %s images" % len(data_dict['id']))
                # get database
                db_in_use = dbs[app_code]

                # parse feature object
                ms_query_feats = np.array(nr_dict[nr_retr]['ms_query_feats'])
                query_feats = np.array(nr_dict[nr_retr]['query_feats'])
                box = nr_dict[nr_retr]['box']
                category = nr_dict[nr_retr]['category']

                # prepare input
                ms_query_feats = np.ascontiguousarray(ms_query_feats.astype('float32'))
                query_feats = np.ascontiguousarray(query_feats.astype('float32'))
                full_query_feats = np.concatenate((ms_query_feats,query_feats), axis=0)
                # search
                retr_num = 10
                # retr_num = nr_retr if len(category) == 0 else 2000
                distances, indices = db_in_use['index'].search(full_query_feats, retr_num)
                
                distances = np.vsplit(distances, 2)
                indices = np.vsplit(indices, 2)
                agg_distances = np.concatenate(distances, axis=1)
                agg_indices = np.concatenate(indices, axis=1)
                sorted_lists = sorted(zip(agg_distances[0], agg_indices[0]))
                sorted_distances, sorted_filenames = zip(*sorted_lists)
                agg_distances = []
                agg_indices = []
                for i in range(len(sorted_filenames)):
                    if sorted_filenames[i] not in agg_indices:
                        agg_indices.append(sorted_filenames[i]) 
                        agg_distances.append(sorted_distances[i])
                agg_distances = np.array([agg_distances])
                agg_indices = np.array([agg_indices])
                # logger.info(f'distances : {distances}')
                # logger.info(f'agg_distances : {agg_distances}')
                # exit()

                # prepare result
                if app_code not in search_result_dict:
                    search_result_dict[app_code] = [{ 'id': data_dict['id'], 'distances': agg_distances, 'indices': agg_indices, 'box': box, 'nr_retr': nr_retr, 'category': category }]
                else:
                    search_result_dict[app_code].append({ 'id': data_dict['id'], 'distances': agg_distances, 'indices': agg_indices, 'box': box, 'nr_retr': nr_retr, 'category': category })

            logger.info(f"Faiss search {len(data_dict['id'])} images of app_code {app_code} done in {time.time() - t1} seconds")

        logger.info(f"Search done in {time.time() - t0} seconds")

    return search_result_dict


def prepare_match_input(dbs, logger, search_result_dict):
    results = []
    for app_code in search_result_dict.keys():
        search_results = search_result_dict[app_code]
        for search_result in search_results:
            # nr_retr = search_result['nr_retr']
            nr_retr = 10
            for img_id, box, distance, indice, category in zip(search_result['id'], search_result['box'], search_result['distances'], search_result['indices'], search_result['category']):
                json_response = {}
                topn = []
                rt_categories = []
                count = 0
                ids = []
                for i, d in zip(indice, distance):
                    if count == nr_retr:
                        break

                    # Update score
                    if i in ids:
                        _index = ids.index(i)
                        score_update = min(float(d), topn[_index][settings.SCORE_KEY])
                        topn[_index][settings.SCORE_KEY] = float(score_update)
                        continue

                    # get master path
                    db_in_use = dbs[app_code]
                    img_path = db_in_use['master_path'][i]
                    prod_path = img_path.split(',')[0]
                    is_use_category = db_in_use['use_category']

                    # Filter by category
                    if is_use_category:
                        # get master data
                        db_product_category = db_in_use['db_product_category']
                        db_category_name = db_in_use['db_category_name']

                        # get product info
                        prod_id = img_path.split(',')[-1]
                        if prod_id in db_product_category:
                            prod_category_id = str(db_product_category[prod_id])
                        else:
                            prod_category_id = '-2'
                        prod_category_name = db_category_name[prod_category_id]

                        # filter category
                        if category != '':  ## Case: filter by certain category
                            rt_categories = [category]
                            if prod_category_name == category:
                                response_dict = { settings.IMAGE_KEY: prod_path,
                                                    settings.SCORE_KEY: float(d),
                                                    settings.SKU_CD: prod_id,
                                                    settings.CATEGORIES_KEY: prod_category_name
                                                  }
                                topn.append(response_dict)
                                ids.append(i)
                                count += 1
                        else:   ## Case: filter by all category
                            response_dict = { settings.IMAGE_KEY: prod_path,
                                                settings.SCORE_KEY: float(d),
                                                settings.SKU_CD: prod_id,
                                                settings.CATEGORIES_KEY: prod_category_name }
                            # append list category
                            if prod_category_name not in rt_categories:
                                rt_categories.append(prod_category_name)
                            topn.append(response_dict)
                            ids.append(i)
                            count += 1
                    else:   ## Case: not filter
                        response_dict = { settings.IMAGE_KEY: img_path,
                                            settings.SCORE_KEY: float(d),
                                            settings.SKU_CD: os.path.splitext(os.path.basename(prod_path))[0].split('_')[0] }
                        topn.append(response_dict)
                        ids.append(i)
                        count += 1

                # create response object
                json_response["topn"] = topn
                json_response['box'] = box
                json_response['category'] = rt_categories
                logger.info(f'rt categories {rt_categories}')

                # create json result
                result = json.dumps(json_response, ensure_ascii=False).encode("utf8")

                results.append((img_id, result))
                logger.info("Request %s done !" % img_id)

    return results

def run(dbs):
    """ Run cluster """
    print("The faiss cluster is ready ...")

    while True:
        # step 1: group requests to dict
        feature_dict = group_request()

        # step 2: faiss search
        if len(feature_dict) > 0:
            search_result_dict = faiss_search(dbs, logger, feature_dict)
        else:
            continue

        # step 3: prepare matching input
        if len(search_result_dict) > 0:
            results = prepare_match_input(dbs, logger, search_result_dict)
            for img_id, result in results:
                # save result to cache db
                print('====',img_id,' ',result)
                db.set(img_id, result)
        else:
            continue

        time.sleep(settings.SERVER_SLEEP)

if __name__ == '__main__':
    print("Starting the faiss cluster")

    # initialize app logger
    logger = AppLogger()

    # initialize redis server
    db = redis.StrictRedis(host=settings.REDIS_HOST,
                           port=settings.REDIS_PORT, db=settings.REDIS_DB)

    # initialize config parser
    config = configparser.ConfigParser(inline_comment_prefixes='#')

    # load config from config file
    _config_path = settings.DATABASE_LIST_PATH
    with open(_config_path, 'r') as json_db:
        db_paths = json.load(json_db)
    # set config object
    multi_db_obj_arr = {}
    for k, v in db_paths.items():
        idx, master_data_paths, desc_mode, use_category, db_category_name, db_product_category = load_config(config, v)
        if idx is not None and master_data_paths is not None:
            multi_db_obj_arr[k] = {
                'db_path': v,
                'index': idx,
                'master_path': master_data_paths,
                'desc_mode': desc_mode,
                'use_category': use_category,
                'db_category_name': db_category_name,
                'db_product_category': db_product_category,
            }
        else:
            logger.error('Database is None')
            raise ValueError('Database is None')
    run(multi_db_obj_arr)
