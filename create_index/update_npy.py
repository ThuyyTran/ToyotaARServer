import sys, getopt, signal, io

sys.path.append('../')

import numpy as np
import os
import faiss
import redis
import settings
import requests
import json
import time
import cv2
import pathlib
import random
import shutil
import subprocess
import shlex
import csv
import helpers
import configparser

from tqdm import tqdm
from extract_cnn import CNN
from logger import AppLogger
from enum import Enum
from urllib.parse import urljoin
from create_descs_db import save_features
from pathlib import Path

class RetCode(Enum):
    NONE = 0
    PROCESS_SUCCESS = 1
    PROCESS_ERROR = 2
    PROCESS_EXCEPTION = 3
    JOB_CONTENT_NULL = 4,
    UPLOAD_ALERT = 5,

class CommonInfo:
    NPY_FOLDER = '/ext_ssd/npy/'
    TRAINED_FILE = 'trained.index'
    IMAGE_ID_UPDATED_PATH = '../common/update_images.txt'
    ROLLBACK_DIR = '../data/rollback/'
    ROLLBACK_INFO = os.path.join(ROLLBACK_DIR, 'rollback.txt')
    ROLLBACK_NPY = os.path.join(ROLLBACK_DIR, 'npy/')
    SAMPLE_DIR = '../request_multi_images/update_test/data/test_images_request_qa/'
    SAMPLE_LIST_DIR = '../request_multi_images/update_test/data/txt/'

class Config:
    def __init__(self, db_config):
        self.index_path = os.path.join(db_path, db_config[settings.INDEX_FILE_KEY])
        self.pca_path = os.path.join(db_path, db_config[settings.PCA_MATRIX_FILE_KEY])
        self.npy_path = os.path.join(db_path, CommonInfo.NPY_FOLDER)
        self.trained_path = os.path.join(db_path, CommonInfo.TRAINED_FILE)
        self.img_list_path = os.path.join(db_path, db_config[settings.IMG_LIST_FILE_KEY])
        self.img_db_path = db_config[settings.MTC_IMAGE_DB_FOLDER_KEY]
        self.descs_db_path = db_config[settings.MTC_DESCS_DB_FOLDER_KEY]
        self.using_pca_key = db_config.getboolean(settings.CNN_IMAGE_FEATURE_USING_PCA_KEY)

def remove_file(files):
    logger.info('Start remove %s files' % len(files))

    arr_deleted = {}
    reverse_lookup = { x:i for i, x in enumerate(master_data_paths) }

    # get index file deleted
    for i, x in enumerate(tqdm(files)):
        # skip S file
        if '_S' in x:
            continue

        # find filename in master paths
        file_id = reverse_lookup.get(x, -1)
        if file_id == -1:
            logger.error('job: %s - file %s doesnt exist in list path' % (current_job, x))
            error_list.append(x)
        else:
            if 'deleted' not in master_data_paths[file_id]:
                # prepare deleted list
                block_idx = file_id // block_size
                at_id = file_id - (block_idx * block_size)

                # set deleted path
                master_data_paths[file_id] = f'deleted({master_data_paths[file_id]})'

                # list block_id = [ids]
                if str(block_idx) in arr_deleted:
                    arr_deleted[str(block_idx)].append(at_id)
                else:
                    arr_deleted[str(block_idx)] = [at_id]

    return remove_feature(arr_deleted)

def crop_foreground(files):
    add_images = []
    block_crop_size = 20
    indexs = len(files) // block_crop_size + 1

    for idx in tqdm(range(indexs)):
        master_data_path_current = files[int(idx * block_crop_size): min(len(files), int((idx + 1) * block_crop_size))]
        res_lst = helpers.detect_key_list(sod_model, _config.img_db_path, master_data_path_current)
        if len(res_lst):
            add_images.extend(res_lst)

    return add_images

def upload_file(files):
    num_blocks = len(os.listdir(_config.npy_path))
    last_block_name = os.path.join(_config.npy_path, 'block_{0}.npy'.format(num_blocks - 1))
    last_block = np.load(last_block_name)

    upload_paths = []

    # Check file is exist
    for file in files:
        upload_paths.append(os.path.join(_config.img_db_path, file))
        master_data_paths.append(file)

    # update feature
    try:
        # fill to last block
        block_offset = block_size - len(last_block)
        append_list = upload_paths[:block_offset]
        if len(append_list) > 0:
            features_offset = model.extract_feat_batch(append_list)
            data = np.concatenate((last_block, features_offset))
            np.save(last_block_name, data)
            logger.info('job: %s - Update block: %s' % (current_job, last_block_name))

        # generate new block
        generate_list = upload_paths[block_offset:]
        if len(generate_list) > 0:
            for i in range(len(generate_list) // block_size + 1):
                master_data_path_current = generate_list[i * block_size: (i + 1) * block_size]
                feature_current = model.extract_feat_batch(master_data_path_current)
                block_path = os.path.join(_config.npy_path, 'block_%d.npy' % num_blocks)
                np.save(block_path, feature_current)
                logger.info('job: %s - Create new block: %s' % (current_job, block_path))
                num_blocks += 1
    except Exception as ex:
        logger.error('Start rollback job: %s - update block npy exception: %s' % (current_job, ex))
        rollback_update(num_blocks - 1, len(last_block))
        return RetCode.PROCESS_EXCEPTION
    return RetCode.PROCESS_SUCCESS

def remove_feature(arr_deleted):
    try:
        for block_idx in arr_deleted:
            # load npy
            block_name = os.path.join(CommonInfo.NPY_FOLDER, 'block_{0}.npy'.format(block_idx))
            block = np.load(block_name)

            # assign a empty npy
            for idx in arr_deleted[block_idx]:
                block[idx] = None

            # save, done
            np.save(block_name, block)
    except Exception as ex:
        logger.error('job: %s - remove exception: %s' % (current_job, ex))
        return RetCode.PROCESS_EXCEPTION
    return RetCode.PROCESS_SUCCESS

def update_index(uploads, removes):
    if len(uploads) > 0:
        ret = upload_file(uploads)
        if ret != RetCode.PROCESS_SUCCESS:
            return ret

    if len(removes) > 0:
        ret = remove_file(removes)
        if ret != RetCode.PROCESS_SUCCESS:
            return ret

    return RetCode.PROCESS_SUCCESS

def get_file_paths(jobContent):
    jobs = json.loads(jobContent)
    files_upload = []
    files_remove = []
    for i in range(len(jobs)):
        for key in jobs[i]:
            job_list = jobs[i][key].split(',')
            job_status = job_list[0]
            job_files = job_list[1:]

            if job_status == "ADD" or job_status == 'UP':
                files_upload.extend(job_files)
            elif job_status == "DEL":
                files_remove.extend(job_files)

    # add crop foreground image to add list
    crop_foreground_added = crop_foreground(files_upload)
    files_upload.extend(crop_foreground_added)

    # add rotate image to remove list
    reverse_lookup = {x:i for i, x in enumerate(master_data_paths)}
    for file in files_remove:
        file_rotate_name = '{}r{}'.format(os.path.splitext(os.path.basename(file))[0], os.path.splitext(os.path.basename(file))[1])
        file_rotate_path = os.path.join('rotate/', os.path.dirname(file), file_rotate_name)
        file_id = reverse_lookup.get(file_rotate_path, -1)
        if file_id != -1:
            files_remove.append(master_data_paths[file_id])

    # add crop foreground image to remove list
    for file in files_remove:
        file_crop_name = '{}c{}'.format(os.path.splitext(os.path.basename(file))[0], os.path.splitext(os.path.basename(file))[1])
        file_crop_path = os.path.join('crop_foreground/', os.path.dirname(file), file_crop_name)
        file_id = reverse_lookup.get(file_crop_path, -1)
        if file_id != -1:
            files_remove.append(master_data_paths[file_id])

    return files_upload, files_remove

def process_job(jobContent):
    # update current job to rollback info
    update_rollback_info(current_job)

    # backup npy before run update
    src_path = CommonInfo.ROLLBACK_NPY
    Path(src_path).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(src_path):
        file_path = os.path.join(src_path, filename)
        os.remove(file_path)

    pathlib.Path(src_path).mkdir(parents=True, exist_ok=True)
    num_blocks = len(os.listdir(_config.npy_path))
    src_block = os.path.join(_config.npy_path, 'block_{0}.npy'.format(num_blocks - 1))
    dest_block = os.path.join(src_path, 'block_{0}.npy'.format(num_blocks - 1))
    shutil.copy(src=src_block, dst=src_path)
    logger.info(f"Run backup {src_block} to {dest_block} success")

    # get file paths
    files_upload, files_remove = get_file_paths(jobContent)

    # gen PKL
    add_files, error_files = gen_plk(files_upload)
    if len(error_files) > 0:
        error_list.extend(error_files)

    if len(add_files) <= 0:
        return RetCode.PROCESS_ERROR

    # update features index
    ret = update_index(add_files, files_remove)
    if ret == RetCode.PROCESS_SUCCESS:
        # get old path
        sample_paths = []
        for file in files_remove:
            path = get_old_path(file)
            if path != "":
                sample_paths.append(path)
            else:
                sample_paths.append(file)

        # choices unit test files
        unitest_files = []
        if len(add_files) > 0:
            d = { 'status': 'upload', 'files': get_unit_test_files(add_files) }
            unitest_files.append(d)
        if len(sample_paths) > 0:
            d = { 'status': 'remove', 'files': get_unit_test_files(sample_paths) }
            unitest_files.append(d)

        # re-generate images list path
        with open(_config.img_list_path, 'w') as f:
            f.seek(0)
            for item in master_data_paths:
                f.write("%s\n" % item)
            f.truncate()

        # prepare sample
        prepare_sample(unitest_files)

        # add temp to redis
        query_key = '%s_npy_jobs' % settings.APP_DB_NAME
        web_redis.lpush(query_key, current_job)

        # save remove source list
        file_name = os.path.join(CommonInfo.ROLLBACK_DIR, f"removed_job_{current_job}.txt")
        if os.path.isfile(file_name):
            os.remove(file_name)
        with open(file_name, "w") as f:
            for file in sample_paths:
                f.write(f'{file}\n')

        # save remove pkl list
        file_name_pkl = os.path.join(CommonInfo.ROLLBACK_DIR, f"removed_pkl_job_{current_job}.txt")
        if os.path.isfile(file_name_pkl):
            os.remove(file_name_pkl)
        with open(file_name_pkl, "w") as f:
            for file in files_remove:
                f.write(f'{file}.pkl\n')

        # save rollback info
        save_rollback_info(current_job)

    return ret

def main():
    global current_job
    logger.info("Begin update npy")

    # re run job not completed
    # check_job_not_completed()

    # run new job
    pending_key = '%s_upload_pending' % settings.APP_DB_NAME
    while (web_redis.llen(pending_key) > 0):
        query_key = '%s_upload_jobs' % settings.APP_DB_NAME
        query_id = web_redis.lpop(query_key)
        if query_id is not None:
            query_id = query_id.decode('utf-8')

            # get job content
            job_key = "%s_%s_job" % (settings.APP_DB_NAME, str(query_id))
            current_job = query_id
            logger.info("job id: %s" % job_key)
            job_content = web_redis.get(job_key)
            if job_content is not None:
                job_content = job_content.decode('utf-8')
                ret = process_job(str(job_content))
                send_response(ret)
            else:
                logger.error("job: %s - job content is null" % current_job)
                send_response(RetCode.JOB_CONTENT_NULL)
        time.sleep(settings.CLIENT_SLEEP)
    logger.info('Daily update finished. Exit !')

def send_response(ret, msg=""):
    req = {}
    if ret == RetCode.JOB_CONTENT_NULL:
        req = { 'status': 1, 'message': 'Job content is null' }
    elif ret == RetCode.PROCESS_EXCEPTION:
        req = { 'status': 1, 'message': 'Job exception' }
    elif ret == RetCode.PROCESS_ERROR:
        req = { 'status': 1, 'message': 'Job failed' }
    elif ret == RetCode.UPLOAD_ALERT:
        ALERT_API = urljoin(settings.WEB_SERVER_HOST, "/api/mail-alert")
        req = { 'message': json.dumps(msg) }
        res = requests.post(ALERT_API, json=req)
        logger.info(res)
        return
    else:
        if len(error_list) > 0:
            req = { 'status': 2, 'message': ','.join(error_list) }
            del error_list[:]
        else:
            req = { 'status': 0, 'message': 'Step 1 successful' }

    # begin send request
    UPDATE_STATUS_API = "/api/upload_tasks/{0}".format(current_job)
    host_name =  urljoin(settings.WEB_SERVER_HOST, UPDATE_STATUS_API, current_job)
    logger.info('send response to %s: %s' % (host_name, json.dumps(req)))
    res = requests.post(host_name, json=req)
    r = res.json()
    if r['status'] == 1:
        logger.info("Job %s success" % current_job)

def gen_plk(files):
    add_files = []
    error_files = []
    for i in tqdm(range(len(files))):
        f = files[i]

        # create sub path
        sub_path = os.path.dirname(f)
        sub_path = os.path.join(_config.descs_db_path, sub_path)
        pathlib.Path(sub_path).mkdir(parents=True, exist_ok=True)

        f = f.replace('Lashinbang_data/', '')
        input_file = os.path.join(_config.img_db_path, f)
        output_file = os.path.join(_config.descs_db_path, f + ".pkl")

        if os.path.exists(output_file):
            n = Path(output_file).stat().st_size
            if n > 0:
                continue

        im = cv2.imread(input_file)
        if im is None:
            logger.error('job: %s - Cannot read file "%s"' % (current_job, input_file))
            error_files.append(f)
        else:
            save_features(im, output_file)
            add_files.append(f)

    return add_files, error_files

def get_unit_test_files(source):
    result = []
    if len(source) < 5:
        result.extend(source)
    else:
        result.extend(random.choices(source, k=5))
    return result

def prepare_sample(unitest_files):
    sample_path = os.path.join(CommonInfo.SAMPLE_DIR, str(current_job))
    img_list_file = os.path.join(CommonInfo.SAMPLE_LIST_DIR, f'list_origin_images_{str(current_job)}.txt')

    # prepare path
    pathlib.Path(sample_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(img_list_file)).mkdir(parents=True, exist_ok=True)

    # create sample for run unit test
    sample_cout = 0
    for item in unitest_files:
        for file in item['files']:
            if not os.path.isfile(os.path.join(_config.img_db_path, file)):
                continue

            # update list sample file
            with open(img_list_file, 'a+') as f:
                f.write("%s,%s \n" % (item['status'], file))

            if (any(file in err for err in error_list)):
                continue

            src_path = os.path.join(_config.img_db_path, file)
            dst_1_path = os.path.join(sample_path, '%d.jpg' % (sample_cout + 1))
            shutil.copy(src=src_path, dst=dst_1_path)
            dst_2_path = os.path.join(sample_path, '%d.jpg' % (sample_cout + 2))
            shutil.copy(src=src_path, dst=dst_2_path)
            sample_cout += 2

def check_job_not_completed():
    global current_job

    # create new if not exist
    if not os.path.isfile(CommonInfo.ROLLBACK_INFO):
        sub_path = os.path.dirname(CommonInfo.ROLLBACK_INFO)
        pathlib.Path(sub_path).mkdir(parents=True, exist_ok=True)
        save_rollback_info(0)

    # get rollback info
    with open(CommonInfo.ROLLBACK_INFO, 'r') as f:
        info = f.readline()
        info = info.split(',')
    last_queue_job = int(str(info[0]).strip())
    last_job = int(str(info[1]).strip())
    last_block_no = int(str(info[2]).strip())
    last_index = int(str(info[3]).strip())

    # start rollback
    if last_queue_job != last_job:
        logger.info('start rollback')
        files_remove = []
        job_key = f'{settings.APP_DB_NAME}_{str(last_queue_job)}_job'
        job_content = web_redis.get(job_key)
        if job_content is not None:
            job_content = job_content.decode('utf-8')
            _ , files_remove = get_file_paths(job_content)
        else:
            send_response(RetCode.JOB_CONTENT_NULL)
            update_rollback_info(last_job)
            return

        # rollback delete files
        if (len(files_remove) > 0):
            for file in files_remove:
                rollback_remove(file)

        # rollback upload file
        rollback_update(last_block_no, last_index)

        # re-run fail-job
        current_job = last_queue_job
        ret = process_job(str(job_content))
        send_response(ret)

def rollback_remove(file):
    ids = [i for i in range(len(master_data_paths)) if f'deleted({file})' in master_data_paths[i]]
    if len(ids) > 0:
        file_id = ids[0]
        block_idx = file_id // block_size
        idx = file_id - (block_idx * block_size)

        # rollback path file
        master_data_paths[file_id] = file

        # rollback data
        block_name = os.path.join(_config.npy_path, f'block_{block_idx}.npy')
        try:
            block = np.load(block_name)
            features_rollback = model.extract_feat_batch([os.path.join(_config.img_db_path, file)])
            block[idx] = features_rollback
            np.save(block_name, block)
        except Exception as ex:
            # copy backup npy
            src_path = CommonInfo.ROLLBACK_NPY
            src_file = os.path.join(src_path, 'block_{0}.npy'.format(block_idx))
            shutil.copy(src=src_file, dst=_config.npy_path)
            logger.info(f"Copy backup npy {src_file} to {_config.npy_path} success")

        logger.info("Rollback file deleted success: %s" % master_data_paths[file_id])

def rollback_update(b_no, i_no):
    num_npy = len(os.listdir(_config.npy_path))
    last_block_name = os.path.join(_config.npy_path, 'block_{0}.npy'.format(b_no))

    # rollback last block
    try:
        last_block = np.load(last_block_name)
        last_block = np.delete(last_block, range(i_no, len(last_block)), axis=0)
        np.save(last_block_name, last_block)
        logger.info(f"Save backup npy {last_block_name} success")
    except Exception as ex:
        # copy backup npy
        src_path = CommonInfo.ROLLBACK_NPY
        src_file = os.path.join(src_path, 'block_{0}.npy'.format(b_no))
        shutil.copy(src=src_file, dst=_config.npy_path)
        logger.info(f"Copy backup npy {src_file} to {_config.npy_path} success")

    # re-check rollback
    re_check = np.load(last_block_name)
    logger.info("Re-check: block no: %s - index: %s" % (b_no, len(re_check)))

    # delete new block
    for i in range(b_no + 1, num_npy):
        block_name = os.path.join(_config.npy_path, 'block_{0}.npy'.format(i))
        if os.path.isfile(block_name):
            os.remove(block_name)
        else:
            logger.error("Error: %s file not found" % block_name)

    # remove paths
    last_id = b_no * block_size + i_no
    rollback_paths = master_data_paths[:last_id]
    master_data_paths.clear()
    master_data_paths.extend(rollback_paths)
    with open(_config.img_list_path, 'w') as f:
        f.seek(0)
        for item in master_data_paths:
            f.write("%s\n" % item)
        f.truncate()
    logger.info('Rollback update success !')

def save_rollback_info(job_id):
    num_blocks = len(os.listdir(_config.npy_path))
    last_block_name = os.path.join(_config.npy_path, 'block_{0}.npy'.format(num_blocks - 1))
    last_block = np.load(last_block_name)
    queue_job = job_id
    with open(CommonInfo.ROLLBACK_INFO, 'w') as f:
        f.write(f'{queue_job},{job_id},{num_blocks - 1},{len(last_block)}')

def update_rollback_info(job_id):
    if not os.path.isfile(CommonInfo.ROLLBACK_INFO):
        sub_path = os.path.dirname(CommonInfo.ROLLBACK_INFO)
        pathlib.Path(sub_path).mkdir(parents=True, exist_ok=True)
        save_rollback_info(current_job)
        return

    # save rollback info
    with open(CommonInfo.ROLLBACK_INFO, 'r') as f:
        info = f.readline()
        info = info.split(',')
    last_queue_job = job_id
    last_job = int(str(info[1]).strip())
    last_block_no = int(str(info[2]).strip())
    last_index = int(str(info[3]).strip())
    with open(CommonInfo.ROLLBACK_INFO, 'w') as f:
        f.write(f'{last_queue_job},{last_job},{last_block_no},{last_index}')

def get_old_path(file_name):
    result = ""
    for k, v in old_master_paths.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if file_name == v:
            result = k
    return result

def load_config(db_path):
    logger.info('update database at: %s' % db_path)

    # read database config
    config_path = os.path.join(db_path, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    return Config(db_config)

if __name__ == "__main__":
    block_size = 10000
    current_job = 0
    error_list = []

    # Get app_code
    if (len(sys.argv) > 2):
        print('Command error: update_npy.py <APP_CODE>')
        exit(0)

    if len(sys.argv) > 1:
        app_code = sys.argv[-1]
    else:
        app_code = 'lashinbang'

    # initialize model
    model = CNN(useRmac=True, use_solar=True)
    sod_model = helpers.init_sod_model(base_path=os.path.dirname(os.getcwd()))

    # initialize config parser
    config = configparser.ConfigParser(inline_comment_prefixes='#')

    # initialize app logger
    logger = AppLogger('update')

    # load config from config file
    _config_path = settings.DATABASE_LIST_PATH
    with open(_config_path, 'r') as json_db:
        db_paths = json.load(json_db)

    if app_code not in db_paths:
        logger.error('app code is not support!')
        exit(0)

    db_path = db_paths[app_code]
    _config = load_config(db_path)

    # initialize master data
    with open(_config.img_list_path, 'r') as f:
        master_data_paths = f.readlines()
    master_data_paths = list(map(lambda x: x[:-1], master_data_paths))

    # initialize web redis
    web_redis = redis.StrictRedis(host=settings.WEB_REDIS_HOST,
                                port=settings.WEB_REDIS_PORT,
                                db=settings.WEB_REDIS_DB)

    # load old file list
    old_master_paths = {}
    with open(CommonInfo.IMAGE_ID_UPDATED_PATH, 'r') as f:
        old_paths = f.readlines()
    old_paths = list(map(lambda x: x[:-1], old_paths))
    for path in old_paths:
        tmp = path.split(',')
        old_master_paths[tmp[0]] = tmp[-1]

    # start script
    main()
