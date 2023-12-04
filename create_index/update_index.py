import os, sys

sys.path.append('../')

import configparser
import helpers
import csv
import shlex
import subprocess
import shutil
import random
import pathlib
import cv2
import time
import json
import requests
import settings
import redis
import faiss
import numpy as np
import getopt
import signal
import io

from pathlib import Path
from create_descs_db import save_features
from urllib.parse import urljoin
from enum import Enum
from logger import AppLogger
from extract_cnn import CNN
from tqdm import tqdm

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

def re_add_faiss_index():
    d = 2048
    num_block = len(os.listdir(_config.npy_path))

    try:
        sub_index = faiss.read_index(_config.trained_path)
        if _config.using_pca_key:
            pca_matrix = faiss.read_VectorTransform(_config.pca_path)
            index = faiss.IndexPreTransform(pca_matrix, sub_index)
        else:
            index = sub_index
        id_start = 0

        for block_id, i in tqdm(enumerate(range(num_block))):
            block_path = os.path.join(_config.npy_path, f"block_{i}.npy")
            arr = np.ascontiguousarray(np.load(block_path))
            id_end = id_start + arr.shape[0]
            ids = np.arange(id_start, id_end)
            if settings.CNN_IMAGE_FEATURE_FULL_SIZE > settings.CNN_IMAGE_FEATURE_REDUCED_SIZE:
                index.add(arr)
            else:
                index.add_with_ids(arr, ids)
            id_start = id_end

        faiss.write_index(sub_index, _config.index_path)
    except Exception as ex:
        logger.error('Update INDEX exception: %s' % ex)
        return RetCode.PROCESS_EXCEPTION
    return RetCode.PROCESS_SUCCESS


def process_job(job_ids):
    ret = re_add_faiss_index()
    if ret != RetCode.PROCESS_SUCCESS:
        return ret

    # run unit-test
    ret = do_run_test(job_ids)
    if ret != RetCode.PROCESS_SUCCESS:
        return ret

    # remove source image
    for job_id in job_ids:
        job_id = job_id.decode('utf-8')
        logger.info(f"Start remove job: {job_id}")
        file_name = os.path.join(CommonInfo.ROLLBACK_DIR, f"removed_job_{job_id}.txt")
        remove_file(file_name, _config.img_db_path)

        # remove pkl file
        file_pkl_name = os.path.join(CommonInfo.ROLLBACK_DIR, f"removed_pkl_job_{job_id}.txt")
        remove_file(file_pkl_name, _config.descs_db_path)

        # remove sudo file
        os.system(f"cd ../script && sudo python sudo_remove.py")

        # # save rollback info
        # save_rollback_info(job_id)

    return RetCode.PROCESS_SUCCESS


def remove_file(file_name, cwd):
    if not os.path.isfile(file_name):
        return

    delete_exceptions = []

    # load list remove file
    with open(file_name, 'r') as f:
        files_remove = f.readlines()
    files_remove = list(map(lambda x: x[:-1], files_remove))
    if len(files_remove) < 0:
        return

    for file_remove in files_remove:
        src_path = os.path.join(cwd, file_remove)

        # remove source file
        if os.path.isfile(src_path):
            try:
                logger.info("Removed: %s" % src_path)
                os.remove(src_path)
            except Exception as ex:
                delete_exceptions.append(src_path)
                logger.error('Delete file %s error with exception %s' %
                             (src_path, ex))
        else:
            continue

    if len(delete_exceptions) > 0:
        with open(os.path.join(CommonInfo.ROLLBACK_DIR, 'sudo_delete.txt'), 'a+') as f:
            for d_file in delete_exceptions:
                f.write(f'{d_file}\n')

    # remove list file
    os.remove(file_name)


def send_response(ret, job_ids, msg=""):
    req = {}
    if ret == RetCode.JOB_CONTENT_NULL:
        req = {'status': 1, 'message': 'Job content is null'}
    elif ret == RetCode.PROCESS_EXCEPTION:
        req = {'status': 1, 'message': 'Job exception'}
    elif ret == RetCode.PROCESS_ERROR:
        req = {'status': 1, 'message': 'Job failed'}
    elif ret == RetCode.UPLOAD_ALERT:
        ALERT_API = urljoin(settings.WEB_SERVER_HOST, "/api/mail-alert")
        req = {'message': json.dumps(msg)}
        logger.info('send alert to %s: %s' % (ALERT_API, json.dumps(req)))
        res = requests.post(ALERT_API, json=req)
        logger.info(res)
        return
    else:
        if len(error_list) > 0:
            req = {'status': 2, 'message': ','.join(error_list)}
            del error_list[:]
        else:
            req = {'status': 3, 'message': 'Job successful'}

    # begin send request
    for job_id in job_ids:
        UPDATE_STATUS_API = "/api/upload_tasks/{0}".format(job_id)
        host_name = urljoin(settings.WEB_SERVER_HOST,
                            UPDATE_STATUS_API, job_id)
        logger.info('send response to %s: %s' % (host_name, json.dumps(req)))
        res = requests.post(host_name, json=req)
        r = res.json()
        if r['status'] == 1:
            logger.info("Job %s success" % job_id)


def read_test_result(path, is_update):
    try:
        if (len(os.listdir(path)) <= 0):
            logger.error(f'is_update: {is_update} - Statistic file not found')
            return RetCode.PROCESS_EXCEPTION

        result_files = os.listdir(path)
        code = []
        response = []
        alerts = []
        for result_file in result_files:
            job_id = result_file.split('_')[0]
            with open(os.path.join(path, result_file), "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    # init statistic data
                    response.append(row[2])
                    code.append(row[3])

                    # alerts file error
                    if is_update:
                        if ("upload" in row[0]) and (row[3] == "-1"):
                            alerts.append(row[0].replace("/", "|"))
                        elif ("remove" in row[0]) and (row[3] == "0"):
                            alerts.append(row[0].replace("/", "|"))

            if len(alerts) > 0:
                d = { "job_id": job_id, "alerts": alerts }
                send_response(RetCode.UPLOAD_ALERT, d)

            if not is_update:
                logger.info("-----------CONFIRM DATA--------------")
            else:
                logger.info("-----------UPDATE TEST--------------")
            logger.info("Total: %s" % len(code))
            logger.info("First index true: %s" % sum(1 for i in [helpers.try_int(i) for i in code] if i == 0))
            logger.info("Other index true: %s" % sum(1 for i in [helpers.try_int(i) for i in code] if i > 0))
            logger.info("Top 5-9 true: %s" % sum(1 for i in [helpers.try_int(i) for i in code] if i > 4))
            logger.info("Not found: %s" % sum(1 for i in [helpers.try_int(i) for i in code] if i == -1))
            logger.info("Not matches: %s" % sum(1 for i in response if i == "[]"))
            os.remove(os.path.join(path, result_file))
    except Exception as ex:
        logger.error(f'is_update: {is_update} - Run unit test fail')
        return RetCode.PROCESS_EXCEPTION

    return RetCode.PROCESS_SUCCESS


def do_run_test(job_ids):
    for line in os.popen("ps ax | grep faiss_cluster.py | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)

    # re run faiss_cluster and run test
    pro = subprocess.Popen(["python", "-u", "faiss_cluster.py"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, cwd="..")
    with pro.stdout:
        for line in iter(pro.stdout.readline, b''):
            if ("ready" in line.decode("utf-8")):
                pathlib.Path("../request_multi_images/update_test/data/csv/").mkdir(parents=True, exist_ok=True)
                pathlib.Path("../request_multi_images/confirm_test/data/csv/").mkdir(parents=True, exist_ok=True)

                # start run update test
                logger.info("Run update test")
                for job_id in job_ids:
                    os.system(f"cd ../request_multi_images/update_test && python send_multi_requests.py {app_code} {int(job_id)}")

                # start run confirm data test
                logger.info("Run confirm data test")
                os.system(f"cd ../request_multi_images/confirm_test && python send_multi_requests.py {app_code}")

                # kill faiss_cluster
                os.kill(int(pro.pid), signal.SIGKILL)
                logger.info("faiss_cluster killed")

    # read confirm result csv
    ret = read_test_result("../request_multi_images/confirm_test/data/csv", is_update=False)
    if ret != RetCode.PROCESS_SUCCESS:
        return ret

    # read update test csv
    ret = read_test_result("../request_multi_images/update_test/data/csv", is_update=True)
    if ret != RetCode.PROCESS_SUCCESS:
        return ret

    return RetCode.PROCESS_SUCCESS


def save_rollback_info(job_id):
    num_blocks = len(os.listdir(_config.npy_path))
    last_block_name = os.path.join(
        _config.npy_path, 'block_{0}.npy'.format(num_blocks - 1))
    last_block = np.load(last_block_name)
    queue_job = job_id
    with open(CommonInfo.ROLLBACK_INFO, 'w') as f:
        f.write(f'{queue_job},{job_id},{num_blocks - 1},{len(last_block)}')


def load_config(db_path):
    logger.info('update database at: %s' % db_path)

    # read database config
    config_path = os.path.join(db_path, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    return Config(db_config)


def main():
    logger.info("Begin update index")

    # run new job
    query_key = '%s_npy_jobs' % settings.APP_DB_NAME
    job_count = web_redis.llen(query_key)
    if int(job_count) > 0:
        job_ids = helpers.multi_pop(web_redis, query_key, job_count)
        if (len(job_ids) > 0):
            ret = process_job(job_ids)
            send_response(ret, job_ids)
    else:
        logger.info('Daily update finished. Exit !')

if __name__ == "__main__":
    error_list = []

    # initialize app logger
    logger = AppLogger('update')

    # Get app_code
    if (len(sys.argv) > 2):
        logger.error('Command error: update_index.py <APP_CODE>')
        exit(0)

    if len(sys.argv) > 1:
        app_code = sys.argv[-1]
    else:
        app_code = 'lashinbang'

    # initialize config parser
    config = configparser.ConfigParser(inline_comment_prefixes='#')

    # load config from config file
    _config_path = settings.DATABASE_LIST_PATH
    with open(_config_path, 'r') as json_db:
        db_paths = json.load(json_db)

    if app_code not in db_paths:
        logger.error('app code is not support!')
        exit(0)

    db_path = db_paths[app_code]
    _config = load_config(db_path)

    # initialize web redis
    web_redis = redis.StrictRedis(host=settings.WEB_REDIS_HOST,
                                  port=settings.WEB_REDIS_PORT,
                                  db=settings.WEB_REDIS_DB)

    # get system arguments
    main()
