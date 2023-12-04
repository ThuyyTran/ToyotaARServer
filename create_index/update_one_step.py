import sys, getopt, signal, io

sys.path.insert(1, '../')

import numpy as np
import os
import faiss
import redis
import settings
import requests
import json
import time, datetime
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
from create_index.create_descs_db import save_features
from pathlib import Path
from pre_process_kbook import ImagePreProcess

class RetCode(Enum):
    NONE = 0
    PROCESS_SUCCESS = 1
    PROCESS_ERROR = 2
    PROCESS_EXCEPTION = 3
    JOB_CONTENT_NULL = 4,
    UPLOAD_ALERT = 5,

NPY_FOLDER = '/home/ubuntu/efs/npy_kbook'
TRAINED_FILE = 'trained.index'
ROLLBACK_INFO = '../data_kbook/rollback/rollback.txt'
SOURCE_IMG_DIR = '/home/ubuntu/efs/kbook-images-source'
CROP_IMG_DIR = '/home/ubuntu/efs/suruga/images'

def remove_file(files):
    # find in list
    for file in files:
        if ('_S' in file):
            continue

        result = [i for i in range(len(master_data_paths)) if file in master_data_paths[i]]
        if len(result) > 0:
            file_id = result[0]
        else:
            logger.error('job: %s - file %s doesnt exist in list path' % (current_job, file))
            error_list.append(file)
            continue

        # find idx and block_idx
        logger.info("Start remove file %s" % file)
        block_idx = file_id // block_size
        idx = file_id - (block_idx * block_size)

        # set path is null
        master_data_paths[file_id] = f'deleted({file})'

        # re-run generate feature
        ret = remove_feature(idx, block_idx)
        if ret == RetCode.PROCESS_EXCEPTION:
            error_list.append(file)
    return RetCode.PROCESS_SUCCESS

def upload_file(files):
    num_blocks = len(os.listdir(_config.npy_path))
    last_block_name = os.path.join(_config.npy_path, 'block_{0}.npy'.format(num_blocks - 1))
    last_block = np.load(last_block_name)

    for file in files:
        result = [i for i in range(len(master_data_paths)) if file in master_data_paths[i]]
        if len(result) > 0:
            continue

    # crop image
    _, pass_ids, _, err_ids = pre_process_fea.cropBookList(SOURCE_IMG_DIR, files, CROP_IMG_DIR)
    error_list.extend(err_ids)
    add_list.extend([f'images/{item}' for item in pass_ids])

    # update feature
    try:
        # fill to last block
        block_offset = block_size - len(last_block)
        append_list = add_list[:block_offset]
        if len(append_list) > 0:
            features_offset = model.extract_feat_batch([os.path.join(CROP_IMG_DIR, item) for item in append_list])
            data = np.concatenate((last_block, features_offset))
            np.save(last_block_name, data)
            logger.info('job: %s - Update block: %s' % (current_job, last_block_name))

        # generate new block
        generate_list = add_list[block_offset:]
        if len(generate_list) > 0:
            for i in range(len(generate_list) // block_size + 1):
                master_data_path_current = generate_list[i * block_size: (i + 1) * block_size]
                feature_current = model.extract_feat_batch([os.path.join(CROP_IMG_DIR, item) for item in master_data_path_current])
                block_path = os.path.join(_config.npy_path, 'block_%d.npy' % num_blocks)
                np.save(block_path, feature_current)
                logger.info('job: %s - Create new block: %s' % (current_job, block_path))
                num_blocks += 1

        # # gen pkl
        # ret = gen_pkl(add_list)
        # if ret != RetCode.PROCESS_SUCCESS:
        #     return ret

    except Exception as ex:
        logger.error('job: %s - update block npy exception: %s - Start rollback' % (current_job, ex))
        rollback_update(num_blocks - 1, len(last_block))
        return RetCode.PROCESS_EXCEPTION
    return RetCode.PROCESS_SUCCESS

def remove_feature(idx, block_idx):
    try:
        # reload npy contain image
        block_name = os.path.join(_config.npy_path, 'block_{0}.npy'.format(block_idx))
        block = np.load(block_name)

        # assign to empty feature
        block[idx] = np.empty(shape=(2048,))

        # re-save block file
        np.save(block_name, block)
    except Exception as ex:
        logger.error('job: %s - remove exception: %s' % (current_job, ex))
        return RetCode.PROCESS_EXCEPTION
    return RetCode.PROCESS_SUCCESS

def re_add_faiss_index():
    nlist = 1000
    d = 2048
    num_block = len(os.listdir(_config.npy_path))

    try:
        # re-generate images list path
        with open(_config.img_list_path, 'a') as f:
            for item in add_list:
                f.write("%s\n" % item)

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
        logger.error('job: %s - update INDEX exception: %s' % (current_job, ex))
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

    return files_upload, files_remove

def process_job(jobContent):
    # # update current job to rollback info
    # update_rollback_info(current_job)

    # # backup npy before run update
    # src_path = BACKUP_NPY
    # for filename in os.listdir(src_path):
    #     file_path = os.path.join(src_path, filename)
    #     os.remove(file_path)

    # pathlib.Path(src_path).mkdir(parents=True, exist_ok=True)
    # num_blocks = len(os.listdir(_config.npy_path))
    # src_block = os.path.join(_config.npy_path, 'block_{0}.npy'.format(num_blocks - 1))
    # dest_block = os.path.join(src_path, 'block_{0}.npy'.format(num_blocks - 1))
    # shutil.copy(src=src_block, dst=src_path)
    # logger.info(f"Run backup {src_block} to {dest_block} success")

    # get file paths
    error_list.clear()
    add_list.clear()
    files_upload, files_remove = get_file_paths(jobContent)

    # update features index
    ret = update_index(files_upload, files_remove)
    if ret == RetCode.PROCESS_SUCCESS:
        ret = re_add_faiss_index()

        # if ret == RetCode.PROCESS_SUCCESS:
        #     unitest_files = []
        #     # choices unit test files
        #     if len(add_list) > 0:
        #         d = { 'status': 'upload', 'files': get_unit_test_files(add_list) }
        #         unitest_files.append(d)
        #     if len(files_remove) > 0:
        #         d = { 'status': 'remove', 'files': get_unit_test_files(files_remove) }
        #         unitest_files.append(d)

        #     # run unit test
        #     run_unit_test(unitest_files)

        #     # remove file
        #     if len(files_remove) > 0:
        #         for file_remove in files_remove:
        #             src_path = os.path.join(_config.img_db_path, file_remove)
        #             pkl_path = os.path.join(_config.descs_db_path, file_remove + ".pkl")
        #             if (os.path.isfile(src_path)):
        #                 try:
        #                     logger.info("Removed: %s" % src_path)
        #                     os.remove(src_path)
        #                     logger.info("Removed pkl: %s" % pkl_path)
        #                     os.remove(pkl_path)
        #                 except Exception as ex:
        #                     logger.error('Delete file %s error with exception %s' % (src_path, ex))

            # # save rollback info
            # save_rollback_info(current_job)

    return ret

def main():
    global current_job
    logger.info("Begin daily update")

    # # re run job not completed
    # check_job_not_completed()

    # run new job
    while True:

        # get job id
        query_key = '%s_upload_kbook_jobs' % settings.APP_DB_NAME
        query_id = web_redis.lpop(query_key)
        if query_id is not None:
            t0 = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

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

            t1 = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f'Job: {str(query_id)} - Begin at: {t0} - Finished at: {t1}')

        else:
            logger.info('Daily update finished. Exit !')
            break
        time.sleep(settings.CLIENT_SLEEP)

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
            req = { 'status': 0, 'message': 'Job successful' }

    # begin send request
    UPDATE_STATUS_API = "/api/kbook/upload_tasks/{0}".format(current_job)
    host_name =  urljoin(settings.WEB_SERVER_HOST, UPDATE_STATUS_API, current_job)
    logger.info('send response to %s: %s' % (host_name, json.dumps(req)))
    res = requests.post(host_name, json=req)
    r = res.json()
    if r['status'] == 1:
        logger.info("Job %s success" % current_job)

def gen_pkl(files):
    ret = RetCode.PROCESS_SUCCESS
    for i in tqdm(range(len(files))):
        f = files[i]
        f1 = f

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
            ret = RetCode.PROCESS_ERROR
            break
        save_features(im, output_file)

    return ret

def get_unit_test_files(source):
    result = []
    if len(source) < 5:
        result.extend(source)
    else:
        result.extend(random.choices(source, k=5))
    return result

def prepare_sample(unitest_files):
    sample_path = f'../request_multi_images/update_test/data/test_images_{str(app_code)}_qa/{current_job}'
    img_list_path = '../request_multi_images/update_test/data/txt/'

    # prepare path
    pathlib.Path(sample_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(img_list_path).mkdir(parents=True, exist_ok=True)

    sample_cout = 0

    # delete all sample files
    for filename in os.listdir(sample_path):
        file_path = os.path.join(sample_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error('Failed to delete %s. Reason: %s' % (file_path, e))

    # copy images to sample folder
    img_list_file = os.path.join(img_list_path, f'list_origin_images_{str(app_code)}_{str(current_job)}.txt')
    if os.path.isfile(img_list_file):
        os.remove(img_list_file)

    # create sample for run unit test
    for item in unitest_files:
        for file in item['files']:
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

def read_test_result(path, is_update):
    result_file = os.path.join(path, f'{str(app_code)}_{str(current_job)}_report_qa.csv')
    if os.path.isfile(result_file):
        code = []
        response = []
        alerts = []
        with open(result_file, "r") as f:
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
            d = { "job_id": current_job, "alerts": alerts }
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
        os.remove(os.path.join(result_file))

def do_run_test(unitest_files):
    for line in os.popen("ps ax | grep search_cluster.py | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)

    # re run search_cluster and run test
    pro = subprocess.Popen(["python", "-u", "search_cluster.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, cwd="..")
    with pro.stdout:
        for line in iter(pro.stdout.readline, b''):
            if ("ready" in line.decode("utf-8")):
                pathlib.Path("../request_multi_images/update_test/data/csv/").mkdir(parents=True, exist_ok=True)
                pathlib.Path("../request_multi_images/confirm_test/data/csv/").mkdir(parents=True, exist_ok=True)

                # start run update test
                logger.info("Run update test")
                os.system(f"cd ../request_multi_images/update_test && python send_multi_requests.py {app_code} {current_job}")

                # kill search_cluster
                os.kill(int(pro.pid), signal.SIGKILL)
                logger.info("Search_cluster killed")

    # read update test csv
    read_test_result("../request_multi_images/update_test/data/csv", is_update=True)

def run_unit_test(unitest_files):
    if len(unitest_files) <= 0:
        logger.error('Unit test failed')
        return

    # prepare sample
    prepare_sample(unitest_files)

    # start test
    do_run_test(unitest_files)

def check_job_not_completed():
    global current_job

    # create new if not exist
    if not os.path.isfile(ROLLBACK_INFO):
        sub_path = os.path.dirname(ROLLBACK_INFO)
        pathlib.Path(sub_path).mkdir(parents=True, exist_ok=True)
        save_rollback_info(0)

    # get rollback info
    with open(ROLLBACK_INFO, 'r') as f:
        info = f.readline()
        info = info.split(',')
    last_queue_job = int(str(info[0]).strip())
    last_job = int(str(info[1]).strip())
    last_block_no = int(str(info[2]).strip())
    last_index = int(str(info[3]).strip())

    # start rollback
    if last_queue_job != last_job:
        files_remove = []
        job_key = f'{settings.APP_DB_NAME}_{str(last_queue_job)}_job'
        job_content = web_redis.get(job_key)
        if job_content is not None:
            job_content = job_content.decode('utf-8')
            _ , files_remove = get_file_paths(job_content)

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
            src_path = BACKUP_NPY
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
        src_path = BACKUP_NPY
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
    with open(ROLLBACK_INFO, 'w') as f:
        f.write(f'{queue_job},{job_id},{num_blocks - 1},{len(last_block)}')

def update_rollback_info(job_id):
    # save rollback info
    with open(ROLLBACK_INFO, 'r') as f:
        info = f.readline()
        info = info.split(',')
    last_queue_job = job_id
    last_job = int(str(info[1]).strip())
    last_block_no = int(str(info[2]).strip())
    last_index = int(str(info[3]).strip())
    with open(ROLLBACK_INFO, 'w') as f:
        f.write(f'{last_queue_job},{last_job},{last_block_no},{last_index}')

def load_config(db_path):
    logger.info('update database at: %s' % db_path)

    # read database config
    config_path = os.path.join(db_path, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    return Config(db_config)

class Config:
    def __init__(self, db_config):
        self.index_path = os.path.join(db_path, db_config[settings.INDEX_FILE_KEY])
        self.pca_path = os.path.join(db_path, db_config[settings.PCA_MATRIX_FILE_KEY])
        self.npy_path = os.path.join(db_path, NPY_FOLDER)
        self.trained_path = os.path.join(db_path, TRAINED_FILE)
        self.img_list_path = os.path.join(db_path, db_config[settings.IMG_LIST_FILE_KEY])
        self.img_db_path = db_config[settings.MTC_IMAGE_DB_FOLDER_KEY]
        self.descs_db_path = db_config[settings.MTC_DESCS_DB_FOLDER_KEY]
        self.using_pca_key = db_config.getboolean(settings.CNN_IMAGE_FEATURE_USING_PCA_KEY)

if __name__ == "__main__":
    block_size = 10000
    current_job = 0
    error_list = []
    add_list = []

    # Get app_code
    if (len(sys.argv) > 2):
        logger.error('Command error: update_index.py <APP_CODE>')
        exit(0)

    if len(sys.argv) > 1:
        app_code = sys.argv[-1]
    else:
        app_code = 'kbook'

    # initialize model
    model = CNN(useRmac=True)
    pre_process_fea = ImagePreProcess()

    # initialize config parser
    config = configparser.ConfigParser(inline_comment_prefixes='#')

    # initialize app logger
    logger = AppLogger()

    # load config from config file
    _config_path = settings.DATABASE_LIST_PATH
    with open(_config_path, 'r') as json_db:
        db_paths = json.load(json_db)

    if app_code not in db_paths:
        logger.error('app code is not support!')
        exit(0)

    db_path = db_paths[app_code]
    _config = load_config(db_path)
    print('_config.img_db_path: ', _config.img_db_path)

    # initialize master data
    with open(_config.img_list_path, 'r') as f:
        master_data_paths = f.readlines()
    master_data_paths = list(map(lambda x: x[:-1], master_data_paths))

    # initialize web redis
    web_redis = redis.StrictRedis(host=settings.WEB_REDIS_HOST,
                                port=settings.WEB_REDIS_PORT,
                                db=settings.WEB_REDIS_DB)

    # get system arguments
    main()