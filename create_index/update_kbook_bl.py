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

    if app_code != 'kbook':
      add_list.append(os.path.join(_config.img_db_path, file))
      master_data_paths.append(file)

  # crop image
  if app_code == 'kbook':
    _, pass_ids, _, err_ids = pre_process_fea.cropBookList(SOURCE_IMG_DIR, files, CROP_IMG_DIR)
    error_list.extend(err_ids)
    add_list.extend([os.path.join(CROP_IMG_DIR, f'images/{item}') for item in pass_ids])
    master_data_paths.extend([f'images/{item}' for item in pass_ids])

  # update feature
  try:
    # fill to last block
    block_offset = block_size - len(last_block)
    append_list = add_list[:block_offset]
    if len(append_list) > 0:
      features_offset = model.extract_feat_batch(append_list)
      data = np.concatenate((last_block, features_offset))
      np.save(last_block_name, data)
      logger.info('job: %s - Update block: %s' % (current_job, last_block_name))

    # generate new block
    generate_list = add_list[block_offset:]
    if len(generate_list) > 0:
      for i in range(len(generate_list) // block_size + 1):
        master_data_path_current = generate_list[i * block_size: (i + 1) * block_size]
        feature_current = model.extract_feat_batch(master_data_path_current)
        block_path = os.path.join(_config.npy_path, 'block_%d.npy' % num_blocks)
        np.save(block_path, feature_current)
        logger.info('job: %s - Create new block: %s' % (current_job, block_path))
        num_blocks += 1

  except Exception as ex:
    logger.error('job: %s - update block npy exception: %s' % (current_job, ex))
    return RetCode.PROCESS_EXCEPTION

  return RetCode.PROCESS_SUCCESS

def remove_feature(idx, block_idx):
  try:
    # reload npy contain image
    block_name = os.path.join(_config.npy_path, 'block_{0}.npy'.format(block_idx))
    block = np.load(block_name)

    # assign to empty feature
    block[idx] = None

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
    print('re-generate images list path')
    with open(_config.img_list_path, 'w') as f:
      f.seek(0)
      for item in master_data_paths:
        f.write("%s\n" % item)
      f.truncate()

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
  # get file paths
  error_list.clear()
  add_list.clear()
  files_upload, files_remove = get_file_paths(jobContent)

  # update features index
  ret = update_index(files_upload, files_remove)
  if ret == RetCode.PROCESS_SUCCESS:
    ret = re_add_faiss_index()

  if ret == RetCode.PROCESS_SUCCESS:
    # remove file
    if len(files_remove) > 0:
      for file_remove in files_remove:
        src_path = os.path.join(_config.img_db_path, file_remove)
        if (os.path.isfile(src_path)):
          try:
            logger.info("Removed: %s" % src_path)
            os.remove(src_path)
          except Exception as ex:
            logger.error('Delete file %s error with exception %s' % (src_path, ex))

  return ret

def main():
  global current_job
  logger.info("Begin daily update")

  # run new job
  while True:
    # get job id
    query_key = '%s_upload_%s_jobs' % (settings.APP_DB_NAME, app_code)
    # logger.info(f'Queue at {query_key}')

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
      logger.info(f'Job: {str(query_id)} - Begin at: {t0} - Finished at: {t1}')

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
  UPDATE_STATUS_API = f"/api/{app_code}/upload_tasks/{current_job}"
  host_name =  urljoin(settings.WEB_SERVER_HOST, UPDATE_STATUS_API, current_job)
  logger.info('send response to %s: %s' % (host_name, json.dumps(req)))
  res = requests.post(host_name, json=req)
  r = res.json()
  if r['status'] == 1:
      logger.info("Job %s success" % current_job)

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
    self.npy_path = NPY_FOLDER
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
    logger.error('Command error: update_one_step.py <APP_CODE>')
    exit(0)

  if len(sys.argv) > 1:
    app_code = sys.argv[-1]
  else:
    app_code = 'kbook'

  # load global config
  TRAINED_FILE = 'trained.index'
  if app_code == 'kbook':
    NPY_FOLDER = '/home/ubuntu/efs/npy_kbook'
    SOURCE_IMG_DIR = '/home/ubuntu/efs/kbook-images-source'
    CROP_IMG_DIR = '/home/ubuntu/efs/suruga/images'
  else:
    NPY_FOLDER = '/home/ubuntu/efs/npy_kbook_bl'

  print(f'app_code: {app_code} - npy: {NPY_FOLDER}')

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