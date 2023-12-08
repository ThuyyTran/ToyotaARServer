import os
import warnings
import pickle
import socket
import time
import json
import configparser
import numpy as np
import cv2
import requests
import io
import redis
import uuid
import base64
import helpers
import settings

from urllib.parse import urljoin
from flask import Flask, request, jsonify, abort, render_template, send_from_directory
from flask_cors import CORS, cross_origin
from PIL import Image
from logger import AppLogger
from werkzeug.exceptions import HTTPException, default_exceptions
from pre_process_kbook import ImagePreProcess
from datetime import date, datetime
from pytz import timezone, utc
from pathlib import Path
from extract_cnn import CNN
warnings.filterwarnings("ignore")

time_zone = timezone('Asia/Tokyo')
# initialize Flask application
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__,
  static_folder=settings.STATIC_FOLDER,
  static_url_path="/static",
  template_folder=os.path.dirname(__file__)
)

# initialize object
image_process = ImagePreProcess()
logger = AppLogger()
config = configparser.ConfigParser(inline_comment_prefixes='#')
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

# list database supports
with open(settings.DATABASE_LIST_PATH, 'r') as json_db:
    db_support = json.load(json_db)
multi_db_obj_arr = {}
for k, v in db_support.items():
    config_path = os.path.join(v, 'config.ini')
    config.read(config_path)
    db_config = config[settings.DATABASE_KEY]
    image_dir = db_config[settings.MTC_IMAGE_DB_FOLDER_KEY]
    begin_time = '8:00' if settings.TIME_BEGIN not in db_config else db_config[settings.TIME_BEGIN]
    end_time = '21:30' if settings.TIME_END not in db_config else db_config[settings.TIME_END]
    matching_config = True if settings.MATHCHING_CONFIG not in db_config else db_config.getboolean(settings.MATHCHING_CONFIG)
    api_key = '' if settings.API_KEY not in db_config else db_config[settings.API_KEY]

    multi_db_obj_arr[k] = { 'image_dir': image_dir, 'begin_time': begin_time, 'end_time': end_time, 'matching_config': matching_config, 'api_key': api_key }
model1 = CNN(useRmac=True, use_solar=False)
model2 = CNN(useRmac=True, use_solar=True)
clipSeg_processor,clipSeg_model = helpers.init_ClipSegModel()
def isServerOnline(app_code):
    """ Check server online by app code """
    db_selected = multi_db_obj_arr[app_code]
    current_time = datetime.now(tz=time_zone).time()
    time_begin = datetime.strptime(db_selected['begin_time'], '%H:%M').time()
    time_end = datetime.strptime(db_selected['end_time'], '%H:%M').time()

    time_update_begin = datetime.strptime('21:45', '%H:%M').time()
    time_update_end = datetime.strptime('22:00', '%H:%M').time()

    return (time_begin <= current_time and current_time <= time_end) or (time_update_begin <= current_time and current_time <= time_update_end)

def make_tree(path):
  tree = dict(name=os.path.basename(path), children=[])
  try:
    lst = sorted(Path(path).iterdir(), key=os.path.getmtime, reverse=True)
  except OSError:
    pass #ignore errors
  else:
    for fn in lst:
      if os.path.isdir(fn):
        tree['children'].append(make_tree(fn))
      else:
        tree['children'].append({
          'name': os.path.basename(fn),
          'created_date': datetime.strptime(time.ctime(os.path.getctime(fn)), '%a %b %d %H:%M:%S %Y').replace(tzinfo=utc).astimezone(timezone('Asia/Tokyo')),
        })

  return tree
def preProcessText(input_string):
    formatted_string = input_string.replace(" ", "_")
    formatted_string = ''.join(char for char in formatted_string if char.isalnum() or char == '_')
    return formatted_string
@app.route("/online", methods=['GET'])
def homepage():
    accepted = False
    app_code = request.args.get(settings.APP_CODE_KEY)
    if not app_code:
        accepted = True
    else:
        api_key = multi_db_obj_arr[app_code]['api_key']
        if api_key != '':
            header = request.headers
            if header and header.get('x-api-key') == multi_db_obj_arr[app_code]['api_key']:
                accepted = True
        else:
            accepted = True

    if accepted:
        return "Lashinbang Homepage!"
    else:
        abort(401)

@app.route("/videos", methods=['GET'])
def static_videos():
  type = request.args.get('type')
  key = request.args.get('key')
  if not type or not key or type != 'revamp' or key != multi_db_obj_arr['revamp']['api_key']:
    abort(404)

  return render_template("videos.html", tree=make_tree(settings.STATIC_FOLDER))

@app.route('/update_price', methods=['GET', 'POST'])
def update_price():
    json_response = {}
    if request.method == 'POST':
        # get product id, price
        product_id = request.form.get(settings.ID_KEY)
        sale_price = request.form.get(settings.SALE_PRICE)
        purchase_price = request.form.get(settings.PURCHASE_PRICE)
        if (sale_price is None) or (purchase_price is None) or (product_id is None):
            abort(400)

        # get app code
        app_code = request.form.get(settings.APP_CODE_KEY)
        if app_code is None:
            app_code = 'kbook'
        if app_code != 'kbook':
            error_msg = {'message': 'This app is not support', 'status': 2}
            return json.dumps(error_msg, ensure_ascii=False).encode("utf8"), 400

        try:
            # Check product id exists
            product_img_path = os.path.join(multi_db_obj_arr[app_code]['image_dir'], 'images', product_id)
            if not os.path.exists(product_img_path):
                error_msg = {'message': f'{product_id} not found', 'status': 2}
                return json.dumps(error_msg, ensure_ascii=False).encode("utf8"), 400

            # save new price
            logger.info(f'Update price of product: {product_id} with sale_price {sale_price} and purchase_price {purchase_price}')
            image_price_raw = image_process.updatePrice(sale_price, purchase_price, fontpath='/home/ubuntu/Lashinbang-webserver/HGrPrE.ttc')
            _, price_image = cv2.imencode('.jpg', image_price_raw)

            # save thumbnail
            img_raw = cv2.imread(product_img_path)
            _, thumbnail_raw = image_process.getThumbnailsFromImage(img_raw, image_price_raw)
            _, thumbnail_image = cv2.imencode('.jpg', thumbnail_raw)

            # DEBUG
            # img_home_dir = Path(multi_db_obj_arr[app_code]['image_dir']).parent
            # product_id_updated = product_id.replace('.', f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.")
            # image_price_path = os.path.join(img_home_dir, 'price', product_id_updated)
            # cv2.imwrite(image_price_path, image_price_raw)

            # send update price info
            payload_file = { 'price_image': price_image.tobytes(), 'thumbnail_image': thumbnail_image.tobytes() }
            payload_form = { 'id': product_id, 'sell_price': sale_price, 'buy_price': purchase_price, 'token': settings.KBOOK_UPDATE_PRICE_TOKEN }
            res = requests.post(urljoin(settings.WEB_SERVER_HOST, "/api/kbook-update-price"), files=payload_file, data=payload_form)
            json_response = res.json()
        except ValueError as err:
            json_response["status"] = 0
            json_response["message"] = err
            logger.error(err)

        result = json.dumps(json_response, ensure_ascii=False).encode("utf8")
        return result

@app.route('/predict', methods=['GET', 'POST'])
def search():
    json_response = {}
    max_request = round(settings.BATCH_SIZE * 1.5)
    return_code = 0

    if request.method == 'POST':
        # request time out
        time_out = 0

        # app_code
        app_code = request.form.get(settings.APP_CODE_KEY)
        if app_code is None:
            app_code = 'lashinbang'
        if app_code not in db_support:
            error_msg = {'message': 'This app is not support', 'status': 2}
            return json.dumps(error_msg, ensure_ascii=False).encode("utf8"), 400

        # check server online
        if not isServerOnline(app_code):
            error_msg = {'message': 'Server not online at this time', 'status': 'off'}
            return json.dumps(error_msg, ensure_ascii=False).encode("utf8"), 400

        # validate request
        api_key = multi_db_obj_arr[app_code]['api_key']
        if api_key != '':
            header = request.headers
            if not header or header.get('x-api-key') != multi_db_obj_arr[app_code]['api_key']:
                abort(401)

        # Get file data
        if request.files.get(settings.FILE_KEY):
            time_out = time.time()
            img_raw_data = request.files.get(settings.FILE_KEY).read()
            image = Image.open(io.BytesIO(img_raw_data)).convert('RGB')
        elif request.form.get(settings.FILE_KEY):
            time_out = time.time()
            img_base64_str = request.form.get(settings.FILE_KEY)
            img_raw_data = base64.b64decode(img_base64_str)
            image = Image.open(io.BytesIO(img_raw_data))
        else:
            abort(400)

        pre_computed_features = None
        # Get pre-computed features
        if request.files.get(settings.CNN_PRE_COMPUTED_KEY) is not None:
            s = request.files.get(settings.CNN_PRE_COMPUTED_KEY).read()
            logger.info("features' length : %s", len(s))
            feature_length = 2048
            if len(s) == feature_length * 8:
                 _dtype = np.float64
            elif len(s) == feature_length * 4:
                 _dtype = np.float32
            elif len(s) == feature_length:
                 _dtype = np.uint8
            else:
                 logger.error("Incorrect features. Size = %s", len(s))
                 abort(400)
            pre_computed_features = np.frombuffer(s, dtype=_dtype)
            if not (_dtype == np.float32):
                pre_computed_features = np.float32(pre_computed_features)
            logger.info("Pre-computed CNN features are available")

        # Get category
        categories = request.form.get('category') or ''

        # Begin search image
        platform = request.user_agent.platform
        browser = request.user_agent.browser
        logger.info('Request from %s platform - %s browser' %
                    (platform, browser))

        # get shopid, if none return 400
        shopid = request.form.get(settings.SHOP_ID_KEY)
        if shopid is None:
            shopid = "shop_id_default"

        # handle too many requests
        if db.llen(settings.IMAGE_QUEUE) > max_request:
            logger.info('many')
            logger.error("Too Many Requests: %s from %s with shop_id %s" % (db.llen(
                settings.IMAGE_QUEUE), request.remote_addr, shopid))
            abort(429)

        # get nt_retr
        nr_retr = settings.NR_RETR
        if request.form.get(settings.NR_RETR_KEY) is not None:
            nr_retr = helpers.try_int(request.form.get(settings.NR_RETR_KEY))
            if nr_retr > settings.MAX_NR_RETR:
                nr_retr = settings.MAX_NR_RETR

        # save request to cache db
        img_id = str(uuid.uuid4())
        logger.info('Request image with id: %s - from: %s - shopid: %s - app_code: %s - categories: %s' %
                    (img_id, request.remote_addr, shopid, app_code, categories.encode('utf-8')))

        # Crop image in webapp
        if browser is not None:
            left, top, right, bottom = helpers.find_center(image.size)
            image = image.crop((left, top, right, bottom))

        # Save input
        if settings.CBIR_SAVE_QUERY_IMAGES:
            path_to_query_image = img_id + '.jpg'
            path_to_query_image = os.path.join(
                settings.QUERY_IMAGE_FOLDER, path_to_query_image)
            try:
                image.save(path_to_query_image)
            except IOError:
                logger.error(
                    "Cannot save the query image to file '%s'" % path_to_query_image)

        # Create search request
        search_image = np.array(image)
        img_str = helpers.base64_encode_image(search_image)

        if pre_computed_features is not None:
            pre_computed_features_str = helpers.base64_encode_image(pre_computed_features)
        else:
            pre_computed_features_str = ''

        req_obj = {settings.ID_KEY: img_id,
                   settings.IMAGE_WIDTH_KEY: image.width,
                   settings.IMAGE_HEIGHT_KEY: image.height,
                   settings.CNN_PRE_COMPUTED_KEY: pre_computed_features_str,
                   settings.NR_RETR_KEY: nr_retr,
                   settings.IMAGE_KEY: img_str,
                   settings.APP_CODE_KEY: app_code,
                   settings.CATEGORIES_KEY: categories
                   }
        db.rpush(settings.IMAGE_QUEUE, json.dumps(req_obj))

        logger.info("Save %s image to %s success" %
                    (img_id, settings.IMAGE_QUEUE))

        # prepare matching image
        image = image.convert('L')
        match_image, w, h = helpers.resize(
            image, settings.MTC_IMAGE_WIDTH, settings.MTC_IMAGE_HEIGHT, True)

        # received searching & matching reponses
        state = 0  # 0 - searching | 1 - matching
        while True:
            run_time = int(round((time.time() - time_out) * 1000))
            if run_time >= settings.REQ_TIME_OUT:
                logger.error("ALERT: Request %s time out" % img_id)
                abort(408)

            # handler searching responses
            if state == 0:
                output = db.get(img_id)
                if output is not None:
                    db.delete(img_id)
                    output = output.decode("utf-8")
                    search_results = []
                    search_results.extend(json.loads(output)['ResultMatch'])
                    if len(search_results) > 0:
                        json_response["type"] = "searching"
                        json_response['matched_files'] = search_results
                        json_response["status"] = 0
                        json_response["message"] = "successful"
                        return_code = 200
                    else:
                        json_response["status"] = 1
                        json_response["message"] = "No matches found"
                        return_code = 404
                    break
        if not os.path.exists(os.path.join(settings.SAVE_LOG_RESULT_URL,img_id+'.json')):
            with open(os.path.join(settings.SAVE_LOG_RESULT_URL,img_id+'.json'), 'w') as fp:
                json.dump(json_response, fp,indent=4)
        else:
            with open(os.path.join(settings.SAVE_LOG_RESULT_URL,img_id+'_new.json'), 'w') as fp:
                json.dump(json_response, fp,indent=4)
    result = json.dumps(json_response, ensure_ascii=False).encode("utf8")
    logger.info('response: %s, %s, %s, %s' % (img_id, shopid, app_code, result))
    return result, return_code
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     json_response = {}
#     max_request = round(settings.BATCH_SIZE * 1.5)
#     return_code = 0

#     if request.method == 'POST':
#         # request time out
#         time_out = 0

#         # app_code
#         app_code = request.form.get(settings.APP_CODE_KEY)
#         if app_code is None:
#             app_code = 'lashinbang'
#         if app_code not in db_support:
#             error_msg = {'message': 'This app is not support', 'status': 2}
#             return json.dumps(error_msg, ensure_ascii=False).encode("utf8"), 400

#         # check server online
#         if not isServerOnline(app_code):
#             error_msg = {'message': 'Server not online at this time', 'status': 'off'}
#             return json.dumps(error_msg, ensure_ascii=False).encode("utf8"), 400

#         # validate request
#         api_key = multi_db_obj_arr[app_code]['api_key']
#         if api_key != '':
#             header = request.headers
#             if not header or header.get('x-api-key') != multi_db_obj_arr[app_code]['api_key']:
#                 abort(401)

#         # Get file data
#         if request.files.get(settings.FILE_KEY):
#             time_out = time.time()
#             img_raw_data = request.files.get(settings.FILE_KEY).read()
#             image = Image.open(io.BytesIO(img_raw_data)).convert('RGB')
#         elif request.form.get(settings.FILE_KEY):
#             time_out = time.time()
#             img_base64_str = request.form.get(settings.FILE_KEY)
#             img_raw_data = base64.b64decode(img_base64_str)
#             image = Image.open(io.BytesIO(img_raw_data))
#         else:
#             abort(400)

#         pre_computed_features = None
#         # Get pre-computed features
#         if request.files.get(settings.CNN_PRE_COMPUTED_KEY) is not None:
#             s = request.files.get(settings.CNN_PRE_COMPUTED_KEY).read()
#             logger.info("features' length : %s", len(s))
#             feature_length = 2048
#             if len(s) == feature_length * 8:
#                  _dtype = np.float64
#             elif len(s) == feature_length * 4:
#                  _dtype = np.float32
#             elif len(s) == feature_length:
#                  _dtype = np.uint8
#             else:
#                  logger.error("Incorrect features. Size = %s", len(s))
#                  abort(400)
#             pre_computed_features = np.frombuffer(s, dtype=_dtype)
#             if not (_dtype == np.float32):
#                 pre_computed_features = np.float32(pre_computed_features)
#             logger.info("Pre-computed CNN features are available")

#         # Get category
#         categories = request.form.get('category') or ''

#         # Begin search image
#         platform = request.user_agent.platform
#         browser = request.user_agent.browser
#         logger.info('Request from %s platform - %s browser' %
#                     (platform, browser))

#         # get shopid, if none return 400
#         shopid = request.form.get(settings.SHOP_ID_KEY)
#         if shopid is None:
#             shopid = "shop_id_default"

#         # handle too many requests
#         if db.llen(settings.IMAGE_QUEUE) > max_request:
#             logger.info('many')
#             logger.error("Too Many Requests: %s from %s with shop_id %s" % (db.llen(
#                 settings.IMAGE_QUEUE), request.remote_addr, shopid))
#             abort(429)

#         # get nt_retr
#         nr_retr = settings.NR_RETR
#         if request.form.get(settings.NR_RETR_KEY) is not None:
#             nr_retr = helpers.try_int(request.form.get(settings.NR_RETR_KEY))
#             if nr_retr > settings.MAX_NR_RETR:
#                 nr_retr = settings.MAX_NR_RETR

#         # save request to cache db
#         img_id = str(uuid.uuid4())
#         logger.info('Request image with id: %s - from: %s - shopid: %s - app_code: %s - categories: %s' %
#                     (img_id, request.remote_addr, shopid, app_code, categories.encode('utf-8')))

#         # Crop image in webapp
#         if browser is not None:
#             left, top, right, bottom = helpers.find_center(image.size)
#             image = image.crop((left, top, right, bottom))

#         # Save input
#         if settings.CBIR_SAVE_QUERY_IMAGES:
#             path_to_query_image = img_id + '.jpg'
#             path_to_query_image = os.path.join(
#                 settings.QUERY_IMAGE_FOLDER, path_to_query_image)
#             try:
#                 image.save(path_to_query_image)
#             except IOError:
#                 logger.error(
#                     "Cannot save the query image to file '%s'" % path_to_query_image)

#         # Create search request
#         search_image = np.array(image)
#         img_str = helpers.base64_encode_image(search_image)

#         if pre_computed_features is not None:
#             pre_computed_features_str = helpers.base64_encode_image(pre_computed_features)
#         else:
#             pre_computed_features_str = ''

#         req_obj = {settings.ID_KEY: img_id,
#                    settings.IMAGE_WIDTH_KEY: image.width,
#                    settings.IMAGE_HEIGHT_KEY: image.height,
#                    settings.CNN_PRE_COMPUTED_KEY: pre_computed_features_str,
#                    settings.NR_RETR_KEY: nr_retr,
#                    settings.IMAGE_KEY: img_str,
#                    settings.APP_CODE_KEY: app_code,
#                    settings.CATEGORIES_KEY: categories
#                    }
#         db.rpush(settings.IMAGE_QUEUE, json.dumps(req_obj))

#         logger.info("Save %s image to %s success" %
#                     (img_id, settings.IMAGE_QUEUE))

#         # prepare matching image
#         image = image.convert('L')
#         match_image, w, h = helpers.resize(
#             image, settings.MTC_IMAGE_WIDTH, settings.MTC_IMAGE_HEIGHT, True)

#         # received searching & matching reponses
#         state = 0  # 0 - searching | 1 - matching
#         while True:
#             run_time = int(round((time.time() - time_out) * 1000))
#             if run_time >= settings.REQ_TIME_OUT:
#                 logger.error("ALERT: Request %s time out" % img_id)
#                 abort(408)

#             # handler searching responses
#             if state == 0:
#                 output = db.get(img_id)
#                 if output is not None:
#                     db.delete(img_id)
#                     output = output.decode("utf-8")
#                     search_results = []
#                     boxes_sod = []
#                     boxes_sod.extend(json.loads(output)[settings.IMAGE_BOXES])

#                     rt_categories = []
#                     rt_categories.extend(json.loads(output)['category'])
#                     # sorted search results list
#                     search_results.extend(json.loads(output)['topn'])
#                     logger.debug('searching: %s, %s, %s, %s' % (img_id, shopid, app_code, search_results))
#                     search_results = sorted(
#                         search_results, key=lambda x: x[settings.SCORE_KEY], reverse=False)

#                     isMatching = matching_config = multi_db_obj_arr[app_code]['matching_config']
#                     if not isMatching:
#                         if len(search_results) > 0:
#                             json_response["type"] = "searching"
#                             json_response['matched_files'] = search_results
#                             if len(rt_categories) > 0:
#                                 json_response['category'] = rt_categories
#                             json_response["status"] = 0
#                             json_response["message"] = "successful"
#                             return_code = 200
#                         else:
#                             json_response["status"] = 1
#                             json_response["message"] = "No matches found"
#                             return_code = 404
#                         break

#                     # initialize matching request
#                     state = 1
#                     match_image = np.array(match_image)
#                     match_img_str = helpers.base64_encode_image(
#                         match_image)
#                     match_req_obj = {settings.ID_KEY: img_id,
#                                      settings.IMAGE_KEY: match_img_str,
#                                      settings.IMAGE_WIDTH_KEY: w,
#                                      settings.IMAGE_HEIGHT_KEY: h,
#                                      settings.IMAGE_BOXES:boxes_sod,
#                                      settings.SIMILAR_IMAGES_KEY: search_results,
#                                      settings.APP_CODE_KEY: app_code,
#                                      'categories': rt_categories
#                                      }
#                     db.rpush(settings.MTC_IMAGE_QUEUE,
#                              json.dumps(match_req_obj))

#                     logger.info('Save %s image to %s success' %
#                                 (img_id, str(settings.MTC_IMAGE_QUEUE)))
#                     if not matching_config:
#                         break

#             # handle matching response
#             elif state == 1:
#                 matching_output = db.get(img_id)
#                 if matching_output is not None:
#                     matching_output = matching_output.decode("utf-8")
#                     data = json.loads(matching_output)["result"]
#                     box = json.loads(matching_output)["box"]
#                     category_arr = json.loads(matching_output)['categories']
#                     matched_files = []
#                     if app_code == 'lashinbang':
#                         for i in range(len(data)):
#                             if 'crop_foreground' in data[i]['image']:
#                                 data[i]['image'] = data[i]['image'].replace('crop_foreground/', '')
#                             if 'c.' in data[i]['image']:
#                                 data[i]['image'] = data[i]['image'].replace('c', '')
#                             if 'rotate/' in data[i]['image']:
#                                 data[i]['image'] = data[i]['image'].replace('rotate/', '')
#                             if 'r' in data[i]['image']:
#                                 data[i]['image'] = data[i]['image'].replace('r', '')
#                             if any(data[i]['id'] == s['id'] for s in matched_files):
#                                 continue
#                             matched_files.append(data[i])
#                     else:
#                         matched_files.extend(data)

#                     if len(matched_files) > 0:
#                         json_response["type"] = "matching"
#                         json_response["matched_files"] = matched_files
#                         json_response["status"] = 0
#                         if len(category_arr) > 0:
#                             json_response["categories"] = category_arr
#                         json_response["message"] = "successful"
#                         json_response["box"] = box
#                         return_code = 200
#                     else:
#                         json_response["status"] = 1
#                         json_response["message"] = "No matches found"
#                         return_code = 404
#                     db.delete(img_id)
#                     logger.info('matching success')
#                     break

#             # sleep 0.25s
#             time.sleep(settings.SERVER_SLEEP)

#     # save statistic log
#     msg = str(json_response["message"])
#     if (msg == "successful"):
#         arr = json_response["matched_files"]
#         data = []
#         for item in arr:
#             data.append(item["image"])
#         logger.info(
#             f"statistic: {img_id}, {shopid}, {app_code}, {msg}, {'|'.join([str(elem) for elem in data])}")
#     else:
#         logger.info(f'statistic: {img_id}, {shopid}, {app_code}, {msg}, 0')

#     # send response
#     result = json.dumps(json_response, ensure_ascii=False).encode("utf8")
#     logger.info('response: %s, %s, %s, %s' % (img_id, shopid, app_code, result))
#     return result, return_code
@app.route('/set_state/<value>')
def set_state(value):
    db.set('stateSystem', value)
    return f"State set to {value}"

@app.route('/get_state')
def get_state():
    value = db.get('stateSystem')
    return value.decode('utf-8') if value else "State not set"
@app.route('/all_product',methods = ['POST', 'GET'])
def get_all_product():
    with open(db_config['LIST_PRODUCT'], 'r') as file:
        data = json.load(file)
    FilterResult = []
    for i in range(len(data)):
        tmppath = []
        for path in data[i]['List_Image']:
            if '_segment_' not in str(path) and '_Augment_' not in str(path):
                tmppath.append(path)
        FilterResult.append({'id':data[i]['id'],'Product_Name':data[i]['Product_Name'],'Product_Detail':data[i]['Product_Detail'],'List_Image':tmppath})
        
    return jsonify(FilterResult)

@app.route('/<folder>/<filename>')
def get_image(folder, filename):
    filepath = os.path.join(folder, filename)
    # Serve the file from within the 'SearchData' directory
    return send_from_directory('Database_CROP', filepath)
@app.route('/add_product', methods=['POST'])
def add_product():
    # Assuming product_name, product_detail, and product_image are the form field names
    product_name = request.form.get('product_name')
    product_detail = request.form.get('product_detail')
    # Validate inputs
    if not product_name or not product_detail:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    product_name = preProcessText(product_name)
    # Process and save the image
    listImage = []
    listFilenames = []
    try:
        if request.files.get(settings.FILE_KEY):
            for imageobject in request.files.getlist(settings.FILE_KEY):
                img_raw_data = imageobject.read()
                image = Image.open(io.BytesIO(img_raw_data)).convert('RGB')
                listImage.append(image)
                file_name, file_extension = os.path.splitext(imageobject.filename)
                imgName = preProcessText(file_name)+file_extension
                listFilenames.append(imgName)
        elif request.form.get(settings.FILE_KEY):
            for imageobject in request.form.getlist(settings.FILE_KEY):
                img_base64_str = imageobject
                img_raw_data = base64.b64decode(img_base64_str)
                image = Image.open(io.BytesIO(img_raw_data))
                listImage.append(image)
                file_name, file_extension = os.path.splitext(imageobject.filename)
                imgName = preProcessText(file_name)+file_extension
                listFilenames.append(imgName)
        else:
            abort(400)
        resultList = helpers.addIndex(product_name,product_detail,listImage,listFilenames,db_config,model1,model2,clipSeg_processor,clipSeg_model,db)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify(resultList), 200
@app.route('/add_images', methods=['POST'])
def add_images():
    # Assuming product_name, product_detail, and product_image are the form field names
    ItemID = request.form.get('id')
    product_name = request.form.get('product_name')
    product_name = preProcessText(product_name)
    product_detail = request.form.get('product_detail')
    # Validate inputs
    if not ItemID or not product_name or not product_detail:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    # Process and save the image
    listImage = []
    listFilenames = []
    if request.files.get(settings.FILE_KEY):
        for imageobject in request.files.getlist(settings.FILE_KEY):
            img_raw_data = imageobject.read()
            image = Image.open(io.BytesIO(img_raw_data)).convert('RGB')
            listImage.append(image)
            file_name, file_extension = os.path.splitext(imageobject.filename)
            imgName = preProcessText(file_name)+file_extension
            listFilenames.append(imgName)
    elif request.form.get(settings.FILE_KEY):
        for imageobject in request.form.getlist(settings.FILE_KEY):
            img_base64_str = imageobject
            img_raw_data = base64.b64decode(img_base64_str)
            image = Image.open(io.BytesIO(img_raw_data))
            listImage.append(image)
            file_name, file_extension = os.path.splitext(imageobject.filename)
            imgName = preProcessText(file_name)+file_extension
            listFilenames.append(imgName)
    resultList = helpers.addImages(ItemID,product_name,product_detail,listImage,listFilenames,db_config,model1,model2,clipSeg_processor,clipSeg_model,db)
    return jsonify(resultList), 200
@app.route('/remove_product', methods=['POST'])
def remove_product():
    productID = request.form.get('id')
    # Validate inputs
    if not productID:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    resultList = helpers.removeIndex(productID,db_config,db)
    return jsonify(resultList), 200
@app.route('/get_detail', methods=['POST'])
def get_detail():
    productID = request.form.get('id')
    # Validate inputs
    if not productID:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    result = helpers.getDetail(productID,db_config)
    result['status']=1
    result['message']='success'
    return jsonify(result), 200
@app.route('/get_total_index', methods=['POST'])
def get_total_index():
    result = helpers.getTotalIndex(db_config)
    result['status']=1
    result['message']='success'
    return jsonify(result), 200
@app.route('/remove_image', methods=['POST'])
def remove_image():
    productID = request.form.get('id')
    pathImage = request.form.get('imagePath')
    # Validate inputs
    if not productID or not pathImage:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    resultList = helpers.removeImage(productID,pathImage,db_config,db)
    return jsonify(resultList), 200
@app.errorhandler(Exception)
def handle_exception(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    error_str = jsonify(message=str(e), code=code, success=False)
    logger.error("EXCEPTION: %s" % error_str.data.decode("utf-8"))
    return error_str
# register error handler
for ex in default_exceptions:
    app.register_error_handler(ex, handle_exception)

if __name__ == '__main__':
    # run server
    app.run(host="0.0.0.0", port=5000)
