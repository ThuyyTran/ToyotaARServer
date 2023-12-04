import os
import sys
import settings
import logging.config
import errno

from datetime import datetime
from os import path

def set_logger_dir(dirname):
    if not os.path.exists(dirname):
        mkdir_p(dirname)

def mkdir_p(dirname):
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def get_log_path(error):
    log_path = ''
    if error:
        set_logger_dir(settings.LOGGER_ERROR_PATH)
        log_path = os.path.join(
            settings.LOGGER_ERROR_PATH, '{:%Y-%m-%d}-Lashinbang-server-error.log'.format(datetime.now()))
    else:
        set_logger_dir(settings.LOGGER_INFO_PATH)
        log_path = os.path.join(
            settings.LOGGER_INFO_PATH, '{:%Y-%m-%d}-Lashinbang-server-info.log'.format(datetime.now()))

    return log_path

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.exception("Uncaught exception: ", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def AppLogger(name='root'):
    log_conf_file_path = path.join(path.dirname(
        path.abspath(__file__)), settings.LOGGER_CONF_NAME)
    logging.config.fileConfig(log_conf_file_path, disable_existing_loggers=False, defaults={
        'loginfofilename': get_log_path(False), 'logerrorfilename': get_log_path(True)})

    # Complete logging config
    log = logging.getLogger(name)

    return log
