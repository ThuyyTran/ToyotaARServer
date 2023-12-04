import sys, os
sys.path.insert(1, '../')
from logger import AppLogger


SUDO_DELETE_FILE = '../data/rollback/sudo_delete.txt'

if __name__ == "__main__":
    if not os.path.isfile(SUDO_DELETE_FILE):
        sys.exit(0)

    # initialize app logger
    logger = AppLogger('update')

    with open(SUDO_DELETE_FILE, 'r') as f:
        delete_paths = f.readlines()
    delete_paths = list(map(lambda x: x[:-1], delete_paths))

    # determine pkl file is exists
    for d_file in delete_paths:
        if os.path.isfile(d_file):
            try:
                os.remove(d_file)
                logger.info('sudo deleted: %s' % d_file)
            except Exception as ex:
                logger.exception('delete %s -  exception: %s' % (d_file, ex))
        else:
            logger.error('%s not found' % d_file)

    os.remove(SUDO_DELETE_FILE)
