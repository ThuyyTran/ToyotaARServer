import os, sys, io, getopt
import settings
import json
import base64
import csv

from datetime import datetime, timedelta

LOG_PATH = settings.LOGGER_INFO_PATH
STATISTIC_LOG_BY_DATE = "statistic_by_date.csv"
STATISTIC_LOG_BY_SHOP_ID = "statistic_by_shop_id.csv"
START_DATE = "2020/05/07"
END_DATE = "2020/05/08"
SHOP_ID_ARR = []

def get_argument(argv):
    if len(argv) < 2:
        print('python statistics_log.py <start_date> <end_date>')
        sys.exit(2)

    return argv[0], argv[1]

def get_log_file(date):
    date_str = datetime.strftime(date, "%Y-%m-%d")
    log_file_name = os.path.join(LOG_PATH, f"{date_str}-Lashinbang-server-info.log")
    return log_file_name

def write_statistic(path, str):
    with open(path, 'a') as f:
        if os.path.getsize(path) == 0:
            if path == STATISTIC_LOG_BY_DATE:
                f.write('log_name, shop_id, not_matches, matched\n')
            else:
                f.write('shop_id, not_matches, matched\n')
        f.write(f'{str}\n')

def stastic_log_by_date(path):
    with open(path) as f:
        all_stats_arr = []
        stas_dict = {}

        # statistic each line of log
        for line in f:
            if "statistic" in line:
                stas_idx = str(line.strip()).find("statistic")
                stas_time = line[:stas_idx].split(' - ')[0]
                stas_arr = [item.strip().rstrip() for item in line[stas_idx:].split(',')]
                shop_id = stas_arr[1]
                if shop_id not in stas_dict:
                    stas_dict[shop_id] = [{ "time": stas_time, "result": stas_arr[3] }]
                else:
                    stas_dict[shop_id].append({ "time": stas_time, "result": stas_arr[3] })

        # write statistic
        if len(stas_dict) > 0:
            for k, v in stas_dict.items():
                # append to list shop id
                if k not in SHOP_ID_ARR:
                    SHOP_ID_ARR.append(k)

                # sum not matches result
                total = len(v)
                sum_not_matches = sum(x.get("result") == '0' for x in v)
                sum_matched = total - sum_not_matches
                stas_str = str(f'{os.path.basename(path)}, {k}, {sum_not_matches}, {sum_matched}')
                write_statistic(STATISTIC_LOG_BY_DATE, stas_str)

def statistic_log_by_shop():
    arr = []
    with open(STATISTIC_LOG_BY_DATE, "r") as f:
        arr.extend(csv.reader(f))

    # write statistic by shop id
    for shop_id in SHOP_ID_ARR:
        shop_arr = [row for row in arr if row[1].strip() == shop_id]
        sum_not_matches = sum(int(row[2].strip()) for row in shop_arr)
        sum_matched = sum(int(row[3].strip()) for row in shop_arr)
        shop_str = str(f'{shop_id}, {sum_not_matches}, {sum_matched}')
        write_statistic(STATISTIC_LOG_BY_SHOP_ID, shop_str)

if __name__ == "__main__":
    # get command lines arguments
    START_DATE, END_DATE = get_argument(sys.argv[1:])

    # format date
    start_date = datetime.strptime(START_DATE, '%Y/%m/%d')
    end_date = datetime.strptime(END_DATE, '%Y/%m/%d')
    delta = timedelta(days=1)

    # remove old statistic file
    if (os.path.exists(STATISTIC_LOG_BY_DATE)):
        os.remove(STATISTIC_LOG_BY_DATE)

    if (os.path.exists(STATISTIC_LOG_BY_SHOP_ID)):
        os.remove(STATISTIC_LOG_BY_SHOP_ID)

    # start run statistic
    while start_date <= end_date:
        log_path = get_log_file(start_date)
        stastic_log_by_date(log_path)
        start_date += delta

    # check shop
    if (os.path.exists(STATISTIC_LOG_BY_DATE)):
        statistic_log_by_shop()
