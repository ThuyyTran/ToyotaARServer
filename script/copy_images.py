import os
from tqdm import tqdm
import pathlib
import subprocess

lst_img = []
log_path = "../logss/2020-10-25-Lashinbang-server-info.log"

with open(log_path, 'r') as f:
    file_name = os.path.basename(log_path)
    current_date = file_name[0:10]
    pathlib.Path(current_date).mkdir(parents=True, exist_ok=True)

    lines = f.readlines()

    for line in lines:
        if 'with id' in line:
            start_index = line.find("id") + 3
            line = line[start_index:start_index + 37]
            lst_img.append(line)

# copy images
for i in tqdm(range(len(lst_img))):
    image_path = lst_img[i].strip()
    # download_path = os.path.join('anlabadmin@192.168.1.190:/home/anlabadmin/Documents/Lashinbang-test/query_images/', image_path.strip())
    download_path = os.path.join('ubuntu@52.194.200.43:/home/ubuntu/efs/query_images/', image_path.strip() + ".jpg")
    # print(download_path)
    # subprocess.run(["sshpass", "-p", "a22b2212", "scp", download_path, os.path.join(current_date, f"{i}_{image_path}.jpg")])
    subprocess.run(["scp", "-i", "lashinbang.pem", download_path, os.path.join(current_date, f"{i}_{image_path}.jpg")])
