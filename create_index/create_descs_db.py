import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
import time
import settings
import pathlib
from pathlib import Path

AKAZE_THRESH = 3e-4 # AKAZE detection threshold set to locate about 1000 keypoints
RANSAC_THRESH = 10 # RANSAC inlier threshold
NN_MATCH_RATIO = 0.75 # Nearest-neighbour matching ratio

detector = cv2.AKAZE_create()
detector.setThreshold(AKAZE_THRESH)
 
def resize_prop_rect(src):
    MAX_SIZE = (settings.MTC_IMAGE_WIDTH, settings.MTC_IMAGE_HEIGHT)
    xscale = MAX_SIZE[0] / src.shape[0]
    yscale = MAX_SIZE[1] / src.shape[1]
    scale = min(xscale, yscale)
    if scale > 1:
        return src
    dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_LINEAR)
    return dst

def save_features(img, filename):
    if len(img.shape) == 3:
        _, _, c = img.shape
        if c == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
    else:
        gray = img
 
    gray = resize_prop_rect(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kps, features = detector.detectAndCompute(gray, None)
    kps = np.float32([kp.pt for kp in kps])

    # open a file, where you ant to store the data
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump((kps, features, gray.shape[0], gray.shape[1]), file)

    # close the file
    file.close()

def read_features(filename): 
    # open a file, where you ant to store the data
    file = open(filename, 'rb')

    # dump information to that file
    kps, features = pickle.load(file)

    # close the file
    file.close()

    return kps, features


def test_matching():
    t1 = time.time()

    kps1, features1 = read_features("file1.pkl")
    kps2, features2 = read_features("file2.pkl")

    matcher = cv2.DescriptorMatcher.create("FlannBased")

    raw_matches = []
    try:
            raw_matches = matcher.knnMatch(np.asarray(features1, np.float32), np.asarray(features2, np.float32), 2)
    except Exception as e:
            print("raw_matches erro = ", e)

    print("Matching time : ", time.time() - t1)

def test_extracting():
    img1 = cv2.imread("test.jpg")
    img2 = cv2.imread("203510292462_L.jpg")

    save_features(img1, "file1.pkl")
    save_features(img2, "file2.pkl")

img_list_fn = "/media/anlabadmin/big_volume/Lashinbang-CreateIndex/trained-data/images-path-txt/path-each-folder-20200327/Images_800_899.txt"
#img_list_fn = "error_files.txt"
src = "Lashinbang_data/"
src_fullpath = "/media/anlabadmin/big_volume/Lashinbang_data"
dst = "/media/anlabadmin/big_volume/Lashinbang500K/descs"

if __name__ == '__main__':
    files = []
    with open(img_list_fn) as f:
        files = [line[:-1] for line in f]
    files = [line.replace(src, '') for line in files]
    print("Read img_list_fn done, ", len(files))
    print("Sample : '%s'" % files[0])

    folders = {os.path.dirname(f) for f in files}
    folders = {os.path.join(dst, f) for f in folders}
    print("Number of subfolders : ", len(folders))
    for f in folders:
        pathlib.Path(f).mkdir(parents=True, exist_ok=True)

    file1 = open("MyFile.txt","a")

    creating_descs = True
    for i in tqdm(range(len(files))):
        f = files[i]
        f1 = f
        f = f.replace(src, '')
        input_file = os.path.join(src_fullpath, f)
        output_file = os.path.join(dst, f + ".pkl")
#        print("Input : ", input_file)
#        print("Output : ", output_file)

        if creating_descs:
            if os.path.exists(output_file):
                n = Path(output_file).stat().st_size
                if n > 0:
                    continue

            im = cv2.imread(input_file)
            if im is None:
                print('Cannot read file "%s"' % input_file)
                continue
            save_features(im, output_file)
        else:
            # check for consistency
            if not os.path.exists(output_file):
                print("Not found : ", output_file)
                file1.writelines([f1 + "\n"])
                continue
            n = Path(output_file).stat().st_size
            if n == 0:
                file1.writelines([f1 + "\n"])
                print("Empty : ", output_file)
    file1.close()


