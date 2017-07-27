# singapore_weather_prediction dataset generator

import logging
import pickle
import math
import os
import errno
import tarfile
import shutil


import numpy as np

logger = logging.getLogger(__name__)

def get_train():
    return _get_dataset("train")

def get_test():
    return _get_dataset("test")

def get_shape_input():
    return (None, 11, 9, 1)

def get_shape_label():
    return (None,11,9,1)

def num_classes():
    return 10

def _unpickle_file(filename):
    logger.debug("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl


def _get_dataset(split):
    assert split == "test" or split == "train"
    path = "data"
    data_file = "weather_data_file.txt"

    file_path = os.path.join(path, data_file)

    # read the data_file line by line
    file = open(file_path, "r")
    content = file.read().split(sep="\n")

    imgs = []
    time_stamps = []
    targets = []
    # for each image
    for line in range(0,2388,12):
        time_stamp = content[line]
        img = []
        temp1 = content[(line+1):(line+12)]
        # for each line
        for i in range(0,11):

            temp2 = temp1[i].split()
            # for each pixel
            temp3 = [float(x) for x in temp2]
            img.append(temp3)

        imgs.append(img)
        time_stamps.append(time_stamp)

    for hour in range(0,198):
        next_img = imgs[hour+1]
        targets.append(next_img)


    # Now we flatten the arrays
    #imgs = np.concatenate(imgs)
    #time_stamps = np.concatenate(time_stamps)

    # Convert images to [0..1] range
    #imgs = imgs.astype(np.float32)/255.0
    return imgs, targets
