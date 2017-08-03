# CIFAR10 Downloader

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
    return (None, 30, 11, 9)

def get_shape_label():
    return (None,11*9)

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
    content = file.read().split("\n")

    imgs = []
    time_stamps = []
    targets = []
    # for each image
    for line in range(0,(len(content)-1),12):
        time_stamp = content[line]
        img = []
        temp1 = content[(line+1):(line+12)]
        # for each line
        for i in range(0,11):

            temp2 = temp1[i].split()
            # convert each element into a pixel/temperature value
            temp3 = [float(x) for x in temp2]
            img.append(np.array(temp3))
        img = np.reshape(np.concatenate(img), [11, 9])

        img_32_32_3 = [0] * 32
        for i in range(32):
            img_32_32_3[i] = [0] * 32
        for i in range(32):
            for j in range(32):
                img_32_32_3[i][j] = [0] * 3

        for item in range(3):
            for i in range(11):
                for j in range(9):
                    img_32_32_3[i][j][item] = img[i][j]


        #imgs.append(img)
        imgs.append(img_32_32_3)
        time_stamps.append(time_stamp)

    STEP_SIZE = 30
    inpts = []
#    preds = []
    total_hours = np.int((len(content)-1)/12)
    for hour in range(0,(total_hours-STEP_SIZE)):
        print(hour)
        inpt = imgs[hour:hour+STEP_SIZE]
        #pred = imgs[(STEP_SIZE+hour)]
        inpts.append(np.reshape(np.concatenate(inpt),[-1,32,32,3]))
        #preds.append(pred)

    return_item_y = [0] * len(inpts)

    return np.array(inpts) , return_item_y
    #np.reshape(np.array(preds),[-1, 99])
