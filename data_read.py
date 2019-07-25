# coding: utf8
# Aim to: read data for experiments, (codes about images are from keras)



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
# import os
# import sys
import time

import numpy as np
from PIL import Image as pil_image
gc.enable()

from utils_constant import DTY_FLT
from utils_constant import DTY_INT

from utils_constant import GAP_INF
from utils_constant import GAP_MID
from utils_constant import GAP_NAN



#===================================================
# Read Data about - GMM, diversity, diverhuge
#===================================================


#----------------------------------------------
# General
#----------------------------------------------


def convertY2num(labels):
    # X = data
    y = np.zeros(len(labels), dtype=DTY_INT) - 1
    target = np.unique(labels)
    labels = np.array(labels)
    for i in range(len(target)):
        idx = (labels == target[i])
        y[idx] = i
    y = y.tolist()
    return deepcopy(y)


def multi2binary(y):
    labels = np.array(y)
    target = np.unique(y)
    size = [np.mean(labels == j) for j in target]
    cs = np.cumsum(size)
    idx = (cs > 0.5).tolist().index(1) 
    idx = np.max([idx, 1])  # robust
    y = np.zeros_like(labels)
    y[labels >= target[idx]] = 1
    y = y.tolist()
    return deepcopy(y)



#----------------------------------------------
# read MAT
#----------------------------------------------


def read_from_mat(uci_path, uci_name):
    data = [];  labels = []
    #
    with open(uci_path + uci_name + '.data') as f:
        for each in f.readlines():
            current = each.strip().split()
            data.append( list(map(float, current)) )
    with open(uci_path + uci_name + '.label') as f:
        for each in f.readlines():
            current = each.strip()
            labels.append( int(current) )
    #
    if uci_name == "gmm_10D_n1k":
        labels = [i - 1 for i in labels]
    return deepcopy(data), deepcopy(labels)


def part_of_mat(uci_path, uci_name):
    if uci_name in ['Ames', 'sonar', 'spam']:
        X, y = txt_read_MissingValue(uci_path, uci_name=uci_name+'.txt', missing=False, delimiter=None, abaft=True)
    elif uci_name in ['card', 'heart', 'iono', 'liver', 'ringnorm', 'waveform', 'wisconsin']:
        X, y = txt_read_MissingValue(uci_path, uci_name=uci_name+'.log', missing=False, delimiter=None, abaft=True)
    elif uci_name == 'house':
        X, y = [], []
        # pass
    else:
        raise UserWarning("LookupError in ReadMAT. Check the uci_name (diversity).")
    y = convertY2num(y)
    return deepcopy(X), deepcopy(y)



#----------------------------------------------
# Interface
#----------------------------------------------


def previously_on_diversity(uci_path, uci_name, binary=True):
    if uci_name in ['gmm_2D_n4k', 'gmm_3D_n2k', 'gmm_10D_n1k']:
        X, y = read_from_mat(uci_path, uci_name)
    elif uci_name in ['Ames', 'card', 'heart', 'iono', 'liver', 'ringnorm', 'sonar', 'spam', 'waveform', 'wisconsin']:  # 'house'
        X, y = read_from_mat(uci_path, uci_name)
    elif uci_name in ['credit', 'landsat', 'page', 'shuttle', 'wilt']:
        X, y = read_from_mat(uci_path, uci_name)
        if uci_name in ['landsat', 'page', 'shuttle']:
            y = convertY2num(y)
        if binary == True and len(np.unique(y)) > 2:
            y = multi2binary(y)
        #
    else:
        raise UserWarning("LookupError in ReadMAT. Check the `uci_name`.")
    return deepcopy(X), deepcopy(y)



#----------------------------------------------
#
#----------------------------------------------



#===================================================
# Read data about - UCI
#===================================================


#----------------------------------------------
# General
#----------------------------------------------



#----------------------------------------------
# Get different datasets
#----------------------------------------------


def txt_read_MissingValue(uci_path, uci_name, missing=False, delimiter=None, abaft=True):
    data = [];  labels = []
    with open(uci_path + uci_name) as file:
        lines = file.readlines()
        for each in lines:
            if missing == True:
                each = each.replace('?', '0')
            current = each.strip().split(delimiter)
            if abaft == True:
                data.append( list(map(float, current[:-1])) )
                labels.append( current[-1] )
            else:
                data.append( list(map(float, current[1:])) )
                labels.append( current[0] )
    return deepcopy(data), deepcopy(labels)


def txt_read_NominalAttribute(uci_path, uci_name, nominal=[0], missing=False, delimiter=None, abaft=True):
    labels = [];    text = [];  amount = []
    with open(uci_path + uci_name) as file:
        lines = file.readlines()
        for each in lines:
            if missing == True:
                each = each.replace('?', '0')
            current = each.strip().split(delimiter)
            if abaft == True:
                labels.append(current[-1])
                temp = current[:-1]
            else:
                labels.append(current[0])
                temp = current[1:]
            #
            text.append([current[j] for j in nominal])
            idx = deepcopy(nominal);    idx.reverse()
            for j in idx:
                del temp[j]
            amount.append( list(map(float, temp)) )
    #   #   #
    text = np.array(text);  textzero = np.zeros_like(text, dtype=DTY_INT)
    for j in range(text.shape[1]):
        temp = convertY2num(text[:, j].tolist())
        textzero[:, j] = np.array(temp)
    textzero = textzero.tolist()
    data = np.concatenate([textzero, amount], axis=1)
    data = data.tolist()
    #
    del text, amount, textzero
    return deepcopy(data), deepcopy(labels)


def txt_read_DeleteNominal(uci_path, uci_name, nominal=[0], missing=False, delimiter=None, abaft=True):
    data = [];  labels = []
    with open(uci_path + uci_name) as file:
        lines = file.readlines()
        for each in lines:
            if missing == True:
                each = each.replace('?', '0')
            current = each.strip().split(delimiter)
            if abaft == True:
                labels.append(current[-1])
                temp = current[:-1]
            else:
                labels.append(current[0])
                temp = current[1:]
            #
            idx = deepcopy(nominal);    idx.reverse()
            for j in idx:
                del temp[j]
            data.append( list(map(float, temp)) )
            del temp, current
        del each, lines
    del file
    return deepcopy(data), deepcopy(labels)


def txt_read_DifferentFiles(uci_path, uci_name, missing=False, delimiter=None):
    data = [];  labels = []
    with open(uci_path + uci_name + '.data') as f:
        for each in f.readlines():
            if missing == True:
                each = each.replace('?', '0')
            current = each.strip().split(delimiter)
            data.append( list(map(float, current)) )
    with open(uci_path + uci_name + '.labels') as f:
        for each in f.readlines():
            current = each.strip()
            # labels.append( float(current) )
            labels.append( current )
    return deepcopy(data), deepcopy(labels)



#----------------------------------------------
# Get different datasets
#----------------------------------------------


def unitary_interface_on_UCI(uci_path, uci_name, binary=False):
    # path += 'datasets/UCI/'
    #
    if uci_name == 'EEGEyeState':
        data, labels = txt_read_MissingValue(uci_path, uci_name=uci_name+'.txt', missing=False, delimiter=',', abaft=True)
        #
    elif uci_name in ['waveform', 'waveform_noise']:
        data, labels = txt_read_MissingValue(uci_path, uci_name=uci_name+'.data', missing=False, delimiter=',', abaft=True)
    elif uci_name in ['sensor_readings_2', 'sensor_readings_4', 'sensor_readings_24']:  # number of features
        data, labels = txt_read_MissingValue(uci_path, uci_name=uci_name+'.data', missing=False, delimiter=',', abaft=True)
        #
    elif uci_name in ['segmentation_data', 'segmentation_test']:
        data, labels = txt_read_MissingValue(uci_path, uci_name=uci_name+'.txt', missing=False, delimiter=',', abaft=False)
        #
    elif uci_name in ['mammographic_masses']:
        data, labels = txt_read_MissingValue(uci_path, uci_name=uci_name+'.data', missing=True, delimiter=',', abaft=True)
        #
    elif uci_name.startswith('ecoli') or uci_name.startswith('yeast'):
        if uci_name.endswith('+'):
            data, labels = txt_read_NominalAttribute(uci_path, uci_name=uci_name[:-1]+'.data', nominal=[0], missing=False, delimiter=None, abaft=True)
        else:
            data, labels = txt_read_DeleteNominal(uci_path, uci_name=uci_name+'.data', nominal=[0], missing=False, delimiter=None, abaft=True)
        #   #
    elif uci_name in ['madelon_train', 'madelon_valid']:
        data, labels = txt_read_DifferentFiles(uci_path, uci_name=uci_name, missing=False, delimiter=None)
        #
    else:
        raise UserWarning("LookupError in ReadUCI. Check the `uci_name`.")  # 用户代码生成的警告
    #
    X = data;   y = convertY2num(labels);   del data, labels
    if binary == True and len(np.unique(y)) > 2:
        y = multi2binary(y)
    return deepcopy(X), deepcopy(y)



#----------------------------------------------
#
#----------------------------------------------





#===================================================
# Read data about - Image
#===================================================


#----------------------------------------------
#
#----------------------------------------------


# target_size = (height, width, channel) -> (channel, height, width)

def image_read_single(img_name, img_size=(10, 10), dimension=3):
    img_obj = pil_image.open(img_name)  # default: img.mode='RGB'
    target_size = (img_size[1], img_size[0])  # (width, height)
    img_obj = img_obj.resize(target_size)  #, pil_image.ANTIALIAS
    #
    if dimension == 3:
        X = np.asarray(img_obj, dtype=DTY_FLT)  # default: channels_last
        X = X.transpose(2, 0, 1)  # channels_first
    elif dimension == 2:
        img_obj = img_obj.convert('L')
        X = np.asarray(img_obj, dtype=DTY_FLT)
    elif dimension == 1:
        img_obj = img_obj.convert('L')
        X = np.asarray(img_obj, dtype=DTY_FLT)
        X = X.reshape(-1,)
    else:
        raise UserWarning("LookupError in ReadIMG. Check the `dimension` in image_read_single.")
    #
    del target_size, img_obj
    ans = X.tolist();   del X
    return deepcopy(ans)



#----------------------------------------------
#
#----------------------------------------------


def image_read_batch(img_folder, img_idx=(0, 1), img_size=(30, 30), dimension=3):
    start = img_idx[0];     end = img_idx[1]
    # height = img_size[0];   width = img_size[1]
    if start >= end:
        raise UserWarning("LookupError in ReadIMG. Check the `img_idx` in image_read_batch.")
    #
    if img_folder.endswith('cats/'):
        labels = [0] * (end - start)
        img_path = img_folder + 'cat.'
    elif img_folder.endswith('dogs/'):
        labels = [1] * (end - start)
        img_path = img_folder + 'dog.'
    else:
        raise UserWarning("LookupError in ReadIMG. Check the `img_folder` in image_read_batch.")
    #
    img_names = [str(i) if i < 12500 else str(i % 12500) for i in range(start, end)]
    img_names = [img_path + i + '.jpg' for i in img_names]
    images = list(map(image_read_single, img_names, [img_size]*(end-start), [dimension]*(end-start) ))
    del start, end, img_path, img_names
    return deepcopy(images), deepcopy(labels)



#----------------------------------------------
#
#----------------------------------------------


def image_read_catdog(img_path, img_idx=[[0, 1], [1, 2]], img_size=(30, 30), dimension=3):  # , random_seed=None):
    cat_images, cat_labels = image_read_batch(img_path+'cats/', img_idx[0], img_size, dimension)
    dog_images, dog_labels = image_read_batch(img_path+'dogs/', img_idx[1], img_size, dimension)
    images = np.concatenate([cat_images, dog_images], axis=0)
    labels = np.concatenate([cat_labels, dog_labels], axis=0)
    del cat_images, cat_labels, dog_images, dog_labels
    #
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    index = list(range(len(labels)))
    prng.shuffle(index)  # np.random.shuffle
    res1 = images[index].tolist()
    res2 = labels[index].tolist()
    del randseed, prng, index, images, labels
    #
    gc.collect()
    return deepcopy(res1), deepcopy(res2)



#----------------------------------------------
#
#----------------------------------------------


