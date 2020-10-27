import pickle
import os
import numpy as np
from easydict import EasyDict
from sklearn.model_selection import train_test_split


class GenerateDataset:
    description = None
    attr_name = None
    image_name = None
    label = None
    partition = None
    reorder = None
    root = None
    weight_train = None
    weight_trainval = None

attr_name = ['AgeOver60', 'Age18-60', 'AgeLess18', 'Female', 'Male']
"""
attr_name = list of attributes(6) 
description = name of the dataset 
image_name = list of the names of all the images
label = ndarray of image_name(rows) x attr_name(columns) with ones and zeros 
partition = 
reorder = string containing 'group_order'
root = absolute path for the root folder containing images
weight_train = 
weight_trainval
"""

dataset = EasyDict()

sub_classes = ['Age1-16', 'Age17-30', 'Age31-45', 'Age46-60']
main_classes = ['Female', 'Male']
all_classes = ['Age1-16', 'Age17-30', 'Age31-45', 'Age46-60', 'Female', 'Male']

label = []
src = 'custom_data/'
save_dir = 'data/custom/'
image_name = []
for main_class in main_classes:
    for sub_class in sub_classes:
        folder_name = src + main_class + '/' + sub_class
        files = os.listdir(folder_name)
        for file in files:
            file_label = np.zeros((6,), dtype=int)
            file_label[all_classes.index(main_class)] = 1
            file_label[all_classes.index(sub_class)] = 1
            label.append(file_label)
            image_name.append(main_class + '/' + sub_class + '/' + file)

dataset_indexes = np.arange(len(image_name))

trainval, test, __, _t = train_test_split(dataset_indexes, dataset_indexes, test_size=.05)
train, val, __, _t = train_test_split(trainval, trainval, test_size=0.1)
dataset.partition = EasyDict()
dataset.partition.train = train
dataset.partition.val = val
dataset.partition.test = test
dataset.partition.trainval = trainval
dataset.image_name = image_name
dataset.label = np.array(label)
dataset.description = 'custom'
dataset.attr_name = all_classes
dataset.reorder = 'group_order'
dataset.root = src
# cls_weight
dataset.weight_train = np.mean(dataset.label[train], axis=0)
dataset.weight_trainval = np.mean(dataset.label[trainval], axis=0)

with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
    pickle.dump(dataset, f)

# attr_name = ['AgeOver60', 'Age18-60', 'AgeLess18', 'Female', 'Male']
