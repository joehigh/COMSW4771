#!/usr/bin/env python
# coding: utf-8

# In[123]:


import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt
import sklearn as sk
import skimage.io as skim
import skimage.transform as skt
from sklearn.datasets import load_sample_image
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import joblib

import tensorflow as tf
from PIL import Image, ImageOps
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import math
import csv
import pickle as pkl


# In[67]:


train_target = pd.read_csv('train.csv')
test_meta = pd.read_csv('test.csv')

train_path = train_target['filename']
test_path = test_meta['filename']

labels = list(set(train_target['label'].to_list()))


# In[68]:


train_images = [os.path.join('data/train', file) for file in os.listdir('data/train')]
train_imgs_sort = sorted(train_images)  
train_array = np.array(train_imgs_sort)


# In[ ]:


pixel1 = [224,224,3]
pixel2 = [229,229,3]
pixel3 = [255,255,3]


# In[ ]:


def preprocess(data, pixel):
    x_new=np.zeros((len(data),pixel[0],pixel[1],pixel[2]))
    for i,d in enumerate(data):
        x_new[i,:,:,:] = d
    return x_new


# In[ ]:


x = []
for f in train_path.to_list():
    xray = skim.imread(f)
    xray = skt.resize(xray, pixel1)
    x.append(xray)


# In[71]:


labels = list(set(train_target['label'].to_list()))
n_cat = len(labels)
labeler = {i:labels[i] for i in range(n_cat)}
rev_labeler = {labeler[i]:i for i in labeler.keys()}

y = [rev_labeler[i] for i in train_target['label'].to_list()]
y_targ = train_target['label']
y_targ = np.array(y_targ)
labels = np.array(labels)


# In[87]:


ohe_labels = np.zeros((len(y_targ), num_classes))  #one-hot encode target labels
# one-hot encoding the labels
for ii in range(len(y_targ)):
    jj = np.where(labels == y_targ[ii])
    ohe_labels[ii, jj] = 1
ohe_labels


# In[89]:


X_train, X_holdout, y_train, y_holdout = train_test_split(x, ohe_labels, test_size=0.2, random_state=123)


# In[ ]:


X_train = preprocess(X_train, pixel1)
X_train_pp = []

for elem in X_train:
    #elem = np.ndarray.flatten(elem)
    X_train_pp.append(elem)
#X_train_pp = np.array(X_train_pp)

X_test = preprocess(X_test, pixel1)
X_test_pp = []
for elem in X_test:
    #elem = np.ndarray.flatten(elem)
    X_test_pp.append(elem)
#X_test_pp = np.array(X_test_pp)


# ### Keras VGG16 Pre-trained model using SGD optimizer

# In[125]:


vgg16_model = keras.applications.vgg16.VGG16()


# In[126]:


model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)


# In[127]:


model.layers.pop()


# In[128]:


for layer in model.layers:
    layer.trainable = False


# In[129]:


model.add(Dense(4, activation='softmax'))


# In[130]:


model.compile(SGD(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


# In[94]:


X_train_pp = np.array(X_train_pp)
X_holdout_pp = np.array(X_holdout_pp)
y_train = np.array(y_train)
y_holdout = np.array(y_holdout)


# In[91]:


print(y_train)


# In[131]:


gen_train = ImageDataGenerator(rotation_range=15, width_shift_range=0.5, shear_range=0.3, height_shift_range=0.5, zoom_range=0.15)
gen_test = ImageDataGenerator()

train_generator = gen_train.flow(X_train_pp, y_train)
test_generator = gen_test.flow(X_holdout_pp, y_holdout)


# In[132]:


model.fit_generator(train_generator, steps_per_epoch=8, validation_data=test_generator, validation_steps=5, epochs=25, verbose=2)


# ### Keras VGG16 Pre-trained model using Adam optimizer

# In[155]:


model.layers.pop()


# In[156]:


for layer in model.layers:
    layer.trainable = False


# In[157]:


model.add(Dense(4, activation='softmax'))


# In[158]:


model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


# In[159]:


gen_train = ImageDataGenerator(rotation_range=10, width_shift_range=0.5, shear_range=0.2, height_shift_range=0.5, zoom_range=0.1)
gen_test = ImageDataGenerator()

train_generator = train_gen.flow(X_train_new, y_train)
test_generator = test_gen.flow(X_holdout_new, y_holdout)


# In[160]:


model.fit_generator(train_generator, steps_per_epoch=4, validation_data=test_generator, validation_steps=4, epochs=25, verbose=2)


# In[ ]:




