#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn as sk
import skimage.io as skim
import skimage.transform as skt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics
from PIL import Image, ImageOps

import itertools
from tqdm import tqdm
import os
import cv2
import math
from sklearn.datasets import load_sample_image
import sklearn.metrics as skm


# In[ ]:


train_target = pd.read_csv('train.csv')
test_meta = pd.read_csv('test.csv')

train_path = train_target['filename']
test_path = test_meta['filename']

labels = list(set(train_target['label'].to_list()))


# In[ ]:


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


# In[ ]:


labels = list(set(train_target['label'].to_list()))
n_cat = len(labels)
labeler = {i:labels[i] for i in range(n_cat)}
rev_labeler = {labeler[i]:i for i in labeler.keys()}

y = [rev_labeler[i] for i in train_target['label'].to_list()]
y_targ = train_target['label']
y_targ = np.array(y_targ)
labels = np.array(labels)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4321)


# In[ ]:


X_train = preprocess(X_train, pixel1)
X_train_pp = []

for elem in X_train:
    elem = np.ndarray.flatten(elem)
    X_train_pp.append(elem)
X_train_pp = np.array(X_train_pp)

X_test = preprocess(X_test, pixel1)
X_test_pp = []
for elem in X_test:
    elem = np.ndarray.flatten(elem)
    X_test_pp.append(elem)
X_test_pp = np.array(X_test_pp)


# In[ ]:


scaler = StandardScaler()
scaler_fit = scaler.fit(X_train_pp)
X_train2 = scaler_fit.transform(X_train_pp)
X_test2 = scaler_fit.transform(X_test_pp)


# ### Cross Validation

# In[ ]:


x_dmatrix = xgb.DMatrix(X_train2, y_train)


# In[ ]:


params={"objective":"multi:softmax",'max_depth':6, 'num_class':4}


# In[ ]:


cv_results = xgb.cv(params, x_dmatrix, nfold=4, num_boost_round=20, metrics="merror", as_pandas=True)
cv_results


# In[ ]:


#param_dist = {"objective":"multi:softmax", "max_depth":4, "n_estimators":250, "learning_rate":0.01, "num_class":4, 
#              "subsample":0.4, "reg_lambda":2, "reg_alpha":1, "min_child_weight":1,"gamma":0.3, "colsample_bytree":1.0, "num_boost_round":20}


# ### Train an XGBoost Classifier

# In[ ]:


xgb_clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=1000, as_pandas=True)


# In[ ]:


xgb_model = xgb_clf.fit(X_train2, y_train)


# In[ ]:


y_pred = xgb_model.predict(X_test2)
accuracy = float(np.sum(y_pred==y_test))/len(y_test)
accuracy


# In[ ]:


acc = skm.accuracy_score(y_test, y_pred)
bal_accuracy = skm.balanced_accuracy_score(y_test, y_pred)
print("accuracy: %f" % (acc))
print("balanced accuracy: %f" % (bal_accuracy))


# In[ ]:


x_test = []
for im in test_path.to_list():
    i = skim.imread(im)
    i = skt.resize(i,[224,224,3])
    x_test.append(i)


# In[ ]:


x_test = preprocess(x_test, pixel1)
x_test_pp = []
for elem in x_test:
    elem = np.ndarray.flatten(elem)
    x_test_pp.append(elem)
x_test_pp = np.array(x_test_pp)
x2 = scaler_fit.transform(x_test_pp)


# In[ ]:


test_pred = xgb_model.predict(x2)


# In[ ]:


output = {'Id':range(len(test_pred)),'label':[labeler[i] for i in test_pred]}
submission = pd.DataFrame.from_dict(output)

submission.to_csv('jph2185_submission8.csv',index=False)


# ### Hyperparameter Tuning/Optimization using Random Search

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


xg_cl = xgb.XGBClassifier(objective="multi:softmax")

param_dist = {
    'gamma': [0.3,  0.5,  1, 1.5, 2],
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [2,4,7,10],
    'subsample': [0.2, 0.4, 0.5, 0.6, 0.7],
    'colsample_bytree': [0.3, 0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0, 1, 1.5, 2, 3, 4.5],  
}

xgb_rscv = RandomizedSearchCV(xg_cl, param_distributions=param_dist, scoring="balanced_accuracy", cv=5, n_iter=5, verbose=3, random_state=1234)


# In[ ]:


model_xgboost = xgb_rscv.fit(X_train2, y_train)


# In[ ]:


# Model best estimators
print("Learning Rate: ", model_xgboost.best_estimator_.get_params()["learning_rate"])
print("Gamma: ", model_xgboost.best_estimator_.get_params()["gamma"])
print("Max Depth: ", model_xgboost.best_estimator_.get_params()["max_depth"])
print("Subsample: ", model_xgboost.best_estimator_.get_params()["subsample"])
print("Max Features at Split: ", model_xgboost.best_estimator_.get_params()["colsample_bytree"])
print("Alpha: ", model_xgboost.best_estimator_.get_params()["reg_alpha"])
print("Lamda: ", model_xgboost.best_estimator_.get_params()["reg_lambda"])
print("Minimum Sum of the Instance Weight Hessian to Make a Child: ",
      model_xgboost.best_estimator_.get_params()["min_child_weight"])
print("Number of Trees: ", model_xgboost.best_estimator_.get_params()["n_estimators"])


# ### Error Analysis

# In[ ]:




