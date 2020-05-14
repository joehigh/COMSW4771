#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sklearn
from sklearn.decomposition import PCA


# In[2]:


swiss_df = pd.read_csv('swiss_roll.txt', header=None, sep="\s+")
swiss_hole_data = pd.read_csv('swiss_roll_hole.txt', header=None, sep="\s+")


# In[3]:


swiss_df = []
with open('swiss_roll.txt','r') as file:
    for line in file.readlines():
        swiss_df.append([float(x) for x in line.split()])
swiss_df = np.array(swiss_df)


# In[4]:


fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection='3d')
ax.view_init(50, 40)
ax.scatter(swiss_df[:,0],swiss_df[:,1],swiss_df[:,2],
           c=swiss_df[:,0]**2+swiss_df[:,1]**2, cmap = plt.cm.viridis)

plt.savefig('swiss_3d.png')


# In[12]:


pca = PCA(n_components = 2)
swiss_2D = pca.fit_transform(swiss_df)


# In[13]:


plt.figure()
plt.scatter(swiss_2D[:,0],swiss_2D[:,1],c=swiss_df[:,0]**2+swiss_df[:,1]**2,
           cmap=plt.cm.viridis)
plt.savefig('swiss_pca.png')


# In[14]:


swiss_hole_df = []
with open('swiss_roll_hole.txt','r') as file:
    for i in file.readlines():
        swiss_hole_df.append([float(x) for x in i.split()])
swiss_hole_df = np.array(swiss_hole_df)


# In[15]:


fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection='3d')
ax.view_init(50, 40)
ax.scatter(swiss_hole_df[:,0],swiss_hole_df[:,1],swiss_hole_df[:,2],
           c=swiss_hole_df[:,0]**2+swiss_hole_df[:,1]**2, cmap = plt.cm.viridis)

plt.savefig('swiss_hole_3d.png')


# In[16]:


pca_2 = PCA(n_components = 2)
swiss_hole_2D = pca_2.fit_transform(swiss_hole_df)


# In[17]:


plt.figure()
plt.scatter(swiss_hole_2D[:,0],swiss_hole_2D[:,1],c=swiss_hole_df[:,0]**2+swiss_hole_df[:,1]**2,
           cmap=plt.cm.viridis)
plt.savefig('swiss_hole_pca.png')


# In[ ]:




