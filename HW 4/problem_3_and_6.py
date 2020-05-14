#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt


# In[21]:


grid_size = 5
transitions = [[0,-1],[-1,0],[0,1],[1,0]]  # west, north, east, south

gamma = 0.9


# In[22]:


A = [0, 1]
B = [0, 3]

A_prime = [4, 1]
B_prime = [2, 3]


# In[23]:


def move(state, action):
    if state == A:
        return A_prime, 10
    if state == B:
        return B_prime, 5

    s_prime = (np.array(state) + action).tolist()
    x, y = s_prime
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        R_t = -1.0
        s_prime = state
    else:
        R_t = 0
    return s_prime, R_t


# In[29]:


def uniform():
    value = np.zeros((grid_size, grid_size))
    while True:
        val_prime = np.zeros_like(value)
        for i in range(grid_size):
            for j in range(grid_size):
                for action in transitions:
                    (i_next, j_next), R_t = move([i, j], action)
                    val_prime[i, j] += 0.25 * (R_t + gamma * value[i_next, j_next])
        if np.sum(np.abs(value - val_prime)) < .001:
            fig, ax = plt.subplots(figsize=(13, 9))
            ax = sns.heatmap(val_prime, annot=True, square=True, linewidths=0.5, cbar=False)
            ax.set_ylim([5,0])
            fig.savefig('../figures/p3_part1.png')
            plt.close()
            break
        value = val_prime


# In[30]:


def non_uniform():
    value = np.zeros((grid_size, grid_size))
    while True:
        val_prime = np.zeros_like(value)
        for i in range(grid_size):
            for j in range(grid_size):
                for action in transitions:
                    (i_next, j_next), R_t = move([i, j], action)
                    if action == 1:
                        val_prime[i, j] += 0.7 * (R_t + gamma * value[i_next, j_next])
                    else: 
                        val_prime[i, j] += 0.1 * (R_t + gamma * value[i_next, j_next])
        if np.sum(np.abs(value - val_prime)) < .001:
            fig, ax = plt.subplots(figsize=(13, 9))
            ax = sns.heatmap(val_prime, annot=True, square=True, linewidths=0.5, cbar=False)
            ax.set_ylim([5,0])
            fig.savefig('../figures/p3_part2.png')
            plt.close()
            break
        value = val_prime


# In[34]:


def non_uniform_2():
    value = np.zeros((grid_size, grid_size))
    while True:
        val_prime = np.zeros_like(value)
        for i in range(grid_size):
            for j in range(grid_size):
                for action in transitions:
                    (i_next, j_next), R_t = move([i, j], action)
                    if action == 1:
                        val_prime[i, j] += 0.4 * (R_t + gamma * value[i_next, j_next])
                    elif action == 3:
                        val_prime[i, j] += 0.4 * (R_t + gamma * value[i_next, j_next])
                    else:
                        val_prime[i, j] += 0.2 * (R_t + gamma * value[i_next, j_next])
        if np.sum(np.abs(value - val_prime)) < .001:
            fig, ax = plt.subplots(figsize=(13, 9))
            ax = sns.heatmap(val_prime, annot=True, square=True, linewidths=0.5, cbar=False)
            ax.set_ylim([5,0])
            fig.savefig('../figures/p3_part3.png')
            plt.close()
            break
        value = val_prime


# In[32]:


def value_iter():
    value = np.zeros((grid_size, grid_size))
    while True:
        val_prime = np.zeros_like(value)
        for i in range(grid_size):
            for j in range(grid_size):
                values = []
                for action in transitions:
                    (i_next, j_next), R_t = move([i, j], action)
                    values.append(R_t + gamma * value[i_next, j_next])
                val_prime[i, j] = np.max(values)
        if np.sum(np.abs(val_prime - value)) < .001:
            fig, ax = plt.subplots(figsize=(13, 9))
            ax = sns.heatmap(val_prime, annot=True, square=True, linewidths=0.5, cbar=False)
            ax.set_ylim([5,0])
            fig.savefig('../figures/p3_part4.png')
            plt.close()
            break
        value = val_prime


# In[35]:


if __name__ == '__main__':
    uniform()
    non_uniform()
    non_uniform_2()
    value_iter()


# In[ ]:




