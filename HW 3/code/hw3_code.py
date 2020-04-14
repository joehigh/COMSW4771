import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal
import seaborn as sns
sns.set_style('darkgrid')

import math
import matplotlib

#import sys
#sys.path.insert(0, '/figures')


#random.seed(123)
x_min = -10
x_max = 10
y_min = -3
y_max = 3
N = 500
x = np.linspace(x_min, x_max, N)


### Part (v):

# Generating random functions with $\mu = \vec{0}$ and $\Sigma = I\$

sigma = np.eye(N)
mean = np.zeros(N)

y = multivariate_normal.rvs(mean, sigma)



# Define function to draw N samples randomly from MVN distribution with given mean and covariance, establishing
# four random functions and subsequently plot their values against the corresponding x-value.
def rnd_func_plot(X, mu, sigma, title=''):
    fig = plt.figure(figsize = (15,11))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ind = ['1','2','3','4']
    for i in range(4):
        y = multivariate_normal.rvs(mu, sigma)
        plt.subplot(2,2,i+1)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        #plt.xlabel('$x_i$', size = 14)
        plt.ylabel('Function Value', size=13)
        plt.title('Random Function ' + ind[i] + title)
        plt.plot(X,y)

rnd_func_plot(x, mean, sigma, ' for $\mu = 0$ and $\Sigma = I$')
#plt.savefig('./figures/part_v_whitenoise.png', bbox_inches='tight')


# Generating random functions with $\mu = \vec{0}$ and $\Sigma = \mathbb{J}_{500}\$  (i.e., the all ones matrix)

sigma_ones = np.ones((N,N))   # setting covariance matrix to all ones matrix
rnd_func_plot(x, mean, sigma_ones, ' for $\mu = 0$ and $\Sigma = \mathbb{J}_n$')
#plt.savefig('./figures/part_v_correlated.png', bbox_inches='tight')


# Adjusting covariance matrix to evaluate the impact
sigma2 = sigma_ones*0.1 + sigma*0.1
sigma2


rnd_func_plot(x, mean, sigma2)
#plt.savefig('./figures/part_v_mean0_cov01.png', bbox_inches='tight') 


# Adjusting mean and covariance to evaluate the impact
sigma3 = sigma_ones*0.1 + sigma*0.05
mu3 = mean - 1

rnd_func_plot(x, mu3, sigma2)


# Examine impact of increasing the variance of X_i, for all i = 1, ..., 500.
fig = plt.figure(figsize = (20, 14))
spec = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
d = np.linspace(0.1, 1, 8)
for i, alpha in enumerate(d):
    sigma_exp = alpha*sigma 
    y = multivariate_normal.rvs(mean, sigma_exp, size=4)
    for yi in y:
        plt.subplot(2,4,i+1)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.title('$\Sigma = $' + str(alpha)[:3] + ' and $\mu = 0$')
        plt.plot(x,yi)
#plt.savefig('./figures/part_v_sigma.png', bbox_inches='tight') 


# Examine impact of varying the mean 
fig = plt.figure(figsize = (20, 14))
spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
d = [-2, -1.5, -1, 1, 1.5, 2]
for i, alpha in enumerate(d):
    mu_exp = mean + d[i]
    sigma_exp = sigma*0.1
    y = multivariate_normal.rvs(mu_exp, sigma*0.1, size=4)
    for yi in y:
        plt.subplot(2,3,i+1)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.title('$\mu = $' + str(d[i]) + ' and $\Sigma = I*0.1$')
        plt.plot(x,yi)
#plt.savefig('./figures/part_v_mu.png', bbox_inches='tight') 


# Examine impact of varying all entries of the covariance matrix.
fig = plt.figure(figsize = (20, 14))
spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
d = np.linspace(0.1, 1, 6)     # varying the diagonal entries
od = np.linspace(0.05, 2, 6)   # varying the off-diagonal entries
#d = [-2, -1.5, -1, 1, 1.5, 2]
for i, alpha in enumerate(d):
    #mu_exp = mean + d[i]
    sigma_exp = sigma*d[i] + sigma_ones*od[i]
    y = multivariate_normal.rvs(mean, sigma_exp, size=4)
    for yi in y:
        plt.subplot(2,3,i+1)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.title('$\mu = 0, \ \Sigma_{ii} = $' + str(d[i]) +  ' and $\Sigma_{ij} = $'  + str(od[i]))
        plt.plot(x,yi)
#plt.savefig('./figures/part_v_sigma2.png', bbox_inches='tight') 


### Part (vi)


# Define the given kernel
def kernel_fnc(X):
    k = np.zeros((N,N))
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            k[i,j]=np.exp(-((X[i]-X[j])**2)/5)
    return k


k = kernel_fnc(x)
rnd_func_plot(x, mean, k, ' for $\Sigma = K(x_i, x_j) = \exp\{(x_i - x_j)^2/5\}$')
#plt.savefig('./figures/part_vi_kernel.png', bbox_inches='tight') 


### Part (vii)

# Define the periodic kernel
def periodic_kernel(X):
    k = np.zeros((N,N))
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            k[i,j]=np.exp(-(2*np.square(np.sin((X[i]-X[j])/2))/9))
    return k


k2 = periodic_kernel(x)
rnd_func_plot(x, mean, k2, ' with $\Sigma =$ Periodic Kernel')
#plt.savefig('./figures/part_vii.png', bbox_inches='tight') 


### Part (ix) 


# Re-defining kernel with 2 arguments to account for the training data.
def K(X1, X2):
    k = np.zeros((X1.shape[0], X2.shape[0]))
    for i, t in enumerate(X1):
        for j, t in enumerate(X2):
            k[i,j]=np.exp(-np.square(X1[i]-X2[j])/5)
    return k


x_train = np.array([-6, 0, 7])
y_train = np.array([3, -2, 2])
#X_join = np.concatenate((x, x_train), axis=0)


mu_n = np.zeros((x.shape[0]))
mu_m = np.zeros((x_train.shape[0]))

K11 = K(x,x)                             # entry (1,1) in posterior covariance (block) matrix
K12 = K(x,x_train)                       # entry (1,2) in posterior covariance (block) matrix
K21 = K(x_train,x)                       # entry (2,1) in posterior covariance (block) matrix
K22 = K(x_train, x_train)                # entry (2,2) in posterior covariance (block) matrix
K1 = np.concatenate((K11, K12), axis=1)  # adjoining entries (1,1) and (1,2)
K2 = np.concatenate((K21, K22), axis=1)  # adjoining entries (2,1) and (2,2)
K = np.concatenate((K1, K2), axis=0)     # combining into full covariance matrix for the posterior


Mu_K = mu_n + K12@np.linalg.inv(K22)@(y_train - mu_m)   # mean of the posterior of Y|Y_bar
Sigma_K = K11 - K12@np.linalg.inv(K22)@K21              # covariance of the posterior of Y|Y_bar



# Plot all four random functions in same plot with training data points
fig = plt.figure(figsize = (15,10))
for _ in range(4):
    y = multivariate_normal.rvs(Mu_K, Sigma_K)   # generate random values from MVN with posterior mean and covariance
    plt.xlim(x_min,x_max)
    plt.ylim(y_min-0.5,y_max+0.5)
    plt.ylabel('Function Value', size=17)
    plt.xlabel('$x_i$', size=20)
    plt.plot(x, y, marker='.')
    plt.plot(x_train,y_train, 'bo', markersize=12)        # mark/emphasize training points
#plt.savefig('./figures/part_ix.png', bbox_inches='tight') 


### Part (x)

# Re-defining periodic kernel with 2 arguments to account for training data.
def periodic_K(X1, X2):
    k = np.zeros((X1.shape[0], X2.shape[0]))
    for i, t in enumerate(X1):
        for j, t in enumerate(X2):
            k[i,j]=np.exp(-(2*np.square(np.sin((X1[i]-X2[j])/2))/9))   # periodic kernel (sourced from 'Gaussian Processes for Machine Learning')
    return k



K_periodic11 = periodic_K(x,x)                                # entry (1,1) in posterior covariance (block) matrix
K_periodic12 = periodic_K(x,x_train)                          # entry (1,2) in posterior covariance (block) matrix
K_periodic21 = periodic_K(x_train,x)                          # entry (2,1) in posterior covariance (block) matrix
K_periodic22 = periodic_K(x_train, x_train)                   # entry (2,2) in posterior covariance (block) matrix
K_1 = np.concatenate((K_periodic11, K_periodic12), axis=1)    # adjoining entries (1,1) and (1,2)
K_2 = np.concatenate((K_periodic21, K_periodic22), axis=1)    # adjoining entries (2,1) and (2,2)
K_periodic = np.concatenate((K_1, K_2), axis=0)               # combining into full covariance matrix for the posterior


mu_periodic = mu_n + K_periodic12@np.linalg.inv(K_periodic22)@(y_train - mu_m)          # mean of the posterior of Y|Y_bar
sigma_periodic = K_periodic11 - K_periodic12@np.linalg.inv(K_periodic22)@K_periodic21   # covariance of the posterior of Y|Y_bar


# Plotting all four random functions in same plot with training data points
fig = plt.figure(figsize = (15,10))
for _ in range(4):
    y = multivariate_normal.rvs(mu_periodic, sigma_periodic)
    plt.ylabel('Function Value', size=17)
    plt.xlabel('$x_i$', size=20)
    plt.title('All Four Random Functions', size=20)
    plt.plot(x, y, '-')
    plt.plot(x_train,y_train, 'bo', markersize=9)    
#plt.savefig('./figures/part_x_all4.png', bbox_inches='tight') 


# Plotting random functions individually
fig = plt.figure(figsize = (15,11))
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ind = ['1','2','3','4']
for i in range(4):
    y = multivariate_normal.rvs(mu_periodic, sigma_periodic)
    plt.subplot(2,2,i+1)
    plt.ylabel('Function Value', size=13)
    #plt.title('Random Function ' + ind[i])
    plt.plot(x, y)
    plt.plot(x_train,y_train, 'bo', markersize=7)  
#plt.savefig('./figures/part_x_individual.png', bbox_inches='tight') 


### Part (xii)

# Define mean of the posterior for part (ix) and (x), respectively.
mean_posterior = K12@np.linalg.inv(K22)@y_train
mean_periodic_posterior = K_periodic12@np.linalg.inv(K_periodic22)@y_train

# Mean function from part (ix)
fig = plt.figure(figsize = (12,8))
y = multivariate_normal.rvs(mean_posterior, Sigma_K)
plt.ylabel('Function Value', size=17)
plt.xlabel('$x_i$', size=20)
plt.title('Mean Function', size=20)
plt.plot(x, y)   
#plt.savefig('./figures/part_xii_posterior.png', bbox_inches='tight') 


# Mean function from part (x)
fig = plt.figure(figsize = (12,8))
y = multivariate_normal.rvs(mean_periodic_posterior, sigma_periodic)
plt.ylabel('Function Value', size=17)
plt.xlabel('$x_i$', size=20)
plt.title('Mean Periodic Function', size=20)
plt.plot(x, y)  
#plt.savefig('./figures/part_xii_periodic.png', bbox_inches='tight') 




