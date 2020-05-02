import scipy.io as sio
import matplotlib.pyplot as plt
import skimage.io as skio
import numpy as np
from perceptron import multiclassPerceptron
from perceptron_v0 import multiclassPerceptron0
from perceptron_v1 import multiclassPerceptron1
from perceptron_v2 import multiclassPerceptron2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

NUM_FOLDS = 3

mat_contents = sio.loadmat('mnist_digits.mat')

X = np.array(mat_contents['X'])
y = mat_contents['Y']
y = np.array([y_i[0] for y_i in y])
X_images = np.array([x.reshape((28,28)) for x in X])

sizes = [25,100,500,1000]

#p_acc = {k:[] for k in iters}
p0_acc = {k:[] for k in sizes}
p1_acc = {k:[] for k in sizes}
p2_acc = {k:[] for k in sizes}

max_iters = 500
for fold in range(NUM_FOLDS):
    print('fold {}'.format(fold))
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    y_train_1 = y_train.copy()
    y_train_2 = y_train.copy()
    y_train_3 = y_train.copy()
    
    for max_size in sizes:
        print('iter {}'.format(max_size))
        #p = multiclassPerceptron(X_train,y_train,max_iters)
        #pred = p.predict(X_test)
        #p_acc[max_iters].append(accuracy_score(y_test,pred))
        
        p0 = multiclassPerceptron0(X_train[:max_size],y_train_1[:max_size],max_iters)
        p0_pred = p0.predict(X_test)
        p0_acc[max_size].append(accuracy_score(y_test,p0_pred))
        
        p1 = multiclassPerceptron1(X_train[:max_size],y_train_2[:max_size],max_iters)
        p1_pred = p1.predict(X_test)
        p1_acc[max_size].append(accuracy_score(y_test,p1_pred))
        
        p2 = multiclassPerceptron2(X_train[:max_size],y_train_3[:max_size],max_iters)
        p2_pred = p2.predict(X_test)
        p2_acc[max_size].append(accuracy_score(y_test,p2_pred))

#p_mean = [np.mean(p_acc[k]) for k in iters]
#p_std = [np.std(p_acc[k]) for k in iters]
p0_mean = [np.mean(p0_acc[k]) for k in sizes]
p0_std = [np.std(p0_acc[k]) for k in sizes]
p1_mean = [np.mean(p1_acc[k]) for k in sizes]
p1_std = [np.std(p1_acc[k]) for k in sizes]
p2_mean = [np.mean(p2_acc[k]) for k in sizes]
p2_std = [np.std(p2_acc[k]) for k in sizes]

names = ['In class','Perceptron 0','Perceptron 1','Perceptron 2']

plt.figure()
#plt.errorbar(iters,p_mean,yerr=p_std,label='In Class')
plt.errorbar(sizes,p0_mean,yerr=p0_std,label='Perceptron V0')
plt.errorbar(sizes,p1_mean,yerr=p1_std,label='Perceptron V1')
plt.errorbar(sizes,p2_mean,yerr=p2_std,label='Perceptron V2')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Perceptron Algorithm Comparison {}-fold Cross Validation'.format(NUM_FOLDS))
plt.legend()
plt.savefig('perceptron_compare_size.png')