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

CONFIRM_DATA = False
NUM_FOLDS = 3

mat_contents = sio.loadmat('mnist_digits.mat')

X = np.array(mat_contents['X'])
y = mat_contents['Y']
y = np.array([y_i[0] for y_i in y])
X_images = np.array([x.reshape((28,28)) for x in X])

# This stuff just confirms that the data was loaded correctly. It makes an image of a handwritten 9 and the annotation "9"
if CONFIRM_DATA:
    print(X.shape)
    print(X[0].shape)
    print(y.shape)
    print(X_images.shape)
    print(X_images[0].shape)
    skio.imsave('testimage.png',X_images[0])
    print(y[0])

print('loaded data')

#iters = [100,500,1000,5000]
iters = [10,100,500,1000]

#p_acc = {k:[] for k in iters}
p0_acc = {k:[] for k in iters}
p1_acc = {k:[] for k in iters}
p2_acc = {k:[] for k in iters}

for fold in range(NUM_FOLDS):
    print('fold {}'.format(fold))
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    y_train_1 = y_train.copy()
    y_train_2 = y_train.copy()
    y_train_3 = y_train.copy()
    
    for max_iters in iters:
        print('iter {}'.format(max_iters))
        #p = multiclassPerceptron(X_train,y_train,max_iters)
        #pred = p.predict(X_test)
        #p_acc[max_iters].append(accuracy_score(y_test,pred))
        
        p0 = multiclassPerceptron0(X_train,y_train_1,max_iters)
        p0_pred = p0.predict(X_test)
        p0_acc[max_iters].append(accuracy_score(y_test,p0_pred))
        
        p1 = multiclassPerceptron1(X_train,y_train_2,max_iters)
        p1_pred = p1.predict(X_test)
        p1_acc[max_iters].append(accuracy_score(y_test,p1_pred))
        
        p2 = multiclassPerceptron2(X_train,y_train_3,max_iters)
        p2_pred = p2.predict(X_test)
        p2_acc[max_iters].append(accuracy_score(y_test,p2_pred))

#p_mean = [np.mean(p_acc[k]) for k in iters]
#p_std = [np.std(p_acc[k]) for k in iters]
p0_mean = [np.mean(p0_acc[k]) for k in iters]
p0_std = [np.std(p0_acc[k]) for k in iters]
p1_mean = [np.mean(p1_acc[k]) for k in iters]
p1_std = [np.std(p1_acc[k]) for k in iters]
p2_mean = [np.mean(p2_acc[k]) for k in iters]
p2_std = [np.std(p2_acc[k]) for k in iters]

names = ['In class','Perceptron 0','Perceptron 1','Perceptron 2']

plt.figure()
#plt.errorbar(iters,p_mean,yerr=p_std,label='In Class')
plt.errorbar(iters,p0_mean,yerr=p0_std,label='Perceptron V0')
plt.errorbar(iters,p1_mean,yerr=p1_std,label='Perceptron V1')
plt.errorbar(iters,p2_mean,yerr=p2_std,label='Perceptron V2')
plt.xlabel('Maximum T')
plt.ylabel('Accuracy')
plt.title('Perceptron Algorithm Comparison {}-fold Cross Validation'.format(NUM_FOLDS))
plt.legend()
plt.savefig('perceptron_compare.png')