import scipy.io as sio
import matplotlib.pyplot as plt
import skimage.io as skio
import numpy as np
from kernel_perceptron import MulticlassKernelPerceptron
from perceptron_v2 import multiclassPerceptron2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CONFIRM_DATA = False
NUM_FOLDS = 2

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
iters = [10,20,50]

#p_acc = {k:[] for k in iters}
pk_5_acc = {k:[] for k in iters}
pk_10_acc = {k:[] for k in iters}
p2_acc = {k:[] for k in iters}

for fold in range(NUM_FOLDS):
    print('fold {}'.format(fold))
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    y_train_1 = y_train.copy()
    y_train_2 = y_train.copy()
    y_train_3 = y_train.copy()
    
    for max_iters in iters:
        print('iter {}'.format(max_iters))
        print('kernel 5')
        pk_5 = MulticlassKernelPerceptron(X_train,y_train_1,times_through_data=int(max_iters/10.0))
        pk_5_pred = pk_5.predict(X_test)
        pk_5_acc[max_iters].append(accuracy_score(y_test,pk_5_pred))
        print('kernel 10')
        pk_10 = MulticlassKernelPerceptron(X_train,y_train_2,times_through_data=int(max_iters/10.0),polynomial_degree=10)
        pk_10_pred = pk_10.predict(X_test)
        pk_10_acc[max_iters].append(accuracy_score(y_test,pk_10_pred))
        print('v2')
        p2 = multiclassPerceptron2(X_train,y_train_3,max_iters)
        p2_pred = p2.predict(X_test)
        p2_acc[max_iters].append(accuracy_score(y_test,p2_pred))


pk_5_mean = [np.mean(pk_5_acc[k]) for k in iters]
pk_5_std = [np.std(pk_5_acc[k]) for k in iters]
pk_10_mean = [np.mean(pk_10_acc[k]) for k in iters]
pk_10_std = [np.std(pk_10_acc[k]) for k in iters]
p2_mean = [np.mean(p2_acc[k]) for k in iters]
p2_std = [np.std(p2_acc[k]) for k in iters]


plt.figure()
plt.errorbar(iters,pk_5_mean,yerr=pk_5_std,label='Polynomial 5 Kernel')
plt.errorbar(iters,pk_10_mean,yerr=pk_10_std,label='Polynomial 10 Kernel')
plt.errorbar(iters,p2_mean,yerr=p2_std,label='Perceptron V2')
plt.xlabel('Maximum Iterations')
plt.ylabel('Accuracy')
plt.title('Perceptron Algorithm Comparison {}-fold Cross Validation'.format(NUM_FOLDS))
plt.legend()
plt.savefig('perceptron_compare_kernel.png')