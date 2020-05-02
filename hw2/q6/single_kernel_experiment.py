import scipy.io as sio
import matplotlib.pyplot as plt
import skimage.io as skio
import numpy as np
from kernel_perceptron import MulticlassKernelPerceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mat_contents = sio.loadmat('mnist_digits.mat')

X = np.array(mat_contents['X'])
y = mat_contents['Y']
y = np.array([y_i[0] for y_i in y])
print(y)
X_images = np.array([x.reshape((28,28)) for x in X])


X_train,X_test,y_train,y_test = train_test_split(X,y)

pk_5 = MulticlassKernelPerceptron(X_train[:100],y_train[:100],times_through_data=5)
pk_5_pred = pk_5.predict(X_test[:100])


print(accuracy_score(y_test[:100],pk_5_pred))