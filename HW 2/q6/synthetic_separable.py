import numpy as np
from perceptron import binaryPerceptron,multiclassPerceptron
from perceptron_v0 import binaryPerceptron0,multiclassPerceptron0
from perceptron_v1 import binaryPerceptron1,multiclassPerceptron1
from perceptron_v2 import binaryPerceptron2,multiclassPerceptron2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from kernel_perceptron import KernelPerceptron,MulticlassKernelPerceptron

SEP=0.75
NUM_FEATS = 100
NUM_SAMPLES = 1000
NUM_FOLDS = 1

X,y = make_classification(n_samples=NUM_SAMPLES,n_features=NUM_FEATS,n_informative=NUM_FEATS,n_redundant=0,n_repeated=0,n_classes=5,class_sep=SEP,flip_y=0)

#print('y')
#print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y)
#print('labels',set(y_train))
y_train_copy = y_train.copy()
y_train_copy2 = y_train.copy()
y_train_copy3 = y_train.copy()
y_train_copy4 = y_train.copy()
y_train_copy5 = y_train.copy()
y_train_copy6 = y_train.copy()
y_train_copy7 = y_train.copy()
y_train_copy8 = y_train.copy()
y_train_copy9 = y_train.copy()
y_train_copy10 = y_train.copy()
#print('ckpoint1')
#print(y_train)

p = binaryPerceptron(X_train,y_train,1000)
pred = p.predict(X_test)
#print(set(pred))

#print('ckpoint2')
#print(y_train_copy)
pm = multiclassPerceptron(X_train,y_train_copy,1000)
pred2 = pm.predict(X_test)

#print('ckpoint3')
#print(y_train_copy)
p0 = binaryPerceptron0(X_train,y_train_copy,1000)
p0_pred = p0.predict(X_test)

#print('y_train',y_train_copy2)
#print('y_test',y_test)

p0_multi = multiclassPerceptron0(X_train,y_train_copy2,1000)
p0_m_pred = p0_multi.predict(X_test)

print('Binary perceptron {}, multi {}'.format(accuracy_score(y_test,pred),accuracy_score(y_test,pred2)))

p1 = binaryPerceptron1(X_train,y_train_copy3,1000)
p1_pred = p1.predict(X_test)

p1_multi = multiclassPerceptron1(X_train,y_train_copy4,1000)
p1_m_pred = p1_multi.predict(X_test)

p2 = binaryPerceptron2(X_train,y_train_copy5,1000)
p2_pred = p2.predict(X_test)

p2_multi = multiclassPerceptron2(X_train,y_train_copy6,1000)
p2_m_pred = p2_multi.predict(X_test)


pk3 = KernelPerceptron(X_train,y_train_copy9)
pk3_pred = pk3.predict(X_test)

pk3m = MulticlassKernelPerceptron(X_train,y_train_copy10)
pk3m_pred = pk3m.predict(X_test)

print('v0 binary: {}, v0 multi: {}'.format(accuracy_score(y_test,p0_pred),accuracy_score(y_test,p0_m_pred)))
print('v1 binary: {}, v1 multi: {}'.format(accuracy_score(y_test,p1_pred),accuracy_score(y_test,p1_m_pred)))
print('v2 binary: {}, v2 multi: {}'.format(accuracy_score(y_test,p2_pred),accuracy_score(y_test,p2_m_pred)))
print('kernel textbook: {}'.format(accuracy_score(y_test,pk3_pred)))
print('kernel textbook multi: {}'.format(accuracy_score(y_test,pk3m_pred)))
####################################################################

#X,y = make_classification(n_samples=NUM_SAMPLES,n_features=NUM_FEATS,n_informative=NUM_FEATS,n_redundant=0,n_repeated=0,n_classes=3,class_sep=SEP,flip_y=0)

#X_train,X_test,y_train,y_test = train_test_split(X,y)

#p3 = multiclassPerceptron(X_train,y_train,500)
#pred = p3.predict(X_test)

#print('Multiclass perceptron {}'.format(accuracy_score(y_test,pred)))

