# perceptron as done in class

import numpy as np
import random


class binaryPerceptron0:
    def __init__(self,X,y,max_iters):
        n,d = X.shape
        
        y[y==0]=-1
        
        self.w = np.zeros(d)
        t = 1
        while t <= max_iters:
            i = t%(n+1)
            if np.multiply(y[i-1],np.dot(self.w,X[i-1])) <= 0:
                self.w = self.w+np.multiply(y[i-1],X[i-1])
            t+=1
        
    def predict(self,X,return_dist=False):
        if not return_dist:
            return [np.max([np.sign(np.dot(self.w,x)),0]) for x in X]
        else:
            return [np.dot(self.w,x) for x in X]
        
        
class multiclassPerceptron0:
    def __init__(self,X,y,max_iters):
        classes = set(y)
        self.num_classes = len(classes)
        n,d = X.shape
        self.w = np.zeros((self.num_classes,d))
        y_onehot = self.make_one_hot(y)
        self.per_cls_w = []
        
        for cls in list(classes):
            y_all_vs_one = np.array([y_oh[cls] for y_oh in y_onehot])
            p = binaryPerceptron0(X,y_all_vs_one,max_iters)
            self.per_cls_w.append(p.w)
            
    def make_one_hot(self,y):
        y_onehot = np.zeros((y.size,y.max()+1))
        y_onehot[np.arange(y.size),y] = 1
        return y_onehot
        
    def predict(self,X,return_dist=False):
        preds = np.zeros(X.shape[0])
        distances = np.zeros(X.shape[0])
        for i,x in enumerate(X):
            x_dot_w = [np.dot(w,x) for w in self.per_cls_w]
            preds[i] = np.argmax(x_dot_w)
            distances[i] = np.max(x_dot_w)
        if not return_dist:
            return preds
        else:
            return preds,distances
            
            
        
                
                
                