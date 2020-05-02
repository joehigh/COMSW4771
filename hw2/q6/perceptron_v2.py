import numpy as np
import random


class binaryPerceptron2:
    def __init__(self,X,y,max_iters):
        n,d = X.shape
        
        y[y==0]=-1
        
        self.w = [np.zeros(d), np.zeros(d)]
        self.c = [0, 0]
        self.k = 1
        
        t = 1
        while t <= max_iters:
            i = t%(n+1)
            if np.multiply(y[i-1],np.dot(self.w[self.k],X[i-1])) <= 0:
                self.w.append(self.w[self.k]+np.multiply(y[i-1],X[i-1]))
                self.c.append(1)
                self.k +=1
            else:
                self.c[self.k]+=1
            t+=1
    
    def prediction_computation(self,x):
        return [np.multiply(self.c[i],np.sign(np.dot(self.w[i],x))) for i in range(1,self.k+1)]
        
    def predict(self,X,return_dist=False):
        S = [self.prediction_computation(x) for x in X]
        preds = [np.sign(sum(s)) for s in S]
        dists = [sum(s) for s in S]
        if not return_dist:
            return [np.max([pred,0]) for pred in preds]
        else:
            return dists
        
        
class multiclassPerceptron2:
    def __init__(self,X,y,max_iters):
        classes = set(y)
        self.num_classes = len(classes)
        n,d = X.shape
        self.w = np.zeros((self.num_classes,d))
        y_onehot = self.make_one_hot(y)
        self.per_cls_w = []
        self.per_cls_k =[]
        self.per_cls_c = []
        
        for cls in list(classes):
            y_all_vs_one = np.array([y_oh[cls] for y_oh in y_onehot])
            p = binaryPerceptron2(X,y_all_vs_one,max_iters)
            self.per_cls_w.append(p.w)
            self.per_cls_c.append(p.c)
            self.per_cls_k.append(p.k)
            
    def make_one_hot(self,y):
        y_onehot = np.zeros((y.size,y.max()+1))
        y_onehot[np.arange(y.size),y] = 1
        return y_onehot

    def prediction_computation(self,x,w,c,k):
        return [np.multiply(c[i],np.sign(np.dot(w[i],x))) for i in range(1,k+1)]
        
    def predict(self,X,return_dist=False):
        preds = []
        dists = []
        for x in X:
            per_cls_dists = []
            for cls,w in enumerate(self.per_cls_w):
                per_cls_dists.append(sum(self.prediction_computation(x,w,self.per_cls_c[cls],self.per_cls_k[cls])))
            preds.append(np.argmax(per_cls_dists))
            dists.append(np.max(per_cls_dists))
        if not return_dist:
            return preds
        else:
            return preds,dists
            
            
        
                
                
                