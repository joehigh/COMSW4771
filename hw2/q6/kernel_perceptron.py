import numpy as np

class KernelPerceptron:
	
    def __init__(self,X,y,times_through_data=10,polynomial_degree=5):
	
        self.p = polynomial_degree		
        self.n,self.d = X.shape
		
        y[y==0]=-1
		
        self.alpha = np.zeros(self.n,dtype=np.float64)		
        self.X = X
        self.y = y		
        self.bias = 0.0
		
        iter = 0		
        while iter < times_through_data:			
            for j,x_j in enumerate(self.X):	
                print(j)
                a = self.activation(x_j)				
                if np.multiply(self.y[j],a) <= 0:					
                    self.alpha[j] += self.y[j]					
                    self.bias += self.y[j]
            iter +=1
					
    def activation(self,x):		
        return sum([np.multiply(self.alpha[m],self.K(self.X[m],x))+self.bias for m in range(self.n)])
		
    def K(self,x_1,x_2):		
        return (1+np.dot(x_1,x_2))**self.p 
		
    def predict(self,X,return_dists=False):
        if not return_dists:
            return [np.max([np.sign(self.activation(x)),0]) for x in X]
        else:
            return [self.activation(x) for x in X]
            
class MulticlassKernelPerceptron:

    def __init__(self,X,y,times_through_data=10,polynomial_degree=5):
        self.p = polynomial_degree
        classes = set(y)
        self.num_classes = len(classes)
        n,d = X.shape
        self.per_cls_alpha = []
        self.per_cls_bias = []
        self.per_cls_X = []
        y_onehot = self.make_one_hot(y)
        
        for cls in list(classes):
            y_all_vs_one = np.array([y_oh[cls] for y_oh in y_onehot])
            p = KernelPerceptron(X,y_all_vs_one,times_through_data=times_through_data,polynomial_degree=polynomial_degree)
            self.per_cls_alpha.append(p.alpha)
            self.per_cls_bias.append(p.bias)
            self.per_cls_X.append(p.X)
 
    def activation(self,x,alpha,bias,X):
        return sum([np.multiply(alpha[m],self.K(X[m],x))+bias for m in range(X.shape[0])])
		
    def K(self,x_1,x_2):		
        return (1+np.dot(x_1,x_2))**self.p 
 
    def make_one_hot(self,y):
        y_onehot = np.zeros((y.size,y.max()+1))
        y_onehot[np.arange(y.size),y] =1 
        return y_onehot
        
    def predict(self,X,return_dist=False):
        preds = np.zeros(X.shape[0])
        distances = np.zeros(X.shape[0])
        for i,x in enumerate(X):
            print('predicting',i)
            a = [self.activation(x,self.per_cls_alpha[cls],self.per_cls_bias[cls],self.per_cls_X[cls]) for cls in range(len(self.per_cls_bias))]
            print(a,np.argmax(a),np.max(a))
            preds[i] = np.argmax(a)
            distances[i] = np.max(a)
        if not return_dist:
            return preds
        else:
            return preds,distances