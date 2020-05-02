import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import jax.numpy as np
from sklearn.linear_model import LinearRegression

from util import *

n_samples = 1000
n_outliers = 50

def sk(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

def main():

    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                          n_informative=1, noise=25,
                                          coef=True, random_state=0)

    #reg = sk(X, y)

    #plt.scatter(X, y)
    #plt.plot(X, reg.predict(X), color='r')
    #plt.show()

    y = y.reshape(-1, 1)
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((X, bias), axis=1)
    X = np.concatenate((X, y), axis=1)

    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    XX, Y = np.meshgrid(x, y)
    Z = sq_loss(pred(X, XX, Y), y)
    print(Z.shape) # should be 1000 x 1000

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X, Y, Z, 100)
    ax.plot_surface(X, Y, Z, cmap='viridis')

    plt.show()
    return X, Y, Z

def pred(X, w, b):
    return np.dot(X, w) + b

def sq_loss(y_hat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

    #out = sum([func(X)(W) for x in X])
    #print(out)
    #return out

def func(X):
    def f(W):
        print(X.shape)
        print(W.shape)
        return (X[:,2] - np.dot(np.transpose(W), X[:,:2]))**2
    return f

if __name__ == '__main__':
    main()
