import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as np
import numpy as onp
import random
from jax import grad

from scipy.optimize import rosen, rosen_der

from gd import gd
from sgd import sgd, bsgd
from sgdm import sgdm, bsgdm
from adagrad import adagrad
from adadelta import adadelta
from rmsprop import rmsprop
from adam import adam
from util import start_contour, plotter

# GD params
lr = .002
max_iters = 1000
prec = .001

# momentum
alpha = .7


def main():
    comparison()

def mle_test():
    mu = 1
    sigma = 2
    x = onp.random.normal(mu, sigma, 100)
    count, bins, ignored = plt.hist(x, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    plt.show()
    plt.plot(x)
    plt.show()

    #mle
    def g(X):
        return -1* likelihood(x, X[0], X[1])

    start = np.array([.25, 1.75])
    path, opt = gd(g, start, lr, max_iters, prec)
    print("Completed " + str(len(path)) + " iterations.")
    print("Found minimum at: ", path[-1])

    path = np.array(path)
    plt.plot(path[:,0], path[:,1], marker='.', linewidth=1, markersize=3)
    plt.show()

    plt.plot(opt)
    plt.show()

def gd_test():
    plotter(f, [-1, 1])

    fig, ax = plt.subplots()
    levels = np.linspace(0, 1, 15)
    CS = ax.contour(X, Y, Z, levels=levels)
    ax.clabel(CS, inline=1, fontsize=10)
    #ax.set_title('')

    start = np.array([.25, .75])
    path, opt = gd(f, start, lr, max_iters, prec)
    print("Completed " + str(len(path)) + " iterations.")
    print("Found minimum at: ", path[-1])

    path = np.array(path)
    plt.plot(path[:,0], path[:,1], marker='.', linewidth=1, markersize=3)
    plt.show()

    plt.plot(opt)
    plt.show()

def sgd_test():

    X, Y, Z = plotter(sff, [-1, 1], 1000)
    start_contour(X, Y, Z)

    start = np.array([1.25, 1.75])
    path, opt = gd(sff, start, lr, max_iters, prec)
    print("Completed " + str(len(path)) + " iterations.")
    print("Found minimum at: ", path[-1])
    path2, opt2 = sgd(ff, start, lr, max_iters, prec)
    print("Completed " + str(len(path2)) + " iterations.")
    print("Found minimum at: ", path2[-1])

    path = np.array(path)
    path2 = np.array(path2)
    plt.plot(path[:,0], path[:,1], marker='.', linewidth=1, markersize=3)
    plt.plot(path2[:,0], path2[:,1], marker='.', linewidth=1, markersize=3)
    plt.show()

    plt.plot(opt)
    plt.plot(opt2)
    plt.yscale('log')
    plt.show()

def momentum_test():
    X, Y, Z = plotter(sff, [-1, 1], 1000)
    start_contour(X, Y, Z)

    start = np.array([1.25, 1.75])
    path, opt = sgdm(ff, start, lr, alpha, max_iters, prec)
    print("Completed " + str(len(path)) + " iterations.")
    print("Found minimum at: ", path[-1])

    path = np.array(path)
    plt.plot(path[:,0], path[:,1], marker='.', linewidth=1, markersize=3)
    plt.show()

    plt.plot(opt)
    plt.show()

def comparison():
    X, Y, Z = plotter(sff, [-1, 1], 1000)
    start_contour(X, Y, Z)

    start = np.array([1.25, 1.75])
    res = []
    #res.append(gd(sff, start, lr, max_iters, prec))
    #res.append(sgd(ff, start, lr, max_iters, prec))
    #res.append(sgdm(ff, start, lr, alpha, max_iters, prec))
    #res.append(bsgd(ff, start, lr, max_iters, prec))
    #res.append(bsgdm(ff, start, lr, alpha, max_iters, prec))
    res.append(adagrad(ff, start, lr*100, max_iters/25, prec))
    res.append(rmsprop(ff, start, lr, max_iters, prec))
    res.append(adadelta(ff, start, lr*100, max_iters/5, prec))
    res.append(adam(ff, start, lr*20, max_iters/5, prec))
    paths = [r[0] for r in res]
    opts = [r[1] for r in res]

    legend = ["AdaGrad", "RMSProp", "AdaDelta", "Adam"]

    for path in paths:
        print("Completed " + str(len(path)) + " iterations.")
        print("Found minimum at: ", path[-1])
        path = np.array(path)
        plt.plot(path[:,0], path[:,1], marker='.', linewidth=1, markersize=3)
    plt.title("Optimization Paths")
    plt.legend(legend)
    plt.show()

    for opt in opts:
        plt.plot(opt)
    plt.title("Optimization Rate (log)")
    plt.legend(legend)
    plt.yscale('log')
    plt.show()

    for opt in opts:
        plt.plot(opt)#, marker='+')
    plt.title("Optimization Rate")
    plt.legend(legend)
    plt.show()


def f(X):
    return 1 - np.exp(-1 * (10*X[0]**2+X[1]**2))

def sff(W):
    return sum([ff(x)(W) for x in range(100)])

def ff(i):
    def f(W):
        return 1 - np.exp(-1 * (10*W[0]**2+W[1]**2))
    return f
    # function of w[0], w[1] at i
    #return i*np.sin(W[0]**2) * W[1]**2*np.cos(i)**2 + (W[0]+W[1])**2 + 1

def likelihood(X, mu, sig):
    return np.sum([-.5*np.log(np.pi*sig)-(1/(2*sig)*(x-mu)**2) for x in X])


if __name__ == '__main__':
    main()
