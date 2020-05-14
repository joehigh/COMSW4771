import random
from jax import grad
import jax.numpy as np
from util import chunks

def adagrad(f, start, step, max_iter, prec):
    x_hist = []
    y_hist = []
    x = start
    iters = 0
    delta_x = 0
    print(x.shape)
    gti = np.zeros(x.shape[0])
    while True:
        l = [x for x in range(500)]
        #random.shuffle(l)
        for i in l:
            print(x)
            last_x = x

            gradm = grad(f(i))(x)
            gti += gradm**2
            gradm = gradm / np.sqrt(gti + .1)
            x = x - step * gradm

            delta_x = x - last_x
            #if np.linalg.norm(delta_x) < prec or iters > max_iter:
            if iters > max_iter:
                return x_hist, y_hist
            x_hist.append(x)
            y_hist.append(f(i)(x))
            iters += 1

    return x_hist, y_hist

def badagrad(f, start, step, alpha, max_iter, prec, batch=5):
    x_hist = []
    y_hist = []
    x = start
    iters = 0
    delta_x = 0
    while True:
        l = [x for x in range(500)]
        random.shuffle(l)
        l = list(chunks(l, batch))
        for chunk in l:
            def ff(W):
                return sum([f(i)(W) for i in chunk])
            print(x)
            last_x = x
            x = x - step * grad(ff)(x) + alpha*delta_x
            delta_x = x - last_x
            if np.linalg.norm(delta_x) < prec or iters > max_iter:
                return x_hist, y_hist
            x_hist.append(x)
            y_hist.append(ff(x))
            iters += 1
    return x_hist, y_hist
