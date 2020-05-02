import random
from jax import grad
import jax.numpy as np

def rmsprop(f, start, step, max_iter, prec):
    x_hist = []
    y_hist = []
    x = start
    iters = 0
    delta_x = 0
    eps = 1e-8
    beta = .9

    print(x.shape)
    grad_sq = np.zeros(x.shape[0])
    while True:
        l = [x for x in range(500)]
        #random.shuffle(l)
        for i in l:
            print(x)
            last_x = x

            gradm = grad(f(i))(x)
            grad_sq = beta * grad_sq + (1-beta) * gradm**2
            x = x - (step/np.sqrt(grad_sq + eps)) * gradm

            delta_x = x - last_x
            #if np.linalg.norm(delta_x) < prec or iters > max_iter:
            if iters > max_iter:
                return x_hist, y_hist
            x_hist.append(x)
            y_hist.append(f(i)(x))
            iters += 1

    return x_hist, y_hist
