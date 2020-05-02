import random
from jax import grad
import jax.numpy as np

def adadelta(f, start, step, max_iter, prec):
    x_hist = []
    y_hist = []
    x = start
    iters = 0
    delta_x = 0
    eps = 1e-8
    rho = .95

    print(x.shape)
    eg = np.zeros(x.shape[0])
    ed = np.zeros(x.shape[0])
    while True:
        l = [x for x in range(500)]
        #random.shuffle(l)
        for i in l:
            print(x)
            last_x = x

            g = grad(f(i))(x)
            eg = rho * eg + (1 - rho)*g**2
            cur_delta = np.sqrt(ed + eps) / np.sqrt(eg + eps) * g
            ed = rho * ed + (1 - rho) * cur_delta**2
            x -= cur_delta

            delta_x = x - last_x
            #if np.linalg.norm(delta_x) < prec or iters > max_iter:
            if iters > max_iter:
                return x_hist, y_hist
            x_hist.append(x)
            y_hist.append(f(i)(x))
            iters += 1

    return x_hist, y_hist
