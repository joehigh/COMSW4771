from jax import grad
import jax.numpy as np

def gd(f, start, step, max_iter, prec):
    #x_hist = [start]
    #y_hist = [f(start)]
    x_hist = []
    y_hist = []
    x = start
    for _ in range(max_iter):
        print(x)
        last_x = x
        x = x - step * grad(f)(x)
        if np.linalg.norm(x - last_x) < prec:
            break
        x_hist.append(x)
        y_hist.append(f(x))
    return x_hist, y_hist
