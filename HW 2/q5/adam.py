import random
from jax import grad
import jax.numpy as np

def adam(f, start, step, max_iter, prec):
    x_hist = []
    y_hist = []
    x = start
    iters = 0
    eps = 1e-8
    b1 = .9
    b2 = .999

    print(x.shape)
    m = np.zeros(x.shape[0])
    v = np.zeros(x.shape[0])
    while True:
        l = [x for x in range(500)]
        #random.shuffle(l)
        t = 1
        for i in l:
            print(x)
            t += 1
            last_x = x

            g = grad(f(i))(x)

            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * np.power(g, 2)
            m_hat = m / (1 - np.power(b1, t))
            v_hat = v / (1 - np.power(b2, t))
            x = x - step * m_hat / (np.sqrt(v_hat) + eps)

            delta_x = x - last_x
            #if np.linalg.norm(delta_x) < prec or iters > max_iter:
            if iters > max_iter:
                return x_hist, y_hist
            x_hist.append(x)
            y_hist.append(f(i)(x))
            iters += 1

    return x_hist, y_hist
