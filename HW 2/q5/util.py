
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as np

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def plotter(f, square_bounds, mesh_div=100):
    x = np.linspace(square_bounds[0], square_bounds[1], mesh_div)
    y = np.linspace(square_bounds[0], square_bounds[1], mesh_div)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))
    print(Z.shape)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X, Y, Z, 100)
    ax.plot_surface(X, Y, Z, cmap='viridis')

    plt.show()
    return X, Y, Z

def start_contour(X, Y, Z):
    fig, ax = plt.subplots()
    levels = np.linspace(0, 1000, 20)
    CS = ax.contour(X, Y, Z, levels=levels)
    ax.clabel(CS, inline=1, fontsize=10)
