import math
import numpy as np
from matplotlib import pyplot as plt
from bvp_methods import fd, shoot


def f(y, t):
    """ Re-written version of 2nd order ODE y'' = y + 4e^t """
    return_y = np.zeros(np.shape(y))
    return_y[0] = y[1]
    return_y[1] = y[0] + 4*np.exp(t)
    return return_y


def get_y_analytic(t):
    """ Analytic solution to y'' = y + 4e^t with y(0)=1 and y(1/2) = 2e^(1/2)"""
    return np.exp(t)*(1 + 2*t)


def practice():
    """ Solves y'' = y + 4exp(t), y(0)=1, y(1/2) = 2exp(1/2) using both the
         finite difference method and the shooting method. """

    a, b = 0.0, 0.5
    n_points = 10
    t1 = np.linspace(a, b, n_points)
    t2 = np.linspace(a, b, 200)
    y_analytic = get_y_analytic(t2)
    y_fd = fd(4*np.exp(t1), 1, 0, t1, 1, 2*np.exp(0.5))
    y_s = shoot(f, np.exp(a), 2*np.exp(b), 3.0, 4.0, t1, 1e-5)
    plt.plot(t2, y_analytic, label="analytic solution")
    plt.plot(t1, y_fd, ls='--', marker='o', label="finite difference method")
    plt.plot(t1, y_s, ls='--', marker='o', label="shooting method")
    plt.legend()
    plt.show()


practice()
