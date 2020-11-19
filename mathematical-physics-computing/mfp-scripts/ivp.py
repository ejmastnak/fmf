import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import re
import time

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

color_gray_ref = "#AAAAAA"  # light gray for reference quantities

# data_dir = "data/"
data_dir = "/Users/ejmastnak/Documents/Media/academics/fmf-media-winter-3/mafiprak/ivp/"
error_dir = data_dir + "error-step/"
time_dir = data_dir + "times/"
step_global_error_dir = data_dir + "stepsize-vs-global-error/"
step_local_error_dir = data_dir + "stepsize-vs-local-error/"

figure_dir = "../6-initial-value-problem/figures/"
save_figures = True

max_err_str_index = 13  # each data file has the header "# Max error:,[max_error]". The number starts at str index 13

# Temperature equation parameters
k = 0.1
T_ext = -5  # external temperature
T0 = -15.0  # initial temperature [Celsius]
A = 1.0  # amplitude of oscillating temperature term
delta = 10  # phase lag [hours] of oscillating term

# simulation parameters
h = 1  # time step [hours]
tmin = 0  # start time
tmax = 72  # end time
hmin = 1e-5  # universal min step size for adaptive step methods
hmax = 1e1   # universal max step size for adaptive step methods

# -----------------------------------------------------------------------------
# START FIRST ORDER ODE METHODS
# -----------------------------------------------------------------------------


"""A variety of methods to solve first order ordinary differential equations.

AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    Based Octave functions written in the spring of 1999
    Python version: March 2008, October 2008
    
    I implemented the rk3r, rk3ssp, rk438, rk4r, bs23, rkf45, ck45 and dp45 based on Senning's code
"""


def euler(f, x0, t):
    """Euler's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = euler(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        x[i+1] = (x[i] + h * f(x[i], t[i]))

    return x


def heun(f, x0, t):
    """Heun's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = heun(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """
    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + k1, t[i+1])
        x[i+1] = x[i] + (k1 + k2) / 2.0

    return x


def rk2a(f, x0, t):
    """Second-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.
       Also known as the explicit midpoint method

    USAGE:
        x = rk2a(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Analysis", 6th Edition, by Burden and Faires, Brooks-Cole, 1997.
    """

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i]) / 2.0
        x[i+1] = x[i] + h * f(x[i] + k1, t[i] + h / 2.0)

    return x


def rk2b(f, x0, t):
    """Second-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk2b(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Mathematics and Computing" 4th Edition, by Cheney and Kincaid,
        Brooks-Cole, 1999.
    """

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + k1, t[i+1])
        x[i+1] = x[i] + (k1 + k2) / 2.0

    return x


def rk3r(f, x0, t):
    """Ralston's third-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk3r(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(x[i] + 0.75 * k2, t[i] + 0.75 * h)
        x[i+1] = x[i] + (2 * k1/9 + k2/3 + 4 * k3 / 9)

    return x


def rk3ssp(f, x0, t):
    """Third-order strong stability preserving Runge-Kutta method (SSPRK3) to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk3ssp(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + k1, t[i+1])
        k3 = h * f(x[i] + 0.25 * k1 + 0.25 * k2, t[i] + 0.5 * h)
        x[i+1] = x[i] + (k1/6 + k2/6 + 2 * k3/3)

    return x


def rk4(f, x0, t):
    """Classic fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(x[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(x[i] + k3, t[i+1])
        x[i+1] = x[i] + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    return x


def rk438(f, x0, t):
    """ 3/8 Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk438(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + k1/3, t[i] + h/3)
        k3 = h * f(x[i] - k1/3 + k2, t[i] + 2 * h/3)
        k4 = h * f(x[i] + k1 - k2 + k3, t[i+1])
        x[i+1] = x[i] + (k1 + 3.0 * (k2 + k3) + k4) / 8.0

    return x


def rk4r(f, x0, t):
    """ Ralston's fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.
    Coefficients from http://www.mymathlib.com/c_source/diffeq/runge_kutta/runge_kutta_ralston_4.c

    USAGE:
        x = rk4r(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    sqrt5 = np.sqrt(5)

    # Coefficients used to compute the independent variable argument of f

    a2 = 0.4
    a3 = (14.0 - 3.0 * sqrt5) / 16.0

    # Coefficients used to compute the dependent variable argument of f

    b21 = 0.4
    b31 = (-2889.0 + 1428.0 * sqrt5) / 1024.0
    b32 = (3785.0 - 1620.0 * sqrt5) / 1024.0
    b41 = (-3365.0 + 2094.0 * sqrt5) / 6040.0
    b42 = (-975.0 - 3046.0 * sqrt5) / 2552.0
    b43 = (467040.0 + 203968.0 * sqrt5) / 240845.0

    # Coefficients used to compute 3rd order RK estimate

    c1 = (263.0 + 24.0 * sqrt5) / 1812.0
    c2 = (125.0 - 1000.0 * sqrt5) / 3828.0
    c3 = 1024.0 * (3346.0 + 1623.0 * sqrt5) / 5924787.0
    c4 = (30.0 - 4.0 * sqrt5) / 123.0

    n = len(t)
    x = np.array([x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + b21*k1, t[i] + a2*h)
        k3 = h * f(x[i] + b31 * k1 + b32 * k2, t[i] + a3*h)
        k4 = h * f(x[i] + b41 * k1 + b42 * k2 + b43 * k3, t[i+1])
        x[i+1] = x[i] + (c1*k1 + c2*k2 + c3*k3 + c4*k4)

    return x


def rk45(f, x0, t):
    """Fourth-order Runge-Kutta method with error estimate.

    USAGE:
        x, err = rk45(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        err   - np array containing estimate of errors at each step.  If
                a system is being solved, err will be an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Mathematics and Computing" 6th Edition, by Cheney and Kincaid,
        Brooks-Cole, 2008.
    """

    # Coefficients used to compute the independent variable argument of f

    c20 = 2.500000000000000e-01  # 1/4
    c30 = 3.750000000000000e-01  # 3/8
    c40 = 9.230769230769231e-01  # 12/13
    c50 = 1.000000000000000e+00  # 1
    c60 = 5.000000000000000e-01  # 1/2

    # Coefficients used to compute the dependent variable argument of f

    c21 = 2.500000000000000e-01  # 1/4
    c31 = 9.375000000000000e-02  # 3/32
    c32 = 2.812500000000000e-01  # 9/32
    c41 = 8.793809740555303e-01  # 1932/2197
    c42 = -3.277196176604461e+00  # -7200/2197
    c43 = 3.320892125625853e+00  # 7296/2197
    c51 = 2.032407407407407e+00  # 439/216
    c52 = -8.000000000000000e+00  # -8
    c53 = 7.173489278752436e+00  # 3680/513
    c54 = -2.058966861598441e-01  # -845/4104
    c61 = -2.962962962962963e-01  # -8/27
    c62 = 2.000000000000000e+00  # 2
    c63 = -1.381676413255361e+00  # -3544/2565
    c64 = 4.529727095516569e-01  # 1859/4104
    c65 = -2.750000000000000e-01  # -11/40

    # Coefficients used to compute 4th order RK estimate

    a1 = 1.157407407407407e-01  # 25/216
    a2 = 0.000000000000000e-00  # 0
    a3 = 5.489278752436647e-01  # 1408/2565
    a4 = 5.353313840155945e-01  # 2197/4104
    a5 = -2.000000000000000e-01  # -1/5

    b1 = 1.185185185185185e-01  # 16.0/135.0
    b2 = 0.000000000000000e-00  # 0
    b3 = 5.189863547758284e-01  # 6656.0/12825.0
    b4 = 5.061314903420167e-01  # 28561.0/56430.0
    b5 = -1.800000000000000e-01  # -9.0/50.0
    b6 = 3.636363636363636e-02  # 2.0/55.0

    n = len(t)
    x = np.array([x0] * n)
    e = np.array([0 * x0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + c21 * k1, t[i] + c20 * h)
        k3 = h * f(x[i] + c31 * k1 + c32 * k2, t[i] + c30 * h)
        k4 = h * f(x[i] + c41 * k1 + c42 * k2 + c43 * k3, t[i] + c40 * h)
        k5 = h * f(x[i] + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4, t[i] + h)
        k6 = h * f(x[i] + c61 * k1 + c62 * k2 + c63 * k3 + c64 * k4 + c65 * k5, t[i] + c60 * h)

        x[i+1] = x[i] + a1 * k1 + a3 * k3 + a4 * k4 + a5 * k5
        x5 = x[i] + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6

        e[i+1] = abs(x5 - x[i+1])

    return x, e


def bs23(f, a, b, x0, tol, hmax, hmin):
    """Bogacki–Shampine adaptive step method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        t, x = bs23(f, a, b, x0, tol, hmax, hmin)

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        a     - left-hand endpoint of interval (initial condition is here)
        b     - right-hand endpoint of interval
        x0    - initial x value: x0 = x(a)
        tol   - maximum value of local truncation error estimate
        hmax  - maximum step size
        hmin  - minimum step size

    OUTPUT:
        t     - np array of independent variable values
        x     - np array of corresponding solution function values

    NOTES:
        This function implements 4th-5th order Runge-Kutta-Fehlberg Method
        to solve the initial value problem

           dx
           -- = f(x,t),     x(a) = x0
           dt

        on the interval [a,b].

        Adapted from from https://en.wikipedia.org/wiki/Bogacki-Shampine_method
    """

    # Coefficients used to compute 2nd order RK estimate
    c21 = 2.916666666666666e-1  # 7/24
    c22 = 2.500000000000000e-1  # 1/4
    c23 = 3.333333333333333e-1   # 1/3
    c24 = 1.250000000000000e-1   # 1/8

    # Coefficients used to compute 3rd order RK estimate
    c31 = 2.222222222222222e-1   # 2/9
    c32 = 3.333333333333333e-1   # 1/3
    c33 = 4.444444444444444e-1   # 4/9

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
    t = a
    x = x0
    h = hmax

    # Initialize arrays that will be returned
    T = np.array([t])
    X = np.array([x])

    while t < b:

        # Adjust step size when we get to last interval
        if t + h > b:
            h = b - t

        k1 = f(x, t)
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(x + 0.75 * h * k2, t + 0.75 * h)
        x3_next = x + h * (c31 * k1 + c32 * k2 + c33 * k3)  # 3rd order approximation
        k4 = f(x3_next, t + h)
        x2_next = x + h * (c21 * k1 + c22 * k2 + c23 * k3 + c24 * k4)  # 2nd order approximation

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
        r = abs(x3_next - x2_next)
        if len(np.shape(r)) > 0:
            r = max(r)
        if r <= tol:
            t = t + h  # increment time
            x = x3_next
            T = np.append(T, t)
            X = np.append(X, [x], 0)

        # Now compute next step size, and make sure that it is not too big or too small.
        # h = h * min(max(0.84 * (tol / r)**0.25, 0.1), 4.0)
        h = get_next_h(h, r, tol)
        if h > hmax:
            h = hmax
        elif h < hmin:
            print("Error in method BS23 for tolerance {}: Step size should be smaller than {}.".format(tol, hmin))
            break

    return T, X


def rkf45(f, a, b, x0, tol, hmax, hmin):
    """Runge-Kutta-Fehlberg adaptive step method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        t, x = rkf45(f, a, b, x0, tol, hmax, hmin)

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        a     - left-hand endpoint of interval (initial condition is here)
        b     - right-hand endpoint of interval
        x0    - initial x value: x0 = x(a)
        tol   - maximum value of local truncation error estimate
        hmax  - maximum step size
        hmin  - minimum step size

    OUTPUT:
        t     - np array of independent variable values
        x     - np array of corresponding solution function values

    NOTES:
        This function implements 4th-5th order Runge-Kutta-Fehlberg Method
        to solve the initial value problem

           dx
           -- = f(x,t),     x(a) = x0
           dt

        on the interval [a,b].

        Based on pseudocode presented in "Numerical Analysis", 6th Edition,
        by Burden and Faires, Brooks-Cole, 1997.
    """

    # Coefficients used to compute the independent variable argument of f
    a2 = 2.500000000000000e-01  # 1/4
    a3 = 3.750000000000000e-01  # 3/8
    a4 = 9.230769230769231e-01  # 12/13
    a5 = 1.000000000000000e+00  # 1
    a6 = 5.000000000000000e-01  # 1/2

    # Coefficients used to compute the dependent variable argument of f
    b21 = 2.500000000000000e-01   # 1/4
    b31 = 9.375000000000000e-02   # 3/32
    b32 = 2.812500000000000e-01   # 9/32
    b41 = 8.793809740555303e-01   # 1932/2197
    b42 = -3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00   # 7296/2197
    b51 = 2.032407407407407e+00   # 439/216
    b52 = -8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00   # 3680/513
    b54 = -2.058966861598441e-01  # -845/4104
    b61 = -2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00   # 2
    b63 = -1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01   # 1859/4104
    b65 = -2.750000000000000e-01  # -11/40

    # Coefficients used to compute 4th order RK estimate
    c41 = 1.157407407407407e-01   # 25/216
    # c42 = 0.0
    c43 = 5.489278752436647e-01   # 1408/2565
    c44 = 5.353313840155945e-01   # 2197/4104
    c45 = -2.000000000000000e-01  # -1/5
    # c46 = 0.0

    # Coefficients used to compute 5th order RK estimate
    c51 = 1.185185185185185e-01  # 16/135
    # c52 = 0.0
    c53 = 5.189863547758284e-01  # 6656/12825
    c54 = 5.061314903420166e-01  # 28561/56430
    c55 = -1.80000000000000e-01  # -9/50
    c56 = 3.636363636363636e-02  # 2/55

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK estimate.
    r1 = c51 - c41  # (16/135) - (25/216)
    # r2 = 0.0
    r3 = c53 - c43  # (6656/12825) - (1408/2565)
    r4 = c54 - c44  # (28561/56430) - (2197/4104)
    r5 = c55 - c45  # (-9/50) - (-1/5)
    r6 = c56        # (2/55) - 0

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
    t = a
    x = x0
    h = hmax

    # Initialize arrays that will be returned

    T = np.array([t])
    X = np.array([x])

    while t < b:

        # Adjust step size when we get to last interval
        if t + h > b:
            h = b - t

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.
        k1 = h * f(x, t)
        k2 = h * f(x + b21 * k1, t + a2 * h)
        k3 = h * f(x + b31 * k1 + b32 * k2, t + a3 * h)
        k4 = h * f(x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h)
        k5 = h * f(x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h)
        k6 = h * f(x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * h)

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)
        if len(np.shape(r)) > 0:
            r = max(r)
        if r <= tol:
            t = t + h  # increment time
            x = x + c41 * k1 + c43 * k3 + c44 * k4 + c45 * k5
            T = np.append(T, t)
            X = np.append(X, [x], 0)

        # Now compute next step size, and make sure that it is not too big or too small.
        # h = h * min(max(0.84 * (tol / r)**0.25, 0.1), 4.0)
        h = get_next_h(h, r, tol)

        if h > hmax:
            h = hmax
        elif h < hmin:
            print("Error in method RKF45 for tolerance {}: Step size should be smaller than {}.".format(tol, hmin))
            break

    return T, X


def ck45(f, a, b, x0, tol, hmax, hmin):
    """Cash-Karp adaptive step method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        t, x = ck45(f, a, b, x0, tol, hmax, hmin)

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        a     - left-hand endpoint of interval (initial condition is here)
        b     - right-hand endpoint of interval
        x0    - initial x value: x0 = x(a)
        tol   - maximum value of local truncation error estimate
        hmax  - maximum step size
        hmin  - minimum step size

    OUTPUT:
        t     - np array of independent variable values
        x     - np array of corresponding solution function values

    NOTES:
        This function implements 4th-5th order Runge-Kutta-Fehlberg Method
        to solve the initial value problem

           dx
           -- = f(x,t),     x(a) = x0
           dt

        on the interval [a,b].

        Coefficients from https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods#Cash-Karp
    """

    # Coefficients used to compute the independent variable argument of f

    a2 = 2.000000000000000e-01  # 1/5
    a3 = 3.000000000000000e-01  # 3/10
    a4 = 6.000000000000000e-01  # 3/5
    a5 = 1.000000000000000e+00  # 1
    a6 = 8.750000000000000e-01  # 7/8

    # Coefficients used to compute the dependent variable argument of f

    b21 = 2.000000000000000e-01   # 1/5
    b31 = 7.500000000000000e-02   # 3/40
    b32 = 2.250000000000000e-01   # 9/40
    b41 = 3.000000000000000e-01   # 3/10
    b42 = -9.000000000000000e-01  # -9/10
    b43 = 1.200000000000000e+00   # 6/5
    b51 = -2.037037037037037e-01  # -11/54
    b52 = 2.500000000000000e+00   # 5/2
    b53 = -2.592592592592592e+00  # -70/27
    b54 = 1.296296296296296e+00   # 35/27
    b61 = 2.949580439814811e-02   # 1631/55296
    b62 = 3.417968750000000e-01   # 175/512
    b63 = 4.159432870370370e-02   # 575/13824
    b64 = 4.003454137731481e-01   # 44275/110592
    b65 = 6.176757812500000e-02   # 253/4096

    # Coefficients used to compute 4th order RK estimate
    c41 = 1.021773726851851e-01  # 2825/27648
    # c42 = 0.0
    c43 = 3.839079034391534e-01  # 18575/48384
    c44 = 2.445927372685185e-01  # 13525/55296
    c45 = 1.932198660714285e-02  # 277/14336
    c46 = 2.500000000000000e-01  # 1/4

    # Coefficients used to compute 5th order RK estimate
    c51 = 9.788359788359788e-02  # 37/378
    # c52 = 0.0
    c53 = 4.025764895330112e-01  # 250/621
    c54 = 2.104377104377104e-01  # 125/594
    # c55 = 0.000000000000000e+00  # 0
    c56 = 2.891022021456804e-01  # 512/1771

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK estimate.
    r1 = c51 - c41  # (37/378) - (2825/27648)
    # r2 = 0.0
    r3 = c53 - c43  # (250/621) - (18575/48384)
    r4 = c54 - c44  # (125/594) - (13525/55296)
    r5 = -c45       # 0.0 - 277/14336
    r6 = c56 - c46  # (512/1771) - (1/4)

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
    t = a
    x = x0
    h = hmax

    # Initialize arrays that will be returned
    T = np.array([t])
    X = np.array([x])

    while t < b:

        # Adjust step size when we get to last interval
        if t + h > b:
            h = b - t

        # Compute values needed to compute truncation error estimate and the 4th order RK estimate.
        k1 = h * f(x, t)
        k2 = h * f(x + b21 * k1, t + a2 * h)
        k3 = h * f(x + b31 * k1 + b32 * k2, t + a3 * h)
        k4 = h * f(x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h)
        k5 = h * f(x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h)
        k6 = h * f(x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * h)

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)
        if len(np.shape(r)) > 0:
            r = max(r)
        if r <= tol:
            t = t + h  # increment time
            x = x + c41 * k1 + c43 * k3 + c44 * k4 + c45 * k5 + c46 * k6  # 4th order solution
            # x = x + c51 * k1 + c53 * k3 + c54 * k4 + c56 * k6  # 5th order solution
            T = np.append(T, t)
            X = np.append(X, [x], 0)

        # Now compute next step size, and make sure that it is not too big or too small.
        # h = h * min(max(0.84 * (tol / r)**0.25, 0.1), 4.0)
        h = get_next_h(h, r, tol)

        if h > hmax:
            h = hmax
        elif h < hmin:
            print("Error in method CK45 for tolerance {}: Step size should be smaller than {}.".format(tol, hmin))
            break

    return T, X


def dp45(f, a, b, x0, tol, hmax, hmin):
    """Dormand-Prince adaptive step method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        t, x = dp45(f, a, b, x0, tol, hmax, hmin)

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        a     - left-hand endpoint of interval (initial condition is here)
        b     - right-hand endpoint of interval
        x0    - initial x value: x0 = x(a)
        tol   - maximum value of local truncation error estimate
        hmax  - maximum step size
        hmin  - minimum step size

    OUTPUT:
        t     - np array of independent variable values
        x     - np array of corresponding solution function values

    NOTES:
        This function implements 4th-5th order Runge-Kutta-Fehlberg Method
        to solve the initial value problem

           dx
           -- = f(x,t),     x(a) = x0
           dt

        on the interval [a,b].

        Coefficients from https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods#Dormand%E2%80%93Prince

    """

    # Coefficients used to compute the independent variable argument of f
    a2 = 2.000000000000000e-1  # 1/5
    a3 = 3.000000000000000e-1  # 3/10
    a4 = 8.000000000000000e-1  # 4/5
    a5 = 8.888888888888888e-1  # 8/9
    a6 = 1.000000000000000e+0  # 1
    a7 = 1.000000000000000e+0  # 1

    # Coefficients used to compute the dependent variable argument of f
    b21 = 2.000000000000000e-1   # 1/5
    b31 = 7.500000000000000e-2   # 3/40
    b32 = 2.250000000000000e-1   # 9/40
    b41 = 9.777777777777777e-1   # 44/45
    b42 = -3.733333333333333e+1  # -56/15
    b43 = 3.555555555555555e+1   # 32/9
    b51 = 2.952598689224203e+0  # 19372/6561
    b52 = -1.159579332418838e+1   # −25360/2187
    b53 = 9.822892851699436e+0  # 64448/6561
    b54 = -2.9080932784636488e-1   # −212/729
    b61 = 2.846275252525252e+0   # 9017/3168
    b62 = -1.0757575757575757e+1   # −355/33
    b63 = 8.906422717743472e+0   # 46732/5247
    b64 = 2.784090909090909e-1   # 49/176
    b65 = -2.735313036020583e-1   # −5103/18656
    b71 = 9.114583333333333e-2   # 35/384
    # b72 = 0.000000000000000e+00  # 0.0
    b73 = 4.492362982929020e-1   # 500/1113
    b74 = 6.510416666666666e-1   # 125/192
    b75 = -3.223761792452830e-1   # −2187/6784
    b76 = 1.309523809523809e-1   # 11/84

    # Coefficients used to compute 4th order RK estimate
    c41 = 8.991319444444444e-2  # 5179/57600
    # c42 = 0.000000000000000e+00  # 0.0
    c43 = 4.534890685834082e-1  # 7571/16695
    c44 = 6.140625000000000e-1  # 393/640
    c45 = -2.715123820754717e-1  # −92097/339200
    c46 = 8.904761904761905e-2  # 187/2100
    c47 = 2.500000000000000e-2  # 1/40

    # Coefficients used to compute 5th order RK estimate
    c51 = 9.114583333333333e-2  # 35/384
    # c52 = 0.000000000000000e+00  # 0.0
    c53 = 4.492362982929021e-1  # 500/1113
    c54 = 6.510416666666667e-1  # 125/192
    c55 = -3.223761792452830e-1  # −2187/6784
    c56 = 1.309523809523810e-1  # 11/84
    # c57 = 0.000000000000000e+00  # 0.0

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK estimate.
    r1 = c51 - c41  # (35/384) - (5179/57600)
    # r2 = 0.000000000000000e+00  # 0.0
    r3 = c53 - c43  # (500/1113) - (7571/16695)
    r4 = c54 - c44  # (125/192) - (393/640)
    r5 = c55 - c45  # (−2187/6784) - (−92097/339200)
    r6 = c56 - c46  # (11/84) - (187/2100)
    r7 = -c47       # 0 - (1/40)

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
    t = a
    x = x0
    h = hmax

    # Initialize arrays that will be returned
    T = np.array([t])
    X = np.array([x])

    while t < b:
        # Adjust step size when we get to last interval
        if t + h > b:
            h = b - t

        k1 = h * f(x, t)
        k2 = h * f(x + b21 * k1, t + a2 * h)
        k3 = h * f(x + b31 * k1 + b32 * k2, t + a3 * h)
        k4 = h * f(x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h)
        k5 = h * f(x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h)
        k6 = h * f(x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * h)
        k7 = h * f(x + b71 * k1 + b73 * k3 + b74 * k4 + b75 * k5 + b76 * k6, t + a7 * h)

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 + r7 * k7)

        if len(np.shape(r)) > 0:
            r = max(r)
        if r <= tol:
            t = t + h  # increment time
            x = x + c51 * k1 + c53 * k3 + c54 * k4 + c55 * k5 + c56 * k6
            T = np.append(T, t)
            X = np.append(X, [x], 0)

        # Now compute next step size, and make sure that it is not too big or too small.
        # h = h * min(max(0.84 * (tol / r)**0.25, 0.1), 4.0)
        h = get_next_h(h, r, tol)

        if h > hmax:
            h = hmax
        elif h < hmin:
            print("Error in method DP45 for tolerance {}: Step size should be smaller than {}.".format(tol, hmin))
            break

    return T, X


def pc4(f, x0, t):
    """Adams-Bashforth-Moulton 4th order predictor-corrector method

    USAGE:
        x = pc4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a np array.  In this
                case f must return a np array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or np array
                if a system of equations is being solved.
        t     - list or np array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function used the Adams-Bashforth-Moulton predictor-corrector
        method to solve the initial value problem

            dx
            -- = f(x,t),     x(t(1)) = x0
            dt

        at the t values stored in the t array (so the interval of solution is
        [t[0], t[N-1]].  The 4th-order Runge-Kutta method is used to generate
        the first three values of the solution.  Notice that it works equally
        well for scalar functions f(x,t) (in the case of a single 1st order
        ODE) or for vector functions f(x,t) (in the case of multiple 1st order
        ODEs).

    """

    n = len(t)
    x = np.array([x0] * n)

    # Start up with 4th order Runge-Kutta (single-step method).  The extra
    # code involving f0, f1, f2, and f3 helps us get ready for the multi-step
    # method to follow in order to minimize the number of function evaluations
    # needed.
    f1 = f2 = f3 = 0
    for i in range(min(3, n - 1)):
        h = t[i+1] - t[i]
        f0 = f(x[i], t[i])
        k1 = h * f0
        k2 = h * f(x[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(x[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(x[i] + k3, t[i+1])
        x[i+1] = x[i] + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        f1, f2, f3 = (f0, f1, f2)

    # Begin Adams-Bashforth-Moulton steps

    for i in range(3, n - 1):
        h = t[i+1] - t[i]
        f0 = f(x[i], t[i])
        # predictor (Adams-Bashfort)
        w = x[i] + h * (55.0 * f0 - 59.0 * f1 + 37.0 * f2 - 9.0 * f3) / 24.0
        fw = f(w, t[i+1])
        # corrector (Adams-Moulton)
        x[i+1] = x[i] + h * (9.0 * fw + 19.0 * f0 - 5.0 * f1 + f2) / 24.0
        f1, f2, f3 = (f0, f1, f2)

    return x


def get_next_h(h, error_guess, tol):
    """Used to find optimal step size for adaptive step methods.
        Formally correct for rkf45 4(5)th order; I use it also for 2(3) order methods. """
    try:
        return h * min(max(0.84 * (tol / error_guess) ** 0.25, 0.1), 4.0)
    except ZeroDivisionError:
        return 4.0 * h
# -----------------------------------------------------------------------------
# END FIRST ORDER ODE METHODS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START AUXILARY TEMPERATURE DE FUNCTIONS
# -----------------------------------------------------------------------------

def T_simple(t, T0):
    """Analytic solution to the differential equation dT/dt = -k(T - T_ext)"""
    return T_ext + np.exp(-k*t)*(T0 - T_ext)


def dTdt_simple(T, t):
    """ Returns the derivative dT/dt = -k(T - Text)"""
    return -k * (T - T_ext)


def dTdt_oscillating(T, t):
    """ Returns the derivative dT/dt = -k(T - Text) + A*sin[(2pi/24)(t-delta)]"""
    return -k * (T - T_ext) + A*np.sin(2*np.pi*(t-delta)/24)


def nlife_time(n, T0):
    """Returns time when analytical temperature falls to 1/n of its final value"""
    return np.log(n*(T0 - T_ext)/(T0 + (1-n)*T_ext))/k


def timeT(T, T0):
    """Returns time when analytical temperature falls to T"""
    return np.log((T0-T_ext)/(T-T_ext))/k


def get_time_values(h):
    """ Returns the time values on which to solve the differential equation for given fixed step size h
         by referencing the global variable tmin and tmax """
    n = int((tmax - tmin) / h)  # number of points
    return np.linspace(tmin, tmax, n, endpoint=False)


def get_fixed_step_solution(f, T_initial, t, method):
    """
    Returns the solution to the differential equation f = dT/dt based on the inputted fixed-step method
    """
    if method == "euler":
        return euler(f, T_initial, t)
    elif method == "heun":
        return heun(f, T_initial, t)
    elif method == "rk2a":
        return rk2a(f, T_initial, t)
    elif method == "rk2b":
        return rk2b(f, T_initial, t)
    elif method == "rk3r":
        return rk3r(f, T_initial, t)
    elif method == "rk3ssp":
        return rk3ssp(f, T_initial, t)
    elif method == "rk4":
        return rk4(f, T_initial, t)
    elif method == "rk438":
        return rk438(f, T_initial, t)
    elif method == "rk4r":
        return rk4r(f, T_initial, t)
    elif method == "pc4":
        return pc4(f, T_initial, t)
    else:  # return analytic solution and include error message for no match
        print("{} does not match any options. Returning analytic solution to basic DE.".format(method))
        return T_simple(t, T_initial)


def get_adaptive_step_solution(f, t_min, t_max, T_initial, tol, hmax, hmin, method):
    """
    Returns the solution to the differential equation f = dT/dt based on the adaptive step inputted method
    """
    if method == "bs23":
        return bs23(f, t_min, t_max, T_initial, tol, hmax, hmin)
    elif method == "rkf45":
        return rkf45(f, t_min, t_max, T_initial, tol, hmax, hmin)
    elif method == "ck45":
        return ck45(f, t_min, t_max, T_initial, tol, hmax, hmin)
    elif method == "dp45":
        return dp45(f, t_min, t_max, T_initial, tol, hmax, hmin)
    else:  # return analytic solution and include error message for no match
        print("{} does not match any options. Returning analytic solution.".format(method))
        return T_simple(get_time_values((hmax + hmin)/2), T_initial)


# -----------------------------------------------------------------------------
# END AUXILARY TEMPERATURE DE FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START AUXILARY MISC FUNCTIONS
# -----------------------------------------------------------------------------

def natural_sort(l):
    """For sorting filenames in natural order"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


def clean_float(float_number):
    """Hacky way to clean up possible floating point silliness. But it works... Examples:
        0.010000000001 -> 0.01
        13232.00300003 -> 13232.003
        0.001 -> 0.001
        13252 -> 13252
       Returns a float
    """
    triggered = False  # past decimal point or not
    float_string = str(float_number)
    for index, char in enumerate(float_string):
        if char == ".": triggered = True
        if triggered and "1" <= char <= "9":
            return float(float_string[0:min(index+1, len(float_string))])  # avoid index out of bounds
    return float(float_string)


def get_method_color(method):
    """
    Returns a unique color for each DE method. Returns the default "C0" for no match
    """

    if method == "euler":
        return "#244d90"  # darker teal / blue
    elif method == "heun":
        return "#FBC490"  # peach
    elif method == "rk2a":
        return "#A82810"  # scarlet
    elif method == "rk2b":
        return "#F67B50"  # coral
    elif method == "rk3r":
        return "#18A558"  # kelly green
    elif method == "rk3ssp":
        return "#A3EBB1"  # neon green
    elif method == "rk4":
        return "#A91B60"  # pink
    elif method == "rk4r":
        return "#FF0080"  # fuchsia
    elif method == "rk438":
        return "#EC9EC0"  # light pink
    elif method == "rk45":
        return "C0"
    elif method == "bs23":
        return "#D3B1C2"  # muave
    elif method == "rkf45":
        return "#C197D2"  # lavender
    elif method == "ck45":
        return "#613659"  # orchid
    elif method == "dp45":
        return "#211522"  # dark dark purple
    elif method == "pc4":
        return "#3B0918"  # dark dark red
    else:
        print("No color match for method: {}".format(method))
        return "C0"


def get_method_marker(method):
    """
    Returns a unique marker for each DE method. Returns the default "o" for no match
    """
    if method == "euler":
        return "o"
    elif method == "heun":
        return "v"
    elif method == "rk2a":
        return "v"
    elif method == "rk2b":
        return "v"
    elif method == "rk3r":
        return "^"
    elif method == "rk3ssp":
        return "^"
    elif method == "rk4":
        return "d"
    elif method == "rk4r":
        return "d"
    elif method == "rk438":
        return "d"
    elif method == "rk45":
        return "d"
    elif method == "pc4":
        return "P"
    elif method == "rkf":
        return "o"
    elif method == "bs23":
        return "o"
    elif method == "rkf45":
        return "o"
    elif method == "ck45":
        return "o"
    elif method == "dp45":
        return "o"
    else:
        print("No marker match for method: {}".format(method))
        return "o"


# -----------------------------------------------------------------------------
# END AUXILARY MISC FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

def generate_fixed_step_solutions():
    """Saves solution data to local csv files for later analysis...
        ...to avoid calling python solution functions over and over again"""
    methods = ("euler", "heun", "rk2a", "rk2b", "rk3r", "rk3ssp", "rk4", "rk438", "rk4r", "pc4")
    for exponent in (-3, -2, -1, 0):  # powers of 10 for calculating h
        for coefficient in range(1,11):  # coefficients for h
            h = coefficient*(10**exponent)  # i.e. h = coef*10^exponent
            h = clean_float(h)  # cleans up floating point sillyness e.g. 0.7000000000004 -> 0.7
            # n = int((tmax - tmin) / h)  # number of points
            # t = np.linspace(tmin, tmax, n, endpoint=False)
            t = get_time_values(h)
            T_analytic = T_simple(t, T0)
            for method in methods:
                T_de = get_fixed_step_solution(dTdt_simple, T0, t, method)
                err = T_analytic - T_de
                max_err = np.max(np.abs(err))
                header = "Max error:,{}\nTime [h], Temp [C], Error [C]".format(max_err)
                np.savetxt(data_dir + method + "/{}.csv".format(h), np.column_stack([t, T_de, err]), delimiter=',', header=header)


def generate_adaptive_step_solutions():
    """Saves solution data to local csv files for later analysis...
        ...to avoid calling python solution functions over and over again"""
    methods = ("bs23", "rkf45", "ck45")  # dp45 is buggy
    for exponent in range(-10, 0):  # powers of 10 for calculating tolerance
        for coefficient in (1, 3, 5, 7, 9):  # coefficients for tolerance
            tol = coefficient*(10**exponent)  # i.e. tol = coef*10^exponent
            for method in methods:
                t_de, T_de = get_adaptive_step_solution(dTdt_simple, tmin, tmax, T0, tol, hmax, hmin, method)
                T_analytic = T_simple(t_de, T0)
                err = T_analytic - T_de
                max_err = np.max(np.abs(err))
                header = "Max error:,{}\nTime [h], Temp [C], Error [C]".format(max_err)
                np.savetxt(data_dir + method + "/{:.2e}.csv".format(tol), np.column_stack([t_de, T_de, err]), delimiter=',', header=header)


def compile_error_h_table():
    """Compiles a table of maximum error as a function of step size for each method"""
    methods = ("euler", "heun", "rk2a", "rk2b", "rk3r", "rk3ssp", "rk4", "rk438", "rk4r", "pc4")
    for method in methods:
        step_sizes, errors = [], []  # empty arrays to hold h and error for each method
        for filename in sorted(os.listdir(data_dir + method + "/")):
            if ".csv" in filename:
                h = float(filename.replace(".csv", ""))  # extract step size from file name
                with open(data_dir + method + "/" + filename) as f:  # extract max error from first line
                    first_line = f.readline().strip()
                    error = float(first_line[max_err_str_index:len(first_line)])
                    step_sizes.append(h)
                    errors.append(error)
        header = "Step size [h], Max abs. error [C]"
        np.savetxt(error_dir + method + ".csv", np.column_stack([step_sizes, errors]), delimiter=',', header=header, fmt=("%.1e", "%.4e"))
        # print("\n" + method)
        # print(np.column_stack([step_sizes, errors]))


def find_nearest_below(array, target):
    """ Returns index of the element in array that is closest to but still less than target
        Adapted from https://stackoverflow.com/questions/17118350/how-to-find-nearest-value-that-is-greater-in-numpy-array
    """
    difference = array - target
    mask = np.ma.greater_equal(difference, 0)
    # Masks positive differences and zero since we are looking for values below target
    if np.all(mask):
        return None  # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(difference, mask)
    return masked_diff.argmax()


def get_step_size_for_tolerance(method, tol):
    """ Finds the largest step size h for which the inputted DE method
         still finds a solution with maximum error below the tolerance tol

        If the method doesn't find a solution with max error below tol
        even for the smallest tested step size, returns smallest step size available
    """
    data = np.loadtxt(error_dir + method + ".csv", delimiter=',', skiprows=1)
    h = data[:,0]
    errors = data[:, 1]
    index = find_nearest_below(errors, tol)  # find closest match in errors still below tol
    if index is None:
        print("No adequate step size for for {} for tolerance {}. Return NaN.".format(method, tol))
        return np.nan
    return h[index]


def fixed_step_time_trial(method, h):
    """ Finds time from multiple runs each fixed-step method takes to find a solution a the step size h"""
    runs = 10
    times = np.zeros(runs)
    for i in range(0, runs):
        t = time.time()
        get_fixed_step_solution(dTdt_simple, T0, get_time_values(h), method)
        times[i] = time.time() - t
    return np.mean(times)


def adaptive_step_time_trial(method, tol):
    """ Finds time from multiple runs each adaptive-step method takes to find a solution for the tolerance tol"""
    runs = 10
    times = np.zeros(runs)
    for i in range(0, runs):
        t = time.time()
        get_adaptive_step_solution(dTdt_simple, tmin, tmax, T0, tol, hmax, hmin, method)
        times[i] = time.time() - t
    return np.mean(times)


def compile_fixed_step_times():
    """ Finds computation times as a function of tolerance for each method and
         saves the data to local files. To avoid calling code over and over... """
    methods = ("euler", "heun", "rk2a", "rk2b", "rk3r", "rk3ssp", "rk4", "rk438", "rk4r", "pc4")

    exponents = (-7, -6, -5, -4, -3, -2, -1, 0)  # powers of 10 for calculating tolerance
    coefficients = (1, 3, 5, 7, 9)  # coefficients for tolerance
    tolerances = []  # I don't feel like preallocating
    for exponent in exponents:  # powers of 10 for calculating h
        for coefficient in coefficients:  # coefficients for h
            tolerances.append(coefficient * (10 ** exponent)) # i.e. tol = coef*10^exponent
    tolerances = np.asarray(tolerances)  # convert to numpy array

    for method in methods:
        times = np.zeros(len(tolerances))
        for i, tol in enumerate(tolerances):
            h = get_step_size_for_tolerance(method, tol)
            if np.isnan(h):  # if no adequate step size found
                times[i] = np.nan
            else:
                times[i] = fixed_step_time_trial(method, h)

        header = "Tolerance [C], Computation Time [s]"
        np.savetxt(time_dir + method + ".csv", np.column_stack([tolerances, times]), delimiter=',', header=header,
                   fmt=("%.6e", "%.6e"))


def compile_adaptive_step_times():
    """ Finds computation times as a function of tolerance for each method and
         saves the data to local files. To avoid calling code over and over... """

    methods = ("bs23", "rkf45", "ck45")  # , "dp45")  # note that dp45 is buggy and bs23 is just plain wrong
    # methods = ("blank", "dp45")

    coefficients = (1, 3, 5, 7, 9)  # coefficients for tolerance
    tolerances = []  # I don't feel like preallocating
    for exponent in range(-10, 0):  # powers of 10 for calculating h
        for coefficient in coefficients:  # coefficients for h
            tolerances.append(coefficient * (10 ** exponent))  # i.e. tol = coef*10^exponent
    tolerances = np.asarray(tolerances)  # convert to numpy array

    for method in methods:
        times = np.zeros(len(tolerances))
        for i, tol in enumerate(tolerances):
            times[i] = adaptive_step_time_trial(method, tol)

        header = "Tolerance [C], Computation Time [s]"
        np.savetxt(time_dir + method + ".csv", np.column_stack([tolerances, times]), delimiter=',', header=header,
                   fmt=("%.6e", "%.6e"))


def generate_step_size_vs_error():
    """ Generates data for step size versus error with respect to the analytic solution for the classic RK4
        Goal is to show that decreasing beyond an arbitrarily small step size
        increases the methods error because of floating point arithmetic issues

        Saves data to local text file to avoid repetitive calls when plotting """

    methods = ("heun", "rk2a")  # "rk3r", "pc4")
    for method in methods:
        print(method)
        step_sizes = []  # empty array to hold step sizes
        global_errors = []  # empty array to hold global errors
        local_errors = []  # empty array to hold errors
        for exponent in range(-6, 0):  # powers of 10 for calculating tolerance
            for coefficient in (1, 2, 3, 5, 7, 9):  # coefficients for tolerance
                h = coefficient * (10 ** exponent)  # i.e. tol = coef*10^exponent
                print(h)

                t = get_time_values(h)
                T_de = get_fixed_step_solution(dTdt_simple, T0, t, method)
                T_analytic = T_simple(t, T0)
                error = np.abs(T_analytic - T_de)

                step_sizes.append(h)
                global_errors.append(np.sum(error))
                local_errors.append(np.max(error))

        np.savetxt(step_global_error_dir + "{}_.csv".format(method), np.column_stack([step_sizes, global_errors]), delimiter=',', header="Step Size [hours], Global Error [C]")
        np.savetxt(step_local_error_dir + "{}_.csv".format(method), np.column_stack([step_sizes, local_errors]), delimiter=',', header="Step Size [hours], Maximum Local Error [C]")


# -----------------------------------------------------------------------------
# END ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def plot_solution():
    """ Just a simple plot of the basic exponential equation using the analytic solution """

    h = 0.1
    t = get_time_values(h)
    T_sim = T_simple(t, T0)
    t_osc, T_osc = t, pc4(dTdt_oscillating, T0, t)  # use a numerical solution for the oscillating equation

    # t_osc, T_osc = rkf45(dTdt_oscillating, tmin, tmax, T0, 1e-6, 0.1, 0.0001)  # use a numerical solution for the oscillating equation
    # t_osc, T_osc = ck45(dTdt_oscillating, tmin, tmax, T0, 1e-6, 0.1, 0.0001)  # use a numerical solution for the oscillating equation
    # t_osc, T_osc = dp54(dTdt_oscillating, tmin, tmax, T0, 1e-6, 0.1, 0.0001)  # use a numerical solution for the oscillating equation

    plt.figure(figsize=(8, 4))
    plt.xlabel("Time $t$ [hours]", fontsize=13, labelpad=-0.1)
    plt.ylabel("Temperature $T$ [$\degree$C]", fontsize=13)
    plt.scatter(t, T_sim, c=T_sim, cmap="coolwarm", s=20, label="simple")
    plt.scatter(t_osc, T_osc, c=T_osc, cmap="coolwarm", s=20, label="oscillating")
    plt.hlines(T_ext, tmin, tmax, color="#3B4CC0", ls=':', linewidth=5, zorder=-1)


    plt.rc('text', usetex=True)

    if T0 > 0:
        t_text = 5
        T_text = T_simple(t_text, T0)
        xy = (1.06*t_text, 1.08*T_text)
        xytext = (1.8*t_text, 1.5*T_text)
    else:
        t_text = 5
        T_text = T_simple(t_text, T0)
        xy = (1.03 * t_text, 1.08 * T_text)
        xytext = (3 * t_text, 1.3 * T_text)

    plt.annotate(r"$\frac{\mathrm{d} T}{\mathrm{d} t} = -k(T - T_{\mathrm{ext}})$", xy=xy,
                 xytext=xytext, arrowprops=dict(facecolor='black', width=1, headwidth=8), fontsize=16,
                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))


    if T0 > 0:
        t_text = 7.5
        T_text = 3.6
        xy=(1.02*t_text, 1.05*T_text)
        xytext=(1.8*t_text, 2.5*T_text)
    else:
        t_text = 20
        T_text = -10
        xy = (15.2, -8.15)
        xytext = (20, -11)

    plt.annotate(r"$\frac{\mathrm{d} T}{\mathrm{d} t} = -k(T - T_{\mathrm{ext}}) + A\sin\left[\frac{2\pi}{24}(t - \delta)\right]$",
                 xy=xy, xytext=xytext, arrowprops=dict(facecolor='black', width=1, headwidth=8), fontsize=16,
                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    if T0 > 0:
        plt.annotate("$T_{{\mathrm{{ext}}}} = {:.1f}\, ^\circ \mathrm{{C}} $".format(T_ext), xy=(0, T_ext - 1.4), va="top", fontsize=16,
                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    else:
        plt.annotate("$T_{{\mathrm{{ext}}}} = {:.1f}\, ^\circ \mathrm{{C}} $".format(T_ext), xy=(0, T_ext + 1), va="bottom", fontsize=16,
                     bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    plt.rc('text', usetex=False)

    plt.grid()
    plt.title("Temperature in a (Poorly) Insulated Room, $T_0={} \degree$C".format(T0), fontsize=18)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "solution-{}_.png".format(int(T0)), dpi=200)
    plt.show()


def plot_fixed_step_error():
    """ Plots error of each method as a function of time over the course of a solution for a fixed h """

    step_sizes = (0.01, 0.05, 0.1, 1)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, h in enumerate(step_sizes):
        if i == 0:
            ax = axes[0][0]
            methods = ("euler", "heun", "rk3r", "pc4", "rk4")
        elif i == 1:
            ax = axes[0][1]
            methods = ("euler", "rk2a", "rk3ssp", "pc4", "rk438")
        elif i == 2:
            ax = axes[1][0]
            methods = ("euler",  "rk2b", "rk3r", "pc4", "rk4r")
        else:
            ax = axes[1][1]
            methods = ("euler", "heun", "rk3ssp", "pc4", "rk45")

        for method in methods:
            method_dir = data_dir + method + "/"
            filename = method_dir + str(clean_float(h)) + ".csv"
            if not os.path.isfile(filename):  # if file for this h does not exist
                print("Error: " + filename + " not found\n")
                continue

            data = np.loadtxt(filename, skiprows=3, delimiter=",")
            t = data[:,0]
            error = np.abs(data[:,2])
            ax.plot(t, error, ls='--', color=get_method_color(method), marker=get_method_marker(method), label=method)

        if i > 1: ax.set_xlabel("Time $t$ [hours]", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Absolute Error [$\degree$ C]", fontsize=12)
        ax.set_yscale("log")
        ax.set_title("$h={}$".format(h), fontsize=14)
        ax.legend(loc="center right", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Absolute Error for Fixed-Step Methods", fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "error-time-fixed-step_.png", dpi=200)
    plt.show()


def plot_adaptive_step_error():
    """ Plots error of each adaptive step method as a function of time over the course of a solution
        for a specified tolerance tol. Also plots tol as a horizontal line for reference. """

    tolerances = (1e-10, 1e-7, 1e-5, 1e-3)
    methods = ("bs23", "rkf45", "ck45", "dp45")  # note that dp45 is buggy and bs23 is just plain wrong

    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, tol in enumerate(tolerances):
        if i == 0:
            ax = axes[0][0]
        elif i == 1:
            ax = axes[0][1]
        elif i == 2:
            ax = axes[1][0]
        else:
            ax = axes[1][1]

        for method in methods:
            method_dir = data_dir + method + "/"
            filename = method_dir + "{:.2e}.csv".format(tol)
            if not os.path.isfile(filename):  # if file for this h does not exist
                print("Error: " + filename + " not found\n")
                continue

            data = np.loadtxt(filename, skiprows=3, delimiter=",")
            t = data[:,0]
            error = np.abs(data[:,2])
            ax.plot(t, error, ls='--', color=get_method_color(method), marker=get_method_marker(method), label=method)

        ax.hlines(tol, tmin, tmax, color="#ec5858", linewidth=3, label="tolerance")
        if i > 1: ax.set_xlabel("Time $t$ [hours]", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Absolute Error [$\degree$ C]", fontsize=12)
        ax.grid()
        ax.set_yscale("log")
        ax.set_title("$\epsilon={}$".format(tol), fontsize=16)
        ax.legend(loc="center right", framealpha=0.95)

    plt.tight_layout()
    plt.suptitle("Error for Adaptive-Step Methods", fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "error-adaptive-step_.png", dpi=200)
    plt.show()


def plot_fixed_step_time():
    """ Plots computation times as a function of tolerance for each method"""
    methods = ("euler", "rk2a", "rk2b", "rk3r", "rk3ssp", "rk4", "rk4r", "rk438", "pc4")

    plt.figure(figsize=(7, 4))
    for method in methods:
        filename = method + ".csv"
        if not os.path.isfile(time_dir + filename):  # if file for this h does not exist
            print("Error: " + filename + " not found\n")
            continue

        data = np.loadtxt(time_dir + filename, skiprows=1, delimiter=",")
        tolerance = data[:, 0]
        time = data[:, 1]
        plt.plot(tolerance, time, ls='--', color=get_method_color(method), marker=get_method_marker(method), label=method)

    plt.xlabel("Maximum Local Error [$\degree$C]", fontsize=12)
    plt.ylabel("Computation Time $t$ [s]", fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.legend(framealpha=0.95)

    plt.tight_layout()
    plt.rc('text', usetex=True)
    plt.title("Computation Time for Fixed-Step Methods".format(h), fontsize=18)
    plt.rc('text', usetex=False)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "time-fixed_.png", dpi=200)
    plt.show()


def plot_adapative_step_time():
    """ Plots computation times as a function of tolerance for each method"""
    methods = ("bs23", "rkf45", "ck45", "dp45", "rk4")
    markers = ("o", "s", "d", "P", '.')
    plt.figure(figsize=(7, 4))
    for i, method in enumerate(methods):
        filename = method + ".csv"
        if not os.path.isfile(time_dir + filename):  # if file for this h does not exist
            print("Error: " + filename + " not found\n")
            continue

        data = np.loadtxt(time_dir + filename, skiprows=1, delimiter=",")
        tolerance = data[:, 0]
        time = data[:, 1]
        plt.plot(tolerance, time, ls='--', linewidth=3, color=get_method_color(method), marker=markers[i], label=method)

    plt.xlabel("Tolerance $\epsilon$ [$\degree$C]", fontsize=12)
    plt.ylabel("Computation Time $t$ [s]", fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.legend(loc="best", framealpha=0.95)

    plt.tight_layout()
    plt.rc('text', usetex=True)
    plt.title("Computation Time for Adaptive-Step Methods".format(h), fontsize=18)
    plt.rc('text', usetex=False)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "time-adaptive_.png", dpi=200)
    plt.show()


def plot_embedded_error_estimate():
    """ Compares the error estimate from the embedded RK45 method
        to the error with respect to the analytic solution"""

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    min_exp = -6
    max_exp = 1
    step_sizes = (0.01, 0.05, 0.1, 1)
    for i, h in enumerate(step_sizes):
        if i == 0: ax = axes[0][0]
        elif i == 1: ax = axes[0][1]
        elif i == 2: ax = axes[1][0]
        else: ax = axes[1][1]

        t = get_time_values(h)
        T_de, error_de = rk45(dTdt_simple, T0, t)
        T_analytic = T_simple(t, T0)
        error_analytic = np.abs(T_analytic - T_de)

        start = 1
        stop = len(t)
        ax.plot(t[start:stop], error_analytic[start:stop], color="#244d90", ls='--', marker='.', label="true error")
        ax.plot(t[start:stop], error_de[start:stop], color="#91331f", ls='--', marker='.', label="estimate")

        if i > 1: ax.set_xlabel("Time $t$ [hours]", fontsize=11)
        if i % 2 == 0: ax.set_ylabel("Absolute Error [$\degree$ C]", fontsize=11)
        ax.set_yscale("log")
        ax.set_title("$h={}$".format(h), fontsize=12)
        ax.legend(loc="upper right", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.rc('text', usetex=True)
    plt.suptitle(r"\texttt{{rk45}} Embedded Error Estimate vs True Error", fontsize=18)
    plt.rc('text', usetex=False)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "error-embedded-rk45_.png", dpi=200)
    plt.show()


def plot_step_size_vs_error():
    """ Plots step size versus error with respect to the analytic solution for the classic RK4
        Goal is to show that decreasing beyond an arbitrarily small step size
        increases the methods error because of floating point arithmetic issues"""

    methods = ("euler", "rk3r", "rk3ssp", "pc4", "rk4", "rk438", "rk4r")

    for method in methods:
        data = np.loadtxt(step_global_error_dir + "{}.csv".format(method), delimiter=',', skiprows=1)
        global_step = data[:,0]
        global_error = data[:, 1]
        data = np.loadtxt(step_local_error_dir + "{}.csv".format(method), delimiter=',', skiprows=1)
        local_step = data[:,0]
        local_error = data[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        ax = axes[0]
        ax.set_ylabel("Global Error [$\degree$C]")
        ax.set_xlabel("Time Step $h$ [hours]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.plot(global_step, global_error, ls='--', color=get_method_color(method), marker=get_method_marker(method))
        ax.grid()
        ax.set_title("Global Error", fontsize=14)

        ax = axes[1]
        ax.set_ylabel("Max Local Error [$\degree$C]")
        ax.set_xlabel("Time Step $h$ [hours]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.plot(local_step, local_error, ls='--', color=get_method_color(method), marker=get_method_marker(method))
        ax.grid()
        ax.set_title("Max Local Error", fontsize=14)

        plt.tight_layout()

        plt.rc('text', usetex=True)
        plt.suptitle(r'Dependence of Error on Time Step for $\texttt{{{}}}$'.format(method), fontsize=18)
        plt.rc('text', usetex=False)
        plt.subplots_adjust(top=0.85)

        if save_figures: plt.savefig(figure_dir + "error-vs-step-size-{}_.png".format(method), dpi=200)
        plt.show()


def  plot_fixed_step_stability():
    """ Plots the approximate solution of various methods for varying (relatively large)
         step sizes. Used to determine which methods remain stable even for small h """

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    step_sizes = (2, 5, 10, 12)
    methods = ("euler", "rk2a", "pc4", "rk4r")
    colors_div = ("#ec8689", "#ec8689", "#ec8689", "#c93475", "#661482", "#350b52")  # neon
    colors_conv = ("#8fcdc2", "#3997bf", "#244d90", "#141760")  # blue
    markers = (".", "s", "d", "P")

    for i, method in enumerate(methods):

        if i == 0:
            ax = axes[0][0]
        elif i == 1:
            ax = axes[0][1]
        elif i == 2:
            ax = axes[1][0]
        else:
            ax = axes[1][1]

        for j, h in enumerate(step_sizes):
            t = get_time_values(h)
            T_de = get_fixed_step_solution(dTdt_oscillating, T0, t, method)

            if np.abs(T_de[-1]) > 20:
                color = colors_div[j]
            else:
                color = colors_conv[j]

            ax.plot(t, T_de, ls='--', color=color, marker=markers[j], label="$h={}$".format(h))

        if i > 1: ax.set_xlabel("Time $t$ [hours]", fontsize=11)
        if i % 2 == 0: ax.set_ylabel("Temperature [$\degree$ C]", fontsize=11)

        ax.set_title(method, fontsize=14)

        ax.legend(loc="best", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Stability of Fixed-Step Methods for $k={}$, $A={}$, $T_0={} \degree$C".format(k, A, T0), fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "stability-fixed-{}_.png".format(k), dpi=200)
    plt.show()


def plot_adaptive_step_stability():
    """ Plots the approximate solution of various methods for varying (relatively large)
         step sizes. Used to determine which methods remain stable even for small h """

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    tolerances = (10, 20, 30, 50)
    methods = ("ck45", "rkf45")
    colors_div = ("#ec8689", "#ec8689", "#ec8689", "#c93475", "#661482", "#350b52")  # neon
    colors_conv = ("#8fcdc2", "#3997bf", "#244d90", "#141760")  # blue
    markers = (".", "s", "d", "P")

    for i, method in enumerate(methods):

        if i == 0:
            ax = axes[0]
        else:
            ax = axes[1]

        for j, tol in enumerate(tolerances):
            t, T_de = get_adaptive_step_solution(dTdt_simple, tmin, tmax, T0, tol, hmax, hmin, method)

            if np.max(np.abs(T_de)) > 75:
                color = colors_div[j]
            else:
                color = colors_conv[j]

            ax.plot(t, T_de, ls='--', color=color, marker=markers[j], label="$\epsilon={}$".format(tol))

        if i > 1: ax.set_xlabel("Time $t$ [hours]", fontsize=11)
        if i % 2 == 0: ax.set_ylabel("Temperature [$\degree$ C]", fontsize=11)

        ax.set_title(method, fontsize=14)

        ax.legend(loc="best", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Adaptive-Step Methods for $k={}$, $T_0={} \degree$C".format(k, T0), fontsize=18)
    plt.subplots_adjust(top=0.81)
    if save_figures: plt.savefig(figure_dir + "stability-adaptive-{}_.png".format(k), dpi=200)
    plt.show()


# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def practice():
    # x = np.linspace(0, 1, 10)
    # y = np.linspace(10, 12.2342, 10)
    # data = np.column_stack([x, y])
    # print(data)
    # data = np.column_stack([x, y, x])
    # filename = data_dir + "test.csv"
    # header = "Max error:,{}\nTime [h], Temp [C], Abs Error [C]".format(np.max(y))
    # np.savetxt(filename, data, delimiter=',',  header=header)

    a = (1, 2, 3, 4)
    b = np.asarray(a)
    print(b)


def run():
    # practice()

    # generate_fixed_step_solutions()
    # generate_adaptive_step_solutions()

    # compile_error_h_table()



    # compile_fixed_step_times()
    # compile_adaptive_step_times()


    # generate_step_size_vs_error()

    # plot_fixed_step_error()
    # plot_step_size_vs_error()
    # plot_adaptive_step_error()
    # plot_embedded_error_estimate()

    # plot_fixed_step_time()
    # plot_adapative_step_time()

    # plot_fixed_step_stability()
    # plot_adaptive_step_stability()

    plot_solution()


run()


