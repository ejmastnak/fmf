import numpy as np

"""
This python file contains a variety of methods for solving first order ordinary differential equations.
AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    Based Octave functions written in the spring of 1999
    Python version: March 2008, October 2008
    
    I implemented the rk3r, rk3ssp, rk438, rk4r, bs23, rkf45, ck45 and dp45 based on Senning's code
"""


# -----------------------------------------------------------------------------
# START ELEMENTARY METHODS
# -----------------------------------------------------------------------------
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
# -----------------------------------------------------------------------------
# END ELEMENTARY METHODS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START FIXED-STEP RUNGE-KUTTA METHODS
# -----------------------------------------------------------------------------
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
                h = t[i+1]-t[i] determines the step size h.
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
# -----------------------------------------------------------------------------
# END FIXED-STEP RUNGE-KUTTA METHODS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ADAPTIVE STEP METHODS
# -----------------------------------------------------------------------------
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
           --= f(x,t),     x(a) = x0
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
           --= f(x,t),     x(a) = x0
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
           --= f(x,t),     x(a) = x0
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
           --= f(x,t),     x(a) = x0
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
# -----------------------------------------------------------------------------
# END ADAPTIVE STEP METHODS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START MULTISTEP METHODS
# -----------------------------------------------------------------------------
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
                h = t[i+1]-t[i] determines the step size h.
    OUTPUT:
        x     - np array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    NOTES:
        This function used the Adams-Bashforth-Moulton predictor-corrector
        method to solve the initial value problem
            dx
            --= f(x,t),     x(t(1)) = x0
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
# END MULTISTEP METHODS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SYMPLECTIC METHODS
# -----------------------------------------------------------------------------
def verlet(f, x0, v0, t):
    """Verlet's 2nd order symplectic method

    USAGE:
        (x,v) = varlet(f, x0, v0, t)

    INPUT:
        f     - function of x and t equal to d^2x/dt^2.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s) of x.  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        v0    - the initial condition(s) of v = dx/dt.  Specifies the value of v when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h = t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values for x corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        v     - NumPy array containing solution values for v = dx/dt corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function used the Varlet/Stoermer/Encke (symplectic) method
        method to solve the initial value problem

            dx^2
            --= f(x),     x(t(1)) = x0  v(t(1)) = v0
            dt^2

        at the t values stored in the t array (so the interval of solution is
        [t[0], t[N-1]].  The 3rd-order Taylor is used to generate
        the first values of the solution.

    """
    n = len(t)
    x = np.array([x0] * n)
    v = np.array([v0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        x[i+1] = x[i] + h * v[i] + (h*h/2) * f(x[i])
        v[i+1] = v[i] + (h/2) * (f(x[i])+f(x[i+1]))

    return np.array([x,v])


def pefrl(f, x0, v0, t):
    """Position Extended Forest-Ruth Like 4th order symplectic method by Omelyan et al.

    USAGE:
        (x,v) = varlet(f, x0, v0, t)

    INPUT:
        f     - function of x and t equal to d^2x/dt^2.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s) of x.  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        v0    - the initial condition(s) of v = dx/dt.  Specifies the value of v when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h = t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values for x corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        v     - NumPy array containing solution values for v = dx/dt corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function uses the Omelyan et al (symplectic) method
        method to solve the initial value problem

            dx^2
            --= f(x),     x(t(1)) = x0  v(t(1)) = v0
            dt^2

        at the t values stored in the t array (so the interval of solution is
        [t[0], t[N-1]].

    """

    xsi = 0.1786178958448091
    lam = -0.2123418310626054
    chi = -0.6626458266981849e-1
    n = len(t)
    x = np.array([x0] * n)
    v = np.array([v0] * n)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        y = np.copy(x[i])
        w = np.copy(v[i])
        y += xsi*h*w
        w += (1-2*lam)*(h/2)*f(y)
        y += chi*h*w
        w += lam*h*f(y)
        y += (1-2*(chi+xsi))*h*w
        w += lam*h*f(y)
        y += chi*h*w
        w += (1-2*lam)*(h/2)*f(y)
        y += xsi*h*w
        x[i+1] = np.copy(y)
        v[i+1] = np.copy(w)

    return np.array([x,v])
