import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.special import ellipk, ellipj
from numerical_methods_odes import *

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

color_gray_ref = "#AAAAAA"  # light gray for reference quantities

# data_dir = "data/"
data_dir = "/Users/ejmastnak/Documents/Media/academics/fmf-media-winter-3/mafiprak/newton/"
error_dir = data_dir + "error-step/"
time_dir = data_dir + "times/"
figure_dir = "../7-newton/figures/"

save_figures = True

# simulation parameters
h = 1  # time step [hours]
tmin = 0  # start time
tmax = 72  # end time
hmin = 1e-5  # universal min step size for adaptive step methods
hmax = 1e1   # universal max step size for adaptive step methods

# -----------------------------------------------------------------------------
# START DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------
def dxdt_linear_pendulum(x):
    """ Dimensionless differential equation of motion for a linear pendulum
        Returns the pendulum's angular acceleration d/dt[dx/dt]
        Uses the small-angle approximation sin x approx x
        x is the pendulum's angular displacement from equilibrium
    """
    return -x


def dxdt_mat_pendulum(x):
    """ Dimensionless differential equation of motion for a mathematical pendulum"""
    return -np.sin(x)


def dxdt_mat_pendulum_damped_driven(t, x, v, b=0.5, w_d=(2/3), a_d=1.0):
    """ Dimensionless differential equation of motion for a damped and driven mathematical pendulum
        Returns the pendulum's angular acceleration d/dt[dx/dt]
        x and v are the pendulum's angular displacement from equilibrium and angular velocity, respectively

        b is the damping coefficient
        w_d is the driving angular frequency
        a_d is the driving amplitude
    """
    return a_d*np.cos(w_d*t) - b*v - np.sin(x)


def dxdt_van_der_pol(x, v, mu=1.0):
    """ Dimensionless differential equation of motion for a van der Pol oscillator
        Returns the oscillator's angular acceleration d/dt[dx/dt]

        x and v are the oscillator's angular displacement from equilibrium and angular velocity, respectively
        mu is the damping coefficient
    """
    return mu*v*(1-x**2) - x


def dxdt_van_der_pol_driven(t, x, v, mu=1.0, w_d=1.0, a_d=1.0):
    """ Dimensionless differential equation of motion for a driven van der Pol oscillator
        Returns the oscillator's angular acceleration d/dt[dx/dt]

        x and v are the oscillator's angular displacement from equilibrium and angular velocity, respectively
        mu is the damping coefficient
        w_d is the driving angular frequency
        a_d is the driving amplitude
    """
    return a_d*np.cos(w_d*t) + mu*v*(1-x**2) - x


def simple_pendulum_analytic(t, x0, w0=1):
    """ Returns the analytic solution of the simple pendulum on the time array t

        x0 is initial anguluar displacement
        Initial angular velocity is zero: v0 = 0
        Optional parameter w0 is angular frequency sqrt(g/l)

        Uses scipy's elliptic integral and elliptic Jacobi functions
        """
    k = np.sin(x0/2)
    return 2*np.arcsin(k * ellipj(ellipk(k**2) - w0*t, k**2)[0])

# -----------------------------------------------------------------------------
# END DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START AUXILARY TEMPERATURE DE FUNCTIONS
# -----------------------------------------------------------------------------
def get_fixed_step_solution(f, t, x0, v0, T_initial, method):
    """
    Returns the solution to the differential equation f based on the inputted fixed-step method
    f is assumed to be a second-order differential equation.
    x0 and v0 are initial conditions
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
        return np.zeros(len(t))


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
        return np.zeros(len(get_time_values((hmax + hmin)/2)))


def get_time_values(h):
    """ Returns the time values on which to solve the differential equation for given fixed step size h
         by referencing the global variable tmin and tmax """
    n = int((tmax - tmin) / h)  # number of points
    return np.linspace(tmin, tmax, n, endpoint=False)
# -----------------------------------------------------------------------------
# END AUXILARY TEMPERATURE DE FUNCTIONS
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
    practice()


run()
