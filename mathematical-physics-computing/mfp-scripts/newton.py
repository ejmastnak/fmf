# import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.special import ellipk, ellipj
from scipy.integrate import solve_ivp
import os
import re
from numerical_methods_odes import *  # includes numpy as np
from newton_states import *  # also includes numpy as np

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
solutions_dir = data_dir + "solutions/"
error_dir = data_dir + "errors/"
time_dir = data_dir + "times/"
figure_dir = "../7-newton/figures/"

save_figures = True

# simulation parameters
# h = 1  # time step [hours]
tmin = 0  # start time
tmax = 15 * 2 * np.pi  # end time
hmin = 1e-5  # universal min step size for adaptive step methods
hmax = 1e1   # universal max step size for adaptive step methods


# -----------------------------------------------------------------------------
# START PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------
def linear_pendulum_analytic(t, initial_state, w0=1):
    """ Returns linear pendulum's displacement as a function of time
        with the initial state [x0, v0] """
    return initial_state[0]*np.cos(w0*t) + (initial_state[1]/w0)*np.sin(w0*t)


def simple_pendulum_analytic(t, initial_state, w0=1.0):
    """ Returns the analytic solution of the simple pendulum's
        angular displacement as a function of time on the time array t

        x0 is initial anguluar displacement
        Initial angular velocity is zero: v0 = 0
        Optional parameter w0 is angular frequency sqrt(g/l)

        Uses scipy's elliptic integral and elliptic Jacobi functions
        """
    k = np.sin(initial_state[0] / 2)
    return 2*np.arcsin(k * ellipj(ellipk(k**2) - w0*t, k**2)[0])


def get_pendulum_energy(x, v):
    """
    Returns the energy of a dimensionless simple pendulum according to the formula
     E = (1/2)v^2 + 1 - cos x
     (1/2)v^2 is the kinetic energy term
     1 - cos x is the potential energy term
    Note that m=g=l are set to 1

    :param x: angular displacement
    :param v: angular velocity
    :return:
    """
    return 0.5*v**2 + 1 - np.cos(x)
# -----------------------------------------------------------------------------
# END PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------
def get_fixed_step_solution(f, initial_state, t, method):
    """
    Returns the solution to the second-order differential equation f on the 1D time array t
     for the specified initial state using the specified numerical method

    :param f: function of the form f(state, t) return pendulums angular velocity and acceleration where state = [x, v]
    :param t: time values on which to find the solution
    :param initial_state: initial conditions in a 2-tuple of the form (x0, v0)
    :param method: which differential equation method to use
    :return:
    """
    if method == "euler":
        return euler(f, initial_state, t)
    elif method == "heun":
        return heun(f, initial_state, t)
    elif method == "rk2a":
        return rk2a(f, initial_state, t)
    elif method == "rk3r":
        return rk3r(f, initial_state, t)
    elif method == "rk3ssp":
        return rk3ssp(f, initial_state, t)
    elif method == "rk4":
        return rk4(f, initial_state, t)
    elif method == "rk4r":
        return rk4r(f, initial_state, t)
    elif method == "pc4":
        return pc4(f, initial_state, t)
    else:
        print("{} does not match any options. Returning zeros.".format(method))
        return np.zeros(len(t))


def get_adaptive_step_solution(f, t_min, t_max, initial_state, tol, hmax, hmin, method):
    """
    Returns the solution to the second-order differential equation f on the time interval [tmin, tmax]
     for the specified initial state using the specified numerical method

    :param f: function of the form f(state, t) return pendulums angular velocity and acceleration where state = [x, v]
    :param t_min: min time
    :param t_max: max time
    :param initial_state: initial conditions in a 2-tuple of the form (x0, v0)
    :param tol: maximum error tolerance, used to adjust step size
    :param hmax: maximum allowed step size
    :param hmin: minimum allowed step size
    :param method: which differential equation method to use
    :return:
    """
    if method == "bs23":
        return bs23(f, t_min, t_max, initial_state, tol, hmax, hmin)
    elif method == "rkf45":
        return rkf45(f, t_min, t_max, initial_state, tol, hmax, hmin)
    elif method == "ck45":
        return ck45(f, t_min, t_max, initial_state, tol, hmax, hmin)
    elif method == "dp45":
        return dp45(f, t_min, t_max, initial_state, tol, hmax, hmin)
    else:
        print("{} does not match any options. Returning zeros.".format(method))
        return np.zeros(len(get_time_values((hmax + hmin)/2)))


def get_scipy_solution(f, t_min, t_max, initial_state, method, tol=1e-6, max_step=hmax):
    """
    Returns the solution to the second-order differential equation f on the time interval [tmin, tmax]
     using a method from SciPy's solve_ivp API

    :param f: function of the form f(t, state) return pendulums angular velocity and acceleration where state = [x, v]
    :param t_min: min time
    :param t_max: max time
    :param initial_state: initial conditions in a 2-tuple of the form (x0, v0)
    :param method: which differential equation method to use e.g. "RK45", "RK23", "DOP853", "BDF", etc...
    :param tol: maximum error tolerance, used to adjust step size
    :param max_step: maximum allowed step size
    :return:
    """
    return solve_ivp(f, (t_min, t_max), initial_state, method=method, atol=tol, max_step=max_step)


def get_symplectic_solution(f, initial_state, t, method):
    """
    Returns the solution to the second-order differential equation f on the 1D time array t
     for the specified initial state using the specified numerical method

    :param f: function of the form f(state, t) return pendulums angular velocity and acceleration where state = [x, v]
    :param t: time values on which to find the solution
    :param initial_state: initial conditions in a 2-tuple of the form (x0, v0)
    :param method: which differential equation method to use
    :return:
    """
    if method == "verlet":
        return verlet(f, initial_state[0], initial_state[1], t)
    elif method == "pefrl":
        return pefrl(f, initial_state[0], initial_state[1], t)
    else:
        print("{} does not match any options. Returning zeros.".format(method))
        return np.zeros(len(t))


def get_time_values(h):
    """ Returns the time values on which to solve the differential equation for given fixed step size h
         by referencing the global variable tmin and tmax """
    n = int((tmax - tmin) / h)  # number of points
    return np.linspace(tmin, tmax, n, endpoint=False)


def get_initial_state():
    """ Returns a consistent initial pendulum/oscillator state for use across all methods"""
    return [1.0, 0.0]
# -----------------------------------------------------------------------------
# END AUXILARY DE FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SOLUTION GENERATION FUNCITONS
# -----------------------------------------------------------------------------
def generate_fixed_step_solutions(initial_state, function=get_simple_pendulum_state):
    """Saves solution data to local csv files for later analysis...
        ...to avoid calling python solution functions over and over again"""
    methods = ("euler", "heun", "rk2a", "rk3r", "rk3ssp", "rk4", "rk4r", "pc4")
    for exponent in (-2, -1, 0):  # powers of 10 for calculating h
        for coefficient in (1, 2, 3, 5, 7, 9):  # coefficients for h
            h = coefficient*(10**exponent)
            t = get_time_values(h)
            for method in methods:
                x, v = get_fixed_step_solution(function, initial_state, t, method).T  # unpack x and v
                energy = get_pendulum_energy(x, v)
                header = "Time, Angular Displacement, Angular Velocity, Energy"
                np.savetxt(solutions_dir + method + "/{:.1e}.csv".format(h), np.column_stack([t, x, v, energy]), delimiter=',', header=header)


def generate_symplectic_solutions(initial_state, function=get_simple_pendulum_state_symp):
    """Saves solution data to local csv files for later analysis...
        ...to avoid calling python solution functions over and over again"""
    methods = ("verlet", "pefrl")
    for exponent in (-2, -1, 0):  # powers of 10 for calculating h
        for coefficient in (1, 2, 3, 5, 7, 9):  # coefficients for h
            h = coefficient*(10**exponent)
            t = get_time_values(h)
            for method in methods:
                x, v = get_symplectic_solution(function, initial_state, t, method)  # unpack x and v
                energy = get_pendulum_energy(x, v)
                header = "Time, Angular Displacement, Angular Velocity, Energy"
                np.savetxt(solutions_dir + method + "/{:.1e}.csv".format(h), np.column_stack([t, x, v, energy]), delimiter=',', header=header)


def generate_adaptive_step_solutions(initial_state, function=get_simple_pendulum_state):
    """Saves solution data to local csv files for later analysis...
        ...to avoid calling python solution functions over and over again"""
    methods = ("rkf45", "ck45")  # dp45 is buggy
    for exponent in range(-10, 0):  # powers of 10 for calculating tolerance
        for coefficient in (1, 2, 3, 5, 7, 9):  # coefficients for tolerance
            tol = coefficient*(10**exponent)  # i.e. tol = coef*10^exponent
            for method in methods:
                t, solution = get_adaptive_step_solution(function, tmin, tmax, initial_state, tol, hmax, hmin, method)
                x, v = solution.T
                energy = get_pendulum_energy(x, v)
                header = "Time, Angular Displacement, Angular Velocity, Energy"
                np.savetxt(solutions_dir + method + "/{:.1e}.csv".format(tol), np.column_stack([t, x, v, energy]), delimiter=',', header=header)


def generate_built_in_solutions(initial_state, function=get_simple_pendulum_state_tfirst):
    """Saves solution data to local csv files for later analysis...
        ...to avoid calling python solution functions over and over again"""
    methods = ("RK45", "RK23", "DOP853")
    for exponent in range(-10, 0):  # powers of 10 for calculating tolerance
        for coefficient in (1, 2, 3, 5, 7, 9):  # coefficients for tolerance
            tol = coefficient*(10**exponent)  # i.e. tol = coef*10^exponent
            for method in methods:
                solution = get_scipy_solution(function, tmin, tmax, initial_state, method,
                                              tol=tol, max_step=1e-0)
                t, y = solution.t, solution.y
                x, v = y[0], y[1]
                energy = get_pendulum_energy(x, v)
                header = "Time, Angular Displacement, Angular Velocity, Energy"
                np.savetxt(solutions_dir + method + "/{:.1e}.csv".format(tol), np.column_stack([t, x, v, energy]), delimiter=',', header=header)


def generate_fixed_step_errors(initial_state, function=get_simple_pendulum_state):
    """ Finds error in angular displacement of the simple pendulum
         as a function of time for the solutions of each fixed-step and symplectic method
         and saves the data to local csv files
    """
    methods = ("euler", "heun", "rk2a", "rk3r", "rk3ssp", "rk4", "rk4r", "pc4", "verlet", "pefrl")
    E_initial = get_pendulum_energy(*get_initial_state())  # pendulum's conserved energy
    for exponent in (-2, -1, 0):  # powers of 10 for calculating h
        for coefficient in (1, 2, 3, 5, 7, 9):  # coefficients for h
            h = coefficient*(10**exponent)
            t = get_time_values(h)
            x_analytic = simple_pendulum_analytic(t, get_initial_state())  # analytic solution
            for method in methods:
                filename = solutions_dir + method + "/{:.1e}.csv".format(h)
                if not os.path.isfile(filename):  # if file for this h does not exist
                    print("Error: " + filename + " not found\n")
                    continue

                # load numerical solution from first column
                x_num, v_num, E_num = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(1, 2, 3)).T
                x_error = np.abs(x_analytic - x_num)
                E_error = E_initial - get_pendulum_energy(x_num, v_num)
                header = "Time, Displacement Error, Energy Error"
                np.savetxt(error_dir + method + "/{:.1e}.csv".format(h), np.column_stack([t, x_error, E_error]), delimiter=',', header=header)


def generate_adaptive_step_errors(initial_state, function=get_simple_pendulum_state):
    """ Finds error in angular displacement of the simple pendulum
         as a function of time for the solutions of each adaptive step and built-in scipy method
         and saves the data to local csv files
    """
    methods = ("rkf45", "ck45", "RK45", "RK23", "DOP853")
    E_initial = get_pendulum_energy(*get_initial_state())  # pendulum's conserved energy

    for exponent in range(-10, 0):  # powers of 10 for calculating tolerance
        for coefficient in (1, 2, 3, 5, 7, 9):  # coefficients for tolerance
            tol = coefficient*(10**exponent)  # i.e. tol = coef*10^exponent
            for method in methods:
                filename = solutions_dir + method + "/{:.1e}.csv".format(tol)
                if not os.path.isfile(filename):  # if file for this h does not exist
                    print("Error: " + filename + " not found\n")
                    continue
                t, x_num, v_num, E_num = np.loadtxt(filename, delimiter=',', skiprows=1).T
                x_analytic = simple_pendulum_analytic(t, get_initial_state())  # analytic solution
                x_error = np.abs(x_analytic - x_num)
                E_error = E_initial - get_pendulum_energy(x_num, v_num)
                header = "Time, Displacement Error, Energy Error"
                np.savetxt(error_dir + method + "/{:.1e}.csv".format(tol), np.column_stack([t, x_error, E_error]), delimiter=',', header=header)
# -----------------------------------------------------------------------------
# END SOLUTION GENERATIION FUNCTIONS
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


def try_stuff():
    h = 0.01
    t = get_time_values(h)
    initial_state = [1.0, 0.0]
    # x, v = get_fixed_step_solution(get_simple_pendulum_state, initial_state, t, "rk4").T  # unpack x and v
    x, v = get_symplectic_solution(get_simple_pendulum_state_symp, initial_state, t, "pefrl")  # unpack x and v
    # t, solution = get_adaptive_step_solution(get_simple_pendulum_state, tmin, tmax, initial_state, 1e-9, hmax, hmin, "ck45")
    # x, v = solution.T
    plt.plot(t, x, label="position")
    plt.plot(t, v, label="velocity")
    plt.legend()
    plt.grid()
    plt.show()
    solution = get_scipy_solution(get_simple_pendulum_state_tfirst, tmin, tmax, initial_state,
                                  method="RK23", max_step=1e-0)
    t, y = solution.t, solution.y
    plt.plot(t, y[0], label="position")
    plt.plot(t, y[1], label="velocity")
    plt.show()


def try_error_stuff():
    h = 1.0e-1
    t = get_time_values(h)
    x_analytic = simple_pendulum_analytic(t, get_initial_state())  # analytic solution
    E_initial = get_pendulum_energy(*get_initial_state())
    filename = data_dir + "euler" + "/{:.1e}.csv".format(h)

    # load numerical solution from first column
    x_num, v_num, E_num = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(1, 2, 3)).T
    x_error = np.abs(x_analytic - x_num)
    E_error = E_initial - get_pendulum_energy(x_num, v_num)

    # plt.plot(t, x_num, label="position")
    # plt.plot(t, v_num, label="velocity")
    # plt.plot(t, x_error, label="x error")
    # plt.plot(t, E_num, label="energy")
    # plt.hlines(E_initial, tmin, tmax)
    plt.plot(t, E_error, label="energy error")
    plt.legend()
    plt.show()


def practice():
    # x = np.linspace(0, 1, 10)
    # y = np.linspace(10, 12.2342, 10)
    # data = np.column_stack([x, y])
    # print(data)
    # data = np.column_stack([x, y, x])
    # filename = data_dir + "test.csv"
    # header = "Max error:,{}\nTime [h], Temp [C], Abs Error [C]".format(np.max(y))
    # np.savetxt(filename, data, delimiter=',',  header=header)

    x, v = get_initial_state()
    print(x)
    print(v)
    E = get_pendulum_energy(*get_initial_state())
    print(E)
    print(get_pendulum_energy(x, v))


def run():
    # practice()
    # try_stuff()
    # try_error_stuff()
    # generate_fixed_step_solutions(get_initial_state())
    # generate_symplectic_solutions(get_initial_state())
    # generate_adaptive_step_solutions(get_initial_state())
    # generate_built_in_solutions(get_initial_state())
    # generate_fixed_step_errors(get_initial_state())
    generate_adaptive_step_errors(get_initial_state())


run()
