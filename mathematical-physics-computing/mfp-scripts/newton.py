# import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.special import ellipk, ellipj
from scipy.integrate import solve_ivp, odeint
import os
import re
from ivp_methods import *  # includes numpy as np
from newton_states import *  # also includes numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

color_gray_ref = "#AAAAAA"  # light gray for reference quantities

data_dir = "../7-newton/data/"
solutions_dir = data_dir + "solutions/"
error_dir = data_dir + "errors/"
resonance_dir = data_dir + "resonance/"
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


def get_simple_pendulum_period(x0):
    """
    Returns the period of a (dimensionless) simple pendulum
     with initial angular displacement x0
    :param x0: initial angular displacement in radians
    """
    k = np.sin(x0/2)
    return 4*ellipk(k**2)


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
    if method == "ck45":
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
    methods = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")
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
    # methods = ("rkf45", "ck45", "RK45", "RK23", "DOP853")
    methods = ("Radau", "BDF", "LSODA")
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


def generate_resonance_data():
    """ Generates resonance curve data for the dampled, driven simple pendulum
         and saves the data to local csv files for later analysis """
    initial_state = get_initial_state()
    b = 0.5

    # region1 = np.arange(0.2, 0.35, 0.03)
    # region2 = np.arange(0.35, 0.45, 0.01)
    # region3 = np.arange(0.45, 0.46, 0.005)
    # region4 = np.arange(0.46, 0.48, 0.002)
    # region5 = np.arange(0.48, 0.51, 0.005)
    # region6 = np.arange(0.51, 0.65, 0.01)
    # region7 = np.arange(0.65, 1.85, 0.03)
    # driving_frequencies = np.concatenate((region1, region2, region3, region4, region5, region6, region7))

    driving_frequencies = np.arange(0.30, 1.5, 0.005)
    # for a_d in (0.5, 1.0, 1.5, 2.5, 3.5, 5.0):
    # for a_d in (0.5, 0.75, 1.0, 1.25):
    for a_d in (0.25, 0.85, 0.90):

        max_amplitudes = np.zeros(len(driving_frequencies))
        for i in range(len(driving_frequencies)):
            w_d = driving_frequencies[i]
            print(w_d)
            period = 2 * np.pi / w_d
            N_periods = 50  # observe over 50 periods of the driving frequency
            t_start = 0  # start time
            t_end = N_periods * period  # end time

            # built-in solve-ivp
            method = "RK45"
            solution = solve_ivp(get_simple_pendulum_damped_driven_state_tfirst, (t_start, t_end), initial_state, method=method,
                                 max_step=5e-1, atol=1e-6, args=(b, w_d, a_d))
            t, y = solution.t, solution.y
            x, v = y[0], y[1]
            max_amplitude = np.max(np.abs(x))
            max_amplitudes[i] = max_amplitude
        np.savetxt(resonance_dir + "{:.1f}_.csv".format(a_d), np.column_stack([driving_frequencies, max_amplitudes]),
                   delimiter=',', header="Driving Frequency, Maximum Amplitude over 50 Periods")
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
# 3: ("#74c1c4", "#2e6fa7", "#161d63")
# 3: ("#f5c450", "#e1692e", "#91331f")
    if method == "euler" or method == "RK23":
        return "#8fcdc2"
    elif method == "heun" or method == "RK45":
        return "#2e6fa7"
    elif method == "rk2a":
        return "#2e6fa7"
    elif method == "rk3r" or method =="DOP853":
        return "#1d357f"
    elif method == "rk3ssp":
        return "#1d357f"
    elif method == "rk4":
        return "#e1692e"
    elif method == "rk4r":
        return "#e1692e"
    elif method == "rkf45":
        return "C0"  # lavender
    elif method == "ck45":
        return "C0"  # orchid
    elif method == "pc4":
        return "#e1692e"
    elif method == "LSODA":
        return "#f5c450"
    elif method == "BDF" or method == "Radau":
        return "#91331f"
    elif method == "verlet":
        return "#741AAC"
    elif method == "pefrl":
        return "#741aac"
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
    elif method == "rk3r":
        return "^"
    elif method == "rk3ssp":
        return "^"
    elif method == "rk4":
        return "P"
    elif method == "rk4r":
        return "P"
    elif method == "pc4":
        return "P"
    elif method == "rkf45":
        return "P"
    elif method == "ck45":
        return "P"
    elif method == "RK45" or method == "RK23" or method=="DOP853":
        return "P"
    elif method == "LSODA" or method == "BDF" or method=="Radau":
        return "^"
    elif method == "verlet":
        return "d"
    elif method == "pefrl":
        return "d"
    else:
        print("No marker match for method: {}".format(method))
        return "."
# -----------------------------------------------------------------------------
# END AUXILARY MISC FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_fixed_step_x_error():
    """ Plots angular displacement error of each method as a function of time
         over the course of a solution for a fixed h """

    step_sizes = (0.01, 0.05, 0.1, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, h in enumerate(step_sizes):
        if i == 0:
            ax = axes[0][0]
            methods = ("euler", "verlet", "rk4", "pefrl")
        elif i == 1:
            ax = axes[0][1]
            methods = ("rk2a", "verlet", "rk3ssp", "pc4", "pefrl")
        elif i == 2:
            ax = axes[1][0]
            methods = ("euler", "heun", "rk3r", "rk4r", "pefrl")
        else:
            ax = axes[1][1]
            methods = ("euler", "heun", "verlet", "rk3ssp", "pc4", "pefrl")

        for method in methods:
            filename = error_dir + method + "/{:.1e}.csv".format(h)
            if not os.path.isfile(filename):  # if file for this h does not exist
                print("Error: " + filename + " not found\n")
                continue

            t, x_error = np.loadtxt(filename, skiprows=1, delimiter=",", usecols=(0, 1)).T  # load time and displacement error
            x_error = np.abs(x_error)
            ax.plot(t, x_error, ls='--', color=get_method_color(method), marker=get_method_marker(method), label=method)

        if i > 1: ax.set_xlabel("Time $t$", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Displacement Error", fontsize=12)
        ax.set_yscale("log")
        ax.set_title("$h={}$".format(h), fontsize=14)
        ax.legend(loc="center right", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Displacement Error for Fixed-Step Methods", fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "error-x-fixed-step_.png", dpi=200)
    plt.show()


def plot_adaptive_step_x_error():
    """ Plots angular displacement error of each method as a function of time
         over the course of a solution for a fixed tolerance """

    tolerances = (1e-10, 1e-7, 1e-3, 5e-1)
    pefrl_steps = (0.01, 0.05, 0.1, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, tol in enumerate(tolerances):
        if i == 0:
            ax = axes[0][0]
            methods = ("RK23", "rkf45", "BDF", "LSODA")
        elif i == 1:
            ax = axes[0][1]
            methods = ("RK23", "ck45", "Radau", "LSODA")
        elif i == 2:
            ax = axes[1][0]
            methods = ("RK23", "RK45", "BDF", "ck45", "DOP853")
        else:
            ax = axes[1][1]
            methods = ("RK23", "rkf45", "Radau", "LSODA")

        for method in methods:
            filename = error_dir + method + "/{:.1e}.csv".format(tol)
            if not os.path.isfile(filename):  # if file for this tol does not exist
                print("Error: " + filename + " not found\n")
                continue

            t, x_error = np.loadtxt(filename, skiprows=1, delimiter=",", usecols=(0, 1)).T  # load time and displacement error
            x_error = np.abs(x_error)
            ax.plot(t, x_error, ls='--', color=get_method_color(method), marker=get_method_marker(method), label=method)

        # plot pefrl for reference
        h = pefrl_steps[i]
        filename = error_dir + "pefrl/{:.1e}.csv".format(h)
        if not os.path.isfile(filename):  # if file for this h does not exist
            print("Error: " + filename + " not found\n")
        else:
            t, x_error = np.loadtxt(filename, skiprows=1, delimiter=",", usecols=(0, 1)).T  # load time and displacement error
            ax.plot(t, x_error, ls='--', color=get_method_color("pefrl"), marker=get_method_marker("pefrl"), label="pefrl {}".format(h))

        if i > 1: ax.set_xlabel("Time $t$", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Displacement Error", fontsize=12)
        ax.set_yscale("log")
        ax.set_title("$\epsilon={}$".format(tol), fontsize=14)
        ax.legend(loc="center right", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Displacement Error for Adaptive-Step Methods", fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "error-x-adaptive-step_.png", dpi=200)
    plt.show()


def plot_fixed_step_energy_error():
    """ Plots energy error of each method as a function of time over the course of a solution for a fixed h """

    step_sizes = (0.01, 0.1, 0.5, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, h in enumerate(step_sizes):
        if i == 0:
            ax = axes[0][0]
            methods = ("euler", "heun", "verlet", "pefrl", "rk4")
        elif i == 1:
            ax = axes[0][1]
            methods = ("rk2a", "verlet", "rk3ssp", "pc4", "pefrl")
        elif i == 2:
            ax = axes[1][0]
            methods = ("euler", "heun", "rk3r", "rk4r", "pefrl")
        else:
            ax = axes[1][1]
            methods = ("euler", "heun", "verlet", "rk3ssp", "pc4", "pefrl")

        for method in methods:
            filename = error_dir + method + "/{:.1e}.csv".format(h)
            if not os.path.isfile(filename):  # if file for this h does not exist
                print("Error: " + filename + " not found\n")
                continue

            t, E_error = np.loadtxt(filename, skiprows=1, delimiter=",", usecols=(0, 2)).T  # load time and energy error
            E_error = np.abs(E_error)
            ax.plot(t, E_error, ls='--', color=get_method_color(method), marker=get_method_marker(method), label=method)

        if i > 1: ax.set_xlabel("Time $t$", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Energy Error", fontsize=12)
        ax.set_yscale("log")
        ax.set_title("$h={}$".format(h), fontsize=14)
        ax.legend(loc="center right", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Energy Error for Fixed-Step Methods", fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "error-E-fixed-step_.png", dpi=200)
    plt.show()


def plot_adaptive_step_energy_error():
    """ Plots energy error of each method as a function of time
         over the course of a solution for a fixed tolerance """

    tolerances = (1e-10, 1e-7, 1e-3, 5e-1)
    pefrl_steps = (0.01, 0.05, 0.1, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, tol in enumerate(tolerances):
        if i == 0:
            ax = axes[0][0]
            methods = ("RK23", "rkf45", "BDF", "LSODA")
        elif i == 1:
            ax = axes[0][1]
            methods = ("RK23", "ck45", "Radau", "LSODA")
        elif i == 2:
            ax = axes[1][0]
            methods = ("RK23", "RK45", "BDF", "ck45", "DOP853")
        else:
            ax = axes[1][1]
            methods = ("RK23", "rkf45", "Radau", "LSODA")

        for method in methods:
            filename = error_dir + method + "/{:.1e}.csv".format(tol)
            if not os.path.isfile(filename):  # if file for this tol does not exist
                print("Error: " + filename + " not found\n")
                continue

            t, energy_error = np.loadtxt(filename, skiprows=1, delimiter=",", usecols=(0, 2)).T  # load time and energy error
            energy_error = np.abs(energy_error)
            ax.plot(t, energy_error, ls='--', color=get_method_color(method), marker=get_method_marker(method), label=method)

        # plot pefrl for reference
        h = pefrl_steps[i]
        filename = error_dir + "pefrl/{:.1e}.csv".format(h)
        if not os.path.isfile(filename):  # if file for this h does not exist
            print("Error: " + filename + " not found\n")
        else:
            t, energy_error = np.loadtxt(filename, skiprows=1, delimiter=",", usecols=(0, 2)).T  # load time and displacement error
            ax.plot(t, energy_error, ls='--', color=get_method_color("pefrl"), marker=get_method_marker("pefrl"), label="pefrl {}".format(h))

        if i > 1: ax.set_xlabel("Time $t$", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Energy Error", fontsize=12)
        ax.set_yscale("log")
        ax.set_title("$\epsilon={}$".format(tol), fontsize=14)
        ax.legend(loc="center right", framealpha=0.95)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Energy Error for Adaptive-Step Methods", fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "error-E-adaptive-step_.png", dpi=200)
    plt.show()


def plot_long_period():
    """ Plots energy error of each method as a function of time over the course of a solution for a fixed h """

    fig, ax = plt.subplots(1, figsize=(7, 3.5))
    rc('text', usetex=True)

    h = 0.05
    initial_state = get_initial_state()
    E_initial = get_pendulum_energy(*initial_state)
    period = get_simple_pendulum_period(initial_state[0])
    N_periods = 100
    t_start = 0  # start time
    t_end = N_periods * period  # end time
    n = int((t_end - t_start) / h)  # number of points
    t = np.linspace(t_start, t_end, n, endpoint=False)

    # reference energy
    ax.hlines(0, t_start, t_end, ls='--', color=color_orange_dark, linewidth=2, label="reference energy", zorder=1)

    # symplectic pefrl
    x, v = get_symplectic_solution(get_simple_pendulum_state_symp, initial_state, t, "pefrl")
    energy = get_pendulum_energy(x, v) - E_initial
    ax.plot(t, energy, c="#e86f83", linewidth=5, label="pefrl", zorder=-1)

    # fixed-step rk4
    x, v = get_fixed_step_solution(get_simple_pendulum_state, initial_state, t, "rk4").T
    energy = get_pendulum_energy(x, v) - E_initial
    ax.plot(t, energy, c="#94247d", linewidth=5, label="rk4", zorder=-1)

    # built-in odeint
    x, v = odeint(get_simple_pendulum_state, initial_state, t).T
    energy = get_pendulum_energy(x, v) - E_initial
    ax.plot(t, energy, c="#3e0c5f", linewidth=2, label="odeint", zorder=-1)

    # # built-in solve-ivp
    # solution = solve_ivp(get_simple_pendulum_state_tfirst, (t_start, t_end), initial_state)
    # t, y = solution.t, solution.y
    # x, v = y[0], y[1]
    # energy = get_pendulum_energy(x, v) - E_initial
    # ax.plot(t, energy, label="solve\_ivp", zorder=-1)

    ax.set_xticks(np.arange(t_start, t_end + 0.01*t_end, 10 * period))
    labels = []
    for i in range(0, N_periods + 1, 10):
        labels.append(str(i))
    ax.set_xticklabels(labels)

    ax.set_xlabel(r"$t/T_0$", fontsize=12)
    ax.set_ylabel("Energy", fontsize=12)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid()

    plt.title("Energy Over 100 Oscillation Periods", fontsize=18)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "energy-long_.png", dpi=200)
    plt.show()


def plot_simple_motion(h, initial_state):
    """ Plots a few oscillations of the simple pendulum just to give a sense
         of what the solution looks like.

        Plots angular velocity and displacement on the same axes

        :param h: step size with which to compute solution
        :param initial_state: length-2 array of the form [x0, v0]
        """
    period = get_simple_pendulum_period(initial_state[0])
    N_periods = 3
    t_start = 0  # start time
    t_end = N_periods * period  # end time
    n = int((t_end - t_start) / h)  # number of points
    t = np.linspace(t_start, t_end, n, endpoint=False)
    x, v = get_symplectic_solution(get_simple_pendulum_state_symp, initial_state, t, "pefrl")
    xmax, xmin = np.max(x), np.min(x)
    vmax, vmin = np.max(v), np.min(v)
    v_scale = xmax/vmax
    v = v*v_scale  # rescale v to take up entire figure
    fig, ax = plt.subplots(1, figsize=(7, 3))

    rc('text', usetex=True)
    ax.set_xticks(np.arange(t_start, t_end + 0.01*t_end, period))
    labels = []
    for i in range(N_periods + 1):
        labels.append(str(i))
    ax.set_xticklabels(labels)

    ax.plot(t, x, color=color_blue, label="position")
    ax.plot(t, v, color=color_orange_dark, ls='--', label="velocity")
    ax.hlines(0, t_start, t_end, color='#999999', linewidth=0.7)
    ax.text(-0.008, 1.0, "{:.1f}".format(xmax), va="center", ha="right", transform=ax.transAxes)  # max x
    ax.text(-0.008, 0.5, "{:.1f}".format((xmax + xmin)/2), va="center", ha="right", transform=ax.transAxes)  # zero
    ax.text(-0.008, 0.0, "{:.1f}".format(xmin), va="center", ha="right", transform=ax.transAxes)  # min x
    ax.text(1.008, 1.0, "{:.1f}".format(vmax), va="center", ha="left", transform=ax.transAxes)  # max v
    ax.text(1.008, 0.5, "{:.1f}".format((vmax + vmin)/2), va="center", ha="left", transform=ax.transAxes)  # zero
    ax.text(1.008, 0.0, "{:.1f}".format(vmin), va="center", ha="left", transform=ax.transAxes)  # min v

    ax.text(-0.06, 0.5, r"Displacement $\theta(t)$", fontsize=12, va="center", ha="right", transform=ax.transAxes, rotation=90)
    ax.text(1.05, 0.5, r"Velocity $\dot\theta(t)$", fontsize=12, va="center", ha="left", transform=ax.transAxes, rotation=270)

    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_xlabel(r"$t/T_0$", labelpad=-0.1, fontsize=12)
    ax.margins(x=0)
    ax.legend(framealpha=0.95, loc="lower right")
    ax.grid(which='major', axis='x')
    ax.set_title(r"Simple Pendulum $\theta_0={:.1f}$, $\dot\theta_0={:.1f}$".format(initial_state[0], initial_state[1]), fontsize=15)

    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "simple-motion-{:.1f}-{:.1f}_.png".format(initial_state[0], initial_state[1]), dpi=200)
    plt.show()


def plot_simple_phase_space():
    x1D = np.linspace(-3*np.pi, 3*np.pi, 100)
    v1D = np.linspace(-3, 3, 100)
    fig, ax = plt.subplots(1, figsize=(7, 3.5))

    xgrid, vgrid = np.meshgrid(x1D, v1D)
    agrid = get_simple_pendulum_state_symp(xgrid)
    plt.streamplot(xgrid, vgrid, vgrid, agrid, color=color_blue)

    rc('text', usetex=True)
    ax.set_xticks(np.arange(-3*np.pi, 3*np.pi+0.01, np.pi))
    labels = [r"-3$\pi$", r"-2$\pi$", r"-$\pi$", r"0", r"$\pi$", r"2$\pi$", r"3$\pi$"]
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Angular Displacement $\theta$", fontsize=12, labelpad=-0.1)
    ax.set_ylabel(r"Angular Velocity $\dot\theta$", fontsize=12, labelpad=-0.1)
    ax.set_title("Phase Portrait of a Simple Pendulum", fontsize=18)

    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "phase-simple_.png", dpi=200)
    plt.show()


def plot_damped_phase_space():
    x1D = np.linspace(-3*np.pi, 3*np.pi, 100)
    v1D = np.linspace(-3, 3, 100)
    fig, ax = plt.subplots(1, figsize=(7, 3.5))

    xgrid, vgrid = np.meshgrid(x1D, v1D)

    Vgrid, Agrid = np.zeros(np.shape(xgrid)), np.zeros(np.shape(vgrid))
    for i in range(np.shape(xgrid)[0]):
        for j in range(np.shape(xgrid)[1]):
            x = xgrid[i, j]
            v = vgrid[i, j]
            state = get_simple_pendulum_damped_state([x, v], 0)
            # print(state)
            Vgrid[i, j] = state[0]
            Agrid[i, j] = state[1]

    plt.streamplot(xgrid, vgrid, Vgrid, Agrid, color=color_blue)

    rc('text', usetex=True)
    ax.set_xticks(np.arange(-3*np.pi, 3*np.pi+0.01, np.pi))
    labels = [r"-3$\pi$", r"-2$\pi$", r"-$\pi$", r"0", r"$\pi$", r"2$\pi$", r"3$\pi$"]
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Angular Displacement $\theta$", fontsize=12, labelpad=-0.1)
    ax.set_ylabel(r"Angular Velocity $\dot\theta$", fontsize=12, labelpad=-0.1)
    ax.set_title("Phase Portrait of a Damped Simple Pendulum", fontsize=18)

    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "phase-simple-damped_.png", dpi=200)
    plt.show()


def plot_damped_driven_phase_space():
    x1D = np.linspace(-3*np.pi, 3*np.pi, 100)
    v1D = np.linspace(-3, 3, 100)
    fig, ax = plt.subplots(1, figsize=(7, 3.5))

    xgrid, vgrid = np.meshgrid(x1D, v1D)

    Vgrid, Agrid = np.zeros(np.shape(xgrid)), np.zeros(np.shape(vgrid))
    for i in range(np.shape(xgrid)[0]):
        for j in range(np.shape(xgrid)[1]):
            x = xgrid[i, j]
            v = vgrid[i, j]
            state = get_simple_pendulum_damped_driven_state([x, v], 0)
            # print(state)
            Vgrid[i, j] = state[0]
            Agrid[i, j] = state[1]

    plt.streamplot(xgrid, vgrid, Vgrid, Agrid, color=color_blue)

    rc('text', usetex=True)
    ax.set_xticks(np.arange(-3*np.pi, 3*np.pi+0.01, np.pi))
    labels = [r"-3$\pi$", r"-2$\pi$", r"-$\pi$", r"0", r"$\pi$", r"2$\pi$", r"3$\pi$"]
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Angular Displacement $\theta$", fontsize=12, labelpad=-0.1)
    ax.set_ylabel(r"Angular Velocity $\dot\theta$", fontsize=12, labelpad=-0.1)
    ax.set_title("Phase Portrait of a Damped, Driven Simple Pendulum", fontsize=18)

    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "phase-simple-damped-driven_.png", dpi=200)
    plt.show()


def plot_van_der_pol_phase_space():
    x1D = np.linspace(-3*np.pi, 3*np.pi, 100)
    v1D = np.linspace(-3, 3, 100)
    fig, ax = plt.subplots(1, figsize=(7, 3.5))
    mu = 1.0

    xgrid, vgrid = np.meshgrid(x1D, v1D)
    Vgrid, Agrid = np.zeros(np.shape(xgrid)), np.zeros(np.shape(vgrid))
    for i in range(np.shape(xgrid)[0]):
        for j in range(np.shape(xgrid)[1]):
            x = xgrid[i, j]
            v = vgrid[i, j]
            state = get_van_der_pol_state([x, v], 0, mu=mu)
            # print(state)
            Vgrid[i, j] = state[0]
            Agrid[i, j] = state[1]

    plt.streamplot(xgrid, vgrid, Vgrid, Agrid, color=color_blue)

    rc('text', usetex=True)
    ax.set_xticks(np.arange(-3*np.pi, 3*np.pi+0.01, np.pi))
    labels = [r"-3$\pi$", r"-2$\pi$", r"-$\pi$", r"0", r"$\pi$", r"2$\pi$", r"3$\pi$"]
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Angular Displacement $\theta$", fontsize=12, labelpad=-0.1)
    ax.set_ylabel(r"Angular Velocity $\dot\theta$", fontsize=12, labelpad=-0.1)
    ax.set_title(r"Phase Portrait of a Van der Pol Oscillator, $\lambda={}$".format(mu), fontsize=18)

    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "phase-vdp-{}_.png".format(mu), dpi=200)
    plt.show()


def plot_van_der_pol_driven_phase_space():
    x1D = np.linspace(-3*np.pi, 3*np.pi, 100)
    v1D = np.linspace(-3, 3, 100)
    fig, ax = plt.subplots(1, figsize=(7, 3.5))
    a_d = 1.0

    xgrid, vgrid = np.meshgrid(x1D, v1D)
    Vgrid, Agrid = np.zeros(np.shape(xgrid)), np.zeros(np.shape(vgrid))
    for i in range(np.shape(xgrid)[0]):
        for j in range(np.shape(xgrid)[1]):
            x = xgrid[i, j]
            v = vgrid[i, j]
            state = get_van_der_pol_driven_state([x, v], 0, a_d=a_d)
            # print(state)
            Vgrid[i, j] = state[0]
            Agrid[i, j] = state[1]

    plt.streamplot(xgrid, vgrid, Vgrid, Agrid, color=color_blue)

    rc('text', usetex=True)
    ax.set_xticks(np.arange(-3*np.pi, 3*np.pi+0.01, np.pi))
    labels = [r"-3$\pi$", r"-2$\pi$", r"-$\pi$", r"0", r"$\pi$", r"2$\pi$", r"3$\pi$"]
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Angular Displacement $\theta$", fontsize=12, labelpad=-0.1)
    ax.set_ylabel(r"Angular Velocity $\dot\theta$", fontsize=12, labelpad=-0.1)
    ax.set_title(r"Phase Portrait of a Driven Van der Pol Oscillator, $A_d={}$".format(a_d), fontsize=18)

    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "phase-vdp-driven-{}_.png".format(a_d), dpi=200)
    plt.show()


def plot_driven_displacement():
    b = 0.5
    w_d = 5
    a_d = 1.0

    initial_state = get_initial_state()

    period = 2 * np.pi / w_d
    N_periods = 50
    t_start = 0  # start time
    t_end = N_periods * period  # end time

    # x, v = get_fixed_step_solution(get_simple_pendulum_damped_driven_state, initial_state, t, "rk4").T

    # built-in solve-ivp
    method="RK45"
    solution = solve_ivp(get_simple_pendulum_damped_driven_state_tfirst, (t_start, t_end), initial_state, method=method,
                         max_step=5e-1, atol=1e-6, args=(b, w_d, a_d))
    t, y = solution.t, solution.y
    x, v = y[0], y[1]
    plt.plot(t, x)
    plt.show()


def plot_resonance_curve():
    """ Plots damped, driven simple pendulum's resonance curve for various driving amplitudes """

    # amplitudes = (0.5, 1.0, 1.5, 2.5, 3.5, 5.0)
    amplitudes = (0.5, 1.0, 1.5, 5.0)
    # amplitudes = (0.5, 0.75, 0.85, 0.90)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, a_d in enumerate(amplitudes):
        if i == 0:
            ax = axes[0][0]
        elif i == 1:
            ax = axes[0][1]
        elif i == 2:
            ax = axes[1][0]
        else:
            ax = axes[1][1]

        filename = resonance_dir + "{:.2f}.csv".format(a_d)
        w_d, max_amplitude = np.loadtxt(filename, skiprows=1, delimiter=",").T
        ax.plot(w_d, max_amplitude, c=color_blue, ls='--', marker='.')

        if i > 1: ax.set_xlabel("Driving Frequency $w$", fontsize=12)
        if i % 2 == 0: ax.set_ylabel("Maximum Amplitude", fontsize=12)
        ax.set_title("Driving Amplitude $A_d={:.2f}$".format(a_d), fontsize=14)
        ax.grid()

    plt.tight_layout()
    plt.suptitle("Driven Simple Pendulum's Resonance Curve", fontsize=18)
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "resonance_.png", dpi=200)
    plt.show()
# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
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

    a = np.arange(0.1, 0.35, 0.03)
    b = np.arange(0.35, 0.65, 0.01)
    c = np.arange(0.65, 1.5, 0.03)
    d = np.concatenate((a, b, c))
    print(d)


def run():
    # practice()
    # try_stuff()
    # try_error_stuff()
    # generate_fixed_step_solutions(get_initial_state())
    # generate_symplectic_solutions(get_initial_state())
    # generate_adaptive_step_solutions(get_initial_state())
    # generate_built_in_solutions(get_initial_state())
    # generate_fixed_step_errors(get_initial_state())
    # generate_adaptive_step_errors(get_initial_state())
    # generate_resonance_data()

    # plot_simple_motion(0.1, [1.0, 0.0])
    # plot_simple_motion(0.1, [0.99*np.pi, 0.0])
    # plot_simple_motion(0.1, [0.99 * np.pi, 0.1])
    # plot_fixed_step_x_error()
    # plot_fixed_step_energy_error()
    # plot_adaptive_step_x_error()
    # plot_adaptive_step_energy_error()
    # plot_long_period()

    plot_simple_phase_space()
    # plot_damped_phase_space()
    # plot_damped_driven_phase_space()
    # plot_van_der_pol_phase_space()
    # plot_van_der_pol_driven_phase_space()

    plot_resonance_curve()

run()
