import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import re
import time
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


def plot_fixed_step_stability():
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
