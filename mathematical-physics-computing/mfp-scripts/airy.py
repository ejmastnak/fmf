import mpmath
import math
from matplotlib import pyplot as plt
import numpy as np

# Start global constants
figure_directory = "../1-airy-functions/figures/"
log_series = False      # controls whether to log outputs of series calculations
save_figures = True      # controls whether to save figures to avoid accidental overwrites

mpmath.mp.dps = 18      # set mpmath decimal places
mac_epsilon = 1e-10   # epsilon for Maclaurin series calculations
asym_epsilon = 1e-10   # epsilon for asymptotic series calculations
min_epsilon = 1e-16  # limit of numberical accuracy
max_iterations_pow = 40     # maximum iterations for Maclaurin series calculations
max_iterations_asym = 40    # maximum iterations for asymptotic series calculations

asym_to_power = -7    # when to switch from power series to asymptotic approximation
power_to_asym_A = 5.5     # when to switch from power series to asymptotic approximation for ai
power_to_asym_B = 8.2     # when to switch from power series to asymptotic approximation for bi

num_points = 500    # number of points in the simulation
x_min = -5         # minimum x value
x_max = 10         # maximum x value

A = 0.355028053887817239     # A = ai(0) to 18 decimal places. Used in Maclaurin series
B = 0.258819403792806798     # B = -ai'(0)to 18 decimal places. Used in Maclaurin series

ai_color = "#377eb8"    # consistent color for Ai plots. Slightly darker
bi_color = "#e41a1c"    # consistent color for Bi plots. Slightly lighter
light_color1 = "#cccccc"
light_color2 = "#eeeeee"


marker1 = 'o'   # circle
marker2 = '^'   # triangle
marker3 = 's'   # square
# End global constants


# -----------------------------------------------------------------------------
# START ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def plot_true_airy():

    x = np.linspace(x_min, x_max, num_points)
    ai = []
    for i in range(num_points):
        ai = ai + [mpmath.airyai(x[i])]

    bi = []
    for i in range(num_points):
        bi = bi + [mpmath.airybi(x[i])]

    plt.xlabel("x")
    plt.ylabel("Function value")
    plt.plot(x, ai, label="ai(x)")
    plt.plot(x, bi, label="bi(x)")
    plt.legend()
    plt.show()


def mac_f(x):
    """
    The auxiliary function f(x) in the Maclaurin series for ai and bi
    :param x:
    :return:
    """
    next_term = 1       # next term in the series
    current_sum = 1     # cumulative sum
    old_sum = 0
    
    for n in range(1, max_iterations_pow):
        old_sum = current_sum
        old_term = next_term
        next_term *= (1/3 + n - 1) * 3 * (x**3) / ((3*n) * (3*n - 1) * (3*n - 2))
        current_sum += next_term    # update cumulative sum

        # if abs(old_sum - current_sum) < mac_epsilon:
        if abs(old_term - next_term) < mac_epsilon:
            # print("Exiting f(x) at {} iterations\t x = {:.4f} \t f(x) = {:.2e}".format(n, x, current_sum))
            if log_series: print("Exiting at {} iterations for f({:.4f})\t f = {:.16f} \t Step: {:.16e}".format(n, x, current_sum, abs(old_sum - current_sum)))
            return current_sum

    # print("Maxed at {} iterations for f(x) for \t x = {:.4f} \t f(x) = {:.2e}".format(max_iterations_pow, x, current_sum))
    if log_series: print("Maxed at {} iterations for f({:.4f})\t g = {:.16f} \t Step: {:.16e}".format(max_iterations_pow, x, current_sum, abs(old_sum - current_sum)))

    return current_sum


def mac_g(x):
    """
    The auxiliary function g(x) in the Maclaurin series for ai and bi
    :param x:
    :return:
    """
    next_term = x       # next term in the series. Initial value is x at first iteration
    current_sum = x     # cumulative sum. Initial value is x at first iteration
    old_sum = 0

    for n in range(1, max_iterations_pow):
        old_sum = current_sum
        old_term = next_term
        next_term *= (2/3 + n - 1) * 3 * (x**3) / ((3*n + 1) * (3*n) * (3*n - 1))
        current_sum += next_term

        # if abs(old_sum - current_sum) < mac_epsilon:
        if abs(old_term - next_term) < mac_epsilon:
            # print("Exiting g(x) at {} iterations\t x = {:.4f} \t g(x) = {:.2e}".format(n, x, current_sum))
            if log_series: print("Exiting at {} iterations for g({:.4f})\t g = {:.16f} \t Step: {:.16e}".format(n, x, current_sum, abs(old_sum - current_sum)))

            return current_sum

    # print("Maxed at {} iterations for g(x) for \t x = {:.4f} \t g(x) = {:.2e}".format(max_iterations_pow, x, current_sum))
    if log_series: print("Maxed at {} iterations for g({:.4f})\t g = {:.16f} \t Step: {:.16e}".format(max_iterations_pow, x, current_sum, abs(old_sum - current_sum)))

    return current_sum


def L(xi, x):
    """
    Returns a partial sum of the asymptotic series L(x), used in the asymptotic expansion
    of ai and bi for large positive x
    :param xi: the argument
    :param x: used only for logging
    :return:
    """
    next_term = 1   # Next term in the series. Initial value is 1 at first iteration
    current_sum = 1     # cumulative sum. Initial value is 1 at first iteration
    old_sum = 0

    for n in range(1, max_iterations_asym):
        old_term = next_term
        old_sum = current_sum
        next_term *= (6*n - 1)*(6*n - 3)*(6*n - 5)/(216 * n * (2*n-1) * xi)
        current_sum += next_term

        if truncate_asym_expansion(next_term, old_term, current_sum, old_sum, n, x, "L"):
            return current_sum

    return current_sum


def P(xi, x):
    """
    Returns a partial sum of the asymptotic series P(x), used in the asymptotic expansion
    of ai and bi for large negative x
    :param xi: the argument
    :param x: used only for logging
    :return:
    """
    next_term = 1   # Next term in the series. Initial value is 1 at first iteration
    current_sum = next_term     # cumulative sum. Initial value is 1 at first iteration
    old_sum = 0

    for n in range(1, max_iterations_asym):
        old_term = next_term
        old_sum = current_sum
        next_term *= -(12*n - 1)*(12*n - 3)*(12*n - 5)*(12*n - 7)*(12*n - 9)*(12*n - 11)/((216 ** 2) * (2*n) * (2*n - 1) * (4*n - 1) * (4*n - 3) * (xi ** 2))
        current_sum += next_term

        if truncate_asym_expansion(next_term, old_term, current_sum, old_sum, n, x, "P"):
            return current_sum

    return current_sum


def Q(xi, x):
    """
    Returns a partial sum of the asymptotic series Q(x), used in the asymptotic expansion
    of ai and bi for large negative x
    :param xi: the argument
    :param x: used only for logging
    :return:
    """
    next_term = 15/(216 * xi)   # Next term in the series. Initial value is 15/(216x) at first iteration
    current_sum = next_term     # cumulative sum. Initial value is 15/(216x) at first iteration
    old_sum = 0

    for n in range(1, max_iterations_asym):
        old_term = next_term
        old_sum = current_sum
        next_term *= -(12*n + 5)*(12*n + 3)*(12*n + 1)*(12*n - 1)*(12*n - 3)*(12*n - 5)/((216 ** 2) * (2*n) * (2*n + 1) * (4*n + 1) * (4*n - 1) * (xi ** 2))
        current_sum += next_term

        if truncate_asym_expansion(next_term, old_term, current_sum, old_sum, n, x, "Q"):
            return current_sum

    return current_sum


def truncate_asym_expansion(current_term, old_term, current_sum, old_sum, iteration, x, series_name):
    """
    Determines when to optimally truncate the asymptotic expansions for L, P, and Q
    Displays an appropriate exit message
    :param current_term:
    :param old_term:
    :param current_sum:
    :param old_sum:
    :param x
    :param iteration
    :param series_name
    :return:
    """
    if abs(old_term - current_term) < min_epsilon:    # to avoid iterating beyond regime of numerical error
        if log_series: print("{} iterations {}({:.4f})\t L = {:.16f} \t Step: {:.16e} less than {:.2e}".format(iteration, series_name, x,current_sum, abs(old_sum - current_sum), min_epsilon))
        return True
    elif abs(current_term) > abs(old_term):
    # elif abs(old_sum - current_sum) < asym_epsilon:   # gives aweful results for 1e-16 and fine results for 1e-10
    # elif abs(old_term - next_term) < asym_epsilon:
        if log_series: print("{} iterations {}({:.4f})\t L = {:.16f} \t Step: {:.16e}".format(iteration, series_name, x,
                                                                                          current_sum,
                                                                                          abs(old_sum - current_sum)))
        return True
    elif iteration == max_iterations_asym:
        if log_series: print("Max {} iterations {}({:.4f})\t L = {:.16f} \t Step: {:.16e}".format(max_iterations_asym,
                                                                                            series_name, x, current_sum,
                                                                                            abs(old_sum - current_sum)))

    else: return False


def my_airy_AB(x):
    """
    My implementation of the Airy functions ai and bi using finite precision arithmetic
    I calculate Ai and Bi together (instead of implementing separate functions for Ai(x) and Bi(x))
    to only have to calculate the auxiliary values P, Q, f(x) and g(x)

    :param x:
    :return: A two-tuple holding (ai(x), bi(x))
    """
    if x <= asym_to_power:     # asymptotic for both A and B
        xi = 2 / 3 * math.pow(abs(x), 1.5)
        q = Q(xi, x)
        p = P(xi, x)
        ai = (math.sin(xi - (math.pi/4))*q + math.cos(xi - (math.pi/4))*p) / (math.sqrt(math.pi) * math.pow(-x, 0.25))
        bi = (math.cos(xi - (math.pi/4))*q - math.sin(xi - (math.pi/4))*p) / (math.sqrt(math.pi) * math.pow(-x, 0.25))
        return ai, bi

    elif asym_to_power < x <= power_to_asym_A:     # power for both A and B
        f = mac_f(x)
        g = mac_g(x)
        ai = A * f - B * g
        bi = math.sqrt(3) * (A*f + B*g)
        return ai, bi

    elif power_to_asym_A < x <= power_to_asym_B:   # asym for A and power for B
        xi = 2 / 3 * math.pow(abs(x), 1.5)
        ai = L(-xi, x) * math.exp(-xi) / (2 * math.sqrt(math.pi) * math.pow(x, 0.25))

        f = mac_f(x)
        g = mac_g(x)
        bi = math.sqrt(3) * (A * f + B * g)

        return ai, bi

    elif x > power_to_asym_B:     # asymptotic for both A and B
        xi = 2/3 * math.pow(abs(x), 1.5)
        ai = L(-xi, x) * math.exp(-xi) / (2 * math.sqrt(math.pi) * math.pow(x, 0.25))
        bi = L(xi, x) * math.exp(xi) / (math.sqrt(math.pi) * math.pow(x, 0.25))
        return ai, bi


def my_airyai(x):
    """
    My implementation of the Airy function ai  using finite precision arithmetic

    :param x:
    :return: Approximation for ai(x)
    """
    if x <= asym_to_power:     # asymptotic
        xi = 2 / 3 * math.pow(abs(x), 1.5)
        q = Q(xi, x)
        p = P(xi, x)
        return (math.sin(xi - (math.pi/4))*q + math.cos(xi - (math.pi/4))*p) / (math.sqrt(math.pi) * math.pow(-x, 0.25))

    elif asym_to_power < x < power_to_asym_A:     # power
        f = mac_f(x)
        g = mac_g(x)
        return A * f - B * g

    elif x >= power_to_asym_A:   # asymptotic
        xi = 2 / 3 * math.pow(abs(x), 1.5)
        return L(-xi, x) * math.exp(-xi) / (2 * math.sqrt(math.pi) * math.pow(x, 0.25))


def my_airybi(x):
    """
    My implementation of the Airy function bi using finite precision arithmetic

    :param x:
    :return: Approximation for bi(x)
    """
    if x <= asym_to_power:     # asymptotic
        xi = 2 / 3 * math.pow(abs(x), 1.5)
        q = Q(xi, x)
        p = P(xi, x)
        return (math.cos(xi - (math.pi/4))*q - math.sin(xi - (math.pi/4))*p) / (math.sqrt(math.pi) * math.pow(-x, 0.25))

    elif asym_to_power < x < power_to_asym_B:     # power
        f = mac_f(x)
        g = mac_g(x)
        return math.sqrt(3) * (A*f + B*g)
  
    elif x >= power_to_asym_B:     # asymptotic
        xi = 2/3 * math.pow(abs(x), 1.5)
        return L(xi, x) * math.exp(xi) / (math.sqrt(math.pi) * math.pow(x, 0.25))
# -----------------------------------------------------------------------------
# END ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ZERO FUNCTIONS
# -----------------------------------------------------------------------------
def asym_airyaizero(n):
    """
    Returns the kth zero of ai calculated using the asymptotic series in the assignment instructions
    :param n:
    :return:
    """
    xi = 3 * math.pi * (4 * n - 1) / 8
    if n == 1:
        return - airyzero_asym_series(xi, 4)  # 4 terms minimizes error for n = 1
    else:
        return - airyzero_asym_series(xi, 5)


def asym_airybizero(n):
    """
    Returns the kth zero of bi calculated using the asymptotic series in the assignment instructions
    :param n:
    :return:
    """
    xi = 3 * math.pi * (4 * n - 3) / 8
    if n == 1:
        return - airyzero_asym_series(xi, 2)    # 2 terms minimizes error for n = 1
    else:
        return - airyzero_asym_series(xi, 5)


def airyzero_asym_series(xi, number_terms):
    """
    Asymptotic series approximation for airy zeros

    :param xi:
    :param number_terms: controls how many terms to use the expansion
    :return:
    """

    if number_terms == 1:
        return math.pow(xi, 2 / 3)
    elif number_terms == 2:
        return math.pow(xi, 2 / 3) * (1 + 5 / (48 * (xi ** 2)))
    elif number_terms == 3:
        return math.pow(xi, 2 / 3) * (1 + 5 / (48 * (xi ** 2)) - 5 / (36 * (xi ** 4)))
    elif number_terms == 4:
        return math.pow(xi, 2 / 3) * (1 + 5 / (48 * (xi ** 2)) - 5 / (36 * (xi ** 4)) + 77125 / (82944 * (xi ** 6)))
    else:    # uses 5 terms as default case:
        return math.pow(xi, 2 / 3) * (1 + 5 / (48 * (xi ** 2)) - 5 / (36 * (xi ** 4)) + 77125 / (82944 * (xi ** 6)) - 108056875 / (6967296 * (xi ** 8)))


def my_airy_zeros(max_zeros, eps):
    """
    Finds the first max_zeros zeros of the ai and bi using the bisection method
    Returns an tuple array my_airyaizeros, my_airybizeros

    :param max_zeros: how many zeros to find
    :param eps: controls tolerance in bisection method
    :return:
    """
    xmin = 0
    xmax = -61  # linspace parameters reversed on purpose to calculate zeros from 0 toward -61
    numpoints = 500
    X = np.linspace(xmin, xmax, numpoints)

    my_ai_values = []
    my_bi_values = []
    for i in range(numpoints):
        my_airy_tuple = my_airy_AB(X[i])
        my_ai_values.append(my_airy_tuple[0])
        my_bi_values.append(my_airy_tuple[1])

    zero_intervals_a = []      # array of intervals representing the two closest points to each zero
    zero_intervals_b = []
    zero_counter_a = 0        # counts how many zeros have been found
    zero_counter_b = 0

    current_a = my_ai_values[0] # first value
    current_b = my_bi_values[0] # first value
    for i in range(1, numpoints):     # loop through Airy function values
        prev_a = current_a
        prev_b = current_b
        current_a = my_ai_values[i]
        current_b = my_bi_values[i]

        if zero_counter_a < max_zeros and ((prev_a <= 0 <= current_a) or (current_a <= 0 <= prev_a)):  # crossed a zero
            zero_intervals_a.append((X[i-1], X[i]))     # record the surrounding interval
            zero_counter_a += 1     # update zero counter

        if zero_counter_b < max_zeros and ((prev_b <= 0 <= current_b) or (current_b <= 0 <= prev_b)):  # crossed a zero
            zero_intervals_b.append((X[i-1], X[i]))
            zero_counter_b += 1

    my_airyaizeros = []
    my_airybizeros = []
    for i in range(0, max_zeros):   # loop through array of zero intervals, of which there are max_zeros
        zero_a = bisection(my_airyai, zero_intervals_a[i][0], zero_intervals_a[i][1], eps)
        zero_b = bisection(my_airybi, zero_intervals_b[i][0], zero_intervals_b[i][1], eps)
        my_airyaizeros.append(zero_a)
        my_airybizeros.append(zero_b)

    return my_airyaizeros, my_airybizeros


def bisection(f, a, b, eps):
    """
    bisection method modified to work with the Airy functions ai and bi
    :param f: some function
    :param a: left endpoint
    :param b: right endpoint
    :param eps: tolerance
    :return:
    """
    if a == 0.0: return a   # safety check in case intervals are zeros; should never happen in practice
    if b == 0.0: return b

    counter = 0
    max_iterations = 100
    while abs(b-a) > eps:
        c = a + (b-a)/2
        if np.sign(f(a)) == np.sign(f(c)):     # f has same sign at a and c
            a = c       # shift left endpoint to c
        else:       # f has difference sign at a and c
            b = c       # shift right endpoint to c
        counter += 1
        if counter > max_iterations: break

    return a + (b - a)/2
# -----------------------------------------------------------------------------
# END ZERO FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START COEFFICIENT A AND B PRECISION TESTING
# -----------------------------------------------------------------------------
def get_A_mp():
    return 1/(mpmath.power(3, mpmath.fdiv(2,3)) * mpmath.gamma(mpmath.fdiv(2, 3)))


def get_B_mp():
    return -1/(mpmath.power(3, mpmath.fdiv(1,3)) * mpmath.gamma(mpmath.fdiv(1,3)))    # B = -ai'(0), used in Maclaurin series


def get_A_sys():
    return 1/(math.pow(3, 2/3) * math.gamma(2/3))


def get_B_sys():
    return -1/(math.pow(3, 1/3) * math.gamma(1/3))


def test_A_and_B_precision():
    """
    Compares system (finite precision) and mpmath (arbitrary precision) values
    of the coefficiens A and B from the Maclaurin series for ai and bi
    :return:
    """
    print("System A: {:.18f}".format(1/(math.pow(3, 2/3) * math.gamma(2/3))))
    print("mpmath A: " + mpmath.nstr(1/(mpmath.power(3, mpmath.fdiv(2,3)) * mpmath.gamma(mpmath.fdiv(2, 3))), mpmath.mp.dps))

    print("System B: {:.18f}".format(-1/(math.pow(3, 1/3) * math.gamma(1/3))))
    print("mpmath B: " + mpmath.nstr(-1/(mpmath.power(3, mpmath.fdiv(1,3)) * mpmath.gamma(mpmath.fdiv(1,3))), mpmath.mp.dps))
# -----------------------------------------------------------------------------
# END COEFFICIENT A AND B PRECISION TESTING
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_asym_term_optimization():
    """
    Plots the absolute (and relative) error of ai and bi as a function of number of terms
    in the asymptotic expansion for large positive values of x
    :param x:
    :return:
    """

    max_terms = max_iterations_asym
    term_range = range(1, max_terms)
    fig, ax = plt.subplots(figsize=(8, 5))

    counter = 1     # counts loop interations to control which marker to use. Probably there's a better way...
    marker = marker1
    for x in [5, 7, 9]:
        xi = 2 / 3 * math.pow(abs(x), 1.5)
        mp_bi = mpmath.airybi(x)

        abs_errors_bi = []
        rel_errors_bi = []

        for i in term_range:
            next_term = 1  # Next term in the series. Initial value is 1 at first iteration
            current_sum = 1  # cumulative sum. Initial value is 1 at first iteration

            for n in range(1, i):
                next_term *= (6 * n - 1) * (6 * n - 3) * (6 * n - 5) / (216 * n * (2 * n - 1) * xi)
                current_sum += next_term

            my_bi = current_sum * math.exp(xi) / (math.sqrt(math.pi) * math.pow(x, 0.25))

            abs_errors_bi.append(abs(mp_bi - my_bi))
            rel_errors_bi.append(abs(mp_bi - my_bi) / abs(mp_bi))

        plt.subplot(211)

        if counter == 1: marker = marker1
        elif counter == 2: marker = marker2
        elif counter == 3: marker = marker3
        else: marker = '.' # use a dot as default

        plt.ylabel("Absolute Error")
        plt.yscale("log")
        plt.plot(term_range, abs_errors_bi, linestyle='--', marker=marker, label="Bi({:.2f})".format(x))
        plt.legend()

        plt.subplot(212)
        plt.ylabel("Relative Error")
        plt.xlabel("Terms in asymptotic expansion")
        plt.yscale("log")
        plt.plot(term_range, rel_errors_bi, linestyle='--', marker=marker, label="Bi({:.2f})".format(x))
        plt.legend()

        counter += 1

    plt.tight_layout()
    if save_figures: plt.savefig(figure_directory + "asymptotic-term-optimization1.png", dpi=250)
    plt.show()


def plot_my_airy(X, my_ai, my_bi):

    plt.xlabel("x")
    plt.plot(X, my_ai, label="My ai(x)")
    plt.plot(X, my_bi, label="My bi(x)")

    plt.legend()
    plt.show()


def plot_error_both(X, my_ai, my_bi, mp_ai, mp_bi):
    """
    Plots both absolute and relative error of my Airy implementation
    with respect to the arbitrary precision Airy function in the mpmath package

    :param X
    :param my_ai:
    :param my_bi:
    :param mp_ai:
    :param mp_bi:
    :return:
    """

    ai_abs_error = []
    bi_abs_error = []

    for i in range(num_points):
        # Calculate abs error of my implementation with respect to mpmath.airy
        ai_abs_error.append(abs(my_ai[i] - mp_ai[i]))
        bi_abs_error.append(abs(my_bi[i] - mp_bi[i]))

    ai_rel_error = []
    bi_rel_error = []

    for i in range(num_points):
        # Calculate rel error of my implementation with respect to mpmath.airy
        ai_rel_error.append(abs((my_ai[i] - mp_ai[i])/mp_ai[i]))
        bi_rel_error.append(abs((my_bi[i] - mp_bi[i])/mp_bi[i]))

    fig, ax = plt.subplots(figsize=(10, 7))

    plt.subplot(211)
    plt.ylabel("Absolute Error")
    plt.yscale("log")
    plt.plot(X, ai_abs_error, color=ai_color, label="Ai(x) absolute error")
    plt.plot(X, bi_abs_error, color=bi_color, label="Bi(x) absolute error")
    plt.grid()
    plt.tight_layout()
    plt.legend()

    plt.subplot(212) # relative error
    plt.xlabel("x")
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.plot(X, ai_rel_error, color=ai_color, label="Ai(x) relative error")
    plt.plot(X, bi_rel_error, color=bi_color, label="Bi(x) relative error")
    plt.legend()
    plt.tight_layout()
    plt.grid()

    if save_figures: plt.savefig(figure_directory + "rel-abs-errors1.png", dpi=250)
    plt.show()


def plot_compared_airys(X, my_ai, my_bi):
    """
    Plots a comparison of the graphs of my Airy implementation and the arbitrary precision Airy function
    in the mpmath package

    :param X
    :param my_ai:
    :param my_bi:
    :return:
    """

    mp_X = np.linspace(x_min, x_max, num_points) # 10 times for points than my implementation for better resolution
    mp_ai = []
    mp_bi = []

    for i in range(len(mp_X)):
        mp_ai.append(mpmath.airyai(mp_X[i]))
        mp_bi.append(mpmath.airybi(mp_X[i]))


    ai_color = "#100975"  # override global value
    bi_color = "#379ec3"  # override global value

    plt.figure(figsize=(7.5, 4))
    plt.ylabel("Ai(x) and Bi(x)")
    plt.xlabel("x")
    plt.plot(X, my_ai, color=ai_color, label="my Ai(x) implementation", zorder=2)
    plt.plot(X, my_bi, color=bi_color, label="my Bi(x) implementation", zorder=1)
    plt.plot(mp_X, mp_ai, linestyle='none', marker='o', color=light_color1, markersize=6, label="mpmath.airyai", zorder=0)
    plt.plot(mp_X, mp_bi, linestyle='none', marker='s', color=light_color2, markersize=6, label="mpmath.airybi", zorder=0)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    if save_figures: plt.savefig(figure_directory + "Ai-Bi-graphs1.png", dpi=250)
    plt.show()


def plot_airy_zeros(max_zeros):
    """
    Plots the first max_zeros zeros of ai and bi.
    Compres the values of:
        mpmath's arbitrary-precision functions
        asymptotic expansions
        my bisection implementation


    :param max_zeros:
    :return:
    """

    mp_ai_zeros = []
    mp_bi_zeros = []
    asym_ai_zeros = []
    asym_bi_zeros = []
    my_ai_zeros, my_bi_zeros = my_airy_zeros(max_zeros, 1e-5)

    zero_range = range(1, max_zeros+1)
    for i in zero_range:
        mp_ai_zeros.append(mpmath.airyaizero(i))
        mp_bi_zeros.append(mpmath.airybizero(i))
        asym_ai_zeros.append(asym_airyaizero(i))
        asym_bi_zeros.append(asym_airybizero(i))

    markersize1 = 13
    markersize2 = 9
    markersize3 = 4

    colora1 = "#f6cc52"
    colora2 = "#c63b1f"
    colorb1 = "#e97b86"
    colorb2 = "#7b1a7f"

    plt.figure(figsize=(7.5, 4))
    plt.xlabel("Zero Index")
    plt.ylabel("Zeros of Ai and Bi")
    plt.plot(zero_range, mp_ai_zeros, linestyle='none', color=light_color2, marker=marker1, markersize=markersize1, markeredgecolor='#000000', markeredgewidth=0.3, label="mpmath.airyaizero")
    plt.plot(zero_range, mp_bi_zeros, linestyle='none', color=light_color1, marker=marker1, markersize=markersize1, markeredgecolor='#000000', markeredgewidth=0.3, label="mpmath.airybizero")

    plt.plot(zero_range, asym_ai_zeros, linestyle='none', color=colora1, marker=marker3, markersize=markersize2, label="asymptotic Ai")
    plt.plot(zero_range, asym_bi_zeros, linestyle='none', color=colorb1, marker=marker3, markersize=markersize2, label="asymptotic Bi")

    plt.plot(zero_range, my_ai_zeros, linestyle='none', color=colora2, marker=marker2, markersize=markersize3, label="my Ai implementation")
    plt.plot(zero_range, my_bi_zeros, linestyle='none', color=colorb2, marker=marker2, markersize=markersize3, label="my Bi implementation")
    plt.legend()

    plt.tight_layout()
    plt.grid()
    if save_figures: plt.savefig(figure_directory + "zeros-compared1.png", dpi=250)

    plt.show()


def plot_airy_zero_error(max_zeros):
    """
    Plots errors of the first max_zeros zeros of ai and bi.
    Compares the values of the asymptotic expansion formula and my bisection implementation
    to mpmath's arbitrary-precision functions

    :param max_zeros:
    :return:
    """

    mp_ai_zeros = []
    mp_bi_zeros = []
    asym_ai_zeros = []
    asym_bi_zeros = []

    eps1 = 1e-8
    eps2 = 1e-20
    my_ai_zeros1, my_bi_zeros1 = my_airy_zeros(max_zeros, eps1)
    my_ai_zeros2, my_bi_zeros2 = my_airy_zeros(max_zeros, eps2)

    zero_range = range(1, max_zeros)
    for i in zero_range:
        mp_ai_zeros.append(mpmath.airyaizero(i))
        mp_bi_zeros.append(mpmath.airybizero(i))
        asym_ai_zeros.append(asym_airyaizero(i))
        asym_bi_zeros.append(asym_airybizero(i))

    asym_ai_abserror = []
    asym_bi_abserror = []
    asym_ai_relerror = []
    asym_bi_relerror = []
    my_ai_abserror1 = []
    my_bi_abserror1 = []
    my_ai_relerror1 = []
    my_bi_relerror1 = []
    my_ai_abserror2 = []
    my_bi_abserror2 = []
    my_ai_relerror2 = []
    my_bi_relerror2 = []

    for i in range(0, max_zeros-1):
        asym_ai_abserror.append(abs(mp_ai_zeros[i] - asym_ai_zeros[i]))
        asym_bi_abserror.append(abs(mp_bi_zeros[i] - asym_bi_zeros[i]))
        asym_ai_relerror.append(abs(mp_ai_zeros[i] - asym_ai_zeros[i])/abs(mp_ai_zeros[i]))
        asym_bi_relerror.append(abs(mp_bi_zeros[i] - asym_bi_zeros[i])/abs(mp_bi_zeros[i]))

        my_ai_abserror1.append(abs(mp_ai_zeros[i] - my_ai_zeros1[i]))
        my_bi_abserror1.append(abs(mp_bi_zeros[i] - my_bi_zeros1[i]))
        my_ai_relerror1.append(abs(mp_ai_zeros[i] - my_ai_zeros1[i]) / abs(mp_ai_zeros[i]))
        my_bi_relerror1.append(abs(mp_bi_zeros[i] - my_bi_zeros1[i]) / abs(mp_bi_zeros[i]))

        my_ai_abserror2.append(abs(mp_ai_zeros[i] - my_ai_zeros2[i]))
        my_bi_abserror2.append(abs(mp_bi_zeros[i] - my_bi_zeros2[i]))
        my_ai_relerror2.append(abs(mp_ai_zeros[i] - my_ai_zeros2[i]) / abs(mp_ai_zeros[i]))
        my_bi_relerror2.append(abs(mp_bi_zeros[i] - my_bi_zeros2[i]) / abs(mp_bi_zeros[i]))

    fig, ax = plt.subplots(figsize=(6, 7))
    ai_color1 = "#9bd3c0" # light teal
    ai_color2 = "#3997bf" # light teal
    ai_color3 = "#94247d" # dark red

    bi_color1 = "#f5c44f"
    bi_color2 = "#e2692d"
    bi_color3 = "#2e6fa7" # dark blue

    asym_marker = 'D'
    asym_marker_size = 4

    plt.subplot(411)
    plt.ylabel("Ai Absolute Error")
    plt.yscale("log")
    plt.plot(zero_range, asym_ai_abserror, linestyle='none', color=ai_color3, marker=asym_marker, markersize=asym_marker_size, label="asymptotic")
    plt.plot(zero_range, my_ai_abserror1, linestyle='none', color=ai_color1, marker='.', markersize=6, label="my_aizero, $\epsilon$ = {:.0e}".format(eps1))
    plt.plot(zero_range, my_ai_abserror2, linestyle='none', color=ai_color2, marker='.', markersize=6, label="my_aizero, $\epsilon$ = {:.0e}".format(eps2))
    plt.tight_layout()

    plt.legend(loc="upper right")

    plt.subplot(412)
    plt.ylabel("Ai Relative Error")
    plt.yscale("log")
    plt.plot(zero_range, asym_ai_relerror, linestyle='none', color=ai_color3, marker=asym_marker, markersize=asym_marker_size, label="asymptotic")
    plt.plot(zero_range, my_ai_relerror1, linestyle='none', color=ai_color1, marker='.', markersize=6, label="my_aizero, $\epsilon$ = {:.0e}".format(eps1))
    plt.plot(zero_range, my_ai_relerror2, linestyle='none', color=ai_color2, marker='.', markersize=6, label="my_aizero, $\epsilon$ = {:.0e}".format(eps2))
    plt.tight_layout()

    plt.legend(loc="upper right")

    plt.subplot(413)
    # plt.xlabel("Zero Index")
    plt.ylabel("Bi Absolute Error")
    plt.yscale("log")
    plt.plot(zero_range, asym_bi_abserror, linestyle='none', color=bi_color3, marker=asym_marker, markersize=asym_marker_size, label="asymptotic")
    plt.plot(zero_range, my_bi_abserror1, linestyle='none', color=bi_color1, marker='.', markersize=6, label="my_bizero, $\epsilon$ = {:.0e}".format(eps1))
    plt.plot(zero_range, my_bi_abserror2, linestyle='none', color=bi_color2, marker='.', markersize=6, label="my_bizero, $\epsilon$ = {:.0e}".format(eps2))
    plt.tight_layout()
    plt.legend(loc="upper right")

    plt.subplot(414)
    plt.xlabel("Zero Index")
    plt.ylabel("Bi Relative Error")
    plt.yscale("log")
    plt.plot(zero_range, asym_bi_relerror, linestyle='none', color=bi_color3, marker=asym_marker, markersize=asym_marker_size, label="asymptotic")
    plt.plot(zero_range, my_bi_relerror1, linestyle='none', color=bi_color1, marker='.', markersize=6, label="my_bizero, $\epsilon$ = {:.0e}".format(eps1))
    plt.plot(zero_range, my_bi_relerror2, linestyle='none', color=bi_color2, marker='.', markersize=6, label="my_bizero, $\epsilon$ = {:.0e}".format(eps2))
    plt.tight_layout()

    plt.legend(loc="upper right")

    if save_figures: plt.savefig(figure_directory + "zero-errors1.png", dpi=250)

    plt.show()


def plot_zero_opt():
    """
    Plots the absolute (and relative) error of ai and bi's zeros as a function of
    number of terms in the asymptotic expansion
    :param x:
    :return:
    """

    max_terms = 5   # only five terms are given in the expansion
    term_range = range(1, max_terms+1)
    fig, ax = plt.subplots(figsize=(8, 7))

    for n in [1, 2, 3]:
        xi_a = 3 * math.pi * (4 * n - 1) / 8
        xi_b = 3 * math.pi * (4 * n - 3) / 8

        mp_airyaizero = mpmath.airyaizero(n)
        mp_airybizero = mpmath.airybizero(n)

        ai_abserror = []
        ai_relerror = []
        bi_abserror = []
        bi_relerror = []

        for i in term_range:
            asym_airyaizero = - airyzero_asym_series(xi_a, i)
            asym_airybizero = - airyzero_asym_series(xi_b, i)

            ai_abserror.append(abs(mp_airyaizero - asym_airyaizero))
            bi_abserror.append(abs(mp_airybizero - asym_airybizero))
            ai_relerror.append(abs((mp_airyaizero - asym_airyaizero)/mp_airyaizero))
            bi_relerror.append(abs((mp_airybizero - asym_airybizero) / mp_airybizero))

        plt.subplot(221)
        plt.ylabel("Abs. Error ai")
        plt.yscale("log")
        plt.tick_params(  # turn off x axis ticks
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.plot(term_range, ai_abserror, linestyle='--', marker='.', markersize=6, label="n = {}".format(n))
        plt.legend()

        plt.subplot(222)
        plt.ylabel("Rel. Error ai")
        plt.yscale("log")
        # plt.tick_params( # turn off x axis ticks
        #     axis='x',  # changes apply to the x-axis
        #     which='both',  # both major and minor ticks are affected
        #     bottom=False,  # ticks along the bottom edge are off
        #     top=False,  # ticks along the top edge are off
        #     labelbottom=False)  # labels along the bottom edge are off
        plt.plot(term_range, ai_relerror, linestyle='--', marker='^', markersize=6, label="n = {}".format(n))
        plt.legend()

        plt.subplot(223)
        plt.ylabel("Abs. Error bi")
        plt.xlabel("Terms in expansion")
        plt.yscale("log")
        plt.plot(term_range, bi_abserror, linestyle='--', marker='.', markersize=6, label="n = {}".format(n))
        plt.legend()

        plt.subplot(224)
        plt.ylabel("Rel. Error bi")
        plt.xlabel("Terms in expansion")
        plt.yscale("log")
        plt.plot(term_range, bi_relerror, linestyle='--', marker='^', markersize=6, label="n = {}".format(n))
        plt.legend()

    plt.tight_layout()
    plt.show()
# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    X = np.linspace(x_min, x_max, num_points)
    my_ai = []
    my_bi = []
    mp_ai = []
    mp_bi = []

    for i in range(num_points):
        my_airy_tuple = my_airy_AB(X[i])
        my_ai.append(my_airy_tuple[0])
        my_bi.append(my_airy_tuple[1])
        mp_ai.append(mpmath.airyai(X[i]))
        mp_bi.append(mpmath.airybi(X[i]))

    # plot_airy()
    # plot_error_both(X, my_ai, my_bi, mp_ai, mp_bi)
    plot_compared_airys(X, my_ai, my_bi)

    # run_zeros()
    # plot_asym_opt()
    # test_A_and_B_precision()
    # print(mpmath.mp)


def run_zeros():
    max_zeros = 100
    # plot_airy_zeros(max_zeros)
    plot_airy_zero_error(max_zeros)
    # plot_zero_opt()
