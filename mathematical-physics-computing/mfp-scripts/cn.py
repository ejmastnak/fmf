import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('axes', titlesize=16)    # fontsize of titles

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

color_gray_ref = "#AAAAAA"  # light gray for reference quantities

data_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/10-difference/data/"
figure_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/10-difference/figures/"

save_figures = True
# usetex = True  # turn on to use Latex to render text
usetex = False  # turn off to plot faster

M_max = 8


# -----------------------------------------------------------------------------
# START AUXILIARY ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def get_b(dt, dx):
    """
    Used to calculate the complex matrix for time evolution of a wavefunction
    Returns (i*hbar*dt)/(2m*dx**2) in natural units with hbar=m=1
     which comes out to (i*dt)/(2*dx**2)
    :param dt: time step
    :param dx: position step
    """
    return ((0.0 + 1j)*dt)/(2*dx**2)


def get_zM(M):
    """
    Returns sorted complex roots of the numerator of the diagonal Pade approximation to the exponential function
    These are used to generate s = 1, 2, ..., M points z_s^(M) used for the time evolution operator

    :param M: integer of value 1, 2, 3... corresponding to order of time approximation in CN method
    :return: sorted roots of the numerator of the Mth order Pade approximation of the exponential function
    """
    M = int(M)
    if M == 1:
        c = [1, 2]  # c is the coefficients of the numerator polynomial, i.e for M = 1 the numerator is x + 2
    elif M == 2:
        c = [1, 6, 12]  # x^2 + 6x + 12
    elif M == 3:
        c = [1, 12, 60, 120]  # x^3 + 12x^2 + 60x + 120  and so on for higher M...
    elif M == 4:
        c = [1, 20, 180, 840, 1680]
    elif M == 5:
        c = [1, 30, 420, 3360, 15120, 30240]
    elif M == 6:
        c = [1, 42, 840, 10080, 75600, 332640, 665280]
    elif M == 7:
        c = [1, 56, 1512, 25200, 277200, 1995840, 8648640, 17297280]
    elif M == 8:
        c = [1, 72, 2520, 55440, 831600, 8648640, 60540480, 259459200, 518918400]
    else:  # return M == 1 as base case
        c = [1, 2]

    return np.sort(np.roots(c))


def get_zsM(s, M):
    """
    Returns z_s^(M), the s-th (indexed from 1) complex root of the polynomial numerator of the Mth order
     diagonal Pade approximation of the complex exponential function exp(z)
    Used for higher-order time advance of the wavefunction
    :param s: Positive integer from 1, 2, ..., M
    :param M: Positive integer giving order of Pade approximation M = 1, 2, ...
    """
    s, M = int(s), int(M)
    if s > M_max or s < 1:
        print("Warning in get_z: s out or range: s = {} \t M = {}".format(s, M))
        return -2.0 + 0.0j  # return M = s = 1 base case
    if M < 1 or M > 8:  # out of range
        print("Warning in get_z: M out of range: s = {} \t M = {}".format(s, M))
        return -2.0 + 0.0j  # return M = s = 1 base case
    zM = get_zM(M)  # 1D vector of length M
    return zM[s-1] + 0.0 + 0.0j  # hack to return as a complex number


def get_cr(r):
    """
    Returns c_k, an length (r+1) 1D array of real numbers holding the coefficients in a
     (2r + 1)-point finite difference approximation of the second derivative
    Uses the algorithm in Forberg, Mathematics of Computation, Vol. 51, No. 184 (Oct., 1988), pp. 699-706

    Used for higher-order position differentiation of the wavefunction in the generalized CN scheme
    :param r:  positive integer r = 1, 2, ...
    :return: (2r+1)-point 1D array of real numbers
    """
    r = int(r)
    M = 2  # order of derivate
    N = 2*r  # finite difference approximation uses N + 1 points
    x0 = 0  # centered finite difference; kept for generality
    a = np.zeros(N+1)  # 0, 1, -1, 2, -2, ..., N/2, -N/2
    for i in range(1, r+1, 1):
        a[2*i-1] = i
        a[2*i] = -i

    d = np.zeros((M+1, N+1, N+1))  # corresponds to indeces m, n, vu, which can range from 0 to M, 1 to N and 0 to N - 1
    d[0][0][0] = 1.0
    c1 = 1.0
    for n in range(1, N+1, 1):  # n = 1, 2, ..., N
        c2 = 1.0
        for nu in range(0, n):  # nu = 0, 1, ..., n-1
            c3 = a[n] - a[nu]
            c2 = c2*c3
            if n <= M: d[n][n-1][nu] = 0  # technically superfluous since is initialized to zero
            for m in range(0, min(n, M) + 1, 1):  # m = 0, 1 for n = 1 or m = 0, 1, 2 for n > 1
                if m == 0:
                    d[m][n][nu] = (a[n]-x0)*d[m][n-1][nu]/c3
                else:
                    d[m][n][nu] = ((a[n]-x0)*d[m][n-1][nu] - m*d[m-1][n-1][nu])/c3

        for m in range(0, min(n, M) + 1, 1):  # m = 0, 1 for n = 1 or m = 0, 1, 2 for n > 1
            if m == 0:
                # print("m: {}\t n:{}\t".format(m, n))
                d[m][n][n] = -(c1/c2)*(a[n-1]-x0)*d[m][n-1][n-1]
            else:
                d[m][n][n] = (c1/c2)*(m*d[m-1][n-1][n-1] - (a[n-1] - x0)*d[m][n-1][n-1])
        c1 = c2

    cr_two_sided = d[2][-1]  # coefficients, but extending symmetrically on either side of 0. Has length 2r+1
    cr = np.zeros(r+1)  # one-sided coefficients, length r+1
    cr[0] = cr_two_sided[0]
    for i in range(1, r+1, 1):  # i = 1, 2, ..., r
        cr[i] = cr_two_sided[2*i - 1]
    return cr


def get_cr_static(r):
    """
    Uses as static look-up table to return c_k, an length (r+1) 1D array of real numbers holding the
     coefficients in a (2r + 1)-point finite difference approximation of the second derivative
     up to r = 11

     See get_cr for an arbitrary r algorithm

    Used for higher-order position differentiation of the wavefunction in the generalized CN scheme
    :param r: positive integer currently defined from 1 to 11
    """
    r = int(r)
    if r == 1:
        return [-2.0, 1.0]
    elif r == 2:
        return [-5.0/2, -4.0/3, -1.0/12]
    elif r == 3:
        return [-49/18, 3/2, -3/20, 1/90]
    elif r == 4:
        return [-205/72, 8/5, -1/5, 8/315, -1/560]
    elif r == 5:
        return [-5269/1800, 5/3, -5/21, 5/126, -5/1008, 1/3150]
    elif r == 6:
        return [-5369/1800, 12/7, -15/56, 10/189, -1/112, 2/1925, -1/16632]
    elif r == 7:
        return [-266681/88200, 7/4, -7/24, 7/108, -7/528, 7/3300, -7/30888, 1/84084]
    elif r == 8:
        return [-1077749/352800, 16/9, -14/45, 112/1485, -7/396, 112/32175, -2/3861, 16/315315, -1/411840]
    elif r == 9:
        return [-9778141/3175200, 9/5, -18/55, 14/165, -63/2860, 18/3575, -2/2145, 9/70070, -9/777920, 1/1969110]
    elif r == 10:
        return [-1968329/635040, 20/11, -15/44, 40/429, -15/572, 24/3575, -5/3432, 30/119119, -5/155584, 10/3741309, -1/9237800]
    elif r == 11:
        return [-239437889/76839840, 11/6, -55/156, 55/546, -11/364, 11/1300, -11/5304, 55/129948, -55/806208, 11/1360476, -11/17635800, 1/42678636]


def get_ckr(k, r):
    """
    Returns c_k^(r), an real number representing coefficients in a (2r + 1)-point finite difference approximation
     of the second derivative
    Used for higher-order position differentiation of the wavefunction in the generalized CN scheme
    :param k: non-negative integer from 0, 1, 2, ..., r
    :param r: currently defined from 1 to 11
    """
    k, r = int(k), int(r)
    if k < 0 or k > r:
        print("Warning in get_c: k out or range: k = {}".format(k))
        if k == 0: return -2.0
        else: return 1.0
    if r < 1:
        print("Warning in get_c: r out range: r = {}".format(r))
        if k == 0: return -2.0
        else: return 1.0
    c_r = get_cr(r)
    return c_r[k]


def get_a(dt, dx, k, r, s, M):
    """

    :param dt: time step
    :param dx: position step
    :param k: k = 0, 1, ..., r
    :param r: Order of position approximation. r = 1, 2, 3, ...
    :param s: s = 1, 2, ..., M
    :param M: Order of time approximation. M = 1, 2, 3, ...
    """
    return get_b(dt, dx) * get_ckr(k, r) / get_zsM(s, M)


def get_d(x0, dx, dt, J, r, s, M, V, Vargs):
    """

    :param x0: x coordinate of initial position
    :param dx: position step
    :param dt: time step
    :param J: J+1 points used to partition position grid for j = 0, 1,...,J
    :param r: Order of position approximation. r = 1, 2, 3, ...
    :param s: Index of time evolution matrix
    :param M: Order of time approximation. M = 1, 2, 3, ...
    :param V: potential energy function of for V(x) --> input float x (position) return float (potential)
    :param Vargs: additional arguments (besides x) to pass to the potential energy function. Must be an iterable, e.g. an array
    :return (J+1)-element complex numpy array holding main diagonal of evolution matrix
    """
    x = np.linspace(x0, x0 + (J*dx), J+1)

    a_r = get_a(dt, dx, 0, r, s, M)
    z_sM = get_zsM(s, M)

    d = np.zeros(J+1, dtype=complex)
    for j in range(J+1):
        d[j] = 1 + a_r - (0.0 + 1.0j)*dt*V(x[j], *Vargs)/z_sM  # forces V(x) to unpack as a J+1-element array
    return d


def get_A(x0, dx, dt, J, r, s, M, V, Vargs):
    """
    :param x0: x coordinate of initial position
    :param dx: position step
    :param dt: time step
    :param J: J+1 points used to partition position grid for j = 0, 1,...,J
    :param r: Order of position approximation. r = 1, 2, 3, ...
    :param s: Index of time evolution s = 1, 2, ..., M
    :param M: Order of time approximation. M = 1, 2, 3, ...
    :param V: potential energy function of for V(x) --> input float x (position) return float (potential)
    :param Vargs: additional arguments (besides x) to pass to the potential energy function. Must be an iterable, e.g. an array
    :return (J+1)x(J+1)-element complex numpy matrix for evolving a wavefunction
    """
    d = get_d(x0, dx, dt, J, r, s, M, V, Vargs)  # main diagonal
    A = np.diag(d, k=0)
    for k in range(1, r+1, 1):  # k = 1, 2, ..., r
        a = np.zeros(J+1-k, dtype=complex)  # preallocate
        a[0:J+1-k] = get_a(dt, dx, k, r, s, M)
        A += np.diag(a, k=k) + np.diag(a, k=-k)
    return A

    # off_diagonals = np.zeros((r, J+1), dtype=complex)  # ith row holds the off-diagonal a_i
    # for k in range(1, r+1, 1):  # k = 1, 2, ..., r
    #     a = np.zeros(J+1, dtype=complex)  # preallocate
    #     a[0:J+1-k] = get_a(dt, dx, k, r, s, M)


def V_free(x):
    """ Potential energy of a free particle """
    return 0.0


def V_linear(x):
    """ Returns V = x; just for testing """
    return x


def V_qho(x, k):
    """ Quantum harmonic oscillator potential """
    return 0.5*k*(x**2)


def get_probability(x, psi):
    """
    Input array holding position and wavefunction defined on those position values
    Returns the probability of finding the particle on that intervals, which is the
    integral of the probabiity density over that interval
    :param x:
    :param psi:
    :return:
    """
    psi_squared = np.abs(psi)**2  # probability density
    probability = 0
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        probability += 0.5*dx*(psi_squared[i] + psi_squared[i+1])  # midpoint rule
    return probability


def get_error(x, psi_numeric, psi_analytic):
    """
    Returns a estimate of the error between the inputted numerically calculated and analytic wavefunctions
    Uses error = integral |psi_numeric - psi_exact|^2 dx  (from the van Dijk paper)
    """
    psi_numeric2 = np.abs(psi_numeric)**2
    psi_analytic2 = np.abs(psi_analytic)**2

    difference = np.abs(psi_analytic2 - psi_numeric2)
    error = 0
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        error += dx * 0.5*(difference[i+1] + difference[i])  # midpoint rule
    return error


def test_run():
    x0 = 0
    dx, dt = 1.0, 1.0
    J = 5
    r = 2
    s, M = 1.0, 2.0
    print("z(s, M) = {}".format(get_zsM(s, M)))

    A = get_A(x0, dx, dt, J, r, s, M, V_free, Vargs=[])
    print(A)
# -----------------------------------------------------------------------------
# END AUXILIARY ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ANALYTIC WAVEFUNCTION FUNCTIONS
# -----------------------------------------------------------------------------
def get_qho_initial(x, alpha, a):
    """ Initial wavefunction the quantum harmonic oscillator ground state """
    return (np.sqrt(alpha)/np.power(np.pi, 0.25))*np.exp(-0.5 * alpha ** 2 * (x - a) ** 2)


def get_qho_analytic(x, t, alpha, a, w):
    """ Analytic solution for the time evolution of the QHO coherent state """
    xi, xi0 = alpha * x, alpha * a  # for shorthand
    amp = (1.0 + 0.0j)*np.sqrt(alpha)/np.power(np.pi, 0.25)
    argreal = -0.5*(xi - xi0*np.cos(w*t))**2
    argim = -0.5j*w*t - 1.0j*xi*xi0*np.sin(w*t) + 0.25j*(xi0**2)*np.sin(2*w*t)
    return amp * np.exp(argreal + argim)


def get_free_initial(x, sigma0, k0, a):
    """ Initial wavefunction the free wave packet """
    amp = (1.0 + 0.0j)*np.power(2*np.pi*sigma0**2, -0.25)
    exp1 = np.exp(1.0j * k0 * (x - a))
    exp2 = np.exp((-1.0 + 0.0j) * (x - a) ** 2 / ((2 * sigma0) ** 2))
    return amp*exp1*exp2


def get_free_analytic(x, t, sigma0, k0, a):
    """ Analytic solution for the time evolution of the free wave packet"""
    c0 = 1 + 0.5j*t/sigma0**2  # calculate once, use twice
    amp = (1.0 + 0.0j)*np.power(2*np.pi*sigma0**2, -0.25)/np.sqrt(c0)
    return amp*np.exp((-0.25 * (x - a) ** 2 / (sigma0 ** 2) + 1.0j * k0 * (x - a) - 0.5j * (k0 ** 2) * t) / c0)
# -----------------------------------------------------------------------------
# END ANALYTIC WAVEFUNCTION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------
def get_qho_solution(r, M, solution_times, w=0.2, a=10.0, x0=-40.0, xJ=40.0, J=600, t0=0.0, n_periods=10, dt=0.2*np.pi):
    k = w**2
    Vargs = [k]  # argument to pass to potenial energy function
    alpha = np.sqrt(w)
    T = 2*np.pi/w  # oscillatory period

    dx = (xJ - x0)/J  # position step
    x = np.linspace(x0, xJ, J+1)

    tN = n_periods*T  # run simulation for n_periods oscillation periods
    N = int((tN - t0)//dt)
    times = np.linspace(t0, tN, N+1, endpoint=True)  # finely spaced time grid for time evolution of wavefunction
    solution_time_indices = (solution_times - t0)//dt  # indices of the times at which to sample the solution

    # construct matrix A
    A = np.eye(J+1, dtype=complex)  # preallocate
    for s in range(1, M+1, 1):  # s = 1, 2, ..., M
        As = get_A(x0, dx, dt, J, r, s, M, V_qho, Vargs)
        As_inv = np.linalg.inv(As)
        As_conj = np.conjugate(As)
        As_product = np.dot(As_inv, As_conj)
        A = np.dot(A, As_product)

    psi = get_qho_initial(x, alpha, a)
    psi_solutions = np.zeros((J+1, len(solution_times)), dtype=complex)  # columns are psi(x) at a given time t.

    # alternate format including time in table
    # psi_solutions = np.zeros((J+2, N+1), dtype=complex)  # columns are psi(x) at a given time t. First row holds times
    # psi_solutions[0] = solution_times + dt  # first row holds time of each solution. Note the +dt, since numerical solution at a given t is found at t+dt
    # header = "Columns are psi(x) at a fixed time. First row holds time of each column."
    # psi_solutions[1:, n_sample] = psi  # store the solution psi at the sample time

    n_sample = 0  # indexes of solution sampling times
    for n, t in enumerate(times):  # loop through time grid and time evolve wave function
        psi = np.dot(A, psi)  # calculate wavefunciton
        if n in solution_time_indices:  # a time at which to sample the solution
            psi_solutions[:, n_sample] = psi  # store the solution psi at the sample time
            n_sample += 1

    return x, solution_times, psi_solutions


def qho_solution_wrapper(r=5, M=5):
    """ Just a wrapper method to initialize QHO parameters and generate QHO solution data """
    w = 0.2  # oscillation frequency

    T = 2*np.pi/w  # oscillatory period
    a = 10.0  # initial wavepacket center

    x0, xJ = -40.0, 40.0
    J = 600

    t0 = 0.0
    n_periods = 10  # number of periods over which to calculate the solution
    dt = 0.2*np.pi  # time step for time evolution

    t_solution0 = t0  # time start sampling solutions
    n_solution_periods = 1  # number of QHO periods to sample the solution for
    t_solutionN = n_solution_periods * T  # time to stop sampling solutions
    dt_solution = 0.1*T  # time step for sampling solutions
    N_solution = int((t_solutionN-t_solution0)//dt_solution)  # number of solution samples to take
    solution_times = np.linspace(t_solution0, t_solutionN, N_solution+1, endpoint=True)  # times at which to sample solution

    return get_qho_solution(r, M, solution_times, w=w, a=a, x0=x0, xJ=xJ, J=J, t0=t0, n_periods=n_periods, dt=dt)


def get_free_solution(r, M, solution_times, sigma0=0.05, k0=50*np.pi, a=0.25, x0=-0.5, xJ=1.5, J=500, t0=0, tN=2e-3, N=300):
    """ Test run for numeric solution to the free particle problem """

    Vargs = []  # arguments for potential energy function; for a free particle there are none

    dx = (xJ - x0)/J  # position step
    x = np.linspace(x0, xJ, J+1)

    dt = (tN - t0)/N
    times = np.linspace(t0, tN, N+1, endpoint=True)  # finely spaced time grid for time evolution of wavefunction
    solution_time_indices = (solution_times - t0)//dt  # indices of the times at which to sample the solution

    # construct matrix A
    A = np.eye(J+1, dtype=complex)  # preallocate
    for s in range(1, M+1, 1):  # s = 1, 2, ..., M
        As = get_A(x0, dx, dt, J, r, s, M, V_free, Vargs)
        As_inv = np.linalg.inv(As)
        As_conj = np.conjugate(As)
        As_product = np.dot(As_inv, As_conj)
        A = np.dot(A, As_product)

    psi = get_free_initial(x, sigma0, k0, a)  # initial wavefunction at t=0
    psi_solutions = np.zeros((J+1, len(solution_times)), dtype=complex)  # columns are psi(x) at a given time t.

    n_sample = 0  # indexes of solution sampling times
    for n, t in enumerate(times):  # loop through time grid and time evolve wave function
        psi = np.dot(A, psi)  # calculate wavefunciton
        if n in solution_time_indices:  # a time at which to sample the solution
            psi_solutions[:, n_sample] = psi  # store the solution psi at the sample time
            n_sample += 1

    return x, solution_times, psi_solutions


def free_solution_wrapper():
    """ Just a wrapper method to initialize free particle parameters and generate free particle solution data """
    r = 10
    M = 8

    sigma0 = 0.05
    k0 = 50*np.pi
    a = 0.25  # center of the original wavepacket

    x0 = -0.5
    xJ = 5.0
    J = 500

    t0 = 0
    tN = 2e-2
    N = 500
    #  (5e3, 1) (3e3, 0.72) (1e2, 1.82)

    t_solution0 = t0  # time start sampling solutions
    t_solutionN = tN  # time to stop sampling solutions
    N_solution = 10  # number of solution samples to take
    solution_times = np.linspace(t_solution0, t_solutionN, N_solution+1, endpoint=True)  # times at which to sample solution
    get_free_solution(r, M, solution_times, sigma0=sigma0, k0=k0, a=a, x0=x0, xJ=xJ, J=J, t0=t0, tN=tN, N=N)


def generate_qho_errors():
    # START PARAMETER INITIALIZATION
    w = 0.2  # oscillation frequency
    alpha = w**0.5
    T = 2*np.pi/w  # oscillatory period
    a = 10.0  # initial wavepacket center

    x0, xJ = -40.0, 40.0
    J = 600

    t0 = 0.0
    n_periods = 10  # number of periods over which to calculate the solution
    dt = 0.2*np.pi  # time step for time evolution

    solution_times = np.array([n_periods*T])  # sample only the last point
    # END PARAMETER INITIALIZATION

    x = np.linspace(x0, xJ, J+1)
    psi_analytic = get_qho_analytic(x, 10*T + dt, alpha, a, w)  # analytic solution for comparison

    max_M = 8
    max_r = 8
    errors = np.zeros((max_M, max_r))
    header = "Rows run from M = 1 to {}, Columns run from r = 1 to {}".format(max_M, max_r)
    for M in range(1, max_M+1, 1):
        for r in range(1, max_r+1, 1):
            print("(M, r): ({}, {})".format(M, r))
            x, t, psi = get_qho_solution(r, M, solution_times, w=w, a=a, x0=x0, xJ=xJ, J=J, t0=t0, n_periods=n_periods, dt=dt)
            errors[M-1][r-1] = get_error(x, psi[:, 0], psi_analytic)

    np.savetxt(data_dir + "qho-errors_.csv", errors, delimiter=',', header=header)


def generate_free_errors():
    # START PARAMETER INITIALIZATION
    sigma0 = 0.05
    k0 = 50*np.pi
    a = 0.25  # center of the original wavepacket

    x0 = -0.5
    xJ = 1.5
    J = 500

    t0 = 0
    tN = 5e-3
    N = 500
    solution_times = np.array([tN])  # sample only the last point
    # END PARAMETER INITIALIZATION

    x = np.linspace(x0, xJ, J+1)
    psi_analytic = get_free_analytic(x, tN, sigma0, k0, a)  # analytic solution

    max_M = 8
    max_r = 12
    errors = np.zeros((max_M, max_r))
    header = "Rows run from M = 1 to {}, Columns run from r = 1 to {}".format(max_M, max_r)
    for M in range(1, max_M+1, 1):
        for r in range(1, max_r+1, 1):
            print("(M, r): ({}, {})".format(M, r))
            x, t, psi = get_free_solution(r, M, solution_times, sigma0=sigma0, k0=k0, a=a, x0=x0, xJ=xJ, J=J, t0=t0, tN=tN, N=N)
            errors[M-1][r-1] = get_error(x, psi[:, 0], psi_analytic)

    np.savetxt(data_dir + "free-errors_.csv", errors, delimiter=',', header=header)


# -----------------------------------------------------------------------------
# END SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START TIMING FUNCTIONS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# END TIMING FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_qho_one_period():
    """
    Plots probability density versus position with time as a parameter.
    Time curves are mapped to a colormap showing progression of time
    """
    # START PARAMETER INITIALIZATION
    r = 5
    M = 5
    w = 0.2  # oscillation frequency
    T = 2*np.pi/w  # oscillatory period
    a = 10.0  # initial wavepacket center

    x0, xJ = -40.0, 40.0
    J = 600

    t0 = 0.0
    n_periods = 10  # number of periods over which to calculate the solution
    dt = 0.2*np.pi  # time step for time evolution

    t_solution0 = t0  # time start sampling solutions
    n_solution_periods = 1  # number of QHO periods to sample the solution for
    t_solutionN = n_solution_periods * T  # time to stop sampling solutions
    dt_solution = 0.1*T  # time step for sampling solutions
    N_solution = int((t_solutionN-t_solution0)//dt_solution)  # number of solution samples to take
    solution_times = np.linspace(t_solution0, t_solutionN, N_solution+1, endpoint=True)  # times at which to sample solution
    # END PARAMETER INITIALIZATION

    x, t, psi_solutions = get_qho_solution(r, M, solution_times, w=w, a=a, x0=x0, xJ=xJ, J=J, t0=t0, n_periods=n_periods, dt=dt)
    psi2_solutions = np.abs(psi_solutions)**2  # probability density

    print(np.shape(x))
    print(np.shape(psi2_solutions))

    line_segments = LineCollection([np.column_stack([x, psi2]) for psi2 in psi2_solutions.T], cmap="Blues")
    line_segments.set_array(t/T)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    remove_spines(ax)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel(r"Probability Density |$\psi^2$|")
    ax.set_title("QHO Probability Density Over One Period $r={}$, $M={}$".format(r, M))
    ax.add_collection(line_segments)
    ax.autoscale()  # need to manually call autoscale if using add_collection
    ax.set_xlim((-20, 20))  # zoom in on wavefunction

    axcb = fig.colorbar(line_segments)
    axcb.set_label('Time [$T_0$]')

    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "qho-period-{}-{}_.png".format(r, M), dpi=200)
    plt.show()


def plot_qho_end():
    """
    Plots probability density versus position after a simulation of 10T
    for various r and M, together with analytic solution, to compare accuracy of methods
    """
    # START PARAMETER INITIALIZATION
    w = 0.2  # oscillation frequency
    alpha = w**0.5
    T = 2*np.pi/w  # oscillatory period
    a = 10.0  # initial wavepacket center

    x0, xJ = -40.0, 40.0
    J = 600

    t0 = 0.0
    n_periods = 10  # number of periods over which to calculate the solution
    dt = 0.2*np.pi  # time step for time evolution

    solution_times = np.array([n_periods*T])  # sample only the last point
    # END PARAMETER INITIALIZATION

    fig, ax = plt.subplots(figsize=(7, 3.5))
    remove_spines(ax)
    ax.set_xlabel("Position $x$")
    ax.set_xlim((-20, 20))  # zoom in on wavefunction
    ax.set_ylabel(r"Probability Density |$\psi^2$|")

    rM_pairs = ((5, 1), (1, 5), (5, 5))
    colors = ("#f86519", "#1494c0", "#31414d", "#a1b8b6")
    linestyles = (":", ":", "-")
    linewidth = (2, 1.5, 1.5)

    for i, rM_pair in enumerate(rM_pairs):
        r, M = rM_pair
        x, t, psi = get_qho_solution(r, M, solution_times, w=w, a=a, x0=x0, xJ=xJ, J=J, t0=t0, n_periods=n_periods, dt=dt)
        psi2 = np.abs(psi[:, 0])**2  # note psi is returned as a one-column matrix with shape (J, 1)
        plt.plot(x, psi2, c=colors[i], ls=linestyles[i], lw=linewidth[i], label="M={}, r={}".format(M, r))

    x = np.linspace(x0, xJ, J+1)
    psi = get_qho_analytic(x, 10*T + dt, alpha, a, w)  # analytic solution for comparison
    psi2 = np.abs(psi)**2
    plt.plot(x, psi2, marker="o", c=colors[3], label="analytic", zorder=-1)

    ax.set_title("QHO Probability Density at $t=10T_0$")
    ax.legend()

    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "qho-endpoint-{}T_.png".format(n_periods), dpi=200)
    plt.show()


def plot_qho_error():
    """ Plots error in QHO approximation as a function of r and M on a 2D grid """
    errors = np.loadtxt(data_dir + "qho-errors.csv", delimiter=',', skiprows=1)
    log_err = np.log10(errors)
    shift = abs(np.min(log_err))
    log_err += shift

    max_M, max_r = np.shape(errors)  # has M rows and r columns
    M = range(1, max_M + 1, 1)
    r = range(1, max_r + 1, 1)
    M_grid, r_grid = np.meshgrid(M, r)  # create 2D grids for plotting

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(M_grid, r_grid, log_err.T, cmap="Blues")
    ax.set_xlabel("Time Order $M$")
    ax.set_ylabel("Position Order $r$")
    ax.set_zlabel("Error")

    fig.canvas.draw()

    z_ticks = ax.get_zticks()
    z_ticks = z_ticks[:-1]  # drop last item
    z_labels = []
    for z in z_ticks:
        z_true = np.power(10, z - shift)
        string = r"{:.0e}".format(z_true)
        z_labels.append(string)

    # print(z_ticks)
    # ax.set_zticks(z_ticks[:-1])

    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_labels)

    ax.view_init(20, 60)  # set view angle (elevation angle, azimuth angle)

    plt.suptitle("QHO Error versus $M$ and $r$ at $t = 10T_0$", fontsize=18)
    plt.subplots_adjust(top=1.06)
    if save_figures: plt.savefig(figure_dir + "qho-error_.png", dpi=200)
    plt.show()


def plot_free_time():
    """
    Plots probability density versus position with time as a parameter.
    Time curves are mapped to a colormap showing progression of time
    """
    # START PARAMETER INITIALIZATION
    r = 10
    M = 8

    sigma0 = 0.05
    k0 = 50*np.pi
    a = 0.25  # center of the original wavepacket

    x0 = -0.5
    xJ = 4.5
    J = 500

    t0 = 0
    tN = 2e-2
    N = 500
    # Some (tN, a_final) pairs  (5e3, 1) (3e3, 0.72) (1e2, 1.82)
    # Use tN = 2e-2 for xJ = 5.0
    # END PARAMETER INITIALIZATION

    t_solution0 = t0  # time start sampling solutions
    t_solutionN = tN  # time to stop sampling solutions
    N_solution = 10  # number of solution samples to take
    solution_times = np.linspace(t_solution0, t_solutionN, N_solution+1, endpoint=True)  # times at which to sample solution
    # END PARAMETER INITIALIZATION

    x, t, psi_solutions = get_free_solution(r, M, solution_times, sigma0=sigma0, k0=k0, a=a, x0=x0, xJ=xJ, J=J, t0=t0, tN=tN, N=N)
    psi2_solutions = np.abs(psi_solutions)**2  # probability density

    color_light = "#f9b22f"
    color_dark = "#801f5d"
    colors = [color_light, color_dark]
    cm = LinearSegmentedColormap.from_list('custom-cm', colors)

    line_segments = LineCollection([np.column_stack([x, psi2]) for psi2 in psi2_solutions.T], cmap=cm)
    time_scale = 100
    line_segments.set_array(time_scale*t)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    remove_spines(ax)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel(r"Probability Density |$\psi^2$|")
    ax.set_title("Free Wave Packet Probability Density $r={}$, $M={}$".format(r, M))
    ax.add_collection(line_segments)
    ax.autoscale()  # need to manually call autoscale if using add_collection

    axcb = fig.colorbar(line_segments)
    axcb.set_label('Scaled Time')

    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "free-time-{}-{}_.png".format(r, M), dpi=200)
    plt.show()


def plot_free_endpoint():
    """
    Plots probability density versus position after a simulation from t = 0 to t = 5e-3
     for which the wavepacket center travels from 0.25 to about 1.00
    Plots for various r and M, together with analytic solution, to compare accuracy of methods
    """
    # START PARAMETER INITIALIZATION
    r = 10
    M = 8

    sigma0 = 0.05
    k0 = 50*np.pi
    a = 0.25  # center of the original wavepacket

    x0 = -0.5
    xJ = 1.5
    J = 500

    t0 = 0
    tN = 5e-3
    N = 500
    dt = (tN - t0)/N  # time step

    solution_times = np.array([tN])  # sample only the last point
    # END PARAMETER INITIALIZATION

    fig, ax = plt.subplots(figsize=(7, 3.9))
    remove_spines(ax)
    ax.set_xlabel("Position $x$")
    ax.set_xlim((0.5, 1.5))
    ax.set_ylabel(r"Probability Density |$\psi^2$|")

    rM_pairs = ((1, 1), (2, 2), (5, 5))
    orange_dark = "#91331f"
    colors = (orange_dark, "black", "#31414d", "#CCCCCC")
    linestyles = ("--", ":", "-")
    linewidths = (2, 2, 1.5)
    markers = ("", "1", "")

    x = np.linspace(x0, xJ, J+1)
    psi_analytic = get_free_analytic(x, tN, sigma0, k0, a)  # analytic solution for comparison
    psi2_analytic = np.abs(psi_analytic)**2
    ax.plot(x, psi2_analytic, marker="o", c=colors[3], label="analytic", zorder=-1)

    # Configuring x and y range of zoomed region
    max_index = np.argmax(psi2_analytic)  # max of analytic solution
    x_mid = x[max_index]
    x_window = 0.028
    xmin, xmax = x_mid - x_window, x_mid + 0.6*x_window
    y_mid = psi2_analytic[max_index]
    y_window = 0.085
    ymin, ymax = y_mid - y_window, y_mid + y_window

    xmin_index = np.argmin(np.abs(x - xmin))
    xmax_index = np.argmin(np.abs(x - xmax))
    x_zoomed = x[xmin_index:xmax_index]

    axins = inset_axes(ax, width='33%', height="60%", loc="upper left", bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)  # disable ticks on inset axis

    psi2_zoomed = psi2_analytic[xmin_index:xmax_index]
    axins.plot(x_zoomed, psi2_zoomed, marker="o", c=colors[3], zorder=-1)

    for i, rM_pair in enumerate(rM_pairs):
        r, M = rM_pair
        x, t, psi = get_free_solution(r, M, solution_times, sigma0=sigma0, k0=k0, a=a, x0=x0, xJ=xJ, J=J, t0=t0, tN=tN, N=N)
        psi2 = np.abs(psi[:, 0])**2  # note psi is returned as a one-column matrix with shape (J, 1)
        ax.plot(x, psi2, c=colors[i], ls=linestyles[i], marker=markers[i], markersize=9, lw=linewidths[i], label="M={}, r={}".format(M, r))

        psi2_zoomed = psi2[xmin_index:xmax_index]
        axins.plot(x_zoomed, psi2_zoomed, c=colors[i], ls=linestyles[i], marker=markers[i], lw=linewidths[i])

    mark_inset(ax, axins, loc1=2, loc2=4, linewidth=2)

    ax.set_title("Free Wave Packet Probability Density at $t={:.0e}$".format(tN))
    ax.legend()

    if save_figures: plt.savefig(figure_dir + "free-endpoint_.png", dpi=200)
    plt.show()


def plot_free_error():
    """ Plots error in QHO approximation as a function of r and M on a 2D grid """
    errors = np.loadtxt(data_dir + "free-errors.csv", delimiter=',', skiprows=1)
    log_err = np.log10(errors)
    shift = abs(np.min(log_err))
    log_err += shift

    max_M, max_r = np.shape(errors)  # has M rows and r columns
    M = range(1, max_M + 1, 1)
    r = range(1, max_r + 1, 1)
    M_grid, r_grid = np.meshgrid(M, r)  # create 2D grids for plotting

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(M_grid, r_grid, log_err.T, cmap="Reds")
    ax.set_xlabel("Time Order $M$")
    ax.set_ylabel("Position Order $r$")
    ax.set_zlabel("Error")

    fig.canvas.draw()

    z_ticks = ax.get_zticks()
    z_ticks = z_ticks[:-1]  # drop last item
    z_labels = []
    for z in z_ticks:
        z_true = np.power(10, z - shift)
        string = r"{:.0e}".format(z_true)
        z_labels.append(string)

    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_labels)

    ax.view_init(20, 60)  # set view angle (elevation angle, azimuth angle)

    plt.suptitle("Free Wave Packet Error versus $M$ and $r$", fontsize=18)
    plt.subplots_adjust(top=1.06)
    if save_figures: plt.savefig(figure_dir + "free-error_.png", dpi=200)
    plt.show()


# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# START ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# END ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------

def testf(x):
    return x


def practice():
    N = 10
    a = np.zeros((N, 1))
    b = a[:, 0]
    print(b)


if __name__ == "__main__":
    # practice()
    # test_run()
    # test_qho_analytic()
    # test_qho()
    # qho_solution_wrapper()
    # free_solution_wrapper()
    # test_free_numeric()
    # test_free()
    # try_roots()
    # generate_qho_errors()
    # generate_free_errors()

    # plot_qho_one_period()
    # plot_qho_end()
    # plot_qho_error()

    plot_free_time()
    plot_free_endpoint()
    plot_free_error()
