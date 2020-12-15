import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib import cm
import time

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('axes', titlesize=16)    # fontsize of the x and y labels


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

r_max = 11
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
        c = [1, 56, 1512, 25200, 277200, 1995840, 864640, 17297280]
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


def get_cr2(r):
    """
    Returns c_k, an length (r+1) 1D array of real numbers holding the coefficients in a
     (2r + 1)-point finite difference approximation of the second derivative
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
    if k < 0:
        print("Warning in get_c: k out or range: k = {}".format(k))
        if k == 0: return -2.0
        else: return 1.0
    if r < 1 or r > 11:
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
    # a_r = get_a(dt, dx, 0, r, s, M)
    # z_sM = get_zsM(s, M)

    a_r = get_a(dt, dx, 0, r, s, M)
    z_sM = get_zsM(s, M)

    d = np.zeros(J+1, dtype=complex)
    for j in range(J+1):
        d[j] = 1 + a_r - (0.0 + 1.0j)*dt*V(x[j], *Vargs)/z_sM  # forces V(x) to unpack as a J+1-element array

    # d[0: J+1] = 1 + a_r - (0.0 + 1.0j)*dt*V(x)/z_sM  # forces V(x) to unpack as a J+1-element array
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


def get_error(x, psi_numeric, psi_exact):
    """
    Returns a estimate of the error between the inputted numerically calculated and analytic wavefunctions
    Uses error^2 = integral |psi_numeric - psi_exact|^2 dx  (from the van Dijk paper)
    """
    psi = psi_numeric - psi_exact  # calculate difference
    psi_squared = np.abs(psi)**2  # probability density of difference
    error = 0
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        error += 0.5*dx*(psi_squared[i] + psi_squared[i+1])  # midpoint rule
    return error**0.5


def test_run():
    x0 = 0
    dx, dt = 1.0, 1.0
    J = 5
    r = 2
    s, M = 1.0, 2.0
    print("z(s, M) = {}".format(get_zsM(s, M)))

    A = get_A(x0, dx, dt, J, r, s, M, V_free)
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


def get_qho(x, t, alpha, a, w):
    """ Analytic solution for the time evolution of the QHO coherent state """
    xi, xi0 = alpha * x, alpha * a  # for shorthand
    amp = (1.0 + 0.0j)*np.sqrt(alpha)/np.power(np.pi, 0.25)
    argreal = -0.5*(xi - xi0*np.cos(w*t))**2
    # argim = -0.5j - 1.0j*xi*xi0*np.sin(w*t) - 0.25j*(xi0**2)*np.sin(2*w*t)  # van Dijk version, 0.5 vs 0.5*w*t
    argim = -0.5j*w*t - 1.0j*xi*xi0*np.sin(w*t) + 0.25j*(xi0**2)*np.sin(2*w*t)
    return amp * np.exp(argreal + argim)


def get_free_initial(x, sigma0, k0, a):
    """ Initial wavefunction the free wave packet """
    amp = (1.0 + 0.0j)*np.power(2*np.pi*sigma0**2, -0.25)
    exp1 = np.exp(1.0j * k0 * (x - a))
    exp2 = np.exp((-1.0 + 0.0j) * (x - a) ** 2 / ((2 * sigma0) ** 2))
    return amp*exp1*exp2


def get_free(x, t, sigma0, k0, a):
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
def test_qho_analytic():
    """
    Just a test run for the analytic solution to the QHO problem
    :return:
    """
    w = 0.2
    k = w**2
    alpha = np.sqrt(w)
    a = 10
    T = 2*np.pi/w  # oscillatory period

    x0, xJ = -40, 40
    J = 300
    dx = (xJ - x0)/J  # position step
    x = np.linspace(x0, xJ, J)

    t0, tN = 0, 10*T
    dt = np.pi  # time step
    times = np.arange(t0, tN, dt)

    psi0 = get_qho_initial(x, alpha, a)
    psi02 = np.abs(psi0)**2  # probability density

    # for i in np.arange(0, 1, 0.1):
    for i in np.linspace(0, 0.5, 5, endpoint=False):
        t = i*T
        psi = get_qho(x, t, alpha, a, w)
        psi2 = np.abs(psi)**2

        plt.plot(x, psi2, label="{:.2f} T".format(i))
    plt.legend()
    plt.show()


def test_qho():
    """ Test run for numeric solution to the QHO problem """
    w = 0.2
    k = w**2
    Vargs = [k]  # argument to pass to potenial energy function
    alpha = np.sqrt(w)
    a = 10
    T = 2*np.pi/w  # oscillatory period

    x0, xJ = -40, 40
    J = 600
    dx = (xJ - x0)/J  # position step
    x = np.linspace(x0, xJ, J+1)

    t0, tN = 0, 7*T
    N = 1000
    dt = (tN - t0)/N
    # dt = np.pi  # time step
    times = np.arange(t0, tN, dt)

    M = M_max  # order of time approximation
    r = r_max  # order of position approximation
    # construct matrix A
    A = np.eye(J+1, dtype=complex)  # preallocate
    for s in range(1, M+1, 1):  # s = 1, 2, ..., M
        As = get_A(x0, dx, dt, J, r, s, M, V_qho, Vargs)
        As_inv = np.linalg.inv(As)
        As_conj = np.conjugate(As)
        As_product = np.dot(As_inv, As_conj)
        A = np.dot(A, As_product)

    n_plots = 3
    psi = get_qho_initial(x, alpha, a)
    for n, t in enumerate(times):
        psi = np.dot(A, psi)
        if n % int(N/n_plots) == 0:
            psi2 = np.abs(psi)**2
            plt.plot(x, psi2, c='C0', label="t = {:.2f}".format(t))
    plt.legend()

    # for i in np.arange(0, 1, 0.1):
    for n, t in enumerate(times):
        if n % int(N/n_plots) == 0:
            psi = get_qho(x, t, alpha, a, w)
            psi2 = np.abs(psi)**2
            plt.plot(x, psi2, c='C1', label="t = {:.2f}".format(t))

    plt.legend()
    plt.show()


def test_free_analytic():
    """ Test run for numeric solution to the free particle problem """
    sigma0 = 0.05
    k0 = 50*np.pi
    a = 0.25

    x0, xJ = -0.5, 1.5
    J = 300
    dx = (xJ - x0)/J  # position step
    x = np.linspace(x0, xJ, J)

    t0, tN = 0, 4e-3
    dt = 0.1*tN  # time step
    times = np.arange(t0, tN, dt)

    for t in times:
        psi = get_free(x, t, sigma0, k0, a)
        psi2 = np.abs(psi)**2

        plt.plot(x, psi2, label="t = {:.2e}".format(t))
    plt.legend()
    plt.show()
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
    N = 5
    x = np.arange(0, N, 1)
    # args = [0, 2]
    args = []
    for n in range(N):
        print(testf(x[n], *args))
    # x0 = 0
    # J = 5
    # dx = 1
    # x1 = np.arange(x0, x0+(J+1)*dx, dx)
    # x2 = np.linspace(x0, x0 + (J*dx), J+1)
    # print(x2)


if __name__ == "__main__":
    # practice()
    # test_run()
    test_qho()
    # test_qho_analytic()
    # test_free()
    # try_roots()
