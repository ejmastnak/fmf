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


def get_z(s, M):
    """ Returns z_s^(M), as defined in the report.
        Used for higher-order time advance of the wavefunction
        :param s: s = 1, 2, ..., M
        :param M: Positive integer giving order of time approximation M = 1, 2, ...
    """
    if s > M:
        print("Warning in get_z: s > M: s = {} \t M = {}".format(s, M))
    if M == 1:  # s = 1
        return -2.0 + 0.0j
    elif M == 2:  # s = 1, 2
        if s == 1: return -3.0 + 1.73205j
        else: return -3.0 - 1.73205j
    elif M == 3:
        if s == 1: return -4.64437 + 0.0j
        elif s == 2: return -3.67781 - 3.50876j
        else: return -3.67781 + 3.50876j
    else:  # return M == 1 as base case
        print("Returning z for M=1")
        return -2.0 + 0.0j


def get_c(k, r):
    """
    Returns c_k^(r) as defined in the report.
    Used for higher-order position differentiation of the wavefunction
    :param k: runs from 0, 1, 2, ..., r
    :param r: currently defined from 1 to 7
    """
    if k > r:
        print("Warning in get_c: k > r: k = {} \t r = {}".format(k, r))
    if r == 1:
        if k == 0: return -2.0
        else: return 1.0
    elif r == 2:
        if k == 0: return -5.0/2
        elif k == 1: return 4.0/3
        else: return -1.0/12
    else:  # return r = 1 as base case
        print("Returning c for r = 1")
        if k == 0: return -2.0
        else: return 1.0


def get_a(dt, dx, k, r, s, M):
    """

    :param dt: time step
    :param dx: position step
    :param k: k = 0, 1, ..., r
    :param r: Order of position approximation. r = 1, 2, 3, ...
    :param s: s = 1, 2, ..., M
    :param M: Order of time approximation. M = 1, 2, 3, ...
    """
    return get_b(dt, dx) * get_c(k, r) / get_z(s, M)


def get_d(x0, dx, dt, J, r, s, M, V):
    """

    :param x0: x coordinate of initial position
    :param dx: position step
    :param dt: time step
    :param J: J+1 points used to partition position grid for j = 0, 1,...,J
    :param r: Order of position approximation. r = 1, 2, 3, ...
    :param s: Index of time evolution matrix
    :param M: Order of time approximation. M = 1, 2, 3, ...
    :param V: potential energy function of for V(x) --> input float x (position) return float (potential)
    :return (J+1)-element complex numpy array holding main diagonal of evolution matrix
    """
    x = np.linspace(x0, x0 + (J*dx), J+1)
    a_r = get_a(dt, dx, 0, r, s, M)
    z_sM = get_z(s, M)
    d = np.zeros(J+1, dtype=complex)
    d[0: J+1] = 1 + a_r - (0.0 + 1.0j)*dt*V(x)/z_sM  # forces V(x) to unpack as a J+1-element array
    return d


def get_A(x0, dx, dt, J, r, s, M, V):
    """
    :param x0: x coordinate of initial position
    :param dx: position step
    :param dt: time step
    :param J: J+1 points used to partition position grid for j = 0, 1,...,J
    :param r: Order of position approximation. r = 1, 2, 3, ...
    :param s: Index of time evolution s = 1, 2, ..., M
    :param M: Order of time approximation. M = 1, 2, 3, ...
    :param V: potential energy function of for V(x) --> input float x (position) return float (potential)
    :return (J+1)x(J+1)-element complex numpy matrix for evolving a wavefunction
    """
    d = get_d(x0, dx, dt, J, r, s, M, V)  # main diagonal
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


def test_run():
    x0 = 0
    dx, dt = 1.0, 1.0
    J = 5
    r = 2
    s, M = 1.0, 1.0

    A = get_A(x0, dx, dt, J, r, s, M, V_free)
    print(A)
# -----------------------------------------------------------------------------
# END AUXILIARY ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# START SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------
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


def practice():
    # x = np.linspace(0, 1.0, 100)
    # print(x)
    # x0 = 0
    # J = 5
    # dx = 1
    # x1 = np.arange(x0, x0+(J+1)*dx, dx)
    # x2 = np.linspace(x0, x0 + (J*dx), J+1)
    # print(x2)
    r = 3
    for k in range(1, r+1, 1):
        print(k)


if __name__ == "__main__":
    # practice()
    test_run()
