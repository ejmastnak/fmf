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

data_dir = "../9-spectral/data/"
figure_dir = "../9-spectral/figures/"

save_figures = True
usetex = True  # turn on to use Latex to render text
# usetex = False  # turn off to plot faster


# -----------------------------------------------------------------------------
# START AUXILIARY ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def get_time_grid(tmin, tmax, step, factor):
    times = [tmin]
    t = tmin
    while t < tmax:
        t += step
        times.append(t)
        step *= factor
    return np.asarray(times)


def gauss(x, a, mu, sigma):
    """
    Gauss curve with amplitude a, mean mu and standard deviation sigma.
    Used for initial temperature distribution
    """
    return a * np.exp(-(x - mu) ** 2 / sigma ** 2)


def initial_sine(x, a, l):
    """ Used for sine initial temperature distribution along a 1D rod """
    return a * np.sin(x*np.pi/l)


def initial_sine_squared(x, a, l):
    """ Used for sine initial temperature distribution along a 1D rod """
    return a * (np.sin(2*x*np.pi/l)**2)


def initial_abs_val(x, T0):
    """
    Inverted absolute value initial temperature distribution.
    """
    N = len(x)
    if N % 2 == 0:  # if N is even
        T_left = np.linspace(0, T0, N//2)
        T_right = np.linspace(T0, 0, N//2)
        return np.concatenate([T_left, T_right])
    else:  # if N is odd
        T_left = np.linspace(0, T0, N//2 + 1)
        T_right = np.linspace(T0, 0, N//2 + 1)
        return np.concatenate([T_left, T_right[1:len(T_right)]])


def initial_rectangle(x, T0, box_width=0.5):
    """
    Rectangular initial distribution, either T = 0 or T = T0
    Parameter boxwidth is percentage of the x interval taken up by the box
    """
    N = len(x)
    T = np.zeros(N)
    half_width = 0.5 * box_width
    box_start_index = int((0.5 - half_width) * N)
    box_end_index = int((0.5 + half_width) * N)
    T[box_start_index:box_end_index] = T0
    return T


def get_Bspline(x, xm2, xm1, x0, xp1, xp2, dx):
    """
    Returns the cubic spline B_k(x) on a discrete x grid sampled with spacing dx
    xm2 and xm1 are short for x_{k-2} and x_{k-1}; the m is for 'minus'
    xp2 and xp1 are short for x_{k+2} and x_{k+1}; the p is for 'plus'
    x0 is x_{k}
    """
    if x <= xm1:
        return 0
    elif xm2 < x <= xm1:
        return (1/(dx**3))*(x-xm2)**3
    elif xm1 < x <= x0:
        return (1/(dx**3))*(x-xm2)**3 - (4/(dx**3))*(x-xm1)**3
    elif x0 < x <= xp1:
        return (1/(dx**3))*(xp2 - x)**3 - (4/(dx**3))*(xp1 - x)**3
    elif xp1 < x < xp2:
        return (1/(dx**3))*(xp2 - x)**3
# -----------------------------------------------------------------------------
# END AUXILIARY ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------
def spectral_homogeneous(initial_distribution="gauss"):
    """
    Fourier solution to the 1D heat diffusion equation with homogeneous boundary conditions
    :param initial_distribution: optional string for initial temperature distribution
    :return: (x, t, T_grid) tuple where x and t are 1D position arrays and T_grid is 2D array whose columns
                are the temperature along the rod at a given time t
    """
    T0 = 10  # peak temperature of initial gaussian distribution
    l = 1.0  # rod length
    D = 1e-5  # diffusion constant
    sigma = 0.25  # standard deviation of initial temperature distribution

    N_x = 2**10  # 1024 number points, power of two for use with fft
    x = np.linspace(0, l, N_x)  # x interval spanning rod length

    if initial_distribution == "gauss":
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0
    elif initial_distribution == 'sine':
        T = initial_sine(x, T0, l)
    elif initial_distribution == 'absval':
        T = initial_abs_val(x, T0)
    elif initial_distribution == 'box':
        T = initial_rectangle(x, T0)
    else:  # use gauss by default
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)

    T = np.concatenate((T, -T[::-1]))  # concatenate T and -T
    c0 = fft(T)  # initial coefficients are Fourier transform of initial temperature distribution

    tmin = 0
    tmax = 3.0e-2
    N_t = 50
    times = np.linspace(tmin, tmax, N_t)
    T_grid = np.zeros((N_x, N_t))
    for i, t in enumerate(times):
        c = get_ck_homog(t, c0, l, D)
        T = np.real(ifft(c))[:N_x]
        T_grid[:, i] = T

    return x, times, T_grid


def spectral_periodic(initial_distribution="gauss"):
    """
    Fourier solution to the 1D heat diffusion equation with periodic boundary conditions
    :param initial_distribution: optional string for initial temperature distribution
    :return: (x, t, T_grid) tuple where x and t are 1D position arrays and T_grid is 2D array whose columns
                are the temperature along the rod at a given time t
    """
    T0 = 10  # peak temperature of initial gaussian distribution
    l = 1.0  # rod length
    D = 1e-1  # diffusion constant
    sigma = 0.25  # standard deviation of initial temperature distribution

    N_x = 2**10  # 1024 number points
    x = np.linspace(0, l, N_x)  # x interval spanning rod length

    if initial_distribution == "gauss":
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0
        T = fftshift(T)  # ffftshift the Gauss before applying fft
    elif initial_distribution == 'sine':
        T = fftshift(initial_sine(x, T0, l))
    elif initial_distribution == 'sine2':
        T = initial_sine_squared(x, T0, l)
    elif initial_distribution == 'absval':
        T = fftshift(initial_abs_val(x, T0))
    elif initial_distribution == 'box':
        T = fftshift(initial_rectangle(x, T0))
    else:  # use gauss by default
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)
        T = fftshift(T)

    c0 = fft(T)  # initial coefficients are Fourier transform of initial temperature distribution

    # tmin = 1.0e-7
    tmin = 0
    tmid = 1e-6
    tmax = 0.5e-0  # between 5e-1 and 1.0 for D=1e-1
    N_t = 50

    # t_low = np.array([0, 2.5e-8, 5e-8, 7.5e-8, 1e-7, 1.23e-7, 1.5e-7, 1.8e-7, 2.2e-7, 2.7e-7, 3.3e-7, 4.0e-7,
    #                   4.8e-7, 6.4e-7, 8e-3, 3e-2, 5e-2, 7.1e-2])  # for use with tmid = 1e-6 and D = 1e-1
    t_low = np.array([])

    times = np.linspace(tmid, tmax, N_t)
    times = np.concatenate([t_low, times])

    N_t += len(t_low)
    T_grid = np.zeros((N_x, N_t))
    for i, t in enumerate(times):
        c = get_ck_periodic(t, c0, l, D)
        T = np.real(ifftshift(ifft(c)))
        T_grid[:, i] = T

    return x, times, T_grid


def get_ck_homog(t, c0, l, D):
    """
    Returns time-evolution of Fourier coefficients c_k(t) at time t
    Uses the analytic solution c_k(t) = c0 exp(-4 pi^2 f_k^2 D t)
    Uses frequencies from -fs/2 to fs/2
    :param t: time
    :param c0: initial coeffients at time t=0
    :param l: rod length
    :param D: heat diffusion constant
    """
    fc = 0.5*len(c0)/(2*l)  # analog of a Nyquist frequency for sampling position x
    f_k = np.linspace(0, len(c0)-1, len(c0))/(2*l) - fc
    c_k = c0 * np.exp(-4*(np.pi**2)*(f_k**2)*D*t)
    return c_k


def get_ck_periodic(t, c0, l, D):
    """
    Returns time-evolution of Fourier coefficients c_k(t) at time t
    Uses the analytic solution c_k(t) = c0 exp(-4 pi^2 f_k^2 D t)
    Uses frequencies from 0 to position sampling frequency fs
    :param t: time
    :param c0: initial coeffients at time t=0
    :param l: rod length
    :param D: heat diffusion constant
    """
    f_k = np.linspace(0, len(c0)-1, len(c0))/(l)
    c_k = c0 * np.exp(-4*(np.pi**2)*(f_k**2)*D*t)
    return c_k


def get_ck_euler(t_end, n_points, c0, l, D):
    """
    CURRENTLY DOES NOT WORK!!!
    Returns time-evolution of Fourier coefficients c_k(t) at time t
    Uses Euler's method c(t+dt) = c(t) + (dc/dt)dt where dc/dt = -4D pi^2 f_k^2 c_k(t)
    :param t_end: time at which to find c_k(t)
    :param n_points: number of points to use for the time approximation
    :param c0: initial coeffients at time t=0
    :param l: rod length
    :param D: heat diffusion constant
    """
    c_k = np.zeros(len(c0), dtype=complex)
    factor = -4*D*np.pi**2  # constant factor reused over and over in increment dc/dt = -4pi^2 f_k_2 D
    dt = t_end/n_points  # find time step
    t = 0

    f_k = np.linspace(0, len(c0)-1, len(c0))/l
    step = np.ones(len(c0)) - 4*np.pi**2*D*(f_k**2)*dt  # step for numerical integration

    while t < t_end:
        t += dt  # increment time
        c_k *= step  # increment coefficients

    return c_k


def collocation_explicit(initial_distribution="gauss"):
    """
    Collocation method using normal cubic B-splines to the 1D heat equation with homogeneous boundary conditions
    Uses an explicit (basically Euler method) scheme to solve for the time-dependent basis coefficients

    :param initial_distribution: optional string for initial temperature distribution
    :return: (x, t, T_grid) tuple where x and t are 1D position arrays and T_grid is 2D array whose columns
                are the temperature along the rod at a given time t
    """
    T0 = 10  # peak temperature of initial gaussian distribution
    l = 1.0  # rod length
    D = 1e-5  # diffusion constant
    sigma = 0.25  # standard deviation of initial temperature distribution

    N_x = 500  # number of points in the x partition
    x = np.linspace(0, l, N_x)  # N_x pounts spanning rod length
    dx = x[1] - x[0]

    tmin = 0.000
    tmax = 3.0
    dt = 1e-4  # time step for numerical integration
    Nt = (tmax - tmin) / dt  # total number of points in finely-spaced time grid; not all are used in the solution

    if initial_distribution == "gauss":
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0
    elif initial_distribution == 'sine':
        T = initial_sine(x, T0, l)
    elif initial_distribution == 'sine2':
        T = initial_sine_squared(x, T0, l)
    elif initial_distribution == 'absval':
        T = initial_abs_val(x, T0)
    elif initial_distribution == 'box':
        T = initial_rectangle(x, T0)
    else:  # use gauss by default
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)

    A = np.diag(4.0*np.ones(N_x-2), k=0) + np.diag(np.ones(N_x-3), k=1) + np.diag(np.ones(N_x-3), k=-1)  # Nx-2 by Nx-2
    A_inv = np.linalg.inv(A)
    B = 6.0 * D/(dx**2) * np.diag(-2.0*np.ones(N_x-2), k=0) + np.diag(np.ones(N_x-3), k=1) + np.diag(np.ones(N_x-3), k=-1)
    step = np.eye(N_x-2, N_x-2) + dt*np.dot(A_inv, B)  # I + dt*A^{-1}*B
    c0 = np.dot(A_inv, T[1:N_x-1])  # initial coefficients. Note c is of length Nx - 2

    N_time_samples = 50  # number of points at which to sample the solution T(x, t)
    time_indeces = np.linspace(1, Nt*1e-1, N_time_samples)
    times = time_indeces * dt  # convert indeces to times
    T_grid = np.zeros((N_x, N_time_samples))
    for i, n in enumerate(time_indeces):
        c = np.dot(np.power(step, int(n)), c0)  # find coefficients c1 to c_{n-1} at time corresponding to index n
        T = np.concatenate(([0], A.dot(c), [0]))  # T = Ac, plus add endpoints, which are zero
        T_grid[:, i] = T

    return x, times, T_grid


def collocation_implicit(initial_distribution="gauss"):
    """
    Collocation method using normal cubic B-splines to the 1D heat equation with homogeneous boundary conditions
    Uses an implicit (Crank-Nicolson) method to solve for the time-dependent basis coefficients

    :param initial_distribution: optional string for initial temperature distribution
    :return: (x, t, T_grid) tuple where x and t are 1D position arrays and T_grid is 2D array whose columns
                are the temperature along the rod at a given time t
    """
    T0 = 10  # peak temperature of initial gaussian distribution
    l = 1.0  # rod length
    D = 1e-5  # diffusion constant
    sigma = 0.25  # standard deviation of initial temperature distribution

    N_x = 500  # number of points in the x partition
    x = np.linspace(0, l, N_x)  # N_x pounts spanning rod length
    dx = x[1] - x[0]

    tmin = 0.000
    tmax = 3.0
    dt = 1e-4  # time step for numerical integration
    Nt = (tmax - tmin) / dt  # total number of points in finely-spaced time grid; not all are used in the solution

    if initial_distribution == "gauss":
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0
    elif initial_distribution == 'sine':
        T = initial_sine(x, T0, l)
    elif initial_distribution == 'sine2':
        T = initial_sine_squared(x, T0, l)
    elif initial_distribution == 'absval':
        T = initial_abs_val(x, T0)
    elif initial_distribution == 'box':
        T = initial_rectangle(x, T0)
    else:  # use gauss by default
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)

    A = np.diag(4.0*np.ones(N_x-2), k=0) + np.diag(np.ones(N_x-3), k=1) + np.diag(np.ones(N_x-3), k=-1)  # Nx-2 by Nx-2
    A_inv = np.linalg.inv(A)
    B = 6.0 * D/(dx**2) * np.diag(-2.0*np.ones(N_x-2), k=0) + np.diag(np.ones(N_x-3), k=1) + np.diag(np.ones(N_x-3), k=-1)
    step_plus = A + 0.5*dt*B
    step_minus = A - 0.5*dt*B
    step = np.dot(np.linalg.inv(step_minus), step_plus)
    c0 = np.dot(A_inv, T[1:N_x-1])  # initial coefficients. Note c is of length Nx - 2

    N_time_samples = 50  # number of points at which to sample the solution T(x, t)
    time_indeces = np.linspace(1, Nt*1e-1, N_time_samples)
    times = time_indeces * dt  # convert indeces to times
    T_grid = np.zeros((N_x, N_time_samples))
    for i, n in enumerate(time_indeces):
        c = np.dot(np.power(step, int(n)), c0)  # find coefficients c1 to c_{n-1} at time corresponding to index n
        T = np.concatenate(([0], A.dot(c), [0]))  # T = Ac, plus add endpoints, which are zero
        T_grid[:, i] = T

    return x, times, T_grid
# -----------------------------------------------------------------------------
# END SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START TIMING FUNCTIONS
# -----------------------------------------------------------------------------
def fourier_time_trial(N_x=2**10, N_t=50):
    """
    Slimmed-down Fourier method solution to the 1D heat equation with homogeneous boundary conditions
    Used to test timing
    """
    T0 = 10  # peak temperature of initial gaussian distribution
    l = 1.0  # rod length
    D = 1e-5  # diffusion constant
    sigma = 0.25  # standard deviation of initial temperature distribution

    x = np.linspace(0, l, N_x)  # x interval spanning rod length

    T = gauss(x, T0, l/2, sigma)
    T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0
    T = np.concatenate((T, -T[::-1]))  # concatenate T and -T
    c0 = fft(T)  # initial coefficients are Fourier transform of initial temperature distribution

    tmin = 0
    tmax = 2.5e-2
    times = np.linspace(tmin, tmax, N_t)
    for i, t in enumerate(times):
        c = get_ck_homog(t, c0, l, D)
        T = np.real(ifft(c))[:N_x]  # calculate temperature; T is otherwise unused


def collocation_explicit_time_trial(N_x=2*10, N_time_samples=50):
    """
    Slimmed-down explicit collocation method solution to the 1D heat equation with homogeneous boundary conditions
    Used to test timing
    """
    T0 = 10  # peak temperature of initial gaussian distribution
    l = 1.0  # rod length
    D = 1e-5  # diffusion constant
    sigma = 0.25  # standard deviation of initial temperature distribution

    x = np.linspace(0, l, N_x + 1)  # N + 1 pounts spanning rod length
    dx = x[1] - x[0]

    tmin = 0.000
    tmax = 3.0
    dt = 1e-4  # time step for numerical integration
    Nt = (tmax - tmin) / dt  # total number of points in finely-spaced time grid; not all are used in the solution

    T = gauss(x, T0, l/2, sigma)
    T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0

    A = np.diag(4.0*np.ones(N_x-1), k=0) + np.diag(np.ones(N_x-2), k=1) + np.diag(np.ones(N_x-2), k=-1)  # Nx-1 by Nx-1
    A_inv = np.linalg.inv(A)
    B = 6.0 * D/(dx**2) * np.diag(-2.0*np.ones(N_x-1), k=0) + np.diag(np.ones(N_x-2), k=1) + np.diag(np.ones(N_x-2), k=-1)
    step = np.eye(N_x-1, N_x-1) + dt*np.dot(A_inv, B)  # I + dt*A^{-1}*B
    c0 = np.dot(A_inv, T[1:N_x])  # initial coefficients. Note c is of length Nx - 1

    time_indeces = np.linspace(1, Nt*1e-1, N_time_samples)
    for i, n in enumerate(time_indeces):
        c = np.dot(np.power(step, int(n)), c0)  # find coefficients c1 to c_{n-1} at time corresponding to index n
        T = np.concatenate(([0], A.dot(c), [0]))  # T = Ac, plus add endpoints, which are zero


def collocation_implicit_time_trial(N_x=2*10, N_time_samples=50):
    """
    Slimmed-down explicit collocation method solution to the 1D heat equation with homogeneous boundary conditions
    Used to test timing
    """
    T0 = 10  # peak temperature of initial gaussian distribution
    l = 1.0  # rod length
    D = 1e-5  # diffusion constant
    sigma = 0.25  # standard deviation of initial temperature distribution

    x = np.linspace(0, l, N_x + 1)  # N + 1 pounts spanning rod length
    dx = x[1] - x[0]

    tmin = 0.000
    tmax = 3.0
    dt = 1e-4  # time step for numerical integration
    Nt = (tmax - tmin) / dt  # total number of points in finely-spaced time grid; not all are used in the solution

    T = gauss(x, T0, l/2, sigma)
    T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0

    A = np.diag(4.0*np.ones(N_x-1), k=0) + np.diag(np.ones(N_x-2), k=1) + np.diag(np.ones(N_x-2), k=-1)  # Nx-1 by Nx-1
    A_inv = np.linalg.inv(A)
    B = 6.0 * D/(dx**2) * np.diag(-2.0*np.ones(N_x-1), k=0) + np.diag(np.ones(N_x-2), k=1) + np.diag(np.ones(N_x-2), k=-1)
    step_plus = A + 0.5*dt*B
    step_minus = A - 0.5*dt*B
    step = np.dot(np.linalg.inv(step_minus), step_plus)
    c0 = np.dot(A_inv, T[1:N_x])  # initial coefficients. Note c is of length Nx - 1

    time_indeces = np.linspace(1, Nt*1e-1, N_time_samples)
    for i, n in enumerate(time_indeces):
        c = np.dot(np.power(step, int(n)), c0)  # find coefficients c1 to c_{n-1} at time corresponding to index n
        T = np.concatenate(([0], A.dot(c), [0]))  # T = Ac, plus add endpoints, which are zero


def run_time_trial():
    n_loop_runs = 5
    e_min = 4
    e_max = 11
    exponents = np.arange(e_min, e_max + 1, 1)
    n_x_points_two = np.array([int(2**e) for e in exponents])  # generate number of points in x-grid in powers of two
    n_x_points_not_two = np.array([int(2.1**e) for e in exponents])  # non-power of two to see if fourier performs worse
    n_x_points = np.concatenate([n_x_points_two, n_x_points_not_two])
    n_x_points = np.sort(n_x_points)

    fourier_times = np.zeros(len(n_x_points))
    col_explicit_times = np.zeros(len(n_x_points))
    col_implicit_times = np.zeros(len(n_x_points))
    for i, n_x in enumerate(n_x_points):
        print(n_x)
        t = time.time()
        for _ in range(n_loop_runs):
            fourier_time_trial(N_x=n_x)
        fourier_times[i] = time.time() - t

        t = time.time()
        for _ in range(n_loop_runs):
            collocation_explicit_time_trial(N_x=n_x)
        col_explicit_times[i] = time.time() - t

        t = time.time()
        for _ in range(n_loop_runs):
            collocation_implicit_time_trial(N_x=n_x)
        col_implicit_times[i] = time.time() - t

    header = "Points in x grid, Fourier times [s], Explicit collocation times [s], Implicit collocation times [s]"
    table = np.column_stack([n_x_points, fourier_times, col_explicit_times, col_implicit_times])
    np.savetxt(data_dir + "times_.csv", table, header=header, delimiter=',')
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


def remove_spines_all(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.get_xaxis().tick_bottom()


def clean_axis_3d(ax):
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


def get_title_string(distribution):
    if distribution == "gauss":
        return "Gaussian"
    elif distribution == "sine" or distribution == "sine2":
        return "Sinusoidal"
    elif distribution == "absval":
        return "Absolute Value"
    elif distribution == "box":
        return "Rectangular"
    else:
        return "Gaussian"


def plot_initial_condition_modifications(initial_distribution="gauss"):
    """
    Intended to show the necessary modifications to the initial temperature distribution with the Fourier method
    Plots the asymmetrically expanded distribution (homogeneous BC)
    And the FFT-shifted initial distribution (periodic boundary condtions)
    """
    T0, l, sigma = 10, 1.0, 0.25
    N_x = 2**10  # 1024 number points, power of two for use with fft
    x = np.linspace(0, l, N_x)  # x interval spanning rod length
    x_homog = np.linspace(-l, l, int(2*N_x))

    if initial_distribution == "gauss":
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)  # subtract initial value so edges are at T=0
    elif initial_distribution == 'sine':
        T = initial_sine(x, T0, l)
    elif initial_distribution == 'absval':
        T = initial_abs_val(x, T0)
    elif initial_distribution == 'box':
        T = initial_rectangle(x, T0)
    else:  # use gauss by default
        T = gauss(x, T0, l/2, sigma)
        T -= gauss(0, T0, l/2, sigma)

    T_homog = np.concatenate((T, -T[::-1]))  # concatenate T and -T
    T_periodic = fftshift(T)  # ffftshift the Gauss before applying fft

    fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)
    linewidth = 3

    ax = axes[0]
    remove_spines(ax)
    ax.plot(x, T, c="#91331f", lw=linewidth)
    ax.hlines(0, x[0], x[-1], ls='--', color="#222222")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Temperature $T$")
    ax.set_title("Original")

    ax = axes[1]
    remove_spines(ax)
    ax.plot(x_homog, T_homog, c="#c23a1f", lw=linewidth)
    ax.hlines(0, x_homog[0], x_homog[-1], ls='--', color="#222222")
    ax.set_xlabel("Position $x$")
    ax.set_title("Expanded")

    ax = axes[2]
    remove_spines(ax)
    ax.plot(x, T_periodic, c="#e2692d", lw=linewidth)
    ax.hlines(0, x[0], x[-1], ls='--', color="#222222")
    ax.set_xlabel("Position $x$")
    ax.set_title("Shifted")

    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "initial-{}_.png".format(initial_distribution), dpi=200)
    plt.show()


def plot_2d_colorbar(x, t, T_grid, cmap='Reds_r', time_scale=1.0, method="Fourier", distribution='gauss', bc="Homogeneous"):
    """
    Plots temperature versus position along rod with time as a parameter. Time curves are mapped to
    a colormap showing progression from high to low temperature
    :param x: 1D array of length M of position values spanning rod length
    :param t: 1D array of length N of time values
    :param T_grid: 2D (M x N) matrix holding temperature at each position and time whose
                columns are temperature distrubion with respect to x at a given time t
    :param cmap: String representation of a matplotlib colormap
    :param time_scale: multiply times by this value to reach sensible time values (since times are often small)
    :param method: just for plot title
    :param distribution: just for setting plot title
    :param bc: just for setting plot title
    """
    line_segments = LineCollection([np.column_stack([x, T]) for T in T_grid.T], cmap=cmap)
    line_segments.set_array(t*time_scale)

    # We need to set the plot limits, they will not autoscale
    fig, ax = plt.subplots(figsize=(7, 4))
    # remove_spines(ax)
    y_scale = 1.04
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim((0, y_scale*np.max(T_grid)))
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Temperature $T$")
    ax.set_title("{} Method, {} IC, {} BC".format(method, get_title_string(distribution), bc))
    ax.add_collection(line_segments)
    axcb = fig.colorbar(line_segments)
    axcb.set_label('Scaled Time')

    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "2d-{}-{}_.png".format(method.lower(), distribution), dpi=200)
    plt.show()


def plot_3d_surface(x, t, T_grid, cmap='coolwarm', time_scale=1.0, method="Fourier", distribution='gauss', bc="Homogeneous"):
    """
    Plots temperature on 2 dimensional position-time grid
    :param x: 1D array of length M of position values spanning rod length
    :param t: 1D array of length N of time values
    :param T_grid: 2D (M x N) matrix holding temperature at each position and time whose
                    columns are temperature distrubion with respect to x at a given time t
    :param cmap: String representation of a matplotlib colormap
    :param time_scale: multiply times by this value to reach sensible time values (since times are often small)
    """
    t *= time_scale
    x_grid, t_grid = np.meshgrid(x, t)  # create 2D grids for plotting

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x_grid, t_grid, T_grid.T, cmap=cmap)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(t[0], t[-1])
    ax.set_zlim(np.min(0), np.max(T_grid))
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Time $t$")
    ax.set_zlabel("Temperature $T$")

    ax.view_init(26, -44)  # set view angle (elevation angle, azimuth angle)
    clean_axis_3d(ax)

    if method == "Collocation": plt.suptitle("{} Method, {} IC".format(method, get_title_string(distribution)), fontsize=18)
    else: plt.suptitle("{} Method, {} IC, {} BC".format(method, get_title_string(distribution), bc), fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=1.01)
    if save_figures: plt.savefig(figure_dir + "3d-{}-{}_.png".format(method.lower(), distribution), dpi=200)
    plt.show()


def plot_timing():
    """ Plots time taken by each method to find a solution"""
    n_x, t_fourier, t_col_explicit, t_col_implicit = np.loadtxt(data_dir + "times.csv", skiprows=1, delimiter=',').T

    plt.figure(figsize=(7, 4))
    remove_spines(plt.gca())
    plt.xlabel("Points in Position Grid $N_x$")
    plt.ylabel("Computation Time for 5 Runs $t$ [s]")
    plt.yscale('log')
    color_fourier = color_blue
    color_explicit = "#e1692e"
    color_implicit = "#91331f"
    plt.plot(n_x, t_fourier, c=color_fourier, ls='--', marker='d', label='fourier')
    plt.plot(n_x, t_col_explicit, c=color_explicit, ls='--', marker='o', label='explicit')
    plt.plot(n_x, t_col_implicit, c=color_implicit, ls='--', marker='o', label='implicit')
    plt.title("Computation Time of Fourier and Collocation Methods")
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "times_.png", dpi=200)
    plt.show()
# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------
def animate_graph(x, t, T_grid, cmap='coolwarm_r', time_scale=1.0, distribution="gauss"):
    """
    Plots time evolution of temperature versus position along rod with time as a parameter.
    Time curves are mapped to a colormap showing progression from high to low temperature

    :param x: 1D array of length M of position values spanning rod length
    :param t: 1D array of length N of time values
    :param T_grid: 2D (M x N) matrix holding temperature at each position and time whose
                columns are temperature distrubion with respect to x at a given time t
    :param cmap: String representation of a matplotlib colormap
    :param time_scale: multiply times by this value to reach sensible time values (since times are often small)
    :param distribution: string name of initial distribution, just for naming saved files
    """

    iterations = len(t)
    T_max = np.max(T_grid)
    T0 = T_grid[:, 0]  # initial temperature
    t *= time_scale

    fig, ax = plt.subplots(figsize=(7, 4))
    remove_spines(ax)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Temperature $T$")

    time_label_template = "Time: {:.2f}"
    time_label = ax.text(x[0], T_max, '', va='top', ha='left',
                         bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    scatter = ax.scatter(x, T0, c=cm.coolwarm(T0/T_max))  # initial distribution

    temp_animation = animation.FuncAnimation(fig, update_animate_graph, iterations,
                                             fargs=(x, T_grid, T_max, scatter, time_label, time_label_template, iterations, t[-1]),
                                             interval=50, blit=False, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    temp_animation.save(figure_dir + 'animate-graph-{}_.mp4'.format(distribution), writer=writer)


def update_animate_graph(i, x, T_grid, T_max, scatter, time_label, time_label_template, N_points, end_time):
    """
    Update the data held by the scatter plot and therefore animates it.
    """
    print("Frame {} of {}".format(i, N_points))
    T = T_grid[:, i]  # initial temperature
    colormap = cm.coolwarm(T/T_max)
    scatter.set_offsets(np.column_stack([x, T]))
    scatter.set_color(colormap)
    time_label.set_text(time_label_template.format(i*end_time/N_points))
    return scatter


def animate_rod(x, t, T_grid, cmap='coolwarm_r', time_scale=1.0, distribution="gauss"):
    """
    Plots time evolution of temperature versus position along rod with time as a parameter.
    Time curves are mapped to a colormap showing progression from high to low temperature

    :param x: 1D array of length M of position values spanning rod length
    :param t: 1D array of length N of time values
    :param T_grid: 2D (M x N) matrix holding temperature at each position and time whose
                columns are temperature distrubion with respect to x at a given time t
    :param cmap: String representation of a matplotlib colormap
    :param time_scale: multiply times by this value to reach sensible time values (since times are often small)
    :param distribution: string name of initial distribution, just for naming saved files
    """

    iterations = len(t)
    T_max = np.max(T_grid)
    T0 = T_grid[:, 0]  # initial temperature
    t *= time_scale

    fig, ax = plt.subplots(figsize=(7, 2))
    remove_spines_all(ax)
    ax.set_xlabel("Position $x$")
    y_max = 1.0
    ax.set_ylim((0, y_max))  # arbitrary y scale from 0 to 1; rod is placed at y = 0.5
    ax.set_title("1D Heat Diffusion, {} IC".format(get_title_string(distribution)), fontsize=18)

    plt.tight_layout()

    time_label_template = "Time: {:.2f}"
    time_label = ax.text(x[0], y_max, '', va='top', ha='left',
                         bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    rod_points = 0.5*y_max*np.ones(len(x))  # just a horizontal line representing a rod halfway along the y axis
    scatter = ax.scatter(x, rod_points, c=cm.coolwarm(T0/T_max), s=900)  # initial distribution

    temp_animation = animation.FuncAnimation(fig, update_animate_rod, iterations,
                                             fargs=(x, T_grid, T_max, y_max, scatter, time_label, time_label_template, iterations, t[-1]),
                                             interval=50, blit=False, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    temp_animation.save(figure_dir + 'animate-rod-{}_.mp4'.format(distribution), writer=writer)


def update_animate_rod(i, x, T_grid, T_max, y_max, scatter, time_label, time_label_template, N_points, end_time):
    """
    Update the data held by the scatter plot and therefore animates it.
    """
    print("Frame {} of {}".format(i, N_points))
    T = T_grid[:, i]  # temperature distribution
    colormap = cm.coolwarm(T/T_max)  # calculate colormap

    rod = 0.5*y_max*np.ones(len(x))
    scatter.set_offsets(np.column_stack([x, rod]))

    scatter.set_color(colormap)
    time_label.set_text(time_label_template.format(i*end_time/N_points))

    return scatter
# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# START ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# END ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def practice():
    x = np.linspace(0, 1.0, 100)
    y = initial_sine_squared(x, 10, 1.0)
    plt.plot(x, y)
    plt.show()
    # N = 11
    # x = np.linspace(1, N, N)
    # y = np.linspace(1, N, N)
    # A = np.column_stack([x, y])
    # print(A)
    # T = initial_abs_val(x, N)
    # plt.plot(x, T, ls='--', marker='o')
    # plt.show()
    # A = np.diag(a, k=0)
    # b = np.dot(A, a)
    # times = get_time_grid(0, 10, 0.001, 2)
    # print(times)


def run():
    # practice()
    # run_time_trial()

    distribution = "box"
    # plot_initial_condition_modifications(initial_distribution=distribution)
    # plot_2d_colorbar(*spectral_homogeneous(initial_distribution=distribution), time_scale=1e2, method='Fourier', distribution=distribution)
    # plot_2d_colorbar(*spectral_periodic(initial_distribution=distribution), time_scale=1e2, distribution=distribution, bc="Periodic")
    # plot_2d_colorbar(*collocation_explicit(initial_distribution=distribution), time_scale=1e1, method="Collocation", distribution=distribution)

    plot_3d_surface(*spectral_homogeneous(initial_distribution=distribution), time_scale=1e2, method='Fourier', distribution=distribution)
    # plot_3d_surface(*spectral_periodic(initial_distribution=distribution), time_scale=1e0, distribution=distribution, bc="Periodic")

    # plot_3d_surface(*collocation_explicit(initial_distribution=distribution), time_scale=1e1, method='Collocation', distribution=distribution)
    # plot_3d_surface(*collocation_implicit(initial_distribution=distribution), time_scale=1e1, method='Collocation', distribution=distribution)
    # plot_timing()

    # animate_graph(*spectral_homogeneous(initial_distribution='gauss'), time_scale=1e2)
    # animate_graph(*spectral_periodic(initial_distribution=distribution), time_scale=1e1, distribution=distribution)
    # animate_rod(*spectral_homogeneous(initial_distribution='gauss'), time_scale=1e2)
    animate_rod(*spectral_periodic(initial_distribution=distribution), time_scale=1e1, distribution=distribution)


run()
