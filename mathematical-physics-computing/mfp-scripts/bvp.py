import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import newton, root_scalar
from scipy.linalg import eigh_tridiagonal, solve
from scipy.integrate import solve_bvp

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

color_gray_ref = "#AAAAAA"  # light gray for reference quantities

data_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/8-boundary-value-problem/data/"
figure_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/8-boundary-value-problem/figures/"

error_dir = data_dir + "error-step/"
time_dir = data_dir + "times/"
step_global_error_dir = data_dir + "stepsize-vs-global-error/"
step_local_error_dir = data_dir + "stepsize-vs-local-error/"

save_figures = True
usetex = True  # turn off to plot faster
# usetex = False  # turn off to plot faster


# -----------------------------------------------------------------------------
# START SCHRODINGER EQUATIONS AND POTENTIALS
# -----------------------------------------------------------------------------
def V_fpw(x, V0=50.0, l=1.0):
    """
    Potential energy for finite potential well of height V0 and width l
    :param x: x coordinate
    :param V0: well height
    :param l: well width/length
    """
    if np.iterable(x):  # for array input
        return np.array([V_fpw(x_i, V0, l) for x_i in x])
    elif np.abs(x) < l / 2:  # scalar input inside the well
        return 0
    else:  # scalar input outside the well
        return V0


def schro_ipw(state, x, E):
    """
    Stationary Schrodinger equation for a particle in a infinite potential well
    psi'' = -2*m*E*psi(x)/(hbar^2)
    Uses natural units, so m = hbar = 1

    :param x: position
    :param state: 2-element array containing [psi, psi']
    :param E: particle's energy as a 1-element ndarray with shape (1,)
    :return: 2-element array containing [psi', psi'']
    """
    return_state = np.zeros(2)
    return_state[0] = state[1]
    return_state[1] = -2 * E * state[0]
    return return_state


def schro_fpw(state, x, E):
    """
    Stationary Schrodinger equation for a particle in a finite potential well
    psi'' = -2*m*(E-V(x))*psi(x)/(hbar^2)
    Uses natural units, so m = hbar = 1

    :param x: position
    :param state: 2-element array containing [psi, psi']
    :param E: particle's energy as a 1-element ndarray with shape (1,)
    :return: 2-element array containing [psi', psi'']
    """
    return_state = np.zeros(2)
    return_state[0] = state[1]
    return_state[1] = -2*(E-V_fpw(x))*state[0]
    return return_state


def schro_fpw_xfirst(x, state, E):
    return np.vstack((state[1], -2*(E[0]-V_fpw(x))*state[0]))


def rk4_schro(f, initial_state, x, E):
    """
    RK4 integrator for use with the schroedinger equation --- includes the energy parameter
    :param f: function (should be a schrodinger equation) with signature f(state, x, E)
    :param initial_state: initial state [psi, psi']
    :param x: array of x values at which to find the solution
    :return: 2-column matrix; first column holds psi and second column holds psi' at each value of x
    """
    n = len(x)
    states = np.array(n * [initial_state])  # initialize psi with initial state
    for i in range(n - 1):
        h = x[i + 1] - x[i]
        k1 = h * f(states[i], x[i], E)
        k2 = h * f(states[i] + 0.5 * k1, x[i] + 0.5 * h, E)
        k3 = h * f(states[i] + 0.5 * k2, x[i] + 0.5 * h, E)
        k4 = h * f(states[i] + k3, x[i + 1], E)
        states[i + 1] = states[i] + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
    return states


def normalize(x, psi):
    psi_squared = psi**2
    integral = 0
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        integral += 0.5*dx*(psi_squared[i] + psi_squared[i+1])
    return psi / np.sqrt(integral)
# -----------------------------------------------------------------------------
# END SCHRODINGER EQUATIONS AND POTENTIALS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ANALYTIC SOLUTION FUNCTIONS
# -----------------------------------------------------------------------------
def get_fpw_analytic_energies(V0=50.0, l=1.0, n_points=100):
    """ Solves for the energies of the bounds states of a particle in a
        finite potential well with height V0 and width l

        Finds the energies by finding the roots of the function energy_root_fun_fpw, defined below
    """
    rounding_places = 8
    energy_grid = np.linspace(0.0, V0, n_points)  # energy grid between 0 and V0 to use with the root finding function
    energies = []  # array to hold energy solutions
    for n in [1, 2]:  # test both even/odd or symmetric/asymmetric states
        for i in range(n_points - 1):
            E_solution = root_scalar(energy_root_fun_fpw, args=(V0, l, n), x0=energy_grid[i], x1=energy_grid[i+1])
            if E_solution.converged:  # only take convergent solutions (uses root_scalar's convenience solution object)
                if np.around(E_solution.root, rounding_places) not in energies:  # reject already-found energies
                    energies.append(np.around(E_solution.root, rounding_places))

    return np.sort(energies)  # sort and return energy solutions


def energy_root_fun_fpw(E, V0=50, l=1, n=1):
    """
    Auxiliary function used when solving the transcendential equation for the finite potential well energy
     For odd n the equation is k2 = k * tan(k1*l/2)
     For even n the equation is k2 = -k * cot(k1*l/2)
    where k1 and k2 are calculated in natural units with hbar = m = 1

    Finding E comes down to finding the roots of the function f(E) where
     f(E) = k2 - k*tan(k1*l/2) for odd n
     f(E) = k2 + k*cot(k1*l/2) for even n

    :param E: particle energy
    :param V0: well height
    :param l: well width/length
    :param n: state number (e.g. 1st excited state, 2nd excited state, etc...)
    :return f(E) where f(E) is defined above
    """
    if (E < 0) or (V0 - E < 0): return np.nan  # used to break out of a diverging root_finding algorithm that led to non-physical negative energies
    k1 = np.sqrt(2*E)
    k2 = np.sqrt(2*(V0 - E))
    if n % 2 == 1:  # for odd n
        return k2 - k1*np.tan(k1*l/2)
    else:  # for even n
        return k2 + k1/np.tan(k1*l/2)


def get_ipw_energy_analytic(n, l=1.0):
    return (np.pi*n/l)**2/2


def get_ipw_psi_analytic(x, n, l=1.0):
    return np.sqrt(2/l)*np.sin(n * np.pi * (x + l / 2) / l)
# -----------------------------------------------------------------------------
# START ANALYTIC SOLUTION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SHOOTING FUNCTIONS
# -----------------------------------------------------------------------------
def ipw_energy_root_fun(E, psi0, x):
    psi = rk4_schro(schro_ipw, psi0, x, E)
    return psi[-1][0]


def get_ipw_endpoint(initial_state, x, E):
    psi = rk4_schro(schro_ipw, initial_state, x, E)
    return psi[-1][0]  # the value of psi at the endpoint


def shoot_ipw_sym_asym(l=1.0, n_points=300, E_max=200):
    """
    Shooting method for infinte potential well. Finds both symmetric and asymmetric functions in one run.
    """
    a, b = -l/2, l/2
    x = np.linspace(a, b, n_points)

    initial_state = [0.0, 1.0]
    test_energies = np.arange(1.0, E_max, 5.0)  # energies to test with shooting method

    psi_endpoint_values = np.zeros(len(test_energies))
    for i, E in enumerate(test_energies):
        psi_endpoint_values[i] = get_ipw_endpoint(initial_state, x, E)

    zero_crossings = np.where(np.diff(np.sign(psi_endpoint_values)))[0]  # indeces where endpoint values cross zero
    eig_energies = np.zeros(len(zero_crossings))  # preallocate
    for i, cross in enumerate(zero_crossings):
        eig_energies[i] = newton(ipw_energy_root_fun, test_energies[cross], args=(initial_state, x))

    psi_list = np.zeros((len(x), len(eig_energies)))
    for i, E_n in enumerate(eig_energies):
        psi_n = rk4_schro(schro_ipw, initial_state, x, E_n)[:, 0]
        psi_n = normalize(x, psi_n)
        psi_list[:, i] = psi_n

    return x, eig_energies, psi_list


def fpw_energy_root_fun(E, psi0, x):
    psi = rk4_schro(schro_fpw, psi0, x, E)
    return psi[-1][0]


def get_fpw_endpoint(initial_state, x, E):
    psi = rk4_schro(schro_fpw, initial_state, x, E)
    return psi[-1][0]  # the value of psi at the endpoint


def shoot_fpw_sym_asym(l=1.0, n_points=300, V0=50):
    """
    Shooting method for infinte potential well. Finds both symmetric and asymmetric solutions in one shot
    """
    x = np.linspace(-2*l, 2*l, n_points)

    initial_state = [0.0, 0.001]  # small non-zero initial derivative value
    test_energies = np.arange(0.0, V0, 1.0)  # energies to test with shooting method

    psi_endpoint_values = np.zeros(len(test_energies))
    for i, E in enumerate(test_energies):
        psi_endpoint_values[i] = get_fpw_endpoint(initial_state, x, E)

    zero_crossings = np.where(np.diff(np.sign(psi_endpoint_values)))[0]  # indeces where endpoint values cross zero
    eig_energies = np.zeros(len(zero_crossings))  # preallocate
    for i, cross in enumerate(zero_crossings):
        eig_energies[i] = newton(fpw_energy_root_fun, test_energies[cross], args=(initial_state, x))

    psi_list = np.zeros((len(x), len(eig_energies)))
    for i, E_n in enumerate(eig_energies):
        psi_n = rk4_schro(schro_fpw, initial_state, x, E_n)[:, 0]
        psi_n = normalize(x, psi_n)
        psi_list[:, i] = psi_n

    return x, eig_energies, psi_list
# -----------------------------------------------------------------------------
# END SHOOTING FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START FINITE DIFFERENCE FUNCTIONS
# -----------------------------------------------------------------------------
def ipw_fd(steps=2000, l=1.0, N_max=10):
    x = np.linspace(-l/2, l/2,steps)
    h = x[1]-x[0]

    d = 1.0*np.ones(steps-0)/h**2
    e = -0.5*np.ones(steps-1)/h**2
    eig_energies, psi_list = eigh_tridiagonal(d, e)
    for i in range(N_max):
        psi_list[:,i] = normalize(x, psi_list[:,i])
    return eig_energies[0:N_max], psi_list[:,0:N_max]


def fpw_fd1(n_points=2000, l=1.0, V0=50.0):
    x = np.linspace(-1.5 * l, 1.5 * l, n_points)
    h = x[1]-x[0]
    d = 1.0 * np.ones(n_points) / h ** 2 + V_fpw(x, V0=V0, l=l)
    e = -0.5 * np.ones(n_points - 1) / h ** 2
    eig_energies, psi_list = eigh_tridiagonal(d, e)
    N_max = 0
    for i, E in enumerate(eig_energies):  # keep only bound energies
        if E > V0:
            N_max = i
            break
    for i in range(N_max):
        psi_list[:,i] = normalize(x, psi_list[:,i])
        if i % 2 == 1: psi_list[:,i] = -1.0 * psi_list[:, i]  # flip sign of odd solutions
    return x, eig_energies[0:N_max], psi_list[:,0:N_max]


def fpw_fd2(steps=2000, l=1.0, V0=50):
    x = np.linspace(-l/2, l/2, steps)
    h = x[1]-x[0]
    energies = get_fpw_analytic_energies(V0=V0, l=l, n_points=100)
    for E in energies:
        k = np.sqrt(2*(V0 - E))
        upper_diag = -1*np.ones(steps - 1)/h**2
        upper_diag[0] = - 1/h  # boundary condition
        lower_diag = -1*np.ones(steps - 1)/h**2
        lower_diag[-1] = -1/h  # boundary condition
        main_diag = 2*np.ones(steps)/h**2 - E  # V(x) = 0 inside the well
        main_diag[0] = -k - 1/h
        main_diag[-1] = -k + 1/h
        H = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)

        b = E * np.ones(steps)
        b[0], b[1] = h*E, h*E  # boundary conditions

        psi = solve(H, b)
        plt.plot(x, psi, label="{:.4f}".format(E))
        plt.legend()
        plt.show()


def fpw_fd3(steps=2000, l=1.0, V0=50):
    x = np.linspace(-l/2, l/2, steps)
    h = x[1]-x[0]
    energies = get_fpw_analytic_energies(V0=V0, l=l, n_points=100)
    for E in energies:
        k = np.sqrt(2*(V0 - E))
        upper_diag = np.ones(steps - 1)
        upper_diag[0] = - 1/h  # boundary condition
        lower_diag = np.ones(steps - 1)
        lower_diag[-1] = -1/h  # boundary condition
        main_diag = -2*(1-E*h**2) * np.ones(1)
        main_diag[0] = k + 1/h - h*E
        main_diag[-1] = k + 1/h - h*E
        H = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)
        b = np.zeros(steps)

        psi = solve(H, b)
        plt.plot(x, psi, label="{:.4f}".format(E))
        plt.legend()
        plt.show()
# -----------------------------------------------------------------------------
# END FINITE DIFFERENCE FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SOLVE_BVP FUNCTIONS
# -----------------------------------------------------------------------------
def bc_res_fpw(state_a, state_b, E):
    """
    Returns residuals of the finite potential well boundary conditions for use with scipy's solve_bvp
    Uses the boundary conditions
        psi(a) = psi(b) = 0 and
        psi'(a) = 0.001  (some arbitrary small value)
    :param state_a: 2-element array containing state [psi, psi'] at left boundary
    :param state_b: 2-element array containing state [psi, psi'] at right boundary
    :param E: particle's energy as a 1-element ndarray with shape (1,)
    """
    return np.array([state_a[0], state_b[0], state_a[1]-0.001])  # residuals at the three boundary conditions
    # formally returns state_a[0] - psi(a) and state_b[0] - psi(b) but psi(a) = psi(b) are zero and thus omitted


def get_fpw_solutions_bvp(x, V0=50.0, l=1.0):
    """ Returns bound-state solutions to the FPW at the position values x """
    y_guess = np.zeros((2, np.shape(x)[0]))  # initial guess for solution y (written as a 2-element state vector)
    y_guess[0, 4] = 0.1  # set a small nonzero value for psi(x), offset from the origin
    energies = get_fpw_analytic_energies(V0=V0, l=l)  # used for energy guesses
    print(energies)
    return [solve_bvp(schro_fpw_xfirst, bc_res_fpw, x, y_guess, p=[E_i]) for E_i in energies]  # solve_bvp solution objects
# END SOLVE_BVP FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------
def generate_ipw_shoot_sym_asym_solutions():
    l = 1.0
    E_max = 180  # the first four states
    x, eig_energies, psi_list = shoot_ipw_sym_asym(l=l, E_max=E_max)
    header = "Energy Eigenvalues:,"
    for i in range(len(eig_energies)):  # adds energy eigenvalues to header
        if i == len(eig_energies) - 1:  # last element in array
            header += "{:.6f}".format(eig_energies[i])
        else:
            header += "{:.6f},".format(eig_energies[i])
    header += "\nPosition x, "
    for i in range(len(eig_energies)):  # just generates a header
        if i == len(eig_energies) - 1:  # last element in array
            header += "Psi {}".format(i+1)
        else:
            header += "Psi {},".format(i+1)  # include a comma
    x = np.reshape(x, (len(x), 1))
    np.savetxt(data_dir + "ipw-shoot-sym-asym-{}.csv".format(l), np.hstack((x, psi_list)), delimiter=',', header=header)


def generate_ipw_fd_solutions(N=1000, l=1.0):
    N_max = 10
    x = np.linspace(-l/2, l/2, N)
    eig_energies, psi_list = ipw_fd(steps=N, l=l, N_max=N_max)
    N_max = len(eig_energies)
    header = "Energy Eigenvalues:,"
    for i in range(N_max):  # adds energy eigenvalues to header
        if i == len(eig_energies) - 1:  # last element in array
            header += "{:.6f}".format(eig_energies[i])
        else:
            header += "{:.6f},".format(eig_energies[i])
    header += "\nPosition x, "
    for i in range(N_max):  # just generates a header
        if i == len(eig_energies) - 1:  # last element in array
            header += "Psi {}".format(i+1)
        else:
            header += "Psi {},".format(i+1)  # include a comma
    x = np.reshape(x, (len(x), 1))
    np.savetxt(data_dir + "ipw-fd-{}-{}.csv".format(l, N), np.hstack((x, psi_list)), delimiter=',', header=header)


def generate_fpw_fd_solutions(N=1000, l=1.0, V0=50.0):
    x, eig_energies, psi_list = fpw_fd1(n_points=N, l=l, V0=V0)
    N_max = len(eig_energies)
    header = "Energy Eigenvalues:,"
    for i in range(N_max):  # adds energy eigenvalues to header
        if i == len(eig_energies) - 1:  # last element in array
            header += "{:.6f}".format(eig_energies[i])
        else:
            header += "{:.6f},".format(eig_energies[i])
    header += "\nPosition x, "
    for i in range(N_max):  # just generates a header
        if i == len(eig_energies) - 1:  # last element in array
            header += "Psi {}".format(i+1)
        else:
            header += "Psi {},".format(i+1)  # include a comma
    x = np.reshape(x, (len(x), 1))
    np.savetxt(data_dir + "fpw-fd-{}-{:.0f}-{}.csv".format(l, V0, N), np.hstack((x, psi_list)), delimiter=',', header=header)


def generate_fpw_shoot_sym_asym_solutions():
    l = 1.0
    V0 = 50
    x, eig_energies, psi_list = shoot_fpw_sym_asym(l=l, V0=V0)

    header = "Energy Eigenvalues:,"
    for i in range(len(eig_energies)):  # adds energy eigenvalues to header
        if i == len(eig_energies) - 1:  # last element in array
            header += "{:.6f}".format(eig_energies[i])
        else:
            header += "{:.6f},".format(eig_energies[i])
    header += "\nPosition x, "
    for i in range(len(eig_energies)):  # just generates a header
        if i == len(eig_energies) - 1:  # last element in array
            header += "Psi {}".format(i+1)
        else:
            header += "Psi {},".format(i+1)  # include a comma
    x = np.reshape(x, (len(x), 1))
    np.savetxt(data_dir + "fpw-shoot-sym-asym-{}.csv".format(l), np.hstack((x, psi_list)), delimiter=',', header=header)


def generate_fpw_bvp_solutions():
    V0, l = 50.0, 1.0
    xmin, xmax = -1.5, 1.5
    n_points_bvp = 11  # only use a few points for bvp
    x = np.linspace(xmin, xmax, n_points_bvp)
    solutions = get_fpw_solutions_bvp(x, V0=V0, l=l)

    n_points_x = 200
    x = np.linspace(xmin, xmax, 200)  # finer x grid for plotting than for solve_bvp
    eig_energies = np.zeros(len(solutions))
    psi_list = np.zeros((n_points_x, len(solutions)))

    for i, solution in enumerate(solutions):
        print(f"n = {i+1}:", "success = {success}; niter = {niter}".format(**solution))
        psi = solution.sol(x)[0]  # interpolate the bvp solutions onto x
        psi = normalize(x, psi)
        psi_list[:,i] = psi
        eig_energies[i] = solution.p[0]

    header = "Energy Eigenvalues:,"
    for i in range(len(eig_energies)):  # adds energy eigenvalues to header
        if i == len(eig_energies) - 1:  # last element in array
            header += "{:.6f}".format(eig_energies[i])
        else:
            header += "{:.6f},".format(eig_energies[i])
    header += "\nPosition x, "
    for i in range(len(eig_energies)):  # just generates a header
        if i == len(eig_energies) - 1:  # last element in array
            header += "Psi {}".format(i+1)
        else:
            header += "Psi {},".format(i+1)  # include a comma
    x = np.reshape(x, (len(x), 1))
    np.savetxt(data_dir + "fpw-solve-bvp-{}.csv".format(l), np.hstack((x, psi_list)), delimiter=',', header=header)


def find_ipw_energy_errors():
    filenames = ("ipw-fd-1.0-100", "ipw-fd-1.0-1000", "ipw-fd-1.0-10000", "ipw-shoot-sym-asym-1.0")
    energy_table = np.zeros((4, 5))
    energy_error_table = np.zeros((4, 4))
    analytic_energies = np.zeros(4)
    for n in range(4):
        analytic_energies[n] = get_ipw_energy_analytic(n+1)
    energy_table[:,0] = analytic_energies

    for k, filename in enumerate(filenames):
        with open(data_dir + filename + ".csv") as f:
            first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
            splitstring = first_line.split(",")
        eig_energies = np.zeros(4)
        for i, str_energy in enumerate(splitstring):
            eig_energies[i] = float(str_energy)
            if i >= 3: break
        energy_table[:,k+1] = eig_energies
        error = np.abs(analytic_energies - eig_energies)
        energy_error_table[:,k] = error

    energy_header = "Analytic, fd-100, fd-1000, fd-10000, shoot"
    error_header = "fd-100, fd-1000, fd-10000, shoot"
    np.savetxt(data_dir + "ipw-energies-50.csv", energy_table, delimiter=',', header=energy_header, fmt="%.6f")
    np.savetxt(data_dir + "ipw-energy-errors-50.csv", energy_error_table, delimiter=',', header=error_header, fmt="%.6e")
    print(energy_table)


def find_fpw_energy_errors():
    filenames = ("fpw-fd-1.0-50-100", "fpw-fd-1.0-50-1000", "fpw-fd-1.0-50-10000", "fpw-shoot-sym-asym-1.0", "fpw-solve-bvp-1.0")
    energy_table = np.zeros((4, 6))
    energy_error_table = np.zeros((4, 5))
    analytic_energies = get_fpw_analytic_energies(V0=50)
    energy_table[:,0] = analytic_energies

    for k, filename in enumerate(filenames):
        with open(data_dir + filename + ".csv") as f:
            first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
            splitstring = first_line.split(",")
        eig_energies = np.zeros(4)
        for i, str_energy in enumerate(splitstring):
            eig_energies[i] = float(str_energy)
            if i >= 3: break
        energy_table[:,k+1] = eig_energies
        error = np.abs(analytic_energies - eig_energies)
        energy_error_table[:,k] = error

    energy_header = "Analytic, fd-100, fd-1000, fd-10000, shoot, solve_bvp"
    error_header = "fd-100, fd-1000, fd-10000, shoot, solve_bvp"
    np.savetxt(data_dir + "fpw-energies-50.csv", energy_table, delimiter=',', header=energy_header, fmt="%.6f")
    np.savetxt(data_dir + "fpw-energy-errors-50.csv", energy_error_table, delimiter=',', header=error_header, fmt="%.6e")
    print(energy_table)

# -----------------------------------------------------------------------------
# END SOLUTION GENERATION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_ipw_shoot():
    l = 1.0
    filename = "ipw-shoot-sym-asym-{:.1f}.csv".format(l)
    with open(data_dir + filename) as f:
        first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
        splitstring = first_line.split(",")
    eig_energies = np.zeros(len(splitstring))
    for i, str_energy in enumerate(splitstring):
        eig_energies[i] = float(str_energy)
    print(eig_energies)
    data = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x = data[:,0]

    oranges = ("#fde961", "#f2b549", "#e78136", "#de5126", "#a7361f", "#562b22")

    plt.figure(figsize=(7, 4))
    if usetex: plt.rc('text', usetex=True)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.get_xaxis().tick_bottom()
    x_text = x[0] - 0.02

    for i in range(np.shape(data)[1] - 1):
        psi = data[:,i+1]
        y_scale = (3 + (i+1)**1.2)
        plt.plot(x, y_scale*psi + eig_energies[i], c=oranges[-i-1])  # plot wave function shifted upward to its energy
        plt.hlines(eig_energies[i], -l/2, l/2, color=oranges[-i-1], ls='--')  #, c=y_plot[0].get_color())
        if i == 0: ax.text(x_text-0.07, eig_energies[i], r"$E = E_1$", va="center", ha="center", fontsize=14)
        else: ax.text(x_text, eig_energies[i], r"$E_{{{}}} = {{{}}}E_1$".format(i+1, (i+1)**2), va="center", ha="right", fontsize=14)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi > 0., color=oranges[-i-1], alpha=0.7)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi < 0., color=oranges[-i-1], alpha=0.7)

    plt.xlim((-l/1.8, l/1.8))
    plt.xlabel(r"Position $x$", fontsize=13)
    plt.title("IPW Wave Functions with Shooting Method", fontsize=18)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "ipw-shoot-plot.png", dpi=200)
    plt.show()


def plot_ipw_shoot_analytic_comparison():
    l = 1.0
    filename = "ipw-shoot-sym-asym-{:.1f}.csv".format(l)
    with open(data_dir + filename) as f:
        first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
        splitstring = first_line.split(",")
    eig_energies = np.zeros(len(splitstring))
    for i, str_energy in enumerate(splitstring):
        eig_energies[i] = float(str_energy)
    data = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x = data[:,0]

    oranges = ("#fce85d", "#eea444", "#e2692d", "#c23a1f")
    oranges = ("#e78136", "#de5126", "#a7361f", "#562b22")
    blues = ("#aad8c0", "#4badc6", "#2e6fa7", "#1d357f", "#111450")

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    if usetex: plt.rc('text', usetex=True)
    for i in range(4):
        if i == 0:
            ax = axes[0][0]
            ax.set_title("Ground State")
        elif i == 1:
            ax = axes[0][1]
            ax.set_title("First Excited State")
        elif i == 2:
            ax = axes[1][0]
            ax.set_title("Second Excited State")
        else:
            ax = axes[1][1]
            ax.set_title("Third Excited State")
        remove_spines(ax)

        if i % 2 == 0:
            ax.set_ylabel(r"Wavefunction $\psi(x)$")
        if i >= 2:
            ax.set_xlabel("Position $x$")
        plt.xlim((-l/1.5, l/1.5))

        psi = data[:,i+1]
        ax.plot(x, get_ipw_psi_analytic(x, i+1), c=color_gray_ref, linewidth=3, label="analytic")
        ax.plot(x, psi, ls=':', c=blues[i+1], linewidth=3, label="numerical")
        # ax.grid()
        ax.legend()

    plt.suptitle("Shooting Method Solutions for IPW", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "ipw-shoot-compared_.png", dpi=200)
    plt.show()


def plot_ipw_shoot_errors():
    l = 1.0
    filename = "ipw-shoot-sym-asym-{:.1f}.csv".format(l)
    with open(data_dir + filename) as f:
        first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
        splitstring = first_line.split(",")
    eig_energies = np.zeros(len(splitstring))
    for i, str_energy in enumerate(splitstring):
        eig_energies[i] = float(str_energy)
    data = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x = data[:,0]

    oranges = ("#e78136", "#de5126", "#a7361f", "#562b22")
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))
    if usetex: plt.rc('text', usetex=True)
    for i in range(4):
        if i == 0:
            ax = axes[0][0]
            ax.set_title("Ground State")
        elif i == 1:
            ax = axes[0][1]
            ax.set_title("First Excited State")
        elif i == 2:
            ax = axes[1][0]
            ax.set_title("Second Excited State")
        else:
            ax = axes[1][1]
            ax.set_title("Third Excited State")
        remove_spines(ax)

        if i % 2 == 0:
            ax.set_ylabel("Absolute Error")
        if i >= 2:
            ax.set_xlabel("Position $x$")
        plt.xlim((-l/1.5, l/1.5))

        psi = data[:,i+1]
        error = np.abs(psi - get_ipw_psi_analytic(x, i+1))
        ax.plot(x, error, ls=':', c=oranges[i], linewidth=3, label="numerical")

    plt.suptitle("Shooting Method Errors for the IPW", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "ipw-shoot-errors_.png", dpi=200)
    plt.show()


def plot_ipw_fd():
    l = 1.0
    filename = "ipw-fd-{:.1f}-1000.csv".format(l)
    with open(data_dir + filename) as f:
        first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
        splitstring = first_line.split(",")
    eig_energies = np.zeros(len(splitstring))
    for i, str_energy in enumerate(splitstring):
        eig_energies[i] = float(str_energy)
    data = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x = data[:, 0]

    oranges = ("#fde961", "#f2b549", "#e78136", "#de5126", "#a7361f", "#562b22")
    blues = ("#aad8c0", "#4badc6", "#2e6fa7", "#1d357f", "#111450")

    plt.figure(figsize=(7, 4))
    if usetex: plt.rc('text', usetex=True)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.get_xaxis().tick_bottom()
    x_text = x[0] - 0.02

    for i in range(5):
        psi = data[:,i+1]
        y_scale = (2.6 + (i+1)**1.5)
        plt.plot(x, y_scale*psi + eig_energies[i], c=blues[-i-1])  # plot wave function shifted upward to its energy
        plt.hlines(eig_energies[i], -l/2, l/2, color=blues[-i-1], ls='--')  #, c=y_plot[0].get_color())
        if i == 0: ax.text(x_text-0.07, eig_energies[i], r"$E = E_1$", va="center", ha="center", fontsize=14)
        else: ax.text(x_text, eig_energies[i], r"$E_{{{}}} = {{{}}}E_1$".format(i+1, (i+1)**2), va="center", ha="right", fontsize=14)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi > 0., color=blues[-i-1], alpha=0.7)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi < 0., color=blues[-i-1], alpha=0.7)
    plt.arrow(-0.5*l, 0, 0, 1.15*eig_energies[4], facecolor="#000000", head_width=0.025, head_length=8, zorder=10)
    plt.arrow(0.5*l, 0, 0, 1.15*eig_energies[4], facecolor="#000000", head_width=0.025, head_length=8, zorder=10)

    x_scale = 0.7
    plt.xlim((-x_scale * l, x_scale * l))
    plt.xlabel(r"Position $x$", fontsize=13)
    plt.title("IPW Wave Functions with Finite Difference Method", fontsize=18)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "ipw-fd-plot_.png", dpi=200)
    plt.show()


def plot_ipw_fd_errors():
    l = 1.0
    filename = "ipw-fd-{:.1f}-{}.csv".format(l, 100)
    data1 = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x1 = data1[:,0]
    filename = "ipw-fd-{:.1f}-{}.csv".format(l, 1000)
    data2 = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x2 = data2[:,0]
    filename = "ipw-fd-{:.1f}-{}.csv".format(l, 10000)
    data3 = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x3 = data3[:,0]

    oranges = ("#e78136", "#de5126", "#a7361f", "#562b22")
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))
    if usetex: plt.rc('text', usetex=True)
    for i in range(4):
        if i == 0:
            ax = axes[0][0]
            ax.set_title("Ground State")
        elif i == 1:
            ax = axes[0][1]
            ax.set_title("First Excited State")
        elif i == 2:
            ax = axes[1][0]
            ax.set_title("Second Excited State")
        else:
            ax = axes[1][1]
            ax.set_title("Third Excited State")
        remove_spines(ax)

        if i % 2 == 0:
            ax.set_ylabel("Absolute Error")
        if i >= 2:
            ax.set_xlabel("Position $x$")
        plt.xlim((-l/1.5, l/1.5))
        ax.set_yscale("log")

        psi1 = data1[:,i+1]
        error1 = np.abs(psi1 - get_ipw_psi_analytic(x1, i+1))
        psi2 = data2[:,i+1]
        error2 = np.abs(psi2 - get_ipw_psi_analytic(x2, i+1))
        psi3 = data3[:,i+1]
        error3 = np.abs(psi3 - get_ipw_psi_analytic(x3, i+1))
        ax.plot(x1, error1, ls=':', c=oranges[i], linewidth=3, label="$N={}$".format(100))
        ax.plot(x2, error2, ls=':', c=oranges[i], linewidth=3, label="$N={}$".format(1000))
        ax.plot(x3, error3, ls=':', c=oranges[i], linewidth=3, label="$N={}$".format(10000))
        ax.legend(framealpha=0.95, loc="lower left")

    plt.suptitle("Finite Difference Errors for the IPW", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    if save_figures: plt.savefig(figure_dir + "ipw-fd-errors_.png", dpi=200)
    plt.show()


def plot_fpw_shoot():
    l = 1.0
    filename = "fpw-shoot-sym-asym-{:.1f}.csv".format(l)
    with open(data_dir + filename) as f:
        first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
        splitstring = first_line.split(",")
    eig_energies = np.zeros(len(splitstring))
    for i, str_energy in enumerate(splitstring):
        eig_energies[i] = float(str_energy)
    data = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x = data[:,0]

    oranges = ("#fde961", "#f2b549", "#e78136", "#de5126", "#a7361f", "#562b22")

    plt.figure(figsize=(7, 4))
    if usetex: plt.rc('text', usetex=True)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.get_xaxis().tick_bottom()
    x_text = -1.8

    plt.plot(x, V_fpw(x, l=l), drawstyle='steps-mid', c='k', alpha=0.85)  # plot potential
    ax.text(1.2, 50, r"$V_0 = 50$", va="bottom", ha="center", fontsize=14)
    for i in range(np.shape(data)[1] - 1):
        psi = data[:,i+1]
        y_scale = (2.5 + (i+1)**1.25)
        plt.plot(x, y_scale*psi + eig_energies[i], c=oranges[-i-1])  # plot wave function shifted upward to its energy
        plt.hlines(eig_energies[i], -l/2, l/2, color=oranges[-i-1], ls='--')  #, c=y_plot[0].get_color())
        ax.text(x_text, eig_energies[i], r"$E_{{{}}} \approx {{{:.2f}}}$".format(i+1, eig_energies[i]), va="center", ha="center", fontsize=14)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi > 0., color=oranges[-i-1], alpha=0.7)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi < 0., color=oranges[-i-1], alpha=0.7)

    plt.xlim((-1.5*l, 1.5*l))
    plt.xlabel(r"Position $x$", fontsize=13)
    plt.title("FPW Wave Functions with Shooting Method", fontsize=18)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "fpw-shoot-plot.png", dpi=200)
    plt.show()


def plot_fpw_fd():
    l, V0, N = 1.0, 100, 1000
    filename = "fpw-fd-{:.1f}-{:.0f}-{}.csv".format(l, V0, N)
    with open(data_dir + filename) as f:
        first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
        splitstring = first_line.split(",")
    eig_energies = np.zeros(len(splitstring))
    for i, str_energy in enumerate(splitstring):
        eig_energies[i] = float(str_energy)
    data = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x = data[:,0]

    oranges = ("#fde961", "#f2b549", "#e78136", "#de5126", "#a7361f", "#562b22", "#562b22")
    # blues = ("#8fcdc2", "#3997bf", "#244d90", "#141760")
    blues = ("#aad8c0", "#4badc6", "#2e6fa7", "#1d357f", "#111450")
    # blues = ("#aad8c0", "#4badc6", "#2e6fa7", "#1d357f", "#111450", "#111450", "#111450")

    plt.figure(figsize=(7, 4))
    if usetex: plt.rc('text', usetex=True)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.get_xaxis().tick_bottom()
    x_text = -1.8

    plt.plot(x, V_fpw(x, l=l, V0=V0), drawstyle='steps-mid', c='k', alpha=0.85)  # plot potential
    ax.text(1.2, V0, r"$V_0 = {:.0f}$".format(V0), va="bottom", ha="center", fontsize=14)
    for i in range(np.shape(data)[1] - 1):
        psi = data[:,i+1]
        if (V0 == 100 or V0 == 200) and i == 1:
            psi = -1.0 * psi
        y_scale = (2.5 + (i+1)**1.25)
        plt.plot(x, y_scale*psi + eig_energies[i], c=blues[-i-1])  # plot wave function shifted upward to its energy
        plt.hlines(eig_energies[i], -l/2, l/2, color=blues[-i-1], ls='--')  #, c=y_plot[0].get_color())
        ax.text(x_text, eig_energies[i], r"$E_{{{}}} \approx {{{:.2f}}}$".format(i+1, eig_energies[i]), va="center", ha="center", fontsize=14)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi > 0., color=blues[-i-1], alpha=0.7)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi < 0., color=blues[-i-1], alpha=0.7)

    plt.xlim((-1.5*l, 1.5*l))
    plt.xlabel(r"Position $x$", fontsize=13)
    plt.title("FPW Wave Functions with Finite Differences", fontsize=18)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "fpw-fd-plot-{:.0f}_.png".format(V0), dpi=200)
    plt.show()


def plot_fpw_solve_bvp():
    l = 1.0
    filename = "fpw-solve-bvp-{:.1f}.csv".format(l)
    with open(data_dir + filename) as f:
        first_line = f.readline().strip().replace("# Energy Eigenvalues:,", "")  # read energies from first file line
        splitstring = first_line.split(",")
    eig_energies = np.zeros(len(splitstring))
    for i, str_energy in enumerate(splitstring):
        eig_energies[i] = float(str_energy)
    data = np.loadtxt(data_dir + filename, delimiter=',', skiprows=2)
    x = data[:,0]

    # oranges = ("#fde961", "#f2b549", "#e78136", "#de5126", "#a7361f", "#562b22")
    neons = ("#ec8689", "#c93475", "#661482", "#350b52")

    plt.figure(figsize=(7, 4))
    if usetex: plt.rc('text', usetex=True)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.get_xaxis().tick_bottom()
    x_text = -1.8

    plt.plot(x, V_fpw(x, l=l), drawstyle='steps-mid', c='k', alpha=0.85)  # plot potential
    ax.text(1.2, 51, r"$V_0 = 50$", va="bottom", ha="center", fontsize=14)
    for i in range(np.shape(data)[1] - 1):
        psi = data[:,i+1]
        y_scale = (2.5 + (i+1)**1.25)
        plt.plot(x, y_scale*psi + eig_energies[i], c=neons[-i-1])  # plot wave function shifted upward to its energy
        plt.hlines(eig_energies[i], -l/2, l/2, color=neons[-i-1], ls='--')  #, c=y_plot[0].get_color())
        ax.text(x_text, eig_energies[i], r"$E_{{{}}} \approx {{{:.2f}}}$".format(i+1, eig_energies[i]), va="center", ha="center", fontsize=14)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi > 0., color=neons[-i-1], alpha=0.7)
        ax.fill_between(x, psi * y_scale + eig_energies[i], eig_energies[i], psi < 0., color=neons[-i-1], alpha=0.7)

    plt.xlim((-1.5*l, 1.5*l))
    plt.xlabel(r"Position $x$", fontsize=13)
    plt.rc('text', usetex=True)
    plt.suptitle(r"FPW Wave Functions with $\texttt{solve\_bvp}$", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if save_figures: plt.savefig(figure_dir + "fpw-solve-bvp-plot_.png", dpi=200)
    plt.show()


def plot_fpw_shooting_roots(l=1.0, V0=50.0, n_points=300):
    a, b = -2*l, 2*l
    x = np.linspace(a, b, n_points)

    initial_state = [0.0, 0.001]
    test_energies = np.arange(0.0, V0, 1.0)  # energies to test with shooting method

    psi_endpoint_values = np.zeros(len(test_energies))
    for i, E in enumerate(test_energies):
        psi_endpoint_values[i] = get_fpw_endpoint(initial_state, x, E)

    energy_roots = get_fpw_analytic_energies(V0=V0)  # cheat and use analytic energies to save computation time

    fix, axes = plt.subplots(1, 2, figsize=(7, 3))
    ax = axes[0]
    ax.plot(test_energies, psi_endpoint_values, ls='-', c=color_blue, marker='.')
    ax.set_ylabel(r"Right Endpoint $\psi(x_n)$", fontsize=12)
    ax.set_xlabel("Energy $E$", fontsize=12)
    ax.set_title("Macroscopic View")
    ax.grid()

    ax = axes[1]
    y_max = 10
    ax.set_ylim((-y_max, y_max))
    ax.set_xlabel("Energy $E$", fontsize=12)
    ax.plot(test_energies, psi_endpoint_values, ls='-', c=color_blue, marker='.')
    for E in energy_roots:
        ax.text(E, 0.8, "{:.2f}".format(E), ha="center", va="bottom", fontsize=11,
                bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.set_title("Zoomed View")
    ax.grid()

    plt.suptitle("Shooting for FPW Energy Roots", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    if save_figures: plt.savefig(figure_dir + "shoot-fpw-roots_.png", dpi=200)
    plt.show()

# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------


def practice():
    # filename = "ipw-shoot-sym-asym-1.0.csv"
    # with open(data_dir + filename) as f:
    #     first_line = f.readline().strip()
    #     first_line = first_line.replace("# Energy Eigenvalues:,", "")
    #     splitstring = first_line.split(",")
    #     print(splitstring)
    # N = 5
    # a = 2*np.ones(N)
    # b = -1*np.ones(N-1)
    # b[0] = 10
    # c = -1*np.ones(N-1)
    # A = np.diag(a) + np.diag(b, 1) + np.diag(c, -1)
    # print(A)
    eig, _ = fpw_fd1()
    print(eig)


def run():
    # practice()
    # generate_ipw_shoot_sym_asym_solutions()
    # generate_ipw_fd_solutions(N=100)
    # generate_fpw_fd_solutions(N=1000, V0=100)
    # generate_fpw_shoot_sym_asym_solutions()
    # generate_fpw_bvp_solutions()
    # find_ipw_energy_errors()
    # find_fpw_energy_errors()
    #
    # plot_ipw_shoot()
    # plot_ipw_shoot_analytic_comparison()
    # plot_ipw_shoot_errors()
    # plot_ipw_fd()
    plot_ipw_fd_errors()
    #
    # plot_fpw_shoot()
    # plot_fpw_fd()
    # plot_fpw_solve_bvp()
    # plot_fpw_shooting_roots()


run()
