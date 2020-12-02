"""
Script for solving the one dimensional Schroedinger equation numerically.
Numerical integration method used is the fourth order Runge Kutta. Counts the nodes of the wave function and determins the harmonic. Then refines the solution until proper energy is found.
Potentials:
Infinite Potential Well
V(x_<0) = inf, V(x_=0,1) = 0, V(x_>1) = inf
Harmonic Oscillator: V(x_) = x_**2
Radial Hydrogen Atom Coulomb attraction: V(r) = 2/r - (L(L+1))/(r**2) a.u.
"""

import numpy as np
import scipy
from scipy import integrate
from scipy.optimize import newton
import matplotlib.pyplot as plt

save_figures = False


def schro_ipw(state, x, E):
    """
    One-liner stationary Schrodinger equation for a particle in a infinite potential well
    psi'' = -2*m*E*psi(x)/(hbar^2)
    Uses natural units, so m = hbar = 1 and also deletes the two!!! TODO edit me

    :param x: position
    :param state: 2-element array containing [psi, psi']
    :param E: particle's energy as a 1-element ndarray with shape (1,)
    :return: 2-element array containing [psi', psi'']
    """
    return_state = np.zeros(2)
    return_state[0] = state[1]
    return_state[1] = -E*state[0]
    return return_state


def schro_hydrogen(state, r, L, E):
    """Odeint calls routine to solve Schroedinger equation of the Hydrogen atom. """
    return_state = np.zeros(2)
    return_state[0] = state[1]
    return_state[1] = state[0] * ((L * (L + 1)) / (r ** 2) - 2. / r - E)
    return return_state


def schro(state, x, V, E):
    """Return one dim Schroedinger eqation with Potential V."""
    return_state = np.zeros(2)
    return_state[0] = state[1]
    return_state[1] = -(E-V)*state[0]
    return return_state


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
        h = x[i+1] - x[i]
        k1 = h*f(states[i], x[i], E)
        k2 = h*f(states[i] + 0.5*k1, x[i] + 0.5*h, E)
        k3 = h*f(states[i] + 0.5*k2, x[i] + 0.5*h, E)
        k4 = h*f(states[i] + k3, x[i+1], E)
        states[i+1] = states[i] + (k1 + 2.0*(k2 + k3) + k4) / 6.0
    return states


def rk4(f, initial_state, x, V, E):
    """
    Fourth-order Runge-Kutta method to solve psi’=f(psi,x) with psi(x[0])=psi0.
    Integrates function f with inital values psi0 and potenital V numerically.
    Output is possible multidimensional (in psi) array with len(x).
    """
    n = len(x)
    states = np.array(n * [initial_state])  # contains two columns, one for psi and one for psi'
    for i in range(n-1):
        h = x[i+1] - x[i]
        k1 = h*f(states[i], x[i], V[i], E)
        k2 = h*f(states[i] + 0.5*k1, x[i] + 0.5*h, V[i], E)
        k3 = h*f(states[i] + 0.5*k2, x[i] + 0.5*h, V[i], E)
        k4 = h*f(states[i] + k3, x[i+1], V[i], E)
        states[i+1] = states[i] + (k1 + 2.0*(k2 + k3) + k4) / 6.0
    return states


def get_endpoint_values_me(fun, initial_state, x, energies):
    """
    Finds a solution for psi using rk4 on the position grid x for each value of energies
    Returns the value of psi at the rightmost endpoint of each solution
    """
    psi_endpoint_values = np.zeros(len(energies))  # preallocate
    for i, energy in enumerate(energies):
        psi = rk4_schro(fun, initial_state, x, energy)
        psi_endpoint_values[i] = psi[-1][0]  # the value of psi at the endpoint

    return psi_endpoint_values


def get_endpoint_values(fun, initial_state, x, V, energies):
    """
    Finds a solution for psi using rk4 on the position grid x for each value of energies
    Returns the value of psi at the rightmost endpoint of each solution
    """
    psi_endpoint_values = np.zeros(len(energies))  # preallocate
    for i, energy in enumerate(energies):
        psi = rk4(fun, initial_state, x, V, energy)
        psi_endpoint_values[i] = psi[-1][0]  # the value of psi at the endpoint

    return psi_endpoint_values


def shoot1_me(E, func, psi0, x):
    """Helper function for optimizing resuts."""
    psi = rk4_schro(func, psi0, x, E)
    return psi[-1][0]


def shoot1(E, func, psi0, x, V):
    """Helper function for optimizing resuts."""
    psi = rk4(func, psi0, x, V, E)
    return psi[-1][0]


def shoot_ode(E, psi_init , x, L):
    """Helper function for optimizing resuts."""
    sol = integrate.odeint(schro_hydrogen, psi_init, x, args=(L,E))
    return sol[-1][0]


def find_ipw_energy_eig(fun, initial_state, x, energies):
    """ Finds first fiew energy eigenvalues of a particle in an infinite potential well """
    psi_endpoint_values = get_endpoint_values_me(fun, initial_state, x, energies)  # values of psi(x[-1]) for each energy
    zero_crossings = np.where(np.diff(np.sign(psi_endpoint_values)))[0]  # indeces where endpoint values cross zero
    energy_list = np.zeros(len(zero_crossings))  # preallocate
    for i, cross in enumerate(zero_crossings):
        energy_list[i] = newton(shoot1_me, energies[cross], args=(fun, initial_state, x))
    return np.asarray(energy_list)


def find_energy_eig(fun, psi0, x, V, energies):
    """Optimize energy value for function using brentq."""
    psi_endpoint_values = get_endpoint_values(fun, psi0, x, V, energies)  # values of psi(x[-1]) for each energy
    zero_crossings = np.where(np.diff(np.sign(psi_endpoint_values)))[0]  # indeces where endpoint values cross zero
    energy_list = np.zeros(len(zero_crossings))  # preallocate
    for i, cross in enumerate(zero_crossings):
        energy_list[i] = newton(shoot1, energies[cross], args=(fun, psi0, x, V))
    return np.asarray(energy_list)


def normalize(output_wavefunc):
    """A function to roughly normalize the wave function."""
    normal = max(output_wavefunc)
    return output_wavefunc*(1/normal)


def shoot_potwell(initial_state, h_):
    """ Shooting method for infinte potential well.
        Returns the numerical and analytical solution as arrays.
    """
    l = 1.0
    n_points = 200
    x = np.linspace(0.0, l, n_points)
    V_ipw = np.zeros(len(x))
    test_energies = np.arange(1.0, 100.0, 5.0)  # energies to test with shooting method

    # eigE = find_energy_eig(schro, initial_state, x, V_ipw, test_energies)
    eig_energies = find_ipw_energy_eig(schro_ipw, initial_state, x, test_energies)
    print(eig_energies)

    ipw_out_list = []
    for E_n in eig_energies:
        psi_n = rk4(schro, initial_state, x, V_ipw, E_n)[:, 0]
        ipw_out_list.append(normalize(psi_n))

    out_arr = np.asarray(ipw_out_list)

    # analytical solution for IPW
    state_numbers = np.linspace(1.0, len(eig_energies), len(eig_energies))
    ipw_sol_ana = []
    for n in state_numbers:
        ipw_sol_ana.append(np.sin(n*np.pi*x))
    ipw_sol_ana_arr = np.asarray(ipw_sol_ana)
    return x, out_arr, ipw_sol_ana_arr


def shoot_qho(psi_init, h_):
    """Shooting method for quantum harmonic oscillator.
    500 mesh points.
    Returns the numerical and analytical solution as arrays. """
    x_arr_qho = np.arange(-5.0, 5.0+h_, h_)
    V_qho = x_arr_qho**2
    E_arr = np.arange(1.0, 15.0, 1.0)
    eigEn = find_energy_eig(schro, psi_init, x_arr_qho, V_qho, E_arr)
    qho_out_list = []
    for EN in eigEn:
        out = rk4(schro, psi_init, x_arr_qho, V_qho, EN)
        qho_out_list.append(normalize(out[:, 0]))
    qho_out_arr = np.asarray(qho_out_list)
    # analytical solution for QHO
    qho_sol_ana_0 = np.exp(-(x_arr_qho**2)/2)
    qho_sol_ana_1 = np.sqrt(2.0)*(x_arr_qho)*np.exp(-(x_arr_qho)**2/2)*(-1)
    qho_sol_ana_2 = (1.0/np.sqrt(2.0))*(2.0*(x_arr_qho)**2-1.0)*np.exp(-(x_arr_qho)**2/2)
    qho_sol_list = []
    qho_sol_list.append(qho_sol_ana_0)
    qho_sol_list.append(qho_sol_ana_1)
    qho_sol_list.append(qho_sol_ana_2)
    return x_arr_qho, qho_out_arr, np.asarray(qho_sol_list)


def plot_wavefunction(fig, title_string, x_arr, num_arr, ana_arr, axis_list):
    """Output plots for wavefunctions."""
    plt.cla()  # clear axis
    plt.clf()  # clear figure
    plt.plot(x_arr, num_arr, 'b:', linewidth=4, label=r"$\Psi(\hat{x})_{num}$")
    plt.plot(x_arr, normalize(ana_arr), 'r-', label=r"$\Psi(\hat{x})_{ana}$")
    plt.ylabel(r"$\Psi(\hat{x})$", fontsize=16)
    plt.xlabel(r'$\hat{x}$', fontsize=16)
    plt.legend(loc='best', fontsize='small')
    plt.axis(axis_list)
    plt.title(title_string)
    plt.grid()
    plt.show()
    if save_figures: fig.savefig("plots/wavefunc_"+title_string+".png")


def run():
    initial_state = [0.0, 1.0]  # asymmetric "sine" initial condition
    h_ = 1.0/200.0  # stepsize for range arrays
    fig = plt.figure()
    ipw_x, ipw_num, ipw_ana = shoot_potwell(initial_state, h_,)
    # qho_x, qho_num, qho_ana = shoot_QuantumHarmonicOscillator(psi_init, h_)

    print("IPW Shooting Method")
    plot_wavefunction(fig, "IPW Ground State", ipw_x, ipw_num[0, :], ipw_ana[0, :], [-0.1, 1.1, -0.2, 1.2])
    plot_wavefunction(fig, "IPW 1st Excited State", ipw_x, ipw_num[1, :], ipw_ana[1, :], [-0.1, 1.1, -0.2, 1.2])
    plot_wavefunction(fig, "IPW 2nd Excited State", ipw_x, ipw_num[2, :], ipw_ana[2, :], [-0.1, 1.1, -0.2, 1.2])

    # print("QHO shooting")
    # plot_wavefunction(fig, "QHO Ground State", qho_x, qho_num[0, :], qho_ana[0, :], [-5.2, 5.2, -1.2, 1.2])
    # plot_wavefunction(fig, "1st Excited State", qho_x, qho_num[1, :], qho_ana[1, :], [-5.2, 5.2, -1.2, 1.2])
    # plot_wavefunction(fig, "2nd Excited State", qho_x, qho_num[2, :], qho_ana[2, :], [-5.2, 5.2, -1.2, 1.2])

run()
