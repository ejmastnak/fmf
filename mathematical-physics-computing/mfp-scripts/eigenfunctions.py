import numpy as np
from math import factorial
import matplotlib.pyplot as plt

figure_directory = data_dir = "../3-eigenvalues/figures/"
save_figures = True
color_above = "#f86519"  # orange
color_below = "#1494c0"  # blue
color_V = "#888888"  # grey


def get_norm(n):  # normalization constant of nth wavefunction
    return 1. / np.sqrt(np.sqrt(np.pi) * 2 ** n * factorial(n))


def get_energy(n):  # energy in \hbar \omega units of nth state
    return n + 0.5


def get_hermite_poly(N):
    """
    Returns the coefficients of the Nth physicist's Hermite polynomials
    Finds coefficients of succesive terms by recursion using H_[n] = 2xH_[n-1] - 2(n-1)H_[n-2]
    :param n:
    :return:
    """
    if N == 0:
        return np.poly1d([1.])
    elif N ==1:
        return np.poly1d([2., 0.])
    factor_2x = np.poly1d([2., 0.0])  # represents 2x term in the formula 2xH_[n-1] - 2(n-1)H_[n-2]

    previous2 = np.poly1d([1., ])  # two terms back
    previous1 = np.poly1d([2., 0.0])  # previous term
    current = np.poly1d([0])  # initially empty
    for n in range(2, N + 1):
        current = factor_2x*previous1 - 2*(n-1)*previous2
        previous2 = previous1
        previous1 = current

    return current


def get_psi_qho(n, q):
    """
    Returns the harmonic oscillator's nth wavefunction on the provided values of q
    :param n:
    :param q:
    :return:
    """
    return get_norm(n) * get_hermite_poly(n)(q) * np.exp(-q * q / 2.)


def get_psi_expanded(eigenvector, q):
    """
    Returns an oscillating system's eigenfunction for the inputed eigenvector by expanding the wavefunction
     in the unperturbed QHO basis (i.e. using the Hermite polynomials)
    """
    psi = np.zeros(len(q))
    for i in range(0, len(eigenvector)):
        psi += eigenvector[i] * get_psi_qho(i, q)

    return psi

def get_V_qho(q):
    """
    Potential energy of regular QHO
    """
    return q**2 / 2


def get_V_perturbed(q, lambd):
    """
    Potential energy of perturbed QHO
    """
    return 0.5*(q**2) + lambd*(q**4)


def get_V_double_well(q):
    """
    Potential energy of double-well QHO
    """
    return -2*(q**2) + 0.1*(q**4)


def scale_and_fill(axis,x, y, y_scaling=1.0, yoffset=0):
    """
    Auxiliary function for plotting. Placed here to reduce clutter
    Plots (x, y*y_scaling) on the given axis with offset yoffset.
    and fills areas between function and axis with color
    """
    axis.plot(x, y * y_scaling + yoffset, color=color_above)
    axis.fill_between(x, y * y_scaling + yoffset, yoffset, y > 0., color=color_above, alpha=0.5)
    axis.fill_between(x, y * y_scaling + yoffset, yoffset, y < 0., color=color_below, alpha=0.5)


def plot_qho(N, plot_psi):
    # Some appearance settings
    x_padding = 1.3  # pads the x axis to fit nicel in frame
    y_scale = 0.7  # Scale down the wavefunctions' y values so they don't overlap
    color_above = "#f86519"  # orange
    color_below = "#1494c0"  # blue

    fig, ax = plt.subplots()
    qmax = np.sqrt(2. * get_energy(N + 0.5))  # determine q limits
    qmin = -qmax
    xmin, xmax = x_padding * qmin, x_padding * qmax

    q = np.linspace(qmin, qmax, 500)
    V = get_V_qho(q)
    ax.plot(q, V, color=color_V, linestyle='-', linewidth=1.5, zorder=0)  # plots the potential V(q)

    # Plot each of the wavefunctions (or probability densities)
    for n in range(N + 1):
        psi_n = get_psi_qho(n, q)
        E_n = get_energy(n)
        if plot_psi:
            scale_and_fill(ax, q, psi_n, y_scaling=y_scale, yoffset=E_n)
        else:
            scale_and_fill(ax, q, psi_n ** 2, y_scaling=y_scale * 1.5, yoffset=E_n)

        ax.text(s=r'$\frac{{{}}}{{2}}\hbar\omega$'.format(2 * n + 1), x=qmax + 0.2, y=E_n, va='center')  # labels energy, E = (2n +1)/(2) * hbar.omega.
        ax.text(s=r'$n={}$'.format(n), x=qmin - 0.2, y=E_n, va='center', ha='right')  # labels quantum numbers

    # The top of the plot, plus a bit.
    ymax = get_energy(N) + 0.5

    ax.set_xlabel('$q$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.spines['left'].set_position('center')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if plot_psi:
        ax.set_title("Unperturbed QHO Eigenfunctions")
    else:
        ax.set_title("Unperturbed QHO Probability Density")

    if save_figures:
        if plot_psi:
            plt.savefig(figure_directory+'eigfunc-qho-psi_.png', dpi=200)
        else:
            plt.savefig(figure_directory+'eigfunc-qho-prob_.png', dpi=200)

    plt.show()


def plot_double_well_even(eigenvalues, eigenvectors, plot_psi):
    """
    Plots eigenfunctions and eigenvalues of double-well oscillator superimposed on potential.
    Input an nx1 vector of energy eigenvalues
    and an nxn matrix of corresponding eigenvectors
    """
    N = len(eigenvalues)

    # Some appearance settings
    x_padding = 1  # pads the x axis to fit nicel in frame
    y_scale = 1.8  # Scale up wavefunctions' y values to fill frame

    fig, ax = plt.subplots()

    qmax = 4.5  # determine q limits
    qmin = -qmax
    q = np.linspace(qmin, qmax, 500)
    V = get_V_double_well(q)

    ax.plot(q, V, color=color_V, linestyle='-', linewidth=1.5, zorder=0)  # plots the potential V(q)

    # Plot each of the wavefunctions (or probability densities)
    for n in range(0, N, 2):
        psi_n = get_psi_expanded(eigenvectors[:,n], q)  # get_psi_qho(n, q)
        E_n = eigenvalues[n]
        if plot_psi:
            scale_and_fill(ax, q, psi_n, y_scaling=y_scale, yoffset=E_n)
        else:
            scale_and_fill(ax, q, psi_n ** 2, y_scaling=y_scale * 2.5, yoffset=E_n)

        ax.text(s=r'$\frac{{{}}}{{2}}\hbar\omega$'.format(2 * n + 1), x=qmax + 0.2, y=E_n, va='center')  # labels energy, E = (2n +1)/(2) * hbar.omega.
        ax.text(s=r'$n={}$'.format(n), x=qmin - 0.2, y=E_n, va='center', ha='right')  # labels quantum numbers

    ax.set_xlabel('$q$')
    ax.set_xlim(qmin - x_padding, qmax + x_padding)
    ax.spines['left'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_psi:
        ax.set_title("Even Double-Well Eigenfunctions")
    else:
        ax.set_title("Even Double-Well Probability Density")

    if save_figures:
        if plot_psi:
            plt.savefig(figure_directory + 'eigfunc-dw-even-psi_.png', dpi=200)
        else:
            plt.savefig(figure_directory + 'eigfunc-dw-even-prob_.png', dpi=200)

    plt.show()


def plot_double_well_odd_even(eigenvalues, eigenvectors, plot_psi):
    """
    Plots eigenfunctions and eigenvalues of double-well oscillator superimposed on potential.
    Input an nx1 vector of energy eigenvalues
    and an nxn matrix of corresponding eigenvectors
    """
    N = len(eigenvalues)

    # Some appearance settings
    x_padding = 1  # pads the x axis to fit nicel in frame
    y_scale = 1.8  # Scale up wavefunctions' y values to fill frame

    fig, axes = plt.subplots(1,2, figsize=(12,5))

    ax = axes[0]  # plot even eigenfunctions

    qmax = 4.5  # determine q limits
    qmin = -qmax
    q = np.linspace(qmin, qmax, 500)
    V = get_V_double_well(q)

    ax.plot(q, V, color=color_V, linestyle='-', linewidth=1.5, zorder=0)  # plots the potential V(q)

    # Plot each of the wavefunctions (or probability densities)
    for n in range(0, N, 2):
        psi_n = get_psi_expanded(eigenvectors[:,n], q)  # get_psi_qho(n, q)
        E_n = eigenvalues[n]
        if plot_psi:
            scale_and_fill(ax, q, psi_n, y_scaling=y_scale, yoffset=E_n)
        else:
            scale_and_fill(ax, q, psi_n ** 2, y_scaling=y_scale * 2.5, yoffset=E_n)

        ax.text(s=r'$\frac{{{}}}{{2}}\hbar\omega$'.format(2 * n + 1), x=qmax + 0.2, y=E_n, va='center')  # labels energy, E = (2n +1)/(2) * hbar.omega.
        ax.text(s=r'$n={}$'.format(n), x=qmin - 0.2, y=E_n, va='center', ha='right')  # labels quantum numbers

    ax.text(s='$V(q)$ = -$2q^{2} + q^{4}/10$', x=qmax, y=eigenvalues[-1]+1.0, ha='center', va='center')  # labels energy, E = (2n +1)/(2) * hbar.omega.

    ax.set_xlabel('$q$')
    ax.set_xlim(qmin - x_padding, qmax + x_padding)
    ax.spines['left'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if plot_psi:
        ax.set_title("Even Double-Well Eigenfunctions")
    else:
        ax.set_title("Even Double-Well Probability Densities")

    # Plot odd eigenfunctions
    ax = axes[1]

    qmax = 4.5  # determine q limits
    qmin = -qmax
    q = np.linspace(qmin, qmax, 500)
    V = get_V_double_well(q)

    ax.plot(q, V, color=color_V, linestyle='-', linewidth=1.5, zorder=0)  # plots the potential V(q)

    # Plot each of the wavefunctions (or probability densities)
    for n in range(1, N, 2):
        psi_n = get_psi_expanded(eigenvectors[:, n], q)  # get_psi_qho(n, q)
        E_n = eigenvalues[n]
        if plot_psi:
            scale_and_fill(ax, q, psi_n, y_scaling=y_scale, yoffset=E_n)
        else:
            scale_and_fill(ax, q, psi_n ** 2, y_scaling=y_scale * 2.5, yoffset=E_n)

        ax.text(s=r'$\frac{{{}}}{{2}}\hbar\omega$'.format(2 * n + 1), x=qmax + 0.2, y=E_n,
                va='center')  # labels energy, E = (2n +1)/(2) * hbar.omega.
        ax.text(s=r'$n={}$'.format(n), x=qmin - 0.2, y=E_n, va='center', ha='right')  # labels quantum numbers

    ax.text(s='$V(q)$ = -$2q^{2} + q^{4}/10$', x=qmax, y=eigenvalues[-1]+1.0, ha='center', va='center')  # labels energy, E = (2n +1)/(2) * hbar.omega.

    ax.set_xlabel('$q$')
    ax.set_xlim(qmin - x_padding, qmax + x_padding)
    ax.spines['left'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if plot_psi:
        ax.set_title("Odd Double-Well Eigenfunctions")
    else:
        ax.set_title("Odd Double-Well Probability Densities")



    plt.tight_layout()

    if save_figures:
        if plot_psi:
            plt.savefig(figure_directory + 'eigfunc-dw-subplots-psi_.png', dpi=200)
        else:
            plt.savefig(figure_directory + 'eigfunc-dw-subplots-prob_.png', dpi=200)

    plt.show()


def run_plot():
    N = 9
    data_dir = "../3-eigenvalues/data/"
    eigvals = np.loadtxt(data_dir + "values-double-well/eig-100.csv")[0: N]
    eigvecs = np.loadtxt(data_dir + "vectors-double-well/eig-100.csv", delimiter=',')[0:N, 0:N]
    # plot_double_well_even(eigvals, eigvecs, False)
    plot_double_well_odd_even(eigvals, eigvecs, False)
    # plot_qho(N, True)

# plot_qho(6)
run_plot()
