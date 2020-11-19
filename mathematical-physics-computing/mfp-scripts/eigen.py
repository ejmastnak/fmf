from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from statistics import mean
from itertools import groupby



# Start global constants
figure_directory = "../3-eigenvalues/figures/"
data_dir = "../3-eigenvalues/data/"

log_to_console = False      # controls whether to log outputs
save_figures = True      # controls whether to save figures to avoid accidental overwrites


# End global constants

def compare_q1q2q4():
    """
    Plots eig1, eig2, eig4 and their average on one plot
    :return:
    """
    dir = data_dir + "q1q2q4/values/"
    N = 100
    for lambd in (0.1, 0.4, 0.7, 1.0):
        eig1 = np.loadtxt(dir + "eig-{:.1f}-{}-H1.csv".format(lambd, N))
        eig2 = np.loadtxt(dir + "eig-{:.1f}-{}-H2.csv".format(lambd, N))
        eig4 = np.loadtxt(dir + "eig-{:.1f}-{}-H4.csv".format(lambd,N))
        avg = np.mean([eig1, eig2, eig4], axis=0)
        plt.xlabel("Index $n$")
        plt.yscale("log")
        plt.ylabel("Eigenvalue $\lambda$")
        plt.plot(eig1, label="(Q1)$^4$")
        plt.plot(eig2, label="(Q2)$^2$")
        plt.plot(eig4, label="Q4")
        plt.plot(avg, label="Average")
        plt.title("N = {} and $\lambda$ = {:.1f}".format(N, lambd))
        plt.legend()
        plt.show()


def compare_q1q2q4_difference():
    """
    Plots the differences of eig1, eig2, eig4 from the value average
    :return:
    """
    dir = data_dir + "q1q2q4/values/"
    N = 500

    fig, ax = plt.subplots(figsize=(8, 5.5))
    i = 1
    colors = ("#74c1c4", "#2e6fa7", "#161d63")

    for lambd in (0.1, 0.4, 0.7, 1.0):
        plt.subplot(2, 2, i)

        eig1 = np.loadtxt(dir + "eig-{:.1f}-{}-H1.csv".format(lambd, N))[0:int(0.5*N)]
        eig2 = np.loadtxt(dir + "eig-{:.1f}-{}-H2.csv".format(lambd, N))[0:int(0.5*N)]
        eig4 = np.loadtxt(dir + "eig-{:.1f}-{}-H4.csv".format(lambd,N))[0:int(0.5*N)]
        avg = np.mean([eig1, eig2, eig4], axis=0)
        if i >= 3: plt.xlabel("Eigenvalue Index $n$")
        if i%2 == 1: plt.ylabel("Abs. Deviation From Average")
        plt.plot(np.abs(eig1-avg), color=colors[2], label="(Q$_{1}$)$^4$")
        plt.plot(np.abs(eig2-avg), color=colors[0], label="(Q$_{2}$)$^2$")
        plt.plot(np.abs(eig4-avg), color=colors[1], label="Q$_{4}$")
        plt.title("$\lambda$ = {:.1f}".format(lambd))
        leg = plt.legend()  # increase legend line thickness
        for line in leg.get_lines():
            line.set_linewidth(3.0)

        i = i+1

    plt.tight_layout()
    plt.suptitle("Testing Eigenvalue Divergence for N = {}".format(N), fontsize=14)
    plt.subplots_adjust(top=0.89)
    if save_figures: plt.savefig(figure_directory + "q1q2q4_.png", dpi=200)
    plt.show()


def get_method_title(method_abbreviation):
    if method_abbreviation == "eig":
        return "eig"
    elif method_abbreviation == "HHqr":
        return "Householder"
    elif method_abbreviation == "qr":
        return "QR Iteration"
    elif method_abbreviation == "jac_max":
        return "Jacobi"
    else:
        return method_abbreviation


def get_method_label(method_abbreviation):
    if method_abbreviation == "eig":
        return "eig"
    elif method_abbreviation == "HHqr":
        return "HH-QR"
    elif method_abbreviation == "qr":
        return "QR"
    elif method_abbreviation == "jac_max":
        return "Jacobi"
    else:
        return method_abbreviation

def plot_eigenvalues():
    dir = data_dir + "values/"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    neons = ("#fce85d", "#eea444", "#e2692d", "#c23a1f", "#5e2c21")
    blues = ("#aad8c0", "#4badc6", "#2e6fa7", "#1d357f", "#111450")
    lambda_values = (0.0, 0.1, 0.4, 0.7, 1.0)
    max_lambda = np.max(lambda_values)
    i = 1
    for method in ("eig", "HHqr", "qr", "jac_max"):
        if method == "jac_max": N = 100  # Jacobi run with N = 100
        else: N = 500

        plt.subplot(2, 2, i)
        plt.title(get_method_title(method) + " $N$ = {}".format(N), fontsize=11)
        if i >= 3: plt.xlabel("Index $n$")
        if i % 2 == 1: plt.ylabel("$E_{n}$ [$\hbar\omega$]")

        color_counter = 0
        cutoff_index = get_cutoff_index(max_lambda, N)
        for lambd in lambda_values:
            eig = np.loadtxt(dir + method + "-{:.1f}-{}.csv".format(lambd, N))[0:cutoff_index]  # only take non-divergent values
            if method == "jac_max": plt.plot(eig, color=neons[color_counter], linestyle='--', marker='.', label="$\lambda: {:.1f}$".format(lambd))
            else: plt.plot(eig, color=blues[color_counter], marker='.', label="$\lambda: {:.1f}$".format(lambd))
            color_counter += 1

        legend = plt.legend()  # increase legend line thickness
        for line in legend.get_lines():
            line.set_linewidth(3.0)

        i = i+1

    plt.tight_layout()
    plt.suptitle("Single-Well Eigenvalues", fontsize=14)
    plt.subplots_adjust(top=0.90)
    if save_figures: plt.savefig(figure_directory + "eigenvalues_.png", dpi=200)
    plt.show()


def plot_eigenvalues_double_well():
    dir1 = data_dir + "values/"
    dir2 = data_dir + "values-double-well/"

    fig, ax = plt.subplots(figsize=(8, 5.5))

    i = 1
    colors = ("#8fcdc2", "#3997bf", "#244d90", "#141760")

    for method in ("eig", "HHqr", "qr", "jac_max"):
        if method == "jac_max":
            N = 100
            cutoff = 50
        else:
            N = 500
            cutoff = 150

        plt.subplot(2, 2, i)
        plt.title(get_method_title(method) + " $N$ = {}".format(N), fontsize=11)
        if i >= 3: plt.xlabel("Eigenvalue Index $n$")
        if i % 2 == 1: plt.ylabel("$E_{n}$ [$\hbar\omega$]")

        eig_ref = np.loadtxt(dir1 + method + "-0.0-{}.csv".format(N))[0:cutoff]  # single well values
        indices_ref = np.linspace(1, len(eig_ref), len(eig_ref))

        eig2 = np.loadtxt(dir2 + method + "-{}.csv".format(N))[0:cutoff]
        indices2 = np.linspace(1, len(eig2), len(eig2))

        plt.plot(indices2, eig2, color=colors[i-1], marker='.', label=get_method_title(method), zorder=1)
        plt.plot(indices_ref, eig_ref, color='#CCCCCC', marker='.', label="single-well reference", zorder=0)
        plt.legend()

        i += 1

    plt.tight_layout()
    plt.suptitle("Double-Well Eigenvalues", fontsize=14)
    plt.subplots_adjust(top=0.90)
    if save_figures: plt.savefig(figure_directory + "eigenvalues-double-well_.png", dpi=200)
    plt.show()


def plot_error():
    """
    Plots error of each eigenvalue implementation relative to eig()
    :return:
    """
    dir = data_dir + "values/"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ("#3e0c5f", "#94247d", "#e86f83")  # neons
    markers = ("o", 's', 'd')

    lambda_values = (0.1, 0.4, 0.7, 1.0)
    max_lambda = np.max(lambda_values)

    i = 1
    for lambd in lambda_values:

        plt.subplot(2, 2, i)
        if i >= 3: plt.xlabel("Eigenvalue Index $n$")
        if i % 2 == 1: plt.ylabel("Error wrt eig [$\hbar\omega$]")
        plt.yscale("log")
        plt.title(" $\lambda$ = {:.1f}".format(lambd))

        eig_reference = np.loadtxt(dir + "eig-{:.1f}-500.csv".format(lambd))[0:get_cutoff_index(max_lambda, 500)]  # only take non-divergent values

        color_counter = 0
        for method in ("jac_max", "HHqr", "qr"):
            if method == "jac_max":
                N = 100  # Jacobi run with N = 100
            else:
                N = 500
            cutoff_index = get_cutoff_index(max_lambda, N)

            indeces = np.linspace(1, cutoff_index, cutoff_index)
            eig = np.loadtxt(dir + method + "-{:.1f}-{}.csv".format(lambd, N))[0:cutoff_index]  # only take non-divergent values
            error = np.abs(eig[0:cutoff_index] - eig_reference[0:cutoff_index])
            plt.plot(error, color=colors[color_counter], marker=markers[color_counter], markersize=5, linestyle="-", label=get_method_label(method))
            color_counter += 1

        legend = plt.legend()  # increase legend line thickness
        for line in legend.get_lines():
            line.set_linewidth(3.0)

        i += 1

    plt.tight_layout()
    plt.suptitle("Eigenvalue Error With Respect to Matlab's eig", fontsize=14)
    plt.subplots_adjust(top=0.89)
    if save_figures: plt.savefig(figure_directory + "error_.png", dpi=200)
    plt.show()


def compare_jacobi_times():
    fig = plt.figure(figsize=(7, 4))

    time_max = np.loadtxt(data_dir + "times/tjac_max.csv", delimiter=',')
    time_cyc = np.loadtxt(data_dir + "times/tjac_cyc.csv", delimiter=',')

    grouper = groupby(time_max, key=lambda x: x[0])
    time_max = np.asarray([[x, mean(yi[1] for yi in y)] for x, y in grouper])
    grouper = groupby(time_cyc, key=lambda x: x[0])
    time_cyc = np.asarray([[x, mean(yi[1] for yi in y)] for x, y in grouper])

    plt.xlabel("Matrix Dimension $N$")
    plt.ylabel("Computation Time [s]")
    plt.plot(time_max[:,0], time_max[:,1], color="#2e6fa7", linestyle='--', marker='o',  label="classical")
    plt.plot(time_cyc[:,0], time_cyc[:,1], color="#e2692d", linestyle='--', marker='o', label="cyclic")
    plt.legend()
    plt.title("Time Comparison of Jacobi Implementations")
    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_directory + "times-jac_.png", dpi=200)
    plt.show()


def plot_times():
    fig, ax = plt.subplots(figsize=(8, 4))

    time_eig = np.loadtxt(data_dir + "times/teig.csv", delimiter=',')
    time_HH = np.loadtxt(data_dir + "times/tHHqr.csv", delimiter=',')
    time_QR = np.loadtxt(data_dir + "times/tqr.csv", delimiter=',')
    time_jac = np.loadtxt(data_dir + "times/tjac_max.csv", delimiter=',')

    grouper = groupby(time_eig, key=lambda x: x[0])
    time_eig = np.asarray([[x, mean(yi[1] for yi in y)] for x, y in grouper])
    grouper = groupby(time_HH, key=lambda x: x[0])
    time_HH = np.asarray([[x, mean(yi[1] for yi in y)] for x, y in grouper])
    grouper = groupby(time_QR, key=lambda x: x[0])
    time_QR = np.asarray([[x, mean(yi[1] for yi in y)] for x, y in grouper])
    grouper = groupby(time_jac, key=lambda x: x[0])
    time_jac = np.asarray([[x, mean(yi[1] for yi in y)] for x, y in grouper])

    plt.subplot(1,2,1)
    plt.xlabel("Matrix Dimension $N$")
    plt.ylabel("Computation Time [s]")
    plt.plot(time_jac[:, 0], time_jac[:, 1], color="#f05454", linestyle='--', marker='d', label="jacobi")
    plt.plot(time_HH[:, 0], time_HH[:, 1], color="#ce6262", linestyle='--', marker='.', label="householder to QR")
    plt.plot(time_QR[:, 0], time_QR[:, 1], color="#af2d2d", linestyle='--', marker='s', label="one-qhase QR")
    plt.plot(time_eig[:, 0], time_eig[:, 1], color="#16697a", linestyle='--', marker='o', label="eig")
    plt.legend()
    plt.title("Linear Scale", fontsize=10)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.xlabel("Matrix Dimension $N$")
    plt.ylabel("Computation Time [s]")
    plt.yscale("log")
    plt.plot(time_jac[:, 0], time_jac[:, 1], color="#f05454", linestyle='--', marker='d', label="jacobi")
    plt.plot(time_HH[:, 0], time_HH[:, 1], color="#ce6262", linestyle='--', marker='.', label="householder to QR")
    plt.plot(time_QR[:, 0], time_QR[:, 1], color="#af2d2d", linestyle='--', marker='s', label="one-qhase QR")
    plt.plot(time_eig[:, 0], time_eig[:, 1], color="#16697a", linestyle='--', marker='o', label="eig")
    plt.legend()
    plt.title("Logarithmic Scale", fontsize=10)
    plt.grid()

    plt.suptitle("Time Comparison of Eigenvalue Implementations", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    if save_figures: plt.savefig(figure_directory + "times_.png", dpi=200)
    plt.show()


def get_cutoff_index(lamb, N):
    if N == 10:
        if lamb == 0.1:
            return 4
        if lamb == 0.4:
            return 4
        if lamb == 0.7:
            return 3
        if lamb == 1.0:
            return 3
    elif N == 50:
        if lamb == 0.1:
            return 21
        if lamb == 0.4:
            return 20
        if lamb == 0.7:
            return 19
        if lamb == 1.0:
            return 18
    elif N == 100:
        if lamb == 0.1:
            return 38
        if lamb == 0.4:
            return 25
        if lamb == 0.7:
            return 21
        if lamb == 1.0:
            return 19
    elif N == 500:
        if lamb == 0.1:
            return 137
        if lamb == 0.4:
            return 97
        if lamb == 0.7:
            return 86
        if lamb == 1.0:
            return 78
    else:
        return int(0.25 * N)


def plot_eigenvectors():
    dir = data_dir + "vectors/"
    N = 500
    cutoff = 20  # get_cutoff_index(lambd, N)

    fig, axes = plt.subplots(2,2, figsize=(8, 6.5))

    i = 0
    for lambd in (0, 0.1, 0.4, 0.7):

        if i == 0: ax = axes[0, 0]
        elif i == 1: ax = axes[0, 1]
        elif i == 2: ax = axes[1, 0]
        else: ax = axes[1, 1]

        # ax.set_title("$\lambda$ = {:.1f}".format(lambd), fontsize=11)
        if i >= 2: ax.set_xlabel("Eigenvector Index $n$", fontsize=13)
        if i % 2 == 0: ax.set_ylabel("Eigenvector Component", fontsize=13)

        data = np.loadtxt(dir + "eig-{:.1f}-{}.csv".format(lambd, N), delimiter=',')[0:cutoff,0:cutoff]  # take only first cutoff vectors and transpose data so eigenvectors are rows

        dn = 1  # space indexes by 1 i.e. n = 1, 2, 3...
        dv = 1  # eigenvector spacing i.e. v1, v2, ...
        nmin, nmax = 1, cutoff+1  # add one extra term on purpose to fit all data values below
        vmin, vmax = 1, cutoff+1  # add one extra term on purpose
        y, x = np.mgrid[slice(vmin, vmax+dv, dv), slice(nmin, nmax+dn, dn)]  # yes, y and x reversed on purpose because sliced returns them in reverse

        z_min, z_max = -np.abs(data).max(), np.abs(data).max()

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        c = ax.pcolor(x, y, data, cmap='seismic', vmin=z_min, vmax=z_max, label="$\lambda$ = {:.1f}".format(lambd))
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.set_ylim(ax.get_ylim()[::-1])  # inverts y axis to go from N to 1
        ax.legend(loc="upper right")
        if i%2 == 1: fig.colorbar(c, ax=ax)

        i = i+1

    plt.tight_layout()
    plt.suptitle("First {} Single-Well Eigenvectors".format(cutoff), fontsize=16)
    plt.subplots_adjust(top=0.94)
    if save_figures: plt.savefig(figure_directory + "vectors{}_.png".format(cutoff), dpi=200)

    plt.show()


def plot_eigenvectors_double_well():
    dir = data_dir + "vectors-double-well/"
    N = 500

    cutoff = 20  # get_cutoff_index(lambd, N)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    ax.set_xlabel("Eigenvector Index $n$", fontsize=13)
    ax.set_ylabel("Eigenvector Component", fontsize=13)
    ax.set_title("First {} Double-Well Eigenvectors".format(cutoff), fontsize=16)
    data = np.loadtxt(dir + "eig-{}.csv".format(N), delimiter=',')[0:cutoff, 0:cutoff]  # take only first cutoff vectors and transpose data so eigenvectors are rows

    dn = 1  # space indexes by 1 i.e. n = 1, 2, 3...
    dv = 1  # eigenvector spacing i.e. v1, v2, ...
    nmin, nmax = 1, cutoff + 1  # add one extra term on purpose to fit all data values below
    vmin, vmax = 1, cutoff + 1  # add one extra term on purpose
    y, x = np.mgrid[slice(vmin, vmax + dv, dv), slice(nmin, nmax + dn,dn)]  # yes, y and x reversed on purpose because sliced returns them in reverse
    z_min, z_max = -np.abs(data).max(), np.abs(data).max()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    c = ax.pcolor(x, y, data, cmap='seismic', vmin=z_min, vmax=z_max)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_ylim(ax.get_ylim()[::-1])  # inverts y axis to go from N to 1
    fig.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    if save_figures: plt.savefig(figure_directory + "vectors-double-well{}_.png".format(cutoff), dpi=200)

    plt.show()


def playground():
    # a, b, c = np.zeros(5), np.zeros(5), np.zeros(5)
    # for i in range(1, 5):
    #     a[i] = i
    #     b[i] = 2*i
    #     c[i] = 3*i
    # print(a, b, c)
    # print(np.mean([a, b, c], axis=0))

    xmax = 3
    ymax = 5
    A = np.zeros((xmax, ymax))
    for i in range(0, xmax):
        for j in range(0, ymax):
            A[i, j] = i
    A = A.T
    print(A)



    dx, dy = 1, 1
    y, x = np.mgrid[slice(1, 5+dy, dy), slice(1, 3+dy, dx)]

    z = y
    print(z)


def run():
    # compare_q1q2q4()
    # compare_q1q2q4_difference()
    # plot_eigenvalues()
    # plot_eigenvalues_double_well()

    # plot_error()
    plot_eigenvectors()
    # plot_eigenvectors_double_well()
    # playground()

    # compare_jacobi_times()
    # plot_times()

run()