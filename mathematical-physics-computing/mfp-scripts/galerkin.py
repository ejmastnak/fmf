import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import block_diag
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time
from scipy.special import beta

plt.rcParams['font.family'] = 'STIXGeneral', 'serif'  # serif as fallback
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('axes', titlesize=18)    # fontsize of titles
plt.rc('text', usetex=True)  # turn on for nice graphs but slower plotting

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

color_gray_ref = "#AAAAAA"  # light gray for reference quantities

data_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/11-galerkin/data/"
figure_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/11-galerkin/figures/"

save_figures = True
save_tables = True


# -----------------------------------------------------------------------------
# START ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def get_Aij_nonzero(m, n1, n2):
    """
    Returns the matrix elements A_{ij} = A_{(m'n')(mn)} (see report for more on notatation and indexing)
    Because A is "block-orthogonal" with respect to m, we can assume m=m', hence the single m argument
    Uses the beta function
    :param m: equivalent to m and m'
    :param n1: equivalent to n'
    :param n2: equivalent to n
    """
    return -0.5*np.pi*beta(n1 + n2 - 1, 3 + 4*m)*(n1*n2*(3 + 4*m))/(2 + 4*m + n1 + n2)


def get_bi(m, n):
    """
    Returns the vector components b_{i} = b_{m'n'} (see report for more on notatation and indexing)
    Uses the beta function
    :param m: equivalent to m'
    :param n: equivalent to n'
    """
    return - 2 * beta(2*m + 3, n + 1)/(2*m + 1)


def get_b_loop(M, N):
    """
    Returns the vector b, the right hand side vector used with the Galerkin method in the report's
     laminar water flow problem
    :param M: max value of index m = 0, 1, ..., M
    :param N: max value of index n = 1, 2, ..., N
    :return:
    """
    b = np.zeros(N*(M+1))  # preallocate
    i = 0  # index to count both m and n
    for m in range(0, M+1, 1):
        for n in range(1, N+1, 1):
            b[i] = get_bi(m, n)
            i += 1  # increment total index

    return b


def get_b_vectorized(M, N):
    """
    Returns the vector b, the right hand side vector used with the Galerkin method in the report's
     laminar water flow problem
    :param M: max value of index m = 0, 1, ..., M
    :param N: max value of index n = 1, 2, ..., N
    :return:
    """
    m = np.arange(0, M+1, 1)  # 0, 1, ..., M
    n = np.arange(1, N+1, 1)  # 1, ..., N
    m_grid, n_grid = np.meshgrid(m, n, indexing="ij")
    B = get_bi(m_grid, n_grid)
    return np.reshape(B, N*(M+1))


def get_A_loop(M, N):
    """
    Returns the matrix A, used with the Galerkin method in the report's laminar water flow problem
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    :return:
    """
    A = np.zeros((N*(M+1), N*(M+1)))  # preallocate complete matrix
    for m in range(0, M+1, 1):  # m = 0, 1, ..., M
        # construct sub-matrix \tilde{A}^{(m)}  (see report)
        A_m = np.zeros((N, N))  # preallocate mth submatrix
        for n1 in range(1, N+1, 1):  # n1 = n' = 1, 2, ..., N  (n1 is n' in report)
            for n2 in range(1, N+1, 1):  # n2 is equivalent to n in report
                A_m[n1-1, n2-1] = get_Aij_nonzero(m, n1, n2)
        A[m*N:(m+1)*N, m*N:(m+1)*N] = A_m

    return A


def get_A_hybrid(M, N):
    """
    Returns the matrix A, used with the Galerkin method in the report's laminar water flow problem
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    :return:
    """

    A = np.zeros((N*(M+1), N*(M+1)))  # preallocate complete matrix

    for m in range(0, M+1, 1):  # m = 0, 1, ..., M
        # construct sub-matrix A^{(m)}  (see report)
        n1v = np.arange(1, N+1, 1)
        n2v = np.arange(1, N+1, 1)
        N1, N2 = np.meshgrid(n1v, n2v, indexing="ij")
        A_m = get_Aij_nonzero(m, N1, N2)
        A[m*N:(m+1)*N, m*N:(m+1)*N] = A_m

    return A


def get_A_vectorized_full(M, N):
    """
    Returns the matrix A, used with the Galerkin method in the report's laminar water flow problem
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    """
    n = np.arange(1, N+1, 1)
    m = np.arange(0, M+1, 1)
    MM, N1, N2 = np.meshgrid(m, n, n, indexing="ij")
    A_vec = get_Aij_nonzero(MM, N1, N2)

    return block_diag(*A_vec)


def get_Psi_mn(m, n, xi, phi):
    """
    Returns the basis function Psi_mn evaluated at the radius xi and angle phi
    :param m: 0, 1, ..., M
    :param n: 1, 2, ..., N
    :param xi: scaled radius of semicircular pipe in range [0, 1]
    :param phi: angle in planar polar coordinates of semicircular pipe in range [0, pi]
    """
    return (xi**(2*m+1)) * ((1-xi)**n) * np.sin((2*m + 1)*phi)


def get_Psi_mn2(xi, phi, m, n):
    """
    Returns the basis function Psi_mn evaluated at the radius xi and angle phi
    :param xi: scaled radius of semicircular pipe in range [0, 1]
    :param phi: angle in planar polar coordinates of semicircular pipe in range [0, pi]
    :param m: 0, 1, ..., M
    :param n: 1, 2, ..., N
    """
    return (xi**(2*m+1)) * ((1-xi)**n) * np.sin((2*m + 1)*phi)


def get_u_loop(K, L, M, N):
    """
    Find velocity profile u(xi, phi) for laminar flow in a pipe with a semicircular cross section
    Uses a planar polar coordinate system with scaled radius xi in [0, 1] and phi in [0, pi]

    Radius discretized as xi_k = xi_0 + k* dxi
    Angle discretized as phi_l = phi_0 + l* dphi
    :param K: number of points in radius grid k = 0, 1, ..., K
    :param L: number of points in angle grid l = 0, 1, ..., L
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    """
    b = get_b_loop(M, N)
    A = get_A_loop(M, N)
    # a = np.linalg.solve(A, b)  # slower than spsolve
    a = spsolve(csr_matrix(A), b)

    xi0, xiK = 0.0, 1.0  # maximum and minimium scaled radius in pipe
    xi = np.linspace(xi0, xiK, K+1)

    phi0, phiL = 0.0, np.pi  # maximum and minimum angle (planar polar coordinate)
    phi = np.linspace(phi0, phiL, L+1)

    psi = np.zeros(N*(M+1))  # N*(M-1)-element vector holding Psi_{mn} at a single point (xi_k, phi_l)
    Psi = np.zeros((K+1, L+1, N*(M+1)))  # (K+1)x(L+1)x(N*(M+1)) 3D array holding psi at every (xi_k, phi_l)

    for k, xi_k in enumerate(xi):  # loop over all xi
        for l, phi_l in enumerate(phi):  # loop over all phi
            i = 0  # combined index of m and n used to index Psi
            for m in range(0, M+1, 1):
                for n in range(1, N+1, 1):
                    psi[i] = get_Psi_mn(m, n, xi_k, phi_l)
                    i += 1  # increment index
            Psi[k][l] = psi  # set the psi vector in the 3D Psi array at the point (xi_k, phi_l)

    U = np.zeros((K+1, L+1))  # holds velocity u at each point in the (xi, phi) grid
    for k in range(K+1):  # k = 0, 1, ..., K
        for l in range(L+1):  # l = 0, 1, ..., L
            U[k][l] = np.dot(a, Psi[k][l])

    return xi, phi, U


def get_u_vectorized(K, L, M, N):
    """
    Find velocity profile u(xi, phi) for laminar flow in a pipe with a semicircular cross section
    Uses a planar polar coordinate system with scaled radius xi in [0, 1] and phi in [0, pi]

    Radius discretized as xi_k = xi_0 + k* dxi
    Angle discretized as phi_l = phi_0 + l* dphi
    :param K: number of points in radius grid k = 0, 1, ..., K
    :param L: number of points in angle grid l = 0, 1, ..., L
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    """
    b = get_b_vectorized(M, N)
    A = get_A_vectorized_full(M, N)

    # a = np.linalg.solve(A, b)  # slower than spsolve
    a = spsolve(csr_matrix(A), b)

    xi0, xiK = 0.0, 1.0  # maximum and minimium scaled radius in pipe
    xi = np.linspace(xi0, xiK, K+1)

    phi0, phiL = 0.0, np.pi  # maximum and minimum angle (planar polar coordinate)
    phi = np.linspace(phi0, phiL, L+1)

    m = np.arange(0, M+1, 1)
    n = np.arange(1, N+1, 1)
    m_grid, n_grid = np.meshgrid(m, n, indexing="ij")
    Psi = np.zeros((K+1, L+1, N*(M+1)))  # (K+1)x(L+1)x(N*(M+1)) 3D array holding psi at every (xi_k, phi_l)
    for k, xi_k in enumerate(xi):  # loop over all xi
        for l, phi_l in enumerate(phi):  # loop over all phi
            psi_vec = get_Psi_mn(m_grid, n_grid, xi_k, phi_l)
            psi_vec = psi_vec.flatten()  # convert to 1D array
            Psi[k][l] = psi_vec  # set the psi vector in the 3D Psi array at the point (xi_k, phi_l)

    # slower than half-vectorized form above for large M, N by factor around 4 or 5
    # slightly faster by factor of about 1-2 for large K, L and small M, N
    # t = time.time()
    # xi_grid, phi_grid, m_grid, n_grid = np.meshgrid(xi, phi, m, n, indexing="ij")
    # Psi_vec = get_Psi_mn2(xi_grid, phi_grid, m_grid, n_grid)
    # Psi_vec = np.reshape(Psi_vec, (K+1, L+1, N*(M+1)))
    # t2 = time.time() - t

    U = np.dot(Psi, a)  # (K+1)x(L+1)x(N(M+1)) dot (,N(M+1)) outputs (K+1)x(L+1)
    return xi, phi, U


def get_C_loop(M, N):
    """
    Calculates the Poiseulle constant C for laminar flow in a pipe with a semi-circular cross section
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    """
    b = get_b_loop(M, N)
    A = get_A_loop(M, N)
    a = spsolve(csr_matrix(A), b)  # convert A to SciPy sparse matrix format
    return - (32/np.pi)*np.dot(b, a)


def get_C_vectorized(M, N, use_spsolve=True):
    """
    Calculates the Poiseulle constant C for laminar flow in a pipe with a semi-circular cross section
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    """
    b = get_b_vectorized(M, N)
    A = get_A_vectorized_full(M, N)
    if use_spsolve: a = spsolve(csr_matrix(A), b)  # convert A to SciPy sparse matrix format
    else: a = np.linalg.solve(A, b)
    return - (32/np.pi)*np.dot(b, a)


def get_C_compared(M, N):
    """
    Calculates the Poiseulle constant C for laminar flow in a pipe with a semi-circular cross section
    Compares the results returned by np.linalg.solve and scipy.sparse.linalg.spsolve
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    """
    b = get_b_vectorized(M, N)
    A = get_A_vectorized_full(M, N)

    a1 = np.linalg.solve(A, b)
    a2 = spsolve(csr_matrix(A), b)  # convert A to SciPy sparse matrix format

    C1 = - (32/np.pi)*np.dot(b, a1)
    C2 = - (32/np.pi)*np.dot(b, a2)

    print("np.linalg.solve: ".format(C1))
    print("scipy.sparse.linalg.spsolve: ".format(C2))
# -----------------------------------------------------------------------------
# END ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START DATA GENERATION FUNCTIONS
# -----------------------------------------------------------------------------
def generate_C_values():
    """ Calculates values of C for various values of M and N and saves the results to a local file """
    MN_max_low = 10
    MN_list = [(N, N) for N in range(1, MN_max_low, 1)]  # low values in intervals of 1

    MN_max_mid = 100
    MN_list_mid = [(N, N) for N in range(MN_max_low, MN_max_mid, 10)]  # mid values in intervals of 10
    for MN in MN_list_mid:
        MN_list.append(MN)

    MN_max_high = 150
    MN_list_high = [(N, N) for N in range(MN_max_mid, MN_max_high+1, 50)]  # high values in intervals of 50
    for MN in MN_list_high:
        MN_list.append(MN)
    #
    C_list = np.zeros(len(MN_list))
    for i, MN in enumerate(MN_list):
        print(MN)
        M, N = MN  # unpack MN pair
        C_list[i] = get_C_vectorized(M, N, use_spsolve=True)

    if save_tables:
        filename = data_dir + "C-table-{}_.csv".format(MN_list[-1][0])
        with open(filename, 'w') as output:  # open file for writing
            output.write("# (M;N),C")
            for i, MN in enumerate(MN_list):
                M, N = MN  # unpack MN pair
                C = C_list[i]
                line = "\n({};{}),{:.15f}".format(M, N, C)
                output.write(line)  # write metadata
    print(C_list)


def generate_C_timing():
    """ Compares computation time needed to find C for a vectorized and Python loop implementation """
    MN_max_low = 10
    MN_list = [(N, N) for N in range(1, MN_max_low, 1)]  # low values in intervals of 1

    MN_max_mid = 100
    MN_list_mid = [(N, N) for N in range(MN_max_low, MN_max_mid+1, 10)]  # mid values in intervals of 10
    for MN in MN_list_mid:
        MN_list.append(MN)

    MN_list = [(100, 100)]

    loop_runs = 10  # number of times to repeat finding C for a given MN
    vec_times = np.zeros(len(MN_list))  # finding C with vectorized functions
    loop_times = np.zeros(len(MN_list))  # finding C with Python loops
    for i, MN in enumerate(MN_list):
        print(MN)
        M, N = MN  # unpack MN pair
        t = time.time()
        for _ in range(loop_runs):
            get_C_vectorized(M, N, use_spsolve=True)
        vec_times[i] = time.time() - t

        t = time.time()
        for _ in range(loop_runs):
            get_C_loop(M, N)
        loop_times[i] = time.time() - t

    if save_tables:
        filename = data_dir + "C-times-{}_.csv".format(MN_list[-1][0])
        with open(filename, 'w') as output:  # open file for writing
            output.write("# M=N,Vectorized Time [s],Python Loop Time [s]")
            for i, MN in enumerate(MN_list):
                M, N = MN  # unpack MN pair
                t_vec = vec_times[i]
                t_loop = loop_times[i]
                line = "\n{},{:.8e},{:.8e}".format(M, t_vec, t_loop)
                output.write(line)  # write metadata


def generate_u_timing():
    """ Compares computation time needed to find velocity profile u for a vectorized and Python loop implementation """
    MN_max_low = 10
    K, L = 50, 50  # on the smaller side to speed up overall timing
    MN_list = [(N, N) for N in range(1, MN_max_low, 1)]  # low values in intervals of 1

    MN_max_mid = 16
    MN_list_mid = [(N, N) for N in range(MN_max_low, MN_max_mid+1, 2)]  # mid values in intervals of 2
    for MN in MN_list_mid:
        MN_list.append(MN)

    loop_runs = 10  # number of times to repeat finding C for a given MN
    vec_times = np.zeros(len(MN_list))  # finding C with vectorized functions
    loop_times = np.zeros(len(MN_list))  # finding C with Python loops
    for i, MN in enumerate(MN_list):
        print(MN)
        M, N = MN  # unpack MN pair
        t = time.time()
        for _ in range(loop_runs):
            get_u_vectorized(K, L, M, N)
        vec_times[i] = time.time() - t

        t = time.time()
        for _ in range(loop_runs):
            get_u_loop(K, L, M, N)
        loop_times[i] = time.time() - t

    if save_tables:
        filename = data_dir + "u-times-{}_.csv".format(MN_list[-1][0])
        with open(filename, 'w') as output:  # open file for writing
            output.write("# M=N,Vectorized Time [s],Python Loop Time [s]")
            for i, MN in enumerate(MN_list):
                M, N = MN  # unpack MN pair
                t_vec = vec_times[i]
                t_loop = loop_times[i]
                line = "\n{},{:.8e},{:.8e}".format(M, t_vec, t_loop)
                output.write(line)  # write metadata


def generate_A_timing():
    """ Compares computation time needed to find the matrix A for a vectorized and Python loop implementation """
    MN_max_low = 10
    MN_list = [(N, N) for N in range(1, MN_max_low, 1)]  # low values in intervals of 1

    MN_max_mid = 100
    MN_list_mid = [(N, N) for N in range(MN_max_low, MN_max_mid+1, 10)]  # mid values in intervals of 10
    for MN in MN_list_mid:
        MN_list.append(MN)

    loop_runs = 10  # number of times to repeat finding C for a given MN
    vec_times = np.zeros(len(MN_list))  # finding C with vectorized functions
    hybrid_times = np.zeros(len(MN_list))  # finding C with hybrid approach
    loop_times = np.zeros(len(MN_list))  # finding C with Python loops
    for i, MN in enumerate(MN_list):
        print(MN)
        M, N = MN  # unpack MN pair
        t = time.time()
        for _ in range(loop_runs):
            get_A_vectorized_full(M, N)
        vec_times[i] = time.time() - t

        t = time.time()
        for _ in range(loop_runs):
            get_A_hybrid(M, N)
        hybrid_times[i] = time.time() - t

        t = time.time()
        for _ in range(loop_runs):
            get_A_loop(M, N)
        loop_times[i] = time.time() - t

    if save_tables:
        filename = data_dir + "A-times-{}_.csv".format(MN_list[-1][0])
        with open(filename, 'w') as output:  # open file for writing
            output.write("# M=N,Vectorized Time [s],Hybrid Time [s], Python Loop Time [s]")
            for i, MN in enumerate(MN_list):
                M, N = MN  # unpack MN pair
                t_vec = vec_times[i]
                t_hybrid = hybrid_times[i]
                t_loop = loop_times[i]
                line = "\n{},{:.8e},{:.8e},{:.8e}".format(M, t_vec, t_hybrid, t_loop)
                output.write(line)  # write metadata
# -----------------------------------------------------------------------------
# END DATA GENERATION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_A(M, N, scale_power=1.0):
    """
    Plots the matrix A on a 2D grid with color proportional to number magnitude
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    :param scale_power: raises A to scale_power. Usually e.g. 0.5 or 0.3333 to make small elements more visible
    """
    A = get_A_vectorized_full(M, N)
    minA = np.min(A[np.nonzero(A)])  # smallest value (is negative because A's elements are negative)
    A = A/minA  # normalize  (and convert to positive values)
    A = A**scale_power
    A[A == 0.0] = np.nan  # hacky way, when plotting the matrix, to plot zero elements in pure white
    # extent = (1, N*(M+1)+1, N*(M+1)+1, 1)
    extent = (0, N*(M+1), N*(M+1), 0)
    plt.matshow(A, cmap="Blues", extent=extent)
    plt.title(r"Matrix $\mathbf{{A}}$, $M={}$, $N={}$".format(M, N))
    if save_figures: plt.savefig(figure_dir + "matrix-{}-{}_.png".format(M, N), dpi=200)
    plt.show()


def plot_u_cartesian(xi, phi, U):

    u_min, u_max = np.min(U), np.max(U)
    n_contours = 10  # number of contour lines to draw
    levels = np.linspace(u_min, u_max, n_contours)  # linearly spaced levels at which to draw contours

    Xi, Phi = np.meshgrid(xi, phi, indexing="ij")  # create meshgrids for polar plotting
    X, Y = Xi*np.cos(Phi), Xi*np.sin(Phi)

    fig, ax = plt.subplots(figsize=(7, 4))
    remove_spines(ax)
    ax.set_xlabel(r"$x=\xi \, \cos \, \phi$")
    ax.set_ylabel(r"$y=\xi \, \sin \, \phi$")
    cp = ax.contourf(X, Y, U, levels, cmap="Blues")
    ax.contour(X, Y, U, levels, colors="black", linewidths=0.5)  # plot contour lines on top of filled contours
    plt.colorbar(cp, format="%.2f")
    plt.suptitle("Fluid Velocity Profile in a Semicircular Pipe", y=0.96, fontsize=21)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "velocity-cartesian_.png", dpi=200)
    plt.show()


def plot_u_polar(xi, phi, U):
    u_min, u_max = np.min(U), np.max(U)
    n_contours = 10  # number of contour lines to draw
    levels = np.linspace(u_min, u_max, n_contours)  # linearly spaced levels at which to draw contours

    Xi, Phi = np.meshgrid(xi, phi, indexing="ij")  # create meshgrids for polar plotting

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_thetagrids(np.linspace(0, 180, 5))  # set the gridlines at intervals of pi/4
    ax.set_xlabel(r"Radius $\xi$", labelpad=-47)

    label_positions = ax.get_xticks()
    labels = list(label_positions)  # convert to list (to subsequently change floats to strings)
    labels = [str(label/np.pi) + r"$\pi$" for label in labels]
    ax.set_xticklabels(labels)

    cp = ax.contourf(Phi, Xi, U, levels, cmap="Blues")  # draws velocity profile in color
    # ax.contour(Phi, Xi, U, levels, colors="#000000", linewidths=0.4)  # draw contour lines
    cp_gap = 0.06  # gap between main axis and colorbar
    cp_width = 0.02  # width of the colorbar
    left = ax.get_position().x1 + cp_gap
    bottom = 0.3  # (hacky) by trial and error to align cb with polar plot. Ideally would work with ax.get_position().y0
    width = cp_width
    height = 0.6*ax.get_position().height
    cax = fig.add_axes([left, bottom, width, height])
    plt.colorbar(cp, cax=cax, shrink=0.5, format="%.2f")  # Similar to fig.colorbar(im, cax = cax)
    ax.grid(color='#000000', alpha=0.5, lw=0.5)  # black gridlines

    plt.suptitle("Fluid Velocity Profile in a Semicircular Pipe", y=0.83, fontsize=18)
    if save_figures: plt.savefig(figure_dir + "velocity-polar_.png", dpi=200)
    plt.show()


def plot_C_error(M_max, N_max):
    C_ref = 0.757722039130460  # reference value at (M, N) = (150, 150)

    M = np.arange(1, M_max+1, 1)
    N = np.arange(1, N_max+1, 1)
    MM, NN = np.meshgrid(M, N, indexing="ij")
    get_C_vec_vec = np.vectorize(get_C_vectorized)  # vectorize get_C_vectorized to allow input of 2D M and N grids
    C_err = np.abs(get_C_vec_vec(MM, NN) - C_ref)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(MM, NN, C_err, rstride=1, cstride=1, antialiased=True, cmap="Blues", linewidths=0.5, edgecolors="k")
    ax.set_xlim(M[0], M[-1])
    ax.set_ylim(N[-1], N[0])
    ax.set_xlabel("$M$")
    ax.set_ylabel("$N$")
    ax.set_zlabel("Error")
    ax.view_init(16, -55)  # set view angle (elevation angle, azimuth angle)
    plt.suptitle("Error in $C$ vs. $M$ and $N$", y=0.83, fontsize=18)

    if save_figures: plt.savefig(figure_dir + "Cerr-{}-{}_.png".format(M_max, N_max), dpi=200)
    plt.show()


def plot_C_error_log(M_max, N_max):
    C_ref = 0.757722039130460  # reference value at (M, N) = (150, 150)

    M = np.arange(1, M_max+1, 1)
    N = np.arange(1, N_max+1, 1)
    MM, NN = np.meshgrid(M, N, indexing="ij")
    get_C_vec_vec = np.vectorize(get_C_vectorized)  # vectorize get_C_vectorized to allow input of 2D M and N grids
    C_err = np.abs(get_C_vec_vec(MM, NN) - C_ref)

    log_err = np.log10(C_err)
    shift = abs(np.min(log_err))
    log_err += shift

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(MM, NN, log_err, rstride=1, cstride=1, antialiased=True, cmap="Blues", linewidths=0.5, edgecolors="k")
    ax.set_xlim(M[0], M[-1])
    ax.set_ylim(N[-1], N[0])
    ax.set_xlabel("$M$")
    ax.set_ylabel("$N$")
    ax.set_zlabel("Error")
    ax.view_init(16, -55)  # set view angle (elevation angle, azimuth angle)

    # rescale ticks to match log error scale
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
    plt.suptitle("Error in $C$ vs. $M$ and $N$ (log scale)", y=0.83, fontsize=18)

    if save_figures: plt.savefig(figure_dir + "Cerr-log-{}-{}_.png".format(M_max, N_max), dpi=200)
    plt.show()


def plot_C_timing():
    N, tvec, tloop = np.loadtxt(data_dir + "C-times-100.csv", delimiter=',').T

    plt.figure(figsize=(7, 4))
    plt.xlabel(r"Number of Basis Functions $I=N\cdot(M+1)$")
    plt.ylabel("Computation Time $t$ [s]")
    plt.title("Computation Time to Find the Coefficient $C$")
    remove_spines(plt.gca())
    plt.plot(N**2, tloop, ls='--', c=color_orange_dark, marker='o', label="python loops")
    plt.plot(N**2, tvec, ls='--', c=color_blue, marker='d', label="vectorized")
    plt.legend()
    plt.tight_layout()

    if save_figures: plt.savefig(figure_dir + "times-C_.png", dpi=200)
    plt.show()


def plot_u_timing():
    N, tvec, tloop = np.loadtxt(data_dir + "u-times-16.csv", delimiter=',').T

    plt.figure(figsize=(7, 4))
    plt.xlabel(r"Number of Basis Functions $I=N\cdot(M+1)$")
    plt.ylabel("Computation Time $t$ [s]")
    plt.title(r"Computation Time to Find the Velocity Profile $u(\xi, \phi)$")
    remove_spines(plt.gca())
    plt.plot(N**2, tloop, ls='--', c=color_orange_dark, marker='o', label="python loops")
    plt.plot(N**2, tvec, ls='--', c=color_blue, marker='d', label="vectorized")
    plt.legend()
    plt.tight_layout()

    if save_figures: plt.savefig(figure_dir + "times-u_.png", dpi=200)
    plt.show()


def plot_A_timing():
    N, tvec, _, tloop = np.loadtxt(data_dir + "A-times-100.csv", delimiter=',').T

    plt.figure(figsize=(7, 4))
    plt.xlabel(r"Number of Basis Functions $I=N\cdot(M+1)$")
    plt.ylabel("Computation Time $t$ [s]")
    plt.title(r"Computation Time to Find the Matrix $\mathbf{A}$")
    remove_spines(plt.gca())
    plt.plot(N**2, tloop, ls='--', c=color_orange_dark, marker='o', label="python loops")
    plt.plot(N**2, tvec, ls='--', c=color_blue, marker='d', label="vectorized")
    plt.legend()
    plt.tight_layout()

    if save_figures: plt.savefig(figure_dir + "times-A_.png", dpi=200)
    plt.show()
# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------
def do_animation(MN_max, K=50, L=50):
    MN_list = [(N, N) for N in range(1, MN_max+1)]

    U_grid = np.zeros((len(MN_list), K+1, L+1))  # holds U(xi, phi) for each MN pair
    xi, phi = np.zeros(K+1), np.zeros(L+1)
    for i, MN in enumerate(MN_list):
        M, N = MN
        if i == 0:
            xi, phi, U = get_u_vectorized(K, L, M, N)  # store xi and phi only once
        else:
            _, _, U = get_u_vectorized(K, L, M, N)
        U_grid[i] = U

    animate_u(MN_list, xi, phi, U_grid)


def animate_u(MN_list, xi, phi, U_grid):
    """
    Animates changing velocity profile as M and N increase
    """

    iterations = len(MN_list)  # number of times at which a solution is found
    U0 = U_grid[0]  # initial velocity profile

    u_min, u_max = np.min(U0), np.max(U0)
    n_contours = 10  # number of contour lines to draw
    levels = np.linspace(u_min, u_max, n_contours)  # linearly spaced levels at which to draw contours

    Xi, Phi = np.meshgrid(xi, phi, indexing="ij")  # create meshgrids for polar plotting
    X, Y = Xi*np.cos(Phi), Xi*np.sin(Phi)

    fig, ax = plt.subplots(figsize=(7, 4))
    remove_spines(ax)
    ax.set_xlabel(r"$x=\xi \, \cos \, \phi$")
    ax.set_ylabel(r"$y=\xi \, \sin \, \phi$")

    cpf = ax.contourf(X, Y, U0, levels, cmap="Blues")
    ax.contour(X, Y, U0, levels, colors="black", linewidths=0.5)
    plt.colorbar(cpf, format="%.2f")

    plt.suptitle("Velocity Profile in a Semicircular Pipe", y=0.96, fontsize=18)
    plt.tight_layout()

    fargs = (ax, X, Y, U_grid, levels, MN_list, iterations)
    temp_animation = animation.FuncAnimation(fig, update_animation, iterations, fargs=fargs, interval=50, blit=False)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, bitrate=1800, extra_args=['-vcodec', 'libx264'])
    filename = "animate-u-{}-{}_.mp4".format(MN_list[-1][0], MN_list[-1][1])
    temp_animation.save(figure_dir + filename, writer=writer)


def update_animation(i, ax, X, Y, U_grid, levels, MN_list, iterations):
    """
    Auxiliary update function for the animation of the velocity profile solution
    """
    print("Frame {} of {}".format(i, iterations))
    ax.clear()  # remove old contour plot; probably not the most efficient, but it works
    
    cpf = ax.contourf(X, Y, U_grid[i], levels, cmap="Blues")
    cp = ax.contour(X, Y, U_grid[i], levels, colors="black", linewidths=0.5)

    label_text = "M: {}, N: {}".format(MN_list[i][0], MN_list[i][1])
    ax.text(0.5, 0.96, label_text, va='center', ha='center', transform=ax.transAxes, fontsize=18,
            bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    
    return cpf, cp
# -----------------------------------------------------------------------------
# END ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------


def practice():
    N = 5
    a = np.zeros((N, 1))
    b = a[:, 0]
    print(b)


def printC_vals():
    C = get_C_vectorized(1, 150)
    print(C)

    C = get_C_vectorized(150, 1)
    print(C)

if __name__ == "__main__":
    # np.set_printoptions(precision=3)
    # do_animation(15)
    # find_C_vectorized(M=10, N=10)
    # generate_C_values()
    # generate_C_timing()
    generate_u_timing()
    # generate_A_timing()

    # plot_A(9, 5, scale_power=1/3)
    plot_A(2, 3)
    # plot_u_polar(*get_u_vectorized(K=100, L=100, M=25, N=25))
    # plot_u_cartesian(*get_u_vectorized(K=100, L=100, M=25, N=25))

    # plot_C_error(10, 10)
    # plot_C_error_log(30, 30)
    # plot_C_timing()
    # plot_u_timing()
    # plot_A_timing()
    # printC_vals()


