import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import beta

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

data_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/11-galerkin/data/"
figure_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/11-galerkin/figures/"

save_figures = True
# usetex = True  # turn on to use Latex to render text
usetex = False  # turn off to plot faster


# -----------------------------------------------------------------------------
# START ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def get_Aij(m, n1, n2):
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


def get_b(M, N):
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

    # indeces = np.linspace(0, N*(M+1) - 1, N*(M+1))
    # plt.plot(indeces, b, ls='--', marker='.')
    # plt.show()


def get_A(M, N):
    """
    Returns the matrix A, used with the Galerkin method in the report's laminar water flow problem
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    :return:
    """
    D = N*(M+1)  # dimension of the matrix, i.e. A is a DxD matrix
    A = np.zeros((D, D))  # preallocate complete matrix
    for m in range(0, M+1, 1):  # m = 0, 1, ..., M
        # construct sub-matrix \tilde{A}^{(m)}  (see report)
        A_m = np.zeros((N, N))  # preallocate mth submatrix
        for n1 in range(1, N+1, 1):  # n1 = n' = 1, 2, ..., N  (n1 is n' in report)
            for n2 in range(1, N+1, 1):  # n2 is equivalent to n in report
                A_m[n1-1, n2-1] = get_Aij(m, n1, n2)
        A[m*N:(m+1)*N, m*N:(m+1)*N] = A_m

    return A

    # min = np.min(np.abs(A[np.nonzero(A)]))
    # max = np.max(np.abs(A[np.nonzero(A)]))
    # # print(max)
    # # print(min)
    # A[A == 0.0] = np.nan  # hacky way, when plotting the matrix, to plot zero elements in pure white
    # plt.matshow(A, cmap="Blues_r", extent=(1, D, D, 1))
    # plt.show()


def get_Psi_mn(m, n, xi, phi):
    """
    Returns the basis function Psi_mn evaluated at the radius xi and angle phi
    :param m: 0, 1, ..., M
    :param n: 1, 2, ..., N
    :param xi: scaled radius of semicircular pipe in range [0, 1]
    :param phi: angle in planar polar coordinates of semicircular pipe in range [0, pi]
    """
    return (xi**(2*m+1)) * ((1-xi)**n) * np.sin((2*m + 1)*phi)


def find_u(K, L, M, N):
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
    b = get_b(M, N)
    A = get_A(M, N)
    a = np.linalg.solve(A, b)

    xi0, xiK = 0.0, 1.0  # maximum and minimium scaled radius in pipe
    dxi = (xiK - xi0)/K  # step size
    xi = np.linspace(xi0, xiK, K+1)

    phi0, phiL = 0.0, np.pi  # maximum and minimum angle (planar polar coordinate)
    dphi = (phiL - phi0)/L  # step size
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

    u_min, u_max = np.min(U), np.max(U)
    n_contours = 10  # number of contour lines to draw
    levels = np.linspace(u_min, u_max, n_contours)

    Xi, Phi = np.meshgrid(xi, phi)  # create meshgrids for polar plotting

    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    # cp = ax.contourf(Phi, Xi, U.T, levels, cmap="Blues")
    # plt.colorbar(cp)
    # plt.show()

    fig, ax = plt.subplots()
    X, Y = Xi*np.cos(Phi), Xi*np.sin(Phi)
    ax.contourf(X, Y, U.T, levels, cmap="Blues")
    ax.contour(X, Y, U.T, levels, colors="black", linewidths=1)  # plot contour lines on top of filled contours
    plt.show()


def find_C(M, N):
    """
    Calculates the Poiseulle constant C for laminar flow in a pipe with a semi-circular cross section
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    """
    # TODO make into a time trial; find times for various M and N e.g. for (M, N) in NM_pairs ((N1,M1), (N2,M2)) etc...
    # save a, b and C along with times to a local file
    t = time.time()
    b = get_b(M, N)
    t_b = time.time() - t
    print("Found b: {:.3e}".format(t_b))

    t = time.time()
    A = get_A(M, N)
    t_A = time.time() - t
    print("Found A: {:.3e}".format(t_A))

    t = time.time()
    a = np.linalg.solve(A, b)
    t_a = time.time() - t
    print("Solved for a: {:.3e}".format(t_a))

    t = time.time()
    C = - (32/np.pi)*np.dot(b, a)
    t_C = time.time() - t
    print("Found C: {:.3e}".format(t_C))

    print(C)
# -----------------------------------------------------------------------------
# END ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


def practice():
    N = 5
    a = np.zeros((N, 1))
    b = a[:, 0]
    print(b)


if __name__ == "__main__":
    # practice()
    M = 1
    N = 1

    K = 101
    L = 100
    # get_b(M, N)
    # get_A(M, N)
    # find_C(M, N)
    find_u(K, L, M, N)

