import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
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
    :param M: max value of index m = 0, 1, ..., M
    :param N: max value of index n = 1, 2, ..., N
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


def find_C(M, N):
    b = get_b(M, N)
    A = get_A(M, N)
    a = np.linalg.solve(A, b)
    C = - (32/np.pi)*np.dot(b, a)
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
    M = 2
    N = 5
    # get_b(M, N)
    # get_A(M, N)
    find_C(M, N)
