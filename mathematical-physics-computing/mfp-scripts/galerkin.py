import numpy as np
import matplotlib.pyplot as plt
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
def get_Aij(m1, n1, m2, n2):
    """
    Returns the matrix elements A_{ij} = A_{(m'n')(mn)} (see report for more on notatation and indexing)
    Uses the beta function
    :param m1: equivalent to m'
    :param n1: equivalent to n'
    :param m2: equivalent to m
    :param n2: equivalent to n
    """
    if m1 != m2:
        return 0  # elements are orthogonal with respect to m
    else:  # if m1 == m2
        return -0.5*np.pi*beta(n1 + n2 - 1, 3 + 4*m1)*(n1*n2*(3 + 4*m1))/(2 + 4*m1 + n1 + n2)


def get_bi(m, n):
    """
    Returns the vector components b_{i} = b_{m'n'} (see report for more on notatation and indexing)
    Uses the beta function
    :param m: equivalent to m'
    :param n: equivalent to n'
    """
    return - 2 * beta(2*m + 3, n + 1)/(2*m + 1)
# -----------------------------------------------------------------------------
# END ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


def practice():
    N = 10
    a = np.zeros((N, 1))
    b = a[:, 0]
    print(b)


if __name__ == "__main__":
    practice()
