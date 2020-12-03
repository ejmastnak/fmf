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

data_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/9-spectral/data/"
figure_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/9-spectral/figures/"

save_figures = True
# usetex = True  # turn off to plot faster
usetex = False  # turn off to plot faster




# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()



def practice():
    N = 5
    a = 2*np.ones(N)
    b = -1*np.ones(N-1)
    # b[0] = 10
    # c = -1*np.ones(N-1)
    # A = np.diag(a) + np.diag(b, 1) + np.diag(c, -1)
    # print(A)


def run():
    practice()


run()
