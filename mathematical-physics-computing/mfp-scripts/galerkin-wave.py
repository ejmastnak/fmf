import numpy as np
import matplotlib.pyplot as plt

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
def get_initial_condition(x):
    """ Returns the initial condition U(x, t) = sin(pi cos(x)) on the 1D position grid x"""
    return np.sin(np.pi * np.cos(x))
# -----------------------------------------------------------------------------
# END ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_initial_condition():
    N = 200 # number of position grid points (well technically there are N+1)
    x0 = 0
    xN = 2*np.pi
    x = np.linspace(x0, xN, N+1)
    u = get_initial_condition(x)

    plt.figure(figsize=(7, 4))
    remove_spines(plt.gca())
    if usetex: plt.rc('text', usetex=True)

    plt.xlabel(r"Position $\xi$")
    plt.ylabel("Amplitude $u$")
    plt.plot(x, u, c=color_blue)
    plt.title("Initial Condition")
    plt.tight_layout()
    plt.show()
# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------


def practice():
    N = 5
    a = np.zeros((N, 1))
    b = a[:, 0]
    print(b)


if __name__ == "__main__":
   print("Hi, you're awesome!")
   plot_initial_condition()
