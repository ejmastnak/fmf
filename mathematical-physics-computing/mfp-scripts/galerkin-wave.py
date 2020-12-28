import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from scipy.special import jv

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
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
# usetex = True  # turn on to use Latex to render text
usetex = False  # turn off to plot faster


# -----------------------------------------------------------------------------
# START ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def get_initial_condition(x):
    """ Returns the initial condition U(x, t) = sin(pi cos(x)) on the 1D position grid x"""
    return np.sin(np.pi * np.cos(x))


def get_analytic_solution(t, x):
    """ Returns the analytic solution for displacement u(x, t) at time t and position x """
    return np.sin(np.pi * np.cos(x + t))


def get_Psi_k(k, x):
    """
    Returns the Galerkin method basis function Psi_k(x) = exp(ikx)/sqrt(2pi)
    Note the basis function is a complex number!
    :param k: wave number
    :param x: position (assumed in the problem to be in the grid (0, 2pi))
    """
    return np.exp((0.0+1.0j)*k*x)/np.sqrt(2*np.pi)


def get_ak_analytic(k, t):
    """
    Returns analytic solution for time-dependent coefficients a_k(t) used with the Galerkin method
    for the linear wave equation.
    Uses the Bessel function and in general returns a complex number
    :param k: wave number  (assumed in problem to take values k = -N/2, -N/2 + 1, ..., N/2 were N is number of position points)
    :param t: time
    """
    return np.sin(0.5*k*np.pi)*jv(k, np.pi)*np.exp((0.0+1.0j)*k*t)


def get_u_numeric(M, N=2**8):
    """
    :param M: number of points in time grid
    :param N: number of points in position grid
    """
    t0 = 0
    tM = 2*np.pi
    t = np.linspace(t0, tM, M)

    x0 = 0
    xN = 2*np.pi
    x = np.linspace(x0, xN, N)
    k = np.linspace(-N/2, N/2, N, endpoint=False)  # create indexes for wave vectors ("frequency" for sampling position)

    T, K = np.meshgrid(t, k, indexing="ij")
    A = get_ak_analytic(K, T)  # (M, N) matrix. Each row holds ak at a fixed time t

    K, X = np.meshgrid(k, x, indexing="ij")
    Psi = get_Psi_k(K, X)

    U = np.real(np.dot(A, Psi))  # (M, N) matrix holding solutions. np.real removes residual complex elements.
    return U
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
    N = 200  # number of position grid points (well technically there are N+1)
    x0 = 0
    xN = 2*np.pi
    x = np.linspace(x0, xN, N+1)
    u = get_initial_condition(x)

    if usetex: plt.rc('text', usetex=True)
    fix, ax = plt.subplots(figsize=(7, 3))
    remove_spines(ax)

    ax.set_xlabel(r"Position $x$", labelpad=-0.3)
    ax.set_ylabel("Amplitude $u$")

    ax.set_xticks(np.linspace(x0, xN, 5))
    # labels = [r"0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    labels = [r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_xticklabels(labels, fontsize="13")

    ax.plot(x, u, c=color_blue)
    ax.hlines(0, x[0], x[-1], color="black", ls=':')
    # ax.vlines(np.pi, np.min(u), np.max(u), color="black", ls=':')
    ax.set_title("Initial Condition for the 1st Order Wave Equation")
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "wave-ic_.png", dpi=200)
    plt.show()


def plot_A(M, N, scale_power=1.0):
    """
    Plots the matrix A on a 2D grid with color proportional to number magnitude
    :param M: Maximum ``m'' index of basis functions Psi_{mn}
    :param N: Maximum ``n'' index of basis functions Psi_{mn}
    :param scale_power: raises A to scale_power. Usually e.g. 0.5 or 0.3333 to make small elements more visible
    """
    t0 = 0
    tM = 2*np.pi
    t = np.linspace(t0, tM, M)
    k = np.linspace(-N/2, N/2, N, endpoint=False)  # create indexes for wave vectors ("frequency" for sampling position)

    T, K = np.meshgrid(t, k, indexing="ij")
    A = get_ak_analytic(K, T)  # (M, N) matrix. Each row holds ak at a fixed time t
    extent = (-N/2, N/2, M, 0)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    ax = axes[0]
    ax.set_ylabel("$m$")
    ax.set_xlabel("$k$", labelpad=8)
    ax.matshow(np.real(A), cmap="seismic", extent=extent)
    ax.set_title(r"Re$[\mathbf{{A}}]$", fontsize=14)

    ax = axes[1]
    ax.set_xlabel("$k$", labelpad=8)
    image = ax.matshow(np.imag(A), cmap="seismic", extent=extent)
    ax.set_title(r"Im$[\mathbf{{A}}]$", fontsize=14)

    cp_gap = 0.01  # gap between right axis and colorbar
    cp_width = 0.02  # width of the colorbar
    left = ax.get_position().x1 + cp_gap
    bottom = 0.12  # (hacky) by trial and error to align cb with image bottom
    width = cp_width
    height = 0.9*ax.get_position().height
    cax = fig.add_axes([left, bottom, width, height])
    plt.colorbar(image, cax=cax, shrink=0.5, format="%.2f")  # Similar to fig.colorbar(im, cax = cax)

    plt.suptitle(r"Matrix $\mathbf{{A}}$, $M={}$, $N={}$".format(M, N), fontsize=18)
    plt.subplots_adjust(top=0.68)
    if save_figures: plt.savefig(figure_dir + "matrix-wave-A-{}-{}_.png".format(M, N), dpi=200)
    plt.show()


def plot_solution_2d():
    M = 4  # number of time grid points (well technically there are N+1)
    t0 = 0
    tM = 1*np.pi
    times = np.linspace(t0, tM, M+1)

    N = 100  # number of position grid points (well technically there are N+1)
    x0 = 0
    xN = 2*np.pi
    x = np.linspace(x0, xN, N+1)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    remove_spines(ax)
    ax.set_xlabel(r"Position $x$", labelpad=-0.3)
    ax.set_ylabel("Amplitude $u$")

    ax.set_xticks(np.linspace(x0, xN, 5))
    labels = [r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_xticklabels(labels, fontsize="13")

    colors = ("#aad8c0", "#4badc6", "#2e6fa7", "#1d357f", "#111450")
    markers = ("o", "d", "P", "s", ".")
    alphas = np.linspace(1.0, 0.0, len(times), endpoint=False)
    for i in range(len(times)):
        t = times[i]
        # u = get_u_numeric(M, N)
        u = get_analytic_solution(t, x)  # use analytic solution for plotting (numeric and analytic give visually identical results but analytic is faster)
        ax.plot(x, u, c=colors[len(colors)-i-1], marker=markers[i], alpha=alphas[i], label=r"$t={:.1f}\pi$".format(t/np.pi), zorder=len(times)-i)
    ax.legend(framealpha=0.95, fontsize=11)

    # ax.set_title(r"1st Order Wave Equation Solution for $t \in [{:.0f}, {:.0f}\pi]$".format(t0/np.pi, tM/np.pi))
    ax.set_title(r"1st Order Wave Equation Solution for $t \in [{:.0f}, \pi]$".format(t0/np.pi))
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "wave-2d_.png", dpi=200)
    plt.show()


def plot_solution_3d():
    M = 80  # number of time grid points (well technically there are N+1)
    t0 = 0
    tM = 4*np.pi
    t = np.linspace(t0, tM, M+1)

    N = 200  # number of position grid points (well technically there are N+1)
    x0 = 0
    xN = 2*np.pi
    x = np.linspace(x0, xN, N+1)

    X, T = np.meshgrid(x, t, indexing="ij")
    U = get_analytic_solution(T, X)  # use analytic solution for plotting (numeric and analytic give visually identical results but analytic is faster)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, T, U, rstride=1, cstride=1, cmap="coolwarm")
    # ax.set_xlim(M[0], M[-1])
    # ax.set_ylim(N[-1], N[0])
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Time $t$")
    ax.set_zlabel("Amplitude $u$")
    # ax.view_init(16, -55)  # set view angle (elevation angle, azimuth angle)
    plt.suptitle("Solution to the 1st Order Wave Equation", y=0.89, fontsize=16)

    if save_figures: plt.savefig(figure_dir + "wave-3d_.png", dpi=200)
    plt.show()
# -----------------------------------------------------------------------------
# END PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------
def do_animation(M, N):
    t0 = 0
    tM = 6*np.pi
    t = np.linspace(t0, tM, M+1)

    x0 = 0
    xN = 2*np.pi
    x = np.linspace(x0, xN, N+1)

    T, X = np.meshgrid(t, x, indexing="ij")
    U = get_analytic_solution(T, X)  # returns M+1 rows of N+1 columns --- each row is u(x) at a fixed time t
    animate_graph(x, t, U, description="{:.2f}".format(tM))


def animate_graph(x, t, U, description=""):
    """
    Animates time evolution of wavedisplacement u as a function of time
    :param t: (, M) array of time values spanning simulation time
    :param x: (, N) array of position values (assumed to be [0, 2pi]
    :param U: 2D (M x N) matrix --- each row is u(x) at a time t
    :param description: for naming animation files when saving as mp4
    """

    iterations = len(t)  # number of times at which a solution is found
    u0 = U[0]  # initial wave distribution

    fig, ax = plt.subplots(figsize=(7, 4))
    remove_spines(ax)

    ax.set_xlabel(r"Position $\xi$", labelpad=-0.3)
    ax.set_ylabel("Amplitude $u$")
    ax.set_xticks(np.linspace(x[0], x[-1], 5))
    # labels = [r"0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    labels = [r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_xticklabels(labels, fontsize="13")
    ax.set_title(r"1st Order Wave Equation Solution for $t \in [{:.0f}, {:.0f}\pi]$".format(t[0]/np.pi, t[-1]/np.pi))

    time_label_template = r"Time: {:.2f}$\pi$"
    time_label = ax.text(0.05, 0.85, '', va='center', ha='left', transform=ax.transAxes,
                         bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    line, = ax.plot(x, u0)
    plt.tight_layout()

    temp_animation = animation.FuncAnimation(fig, update_animation, iterations,
                                             fargs=(U, line, time_label, time_label_template, iterations, t[-1]),
                                             interval=50, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, bitrate=1800, extra_args=['-vcodec', 'libx264'])
    if description != "": filename = "animate-wave-{}_.mp4".format(description)
    else: filename = "animate-wave_.mp4"
    temp_animation.save(figure_dir + filename, writer=writer)


def update_animation(i, U, line, time_label, time_label_template, iterations, end_time):
    """
    Auxiliary update function for the animation of the wave equation solution
    """
    print("Frame {} of {}".format(i, iterations))
    u = U[i]  # initial temperature
    line.set_ydata(u)
    time_label.set_text(time_label_template.format(((i+1) * end_time / iterations)/np.pi))
    return line
# -----------------------------------------------------------------------------
# START ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------


def practice():
    # N = 5
    # a = np.zeros((N, 1))
    # b = a[:, 0]
    # print(b)

    N = 50
    k = np.linspace(-N/2, N/2, N, endpoint=False)  # create indexes for wave vectors ("frequency" for sampling position)
    J = jv(k, np.pi)
    print(J)


if __name__ == "__main__":
    print("Hi, you're awesome!")
    # practice()
    # get_u_numeric(50)

    # plot_initial_condition()
    # plot_A(30, 50)
    # plot_solution_2d()
    # plot_solution_3d()
    do_animation(150, 100)
