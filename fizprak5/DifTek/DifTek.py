import numpy as np
from numpy.fft import fft
from numpy.fft import fftshift
from matplotlib import pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.integrate import quad

save_figures = True
data_dir = "measurements/"
fig_dir = "figures/"

color_blue = "#244d90"  # darker teal / blue
color_dark_orange = "#91331f"  # dark orange

a = 1.5  # [cm] cuvette width
b = 72.0  # [cm] distance between cuvette and screen
c = 35.5  # [cm] distance between laser and cuvette
delta_n = 0.029  # difference in refractive index between water and ethanol


def clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def linear_model_cf(t, k):
    """
    Linear model without y intercept for fitting with curve_fit
    :param t: independent variable
    :param k: slope
    :return:
    """
    return k*t


def linear_model_int_cf(t, k, b):
    """
    Linear model with y intercept for fitting with curve_fit
    :param t: independent variable
    :param k: slope
    :param b: y intercept
    :return:
    """
    return k*t + b


def gauss_model_centered(x, a, b):
    return a * np.exp(-x**2/(2*b**2))


def get_diffusion_curve_model(x, a, b):
    """ Gauss curve minus the line y = -x """
    return a * np.exp(-x**2/(2*b**2)) - x


def guess_curves():
    """ Draws best guesses for the y(x) diffusion curves
        provided with the online lab data. 
        Basically boils down to guessing standard deviations
        of Gauss bell curves
    """
    t, y_max = np.loadtxt(data_dir + "maxima.csv", delimiter=',', skiprows=1).T
    sigma_guess = np.linspace(0.055, 0.247, 9)
    x = np.linspace(-1.0, 1.0, 300)
    A = get_area_under_curve()

    fix, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    for i, sigma in enumerate(sigma_guess):
        y = gauss_model_centered(x, y_max[i], sigma)
        y = y - x  # subtract the line y = x
        ax.plot(x, y, label="t = " + str(t[i]) + " min")
    ax.legend()
    ax.set_xlabel("Position $z$ [cm]")
    ax.set_ylabel("Height $y$ [cm]")
    ax.set_title("Simulated Diffusion Curves", fontsize=16)
    ax.grid()

    time_labels = ["0", "3", "7", "13", "22", "37", "59", "93", "145"]
    time_positions = np.linspace(0, len(t), len(t))

    ax = axes[1]
    ax.bar(time_positions, A)
    ax.set_xlabel("Time $t$ [min]")
    ax.set_ylabel("Area Under Curve $S$ [cm$^2$]")
    ax.set_title("Area Under Curve", fontsize=16)
    ax.set_xticks(time_positions)
    ax.set_xticklabels(time_labels)

    plt.tight_layout()
    plt.grid()
    if save_figures: plt.savefig(fig_dir + "area_.png", dpi=200)
    plt.show()


def get_gauss_params():
    """ Amplitude and standard deviation of Gauss curves 
        modeling the diffusion curves in the provided online simulation data
    """
    a = np.loadtxt(data_dir + "maxima.csv", delimiter=',', skiprows=1, usecols=1)
    sigma = np.linspace(0.055, 0.247, 9)
    return a, sigma


def get_area_under_curve():
    """ Returns the area under the diffusion curves, modeled with Gauss curves) """
    a, sigma = get_gauss_params()
    xmin, xmax = -1.0, 1.0
    x = np.linspace(xmin, xmax, 100)
    t = np.loadtxt(data_dir + "maxima.csv", delimiter=',', skiprows=1, usecols=0)  # time values [min]
    A = np.zeros(len(a))
    for i in range(len(a)):
        A[i] = quad(get_diffusion_curve_model, xmin, xmax, args=(a[i], sigma[i]))[0]  # area under curve
    return A


def plot_y_max():
    t, y_max = np.loadtxt(data_dir + "maxima.csv", delimiter=',', skiprows=1).T
    y_err = 0.05 * y_max  # assume five percent error
    plt.figure(figsize=(7, 4))
    plt.errorbar(t, y_max, yerr=y_err, ls='--', c=color_blue, marker='o')
    plt.xlabel("Time $t$ [min]", fontsize=11)
    plt.ylabel(r"Maximum Height $y_{max}$ [cm]", fontsize=11)
    plt.grid()
    plt.title("Maximum Diffusion Curve Height vs. Time", fontsize=16)
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "ymax-vs-time_.png", dpi=200)
    plt.show()


def diffusion_constant_linear_fit():
    t, y_max = np.loadtxt(data_dir + "maxima.csv", delimiter=',', skiprows=1).T
    y_err = 0.05 * y_max  # assume five percent error
    k = (b + c)/c  # magnification
    S_theoretical = a*b*k*delta_n  # theoretically expected (constant) area under the curve
    # Q1 = ((S_theoretical/y_max)**2)/(4*np.pi*k**2)  # Q just stands for quantity
    Q2 = ((a*b*delta_n)**2)/(4*np.pi*y_max**2)
    Q_err = get_Q_error(y_max, y_err)

    param_guess = (2.0e-4, 0.001)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, t, Q2, p0=param_guess, sigma=Q_err, absolute_sigma=True)
    D, y_int = opt
    D_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
    print(D)
    print(D_err)

    # S_guess = get_area_under_curve()
    # Q_guess = ((S_guess/y_max)**2)/(4*np.pi*k**2)  # Q just stands for quantity
    # plt.plot(t, Q_guess, ls='--', c=color_dark_orange, marker='o', label='measured area')

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    # plt.plot(t, Q1, ls='--', c=color_blue, marker='o', label='Data (simulated)')
    plt.errorbar(t, Q2, yerr=Q_err, ls='--', c=color_blue, marker='o', label='Data (simulated)')

    t_fit = np.linspace(np.min(t), np.max(t), 200)
    Q_fit = linear_model_int_cf(t_fit, D, y_int)
    plt.plot(t_fit, Q_fit, ls='-', c=color_dark_orange, label='Fit: $y = Dt$\n$D = {:.2e}$ [min$^{{-1}}$]\n$\sigma_D = {:.2e}$ [min$^{{-1}}$]'.format(D, D_err))

    plt.xlabel("Time $t$ [min]", fontsize=11)
    plt.ylabel(r"Linearized Quantity $ Q = \frac{(ab\Delta n)^2}{4 \pi y^2}$", fontsize=11)
    plt.title("Finding Diffusion Constant with a Linear Fit", fontsize=18)
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "diffusion-constant-linfit_.png", dpi=200)
    plt.show()


def get_Q_error(y_max, u_y):
    """ Error estimate for the quantity Q plotted in the linear fit used to find diffusion constant
        Uses the definition Q2 = ((a*b*delta_n)**2)/(4*np.pi*y_max**2), with slightly simpler sensitivity coefficients
    """
    c_a = (2*a*(b*delta_n)**2)/(4*np.pi*y_max**2)
    c_b = (2*b*(a*delta_n)**2)/(4*np.pi*y_max**2)
    c_n = (2*delta_n*(a*b)**2)/(4*np.pi*y_max**2)
    c_y = ((a*b*delta_n)**2)/(2*np.pi*y_max**3)
    u_a = 0.1  # [cm]
    u_b = 0.5  # [cm]
    u_n = 0.001  # dimensionless
    return np.sqrt((c_a * u_a)**2 + (c_b * u_b)**2 + (c_n * u_n)**2 + (c_y * u_y)**2)


# guess_curves()
# get_gauss_params()
# get_area_under_curve()
diffusion_constant_linear_fit()
# plot_y_max()
