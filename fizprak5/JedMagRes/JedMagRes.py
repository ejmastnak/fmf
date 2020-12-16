import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
plt.rc('axes', titlesize=16)    # fontsize of the x and y labels

save_figures = True
data_dir = "measurements/"
fig_dir = "figures/"

color_blue = "#244d90"  # darker teal / blue
color_dark_orange = "#91331f"  # dark orange


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


def plot_calibrate1x():
    """
    Calibrates TAU dial position to time for 1x amplification
    """
    tau, t = np.loadtxt(data_dir + "calibration-1x.txt", skiprows=2).T
    d_tau = 0.02 * tau
    dt = 0.02 * t
    k_guess = (t[-1] - t[0])/(tau[-1] - tau[0])
    b_guess = t[0]  # first time at tau = 0 is y intercept

    param_guess = (k_guess, b_guess)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, tau, t, p0=param_guess)
    k, b = opt  # slope, y intercept
    print(k, b)
    # k_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element

    tau_fit = np.linspace(tau[0], tau[-1], 100)
    t_fit = linear_model_int_cf(tau_fit, k, b)

    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    clean_axis(ax)
    plt.xlabel("TAU Dial Position")
    plt.ylabel("Time [ms]")
    plt.title("Time Calibration at 1x Amplification")
    plt.errorbar(tau, t, xerr=d_tau, yerr=dt, ls='none', marker='.', c=color_blue, label='calibration data')
    fit_label = "linear fit\n" + r"$t = {:.2f} \tau + {:.2f}$".format(k, b)
    plt.plot(tau_fit, t_fit, ls='-', linewidth=1, c=color_dark_orange, label=fit_label, zorder=-1)
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "calibration-1x_.png", dpi=200)
    plt.show()


def plot_calibrate100x():
    """
    Calibrates TAU dial position to time for 100x amplification
    """
    tau, t = np.loadtxt(data_dir + "calibration-100x.txt", skiprows=2).T
    d_tau = 0.02 * tau
    dt = 0.02 * t

    k_guess = (t[-1] - t[0])/(tau[-1] - tau[0])
    b_guess = t[0]  # first time at tau = 0 is y intercept

    param_guess = (k_guess, b_guess)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, tau, t, p0=param_guess)
    k, b = opt  # slope, y intercept
    print(k, b)
    # k_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element

    tau_fit = np.linspace(tau[0], tau[-1], 100)
    t_fit = linear_model_int_cf(tau_fit, k, b)

    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    clean_axis(ax)
    plt.xlabel("TAU Dial Position")
    plt.ylabel("Time [ms]")
    plt.title("Time Calibration at 100x Amplification")
    plt.errorbar(tau, t, xerr=d_tau, yerr=dt, ls='none', marker='.', c=color_blue, label='calibration data')
    fit_label = "linear fit\n" + r"$t = {:.2f} \tau + {:.2f}$".format(k, b)
    plt.plot(tau_fit, t_fit, ls='-', linewidth=1, c=color_dark_orange, label=fit_label, zorder=-1)
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "calibration-100x_.png", dpi=200)
    plt.show()


def calibrate(tau, amplification):
    """
    Converts TAU dial position to time for either 1x or 100x amplification
    tau is a 1D array of dial position values
    amplification is either 1 or 100 (int)
    """
    tau_calibration, t = np.loadtxt(data_dir + "calibration-{}x.txt".format(amplification), skiprows=2).T

    k_guess = (t[-1] - t[0])/(tau_calibration[-1] - tau_calibration[0])
    b_guess = t[0]  # first time at tau_calibration = 0 is y intercept

    param_guess = (k_guess, b_guess)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, tau_calibration, t, p0=param_guess)
    k, b = opt  # slope, y intercept
    # k_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element

    t = linear_model_int_cf(tau, k, b)
    return t


def plot_T1_ion_data():
    """ Plots amplitude of paramagnetic water free precession signal versus time tau between pi/2 pulses"""
    tau, U = np.loadtxt(data_dir + "T1-ion.txt", skiprows=2).T
    t = calibrate(tau, 1)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.xlabel(r"Pulse Separation $\tau$ [ms]")
    plt.ylabel("Free Precession Amplitude $U$")
    plt.title("Free Precession Signal of Paramagnetic Water")
    plt.plot(t, U, c=color_blue, ls='--', marker='.')
    plt.tight_layout()

    if save_figures: plt.savefig(fig_dir + "T1-ion-data_.png", dpi=200)
    plt.show()


def plot_T1_distilled_data():
    """ Plots amplitude of distilled water free precession signal versus time tau between pi/2 pulses"""
    tau, U = np.loadtxt(data_dir + "T1-distilled.txt", skiprows=2).T
    t = calibrate(tau, 100)
    t *= 1e-3  # convert ms to s

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.xlabel(r"Pulse Separation $\tau$ [s]")
    plt.ylabel("Free Precession Amplitude $U$")
    plt.title("Free Precession Signal of Distilled Water")
    plt.plot(t, U, c=color_blue, ls='--', marker='.')
    plt.tight_layout()

    if save_figures: plt.savefig(fig_dir + "T1-distilled-data_.png", dpi=200)
    plt.show()


def plot_T1_ion_fit():
    """ Finds spin-lattice relaxation time T1 for paramagnetic water """
    tau, U = np.loadtxt(data_dir + "T1-ion.txt", skiprows=2).T
    t = calibrate(tau, 1)
    max_index = np.argmax(U)
    U0 = U[max_index]  # use largest U value to estimate value of asymptotic approach
    extra_offset = 2  # take off a few more noisy terms
    t, U = t[:max_index - extra_offset], U[:max_index - extra_offset]  # trim last few terms to avoid log(0) when finding Q

    Q = -np.log(1 - U/U0)
    dQ = 0.05 * Q  # update later

    k_guess = (Q[-1] - Q[0])/(t[-1] - t[0])
    b_guess = -1

    param_guess = (k_guess, b_guess)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, t, Q, p0=param_guess)
    k, b = opt  # slope, y intercept
    k_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
    T1 = 1/k
    dT1 = k_err/(k**2)

    t_fit = np.linspace(t[0], t[-1], 100)
    Q_fit = linear_model_int_cf(t_fit, k, b)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.xlabel(r"Pulse Separation $\tau$ [ms]")
    plt.ylabel("Linearized Amplitude $-0.5$ln[$1-U/U_0$]")
    plt.title("Paramagnetic Water Spin-Lattice Relaxation Time $T_1$")
    plt.errorbar(t, Q, yerr=dQ, ls='--', marker='.', c=color_blue, label="data")
    fit_label = "linear fit\n" + r"$y = {:.2f} \tau {:.2f}$".format(k, b)  # note \tau b not \tau + b b/c b is negative with a minus sign built-in
    plt.plot(t_fit, Q_fit, ls='-', linewidth=2, c=color_dark_orange, label=fit_label, zorder=-1)
    plt.text(0.02, 0.7, r"$T_1 \approx {:.2f} \pm {:.2f}$ ms".format(T1, dT1),
             va='center', ha='left', transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "T1-ion-fit_.png", dpi=200)
    plt.show()


def plot_T1_distilled_fit():
    """ Finds spin-lattice relaxation time T1 for distilled water """
    tau, U = np.loadtxt(data_dir + "T1-distilled.txt", skiprows=2).T
    t = calibrate(tau, 100)
    t *= 1e-3  # ms to s
    max_index = np.argmax(U)
    U0 = 1.1*U[max_index]  # use a bit larger (data isn't at its asymptote yet) than largest U value to estimate value of asymptotic approach
    extra_offset = 1  # take off a few more noisy terms
    t, U = t[:max_index - extra_offset], U[:max_index - extra_offset]  # trim last few terms to avoid log(0) when finding Q

    Q = -np.log(1 - U/U0)
    dQ = 0.05 * Q  # update later

    k_guess = (Q[-1] - Q[0])/(t[-1] - t[0])
    b_guess = -1

    param_guess = (k_guess, b_guess)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, t, Q, p0=param_guess, sigma=dQ, absolute_sigma=True)
    k, b = opt  # slope, y intercept
    k_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
    T1 = 1/k
    dT1 = k_err/(k**2)

    t_fit = np.linspace(t[0], t[-1], 100)
    Q_fit = linear_model_int_cf(t_fit, k, b)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.xlabel(r"Pulse Separation $\tau$ [s]")
    plt.ylabel("Linearized Amplitude $-0.5$ln[$1-U/U_0$]")
    plt.title("Distilled Water Spin-Lattice Relaxation Time $T_1$")
    plt.errorbar(t, Q, yerr=dQ, ls='--', marker='.', c=color_blue, label="data")
    fit_label = "linear fit\n" + r"$y = {:.2f} \tau + {:.2f}$".format(k, b)
    plt.plot(t_fit, Q_fit, ls='-', linewidth=2, c=color_dark_orange, label=fit_label, zorder=-1)
    plt.text(0.02, 0.7, r"$T_1 \approx {:.2f} \pm {:.2f}$ s".format(T1, dT1),
             va='center', ha='left', transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "T1-distilled-fit_.png", dpi=200)
    plt.show()


def plot_T2_fit():
    """ Finds spin-spin relaxation time T2 for paramagnetic water """
    tau, U = np.loadtxt(data_dir + "T2-ion.txt", skiprows=2).T
    t = calibrate(tau, 1)
    t, U = t[:-1], U[:-1]  # trim last (noisy) term with negative voltage (no negative logarithms!)

    Q = -0.5 * np.log(U)
    dQ = 0.05 * Q  # update later

    k_guess = (Q[-1] - Q[0])/(t[-1] - t[0])
    b_guess = -1

    param_guess = (k_guess, b_guess)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, t, Q, p0=param_guess)
    k, b = opt  # slope, y intercept
    k_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
    T2 = 1/k
    dT2 = k_err/(k**2)

    t_fit = np.linspace(t[0], t[-1], 100)
    Q_fit = linear_model_int_cf(t_fit, k, b)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.xlabel(r"Pulse Separation $\tau$ [ms]")
    plt.ylabel("Linearized Amplitude $-0.5$ln$U$")
    plt.title("Finding Spin-Spin Relaxation Time $T_2$")
    plt.errorbar(t, Q, yerr=dQ, ls='--', marker='.', c=color_blue, label="data")
    fit_label = "linear fit\n" + r"$y = {:.2f} \tau + {:.2f}$".format(k, b)
    plt.plot(t_fit, Q_fit, ls='-', linewidth=2, c=color_dark_orange, label=fit_label, zorder=-1)
    plt.text(0.02, 0.7, r"$T_2 \approx {:.2f} \pm {:.2f}$ ms".format(T2, dT2),
             va='center', ha='left', transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "T2-fit_.png", dpi=200)
    plt.show()


def plot_T2_data():
    """ Plots amplitude of spin echo signal versus time tau between pi/2 and pi pulses"""
    tau, U = np.loadtxt(data_dir + "T2-ion.txt", skiprows=2).T
    t = calibrate(tau, 1)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.xlabel(r"Pulse Separation $\tau$ [ms]")
    plt.ylabel("Spin-Echo Amplitude $U$")
    plt.title("Spin-Echo Signal of Paramagnetic Water")
    plt.plot(t, U, c=color_blue, ls='--', marker='.')
    plt.tight_layout()

    if save_figures: plt.savefig(fig_dir + "T2-data_.png", dpi=200)
    plt.show()


# plot_calibrate1x()
# plot_calibrate100x()
# plot_T1_ion_data()
# plot_T1_distilled_data()
# plot_T1_ion_fit()
# plot_T1_distilled_fit()
plot_T2_fit()
# plot_T2_data()
