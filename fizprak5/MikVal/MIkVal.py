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

h_res = 408  # resonator cavity screw position [dimensionless] at resonance
x_min, x_max = 0.0, 5.7  # [cm] start and end positions of sliding measurement probe
waveguide_width = 2.3  # [cm]
c_vacuum = 3.0e8  # [m/s]


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


def fit_microwave_frequency():
    """ Estimates microwave frequency from resonance cavity screw position
        and the corresponding (but rather meager) calibration table
    """
    h = [100, 300, 500]  # screw positions
    f = [10.0, 9.0, 8.0]  # frequency [GHz]

    param_guess = (0.005, 1)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, h, f, p0=param_guess)
    k, b = opt  # slope, y intercept
    print(k, b)
    # k_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
    f_res = linear_model_int_cf(h_res, k, b)

    h_fit = np.linspace(h[0], h[-1], 100)
    f_fit = linear_model_int_cf(h_fit, k, b)

    plt.figure(figsize=(7, 3.5))
    ax = plt.gca()
    clean_axis(ax)
    plt.plot(h, f, ls='--', marker='o', c=color_blue, label='calibration data')
    plt.plot(h_fit, f_fit, ls='-', linewidth=1, c=color_dark_orange, label='linear fit', zorder=-1)
    plt.scatter(h_res, f_res, marker='o', edgecolors=color_blue, linewidths=2, facecolors='none', s=900)
    plt.scatter(h_res, f_res, marker='+', color=color_blue, s=100)

    plt.annotate(r"Microwave frequency: $f \approx {:.2f}$ GHz".format(f_res), xy=(h_res, f_res),
                 xytext=(330, 8.15), fontsize=11, va="center", ha="right",
                 arrowprops=dict(facecolor='black', width=1, headwidth=10),
                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    plt.xlabel("Screw Position $h$")
    plt.ylabel("Frequency [GHz]")
    plt.title("Estimating Microwave Frequency")
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "frequency_.png", dpi=200)
    plt.show()


def tuning_curve_analysis():
    U_x, U_sc, U_bol = np.loadtxt(data_dir + "signals.txt", delimiter=" ", skiprows=1).T  # position, short-circuit termination and bolometer signals, all in volts
    x = np.linspace(x_min, x_max, len(U_x))
    bol_max_indeces = find_peaks(U_bol, prominence=1)  # indeces of bolometer maxima
    bol_min_indeces = find_peaks(-1.0*U_bol, prominence=1.5)  # indeces of bolometer minima
    sc_min_indeces = find_peaks(-1.0*U_sc, prominence=3)  # indeces of short-circuit minima

    bol_max = U_bol[bol_max_indeces[0]][0]  # first bol maximum
    bol_min = U_bol[bol_min_indeces[0]][0]  # first bol minimum
    sc_min_x = x[sc_min_indeces[0]][0]
    bol_max_x = x[bol_max_indeces[0]][0]
    bol_min_x = x[bol_min_indeces[0]][0]
    x_min_difference = sc_min_x - bol_min_x
    print("h_min: {:.2f}\t h_max: {:.2f}\t x'_min: {:.2f}".format(bol_min, bol_max, x_min_difference))

    # standing wave ratio
    s = np.sqrt(bol_min) / np.sqrt(bol_max)
    c_min = np.sqrt(bol_max/bol_min)/(2*bol_max)
    c_max = np.sqrt(bol_max/bol_min)/(2*(bol_max**2))
    u_min, u_max = 0.03*bol_min, 0.03*bol_max
    s_err = np.sqrt((c_min * u_min)**2 + (c_max*u_max)**2)
    print("SWR: {:.3f} +/- {:.3f}".format(s, s_err))

    delta_x_sc = (x[sc_min_indeces[0]][1] - x[sc_min_indeces[0]][0])
    l_wg = 2*delta_x_sc  # wavelength in the waveguide
    print("SC minima separation: {:.3f} [cm]\t Wavelength': {:.3f} [cm]".format(l_wg/2, l_wg))

    # frequency error analysis
    c_xsc = - 1e2*c_vacuum/(2*(delta_x_sc**3)*np.sqrt(1/(delta_x_sc**2) + 1/(waveguide_width**2)))  # convert m/s to cm/s
    c_a = - 1e2*c_vacuum/(2*(delta_x_sc**3)*np.sqrt(1/(delta_x_sc**2) + 1/(waveguide_width**2)))  # waveguide width
    u_xsc = 0.04 * delta_x_sc
    u_a = 0.05  # [cm]
    f_err = np.sqrt((c_xsc*u_xsc)**2 + (c_a*u_a)**2)

    l_vacuum = l_wg / np.sqrt(1 + 0.25*(l_wg/waveguide_width)**2)
    print("Wavelength: {:.2f} [cm]".format(l_vacuum))
    f = (c_vacuum/l_vacuum) * 1e2  # convert to m/s
    print("Frequency: {:.2e} +/- {:.2e} [Hz]".format(f, f_err))
    print("")

    # product \beta xmin
    beta_xmin = 2 * np.pi * x_min_difference / l_wg
    c_sc = np.pi * x_min_difference / (delta_x_sc**2)
    c_min = np.pi / delta_x_sc
    u_min = 0.04 * x_min_difference
    beta_xmin_err = np.sqrt((c_sc * u_xsc)**2 + (c_min * u_min)**2)
    print("beta*x'_min: {:.2f} +/- {:.2f}".format(beta_xmin, beta_xmin_err))

    # reactance
    reactance = (s**2 - 1)*np.tan(beta_xmin)/(1 + (s**2)*(np.tan(beta_xmin)**2))  # reactance normalized by Z0
    tan_b = np.tan(beta_xmin)
    sec2_b = 1/(np.cos(beta_xmin)**2)
    c_s = 2*s*tan_b*(1 + tan_b**2)/((1 + (s**2)*(tan_b**2))**2)
    c_beta = (sec2_b - (s**2)*sec2_b*(tan_b**2))*(s**2 - 1)/((1 + (s**2)*(tan_b**2))**2)
    reactance_err = np.sqrt((c_s*s_err)**2 + (c_beta*beta_xmin_err)**2)

    # resistance
    resistance = s*(1 - reactance*np.tan(beta_xmin))
    c_s = 1 - reactance * tan_b
    c_beta = s*reactance * sec2_b
    c_reactance = s * tan_b
    resistance_err = np.sqrt((c_s*s_err)**2 + (c_beta*beta_xmin_err)**2 + (c_reactance*reactance_err)**2)

    print("Resistance/Z0: {:.2f} +/- {:.2f}".format(resistance, resistance_err))
    print("Reactance/Z0: {:.2f} +/- {:.2f}".format(reactance, reactance_err))


def plot_tuning_curve():
    U_x, U_sc, U_bol = np.loadtxt(data_dir + "signals.txt", delimiter=" ", skiprows=1).T  # position, short-circuit termination and bolometer signals, all in volts
    x = np.linspace(x_min, x_max, len(U_x))
    bol_max_indeces = find_peaks(U_bol, prominence=1)  # indeces of bolometer nuumaxima
    bol_min_indeces = find_peaks(-1.0*U_bol, prominence=1.5)  # indeces of bolometer minima
    sc_min_indeces = find_peaks(-1.0*U_sc, prominence=3)  # indeces of short-circuit minima

    bol_max = U_bol[bol_max_indeces[0]][0]  # first bol maximum
    bol_min = U_bol[bol_min_indeces[0]][0]  # first bol minimum
    sc_min_x = x[sc_min_indeces[0]][0]
    bol_max_x = x[bol_max_indeces[0]][0]
    bol_min_x = x[bol_min_indeces[0]][0]
    x_min_difference = sc_min_x - bol_min_x

    # plt.plot(U_x, ls='--', marker='.')
    # plt.plot(U_sc, ls='--', marker='.')

    plt.figure(figsize=(7, 3.5))
    clean_axis(plt.gca())
    plt.plot(x, U_sc, ls='--', marker='.', c=color_blue, label="short-circuit")
    plt.plot(x, U_bol, ls='--', marker='d', c=color_dark_orange, markersize=5, label="external load")
    plt.text(bol_min_x, bol_min - 0.15, r"$h_{{min}} = {:.2f}$".format(bol_min), va="top", ha="center",
             bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    plt.text(bol_max_x, bol_max + 0.15, r"$h_{{max}} = {:.2f}$".format(bol_max), va="bottom", ha="center",
             bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    bol_min_x = x[bol_min_indeces[0]][1]
    plt.annotate('', xy=(sc_min_x, 0), xycoords='data',
                 xytext=(bol_min_x, 0), textcoords='data',
                 arrowprops=dict(facecolor='black', lw=2.0, arrowstyle='<->'))
    plt.vlines(bol_min_x, 0.0, bol_min-0.1, ls=':', color="black", linewidth=2.0)
    plt.text((bol_min_x + sc_min_x)/2, 0.18, r"$x'_{{min}} = {:.2f}$".format(x_min_difference), va="bottom", ha="center",
             bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    plt.xlabel("Position $x$ [cm]", labelpad=-0.1)
    plt.ylabel("Signal $U$ [V]")
    plt.title("Short-Circuit and Load-Terminated Signals")
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "curves_.png", dpi=200)
    plt.show()


def plot_mode_powers():
    """ Plots power at each klystron mode """
    U_x, U_sc, U_bol = np.loadtxt(data_dir + "signals.txt", delimiter=" ", skiprows=1).T  # position, short-circuit termination and bolometer signals, all in volts
    bol_max_indeces = find_peaks(U_bol, prominence=1)  # indeces of bolometer maxima
    bol_min_indeces = find_peaks(-1.0*U_bol, prominence=1.5)  # indeces of bolometer minima
    bol_max = U_bol[bol_max_indeces[0]][0]  # first bol maximum
    bol_min = U_bol[bol_min_indeces[0]][0]  # first bol minimum

    s = np.sqrt(bol_min) / np.sqrt(bol_max)  # SWR for load termination
    c_min = np.sqrt(bol_max/bol_min)/(2*bol_max)
    c_max = np.sqrt(bol_max/bol_min)/(2*(bol_max**2))
    u_min, u_max = 0.03*bol_min, 0.03*bol_max
    s_err = np.sqrt((c_min * u_min)**2 + (c_max*u_max)**2)

    reflection = ((1-s)/(1+s))**2  # squared absolute value of reflection coefficient for load-termination

    # load reflection voltage at each mode and power measured on the bolometer
    U_mode, P_bol = np.loadtxt(data_dir + "voltage-power.csv", delimiter=',', skiprows=1).T  # [V] and [mW]
    u_U = 2  # assume 2 V error for mode voltage
    u_bol = 0.2  # assume 0.2 mW error for bolometer power
    P_wave = P_bol / (1 - reflection)  # convert measured power to wave power

    c_bol = (s**2 + 2*s + 1)/(4*s)
    c_s = P_bol * (s+1)*(s-1)/(4*s**2)
    P_err = np.sqrt((c_bol*u_bol)**2 + (c_s * s_err)**2)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.errorbar(np.abs(U_mode), P_wave, xerr=u_U, yerr=P_err, ls='none', lw=2, color=color_blue)
    for i, P in enumerate(P_wave):
        U = np.abs(U_mode[i])
        plt.text(U, P + 0.4, r"${:.2f} $ mW".format(P), va='bottom', ha='center',
                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    plt.xlabel("Reflection Voltage $U$ [V]")
    plt.ylabel("Microwave Power $P$ [mW]")
    plt.title("Microwave Power at Klystron Modes")
    plt.tight_layout()
    plt.savefig(fig_dir + "mode-powers_.png", dpi=200)
    plt.show()


# guess_curves()
# fit_microwave_frequency()
# plot_tuning_curve()
# tuning_curve_analysis()
plot_mode_powers()
