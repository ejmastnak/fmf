import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
plt.rc('axes', titlesize=16)    # fontsize of the x and y labels

save_figures = True
data_dir = "measurements/"
fig_dir = "figures/"

color_blue = "#244d90"  # darker teal / blue
color_dark_orange = "#91331f"  # dark orange

U_battery = 1.5  # [V] voltage of battery applied to Hall probe
sc_thickness = 0.95e-3  # [m]  semiconductor thickness
B = 0.173  # [T]
electron_charge = 1.602e-19  # [C]  elementary charge
kB = 8.617e-5  # [ev/K]  Boltzmann constant
# kB = 1.38e-23  # [J/K]  Boltzmann constant


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


def plot_hall_voltage():
    T, _, U1, U2 = np.loadtxt(data_dir + "measurements.txt", skiprows=1).T
    U_hall = 0.5*(U1 - U2)  # find hall voltage from U1 and U2; removes asymmetric voltage offset
    U_hall *= 1e3  # [V] to [mV]

    u_rel_err = 0.01
    u1, u2 = u_rel_err*U1, u_rel_err*U2  # one percent errors
    c1, c2 = 0.5, -0.5
    uH = np.sqrt((u1*c1)**2 + (u2*c2)**2)*1e3  # error in hall voltage, from V to mV

    plt.figure(figsize=(7, 3.5))
    # clean_axis(plt.gca())
    plt.errorbar(T, U_hall, xerr=0.02*T, yerr=uH, ls='--', marker='.', c=color_blue)
    plt.xlabel("Temperature $T$ [C]")
    plt.ylabel("Hall Voltage $U_H$ [mV]")
    plt.title("Dependence of Hall Voltage on Temperature")
    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "hall-voltage_.png", dpi=200)
    plt.show()


def plot_resistance():
    T, I = np.loadtxt(data_dir + "measurements.txt", skiprows=1, usecols=(0, 1)).T  # [C] and [mA]
    I *= 1e-3  # convert current from mA to A
    R = U_battery / I  # Ohm's law

    uU = 0.05  # [V] error of U_battery
    uI = 0.015 * I
    cU = 1/I
    cI = U_battery/(I**2)
    uR = np.sqrt((cU*uU)**2 + (cI*uI)**2)

    plt.figure(figsize=(7, 3.5))
    # clean_axis(plt.gca())
    plt.errorbar(T, R, xerr=0.02*T, yerr=uR, ls='--', marker='o', c=color_blue)
    plt.xlabel("Temperature $T$ [C]")
    plt.ylabel(r"Ohmic Resistance $R$ [$\Omega$]")
    plt.title("Semiconductor Resistance versus Temperature")
    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "resistance_.png", dpi=200)
    plt.show()


def plot_hall_coefficient():
    T, I, U1, U2 = np.loadtxt(data_dir + "measurements.txt", skiprows=1).T
    I *= 1e-3  # [mA] to [A]
    U_hall = 0.5*(U1 - U2)  # [V] find hall voltage from U1 and U2; removes asymmetric voltage offset
    R_hall = U_hall * sc_thickness / (I*B)
    R_hall_avg = np.mean(R_hall)

    u_rel_err = 0.01
    u1, u2 = u_rel_err*U1, u_rel_err*U2  # one percent errors
    c1, c2 = 0.5, -0.5
    uU = np.sqrt((u1*c1)**2 + (u2*c2)**2)  # [V]
    uc = 0.05 * 1e-3  # [mm] to [m]
    uI = 0.015 * I  # [A]
    uB = 0.002  # [T]

    cU = sc_thickness/(I*B)
    cc = U_hall/(I*B)
    cI = U_hall*sc_thickness/(B*I**2)
    cB = U_hall*sc_thickness/(I*B**2)

    R_hall_err = np.sqrt((uU*cU)**2 + (uc*cc)**2 + (uI*cI)**2 + (uB*cB)**2)

    plt.figure(figsize=(7, 3.5))
    plt.errorbar(T, R_hall, xerr=0.02*T, yerr=R_hall_err, ls='--', marker='o', c=color_blue, label="data")
    plt.hlines(R_hall_avg, T[0], T[-1], ls='--', zorder=-1, label="average")
    plt.text(T[0], 0.96*R_hall_avg, r"Average: $R_H \approx {:.2e}$ m$^3$/C".format(R_hall_avg), va='bottom', ha='left',
             bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlabel("Temperature $T$ [C]")
    plt.ylabel(r"Hall Coefficient $R_H$ [m$^3$/C]")
    plt.title("Hall Coefficient versus Temperature")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "hall-coefficient_.png", dpi=200)
    plt.show()


def plot_carrier_density_midT():
    """ Test validity of relationship n_carrier = N_donor for medium temperatures k_B T \approx E_d"""
    T, I, U1, U2 = np.loadtxt(data_dir + "measurements.txt", skiprows=1).T
    I *= 1e-3  # mA to A
    U_hall = 0.5*(U2 - U1)  # find hall voltage from U1 and U2; removes asymmetric voltage offset
    n_carrier = I*B / (sc_thickness * electron_charge * U_hall)
    T += 273  # convert Celsius to Kelvin

    U_rel_err = 0.01  # voltage relative error
    u1, u2 = U_rel_err*U1, U_rel_err*U2  # one percent errors
    c1, c2 = 0.5, -0.5
    uU = np.sqrt((u1*c1)**2 + (u2*c2)**2)  # [V]  Hall voltage error
    uc = 0.05 * 1e-3  # [mm] to [m]
    uI = 0.015 * I  # [A]
    uB = 0.002  # [T]

    cU = (I*B)/(sc_thickness*electron_charge*U_hall**2)
    cc = (I*B)/(U_hall*electron_charge*sc_thickness**2)
    cI = B/(sc_thickness*electron_charge*U_hall)
    cB = I/(sc_thickness*electron_charge*U_hall)
    n_err = np.sqrt((uU*cU)**2 + (uc*cc)**2 + (uI*cI)**2 + (uB*cB)**2)

    N_midT = 7  # number of points to take
    T_midT = T[0:N_midT]
    n_midT = n_carrier[0:N_midT]
    n_err_midT = n_err[0:N_midT]
    n_avg = np.mean(n_midT)
    n_err_avg = np.mean(n_err_midT)

    # hacky way to neatly write carrier density, with error, in scientific notation
    n_mantissa = n_avg/1e20  # I know ahead of time result is of order 1e20
    err_mantissa = n_err_avg/1e20  # I know ahead of time result is of order 1e20
    annotation = r"Average: $N_d \approx ({:.2f} \pm {:.2f}) \cdot 10^{{20}}$ m$^{{-3}}$".format(n_mantissa, err_mantissa)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.plot(T, n_carrier, ls='--', marker='.', c="#999999", label="complete data")
    plt.errorbar(T_midT, n_midT, yerr=n_err_midT, ls='--', marker='d', c=color_dark_orange, label="mid-temp data")
    plt.hlines(n_avg, T[0], T[-1], ls='--', zorder=-1, label="mid-temp average")
    plt.annotate(annotation, xy=(301.5, 2.65e20), xytext=(T[6], 2*n_avg),
                 va='bottom', ha='center', fontsize=11, arrowprops=dict(facecolor='black', width=1, headwidth=10),
                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlabel("Temperature $T$ [K]")
    plt.ylabel(r"Carrier Density $n$ [m$^{-3}$]")
    plt.title("Mid-Temperature Charge Carrier Density")
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "carrier-density-mid_.png", dpi=200)
    plt.show()


def plot_carrier_density_highT():
    T, I, U1, U2 = np.loadtxt(data_dir + "measurements.txt", skiprows=1).T
    I *= 1e-3  # mA to A
    U_hall = 0.5*(U2 - U1)  # find hall voltage from U1 and U2; removes asymmetric voltage offset
    n_carrier = I*B / (sc_thickness * U_hall)  # work in eV units with e0 = 1
    log_n = np.log(n_carrier)
    T += 273  # convert Celsius to Kelvin
    kBT = kB * T

    U_rel_err = 0.01  # voltage relative error
    u1, u2 = U_rel_err*U1, U_rel_err*U2  # one percent errors
    c1, c2 = 0.5, -0.5
    uU = np.sqrt((u1*c1)**2 + (u2*c2)**2)  # [V]  Hall voltage error
    uc = 0.05 * 1e-3  # [mm] to [m]
    uI = 0.015 * I  # [A]
    uB = 0.002  # [T]

    cU = 1/U_hall
    cc = 1/sc_thickness
    cI = 1/I
    cB = 1/B
    log_n_err = np.sqrt((uU*cU)**2 + (uc*cc)**2 + (uI*cI)**2 + (uB*cB)**2)

    N_highT = 8  # number of high temperature points to take
    kBT_highT = kB * T[-N_highT:]  # high temperature terms
    log_n_highT = np.log(n_carrier[-N_highT:])
    log_n_err_highT = log_n_err[-N_highT:]

    slope_guess = (kBT_highT[-1] - kBT_highT[0])/(log_n_highT[-1] - log_n_highT[0])
    intercept_guess = 50
    opt, cov = curve_fit(linear_model_int_cf, 1/kBT_highT, log_n_highT, p0=(slope_guess, intercept_guess), sigma=log_n_err_highT, absolute_sigma=True)
    slope, intercept = opt
    slope_err = cov[0][0] ** 0.5  # square root of diagonal entry

    # E_g = -2.0*slope/electron_charge
    Eg = -2.0*slope  # [eV]
    Eg_err = 2*slope_err
    print(slope_err)
    print(Eg_err)

    xx = np.linspace(np.min(1/kBT_highT) - 0.5, np.max(1/kBT_highT+0.8), 100)
    yy = linear_model_int_cf(xx, slope, intercept)

    plt.figure(figsize=(7, 4))
    clean_axis(plt.gca())
    plt.plot(1/kBT, log_n, ls=':', marker='.', c="#999999", label="complete data")
    plt.errorbar(1/kBT_highT, log_n_highT, yerr=log_n_err_highT, ls='--', marker='d', markersize=6, c=color_dark_orange, label="high temp. data")
    plt.plot(xx, yy, ls='--', c=color_blue, lw="2", label="high temp. fit")
    plt.annotate(r"Slope: ${:.2f} \pm {:.2f}$ eV".format(slope, slope_err) + "\n" +
                 r"Band gap: $E_g \approx {:.2f} \pm {:.2f}$ eV".format(Eg, Eg_err),
                 xy=(33.86, 4.52), xytext=(34.5, 4.75),
                 va='center', ha='left', fontsize=11, arrowprops=dict(facecolor='black', width=1, headwidth=8),
                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlabel("Inverse Thermal Energy $1/k_BT$ [eV$^{-1}$]")
    plt.ylabel(r"Carrier Density ln$\,n$")
    plt.title("High-Temperature Charge Carrier Density")
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(fig_dir + "carrier-density-high_.png", dpi=200)
    plt.show()


def practice():
    N = 10
    a = np.arange(1, N + 1, 1)
    print(a)
    b = a[-5:]
    print(b)


# practice()
plot_hall_voltage()
# plot_resistance()
# plot_hall_coefficient()
# plot_carrier_density_highT()
# plot_carrier_density_midT()
