import numpy as np
import matplotlib.pyplot as plt
import os

measurement_directory = "measurements/"
figure_directory = "figures/"
filename = "U-I-1.txt"
save_figures = False

d = .177  # [m] inductor diagonal distance
d_error = 0.03  # [m] error in d
mu0 = 1.2566e-6  # SI units vacuum permeability
muB = 9.274e-24  # [SI] Bohr magneton
h = 6.626e-34  # [SI] Planck's constant
N = 1557  # number of inductor coils

def I_to_B(I):
    """
    Converts current through the inductor to magnetic field along the central axis
    Current I should be in amperes!
    """
    return N * mu0 * I / d

def plot_data():
    data = np.loadtxt(measurement_directory + filename, delimiter=',', skiprows=1)
    I = data[:, 0]
    U1 = data[:, 1]
    U2 = data[:, 2]
    U = np.mean([U1, U2], axis=0)
    plt.xlabel("Inductor Current I")
    plt.ylabel("Lock-In Voltage U")
    plt.plot(I, U, marker=".", linestyle='--', label=filename.replace(".txt", ""))
    plt.legend(loc="best")
    plt.show()


def plot_U_B():
    """
    Plots lock-in amplifier voltage U vs magnetic field B in the inductor
    Input a file with columns I[mA], U_min [mV], U_max [mV]
    """
    colors = ("#74c1c4", "#2e6fa7", "#161d63")
    i = 0
    plt.figure(figsize=(7.5, 4))
    for filename in sorted(os.listdir(measurement_directory)):
        if "U-I-" in filename:
            data = np.loadtxt(measurement_directory + filename, delimiter=',', skiprows=1)
            I = data[:, 0] / 1000  # convert mA to A
            U1 = data[:, 1]  # [mV]
            U2 = data[:, 2]  # [mV]
            U_error = np.abs(U2-U1)/2
            U = np.mean([U1, U2], axis=0)  # take average of min and max values
            B = I_to_B(I)*1000  # convert current to magnetic field and T to mT
            B_error = get_B_error(I, 0.5e-3, d_error) 
            print(B_error)
            w = float(filename.replace(".csv","").replace("U-I-", ""))  # extract frequency from file name 
            plt.title("Absorption Line Derivative")
            plt.xlabel("Inductor Magnetic Field $B$ [mT]")
            plt.ylabel("Lock-In Voltage $U$ [mV]")
            plt.errorbar(B, U, xerr=B_error, yerr=U_error, linestyle="--", color=colors[i], marker='.', label="Frequency: {:.1f} [MHz]".format(w))
            i += 1


    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_figures: plt.savefig(figure_directory + "lockin-vs-B_.png", dpi=200)
    plt.show()


def get_c_I():
    return N * mu0 / d  # sensitivy coefficient for current when finding B


def get_c_d(I):
    return N * mu0 * I / (d**2)  # coefficient for diagonal distance d when finding B


def get_B_error(I, u_I, u_d):
    c_I = get_c_I()
    c_d = get_c_d(I*1e-3)  # convert mA to A
    return np.sqrt((c_I*u_I)**2 + (c_d*u_d)**2) * 1e3  # convert T to mT
    

def get_c_w(B0):
    return 1/B0


def get_c_B0(w0, B0):
    return w0/(B0**2)


def get_R_error(w0, u_w, B0, u_B):
    c_w = get_c_w(B0)
    c_B = get_c_B0(w0, B0)
    return np.sqrt((c_w*u_w)**2 + (c_B*u_B)**2)


def get_g_error(w0, u_w, B0, u_B):
    c_w = get_c_w(B0)
    c_B = get_c_B0(w0, B0)
    return (h/muB) * np.sqrt((c_w*u_w)**2 + (c_B*u_B)**2) * 1e9  # fix SI units

def interpolate_B0():
    extrema_data = np.loadtxt(measurement_directory + "extrema.csv", delimiter=',', skiprows=1)
    w = extrema_data[:,0]
    B_min, B_max = extrema_data[:,1], extrema_data[:,3]
    U_min, U_max = extrema_data[:,2], extrema_data[:,4]
    k = (U_max - U_min)/(B_max - B_min)
    B0 = B_min - (U_min / k)  # finds intersection of U with B = 0
    # print(B0)
    return w, B0  # returns resonance frequency [MHz] and B field [mT]


def get_g_factor():
    u_w = 0.5  # MHz
    u_B0 = 0.02  # mT
    w, B0 = interpolate_B0()
    return h * (w*1e6) / (muB * (B0*1e-3)), get_g_error(w, u_w, B0, u_B0)


def get_resonance_ratio():
    w, B0 = interpolate_B0()  # MHz and mT
    u_w = 0.5  # MHz
    u_B0 = 0.02  # mT
    return (w/B0), get_R_error(w, u_w, B0, u_B0)  # [GHz/T]


plot_U_B()
# interpolate_B0()
print(get_g_factor())
# print(get_resonance_ratio())
