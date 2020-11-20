import numpy as np
import matplotlib.pyplot as plt
import os

measurement_directory = "../measurements/"
ionization_dir = measurement_directory + "ionization/"
polarization_dir = measurement_directory + "polarization/"
figures_dir = "../figures/"
save_figures = True
R = 1e9  # [Ohm]  output resistor
V_cell = 16*3*9*1e-6  # [m^3]
rho_air = 1.225  # [kg/m^3]

def voltage_to_current(V):
    """
    Converts voltage drop across output resistor to ionization current through cell
    """
    return V / R

def plot_ionization():
    plt.figure(figsize=(7,4))
    colors = ("#74c1c4", "#2e6fa7", "#161d63")
    i = 0
    for filename in sorted(os.listdir(ionization_dir)):
        if ".csv" in filename:
            data = np.loadtxt(ionization_dir + filename, delimiter=',', skiprows=1)
            U_cell = data[:,0]
            U_out = (data[:,1] + data[:,2])/2 # average of min and max value 
            U_error = 0.1 + np.abs(data[:,1] - data[:,2])/2  # half of difference between max and min value, plus 0.1 error [V]
            I = 1e9*voltage_to_current(U_out)  # convert A to nA
            I_error = 1e9 * get_I_error(U_out, U_error)  # convert A to nA
            plt.errorbar(U_cell, I, xerr=0.1, yerr=I_error, marker='o',linestyle='--', color=colors[i], label=filename.replace("ionization-","").replace(".csv"," kV"))
            i = i + 1
    plt.xlabel("Applied Cell Voltage [V]")
    plt.ylabel("Ionization Current [nA]")
    plt.title("Ionization Curves", fontsize=16)
    plt.legend()
    if save_figures: plt.savefig(figures_dir + "ionization_.png", dpi=200)
    plt.show()


def plot_dose_rate():
    """
    Plots exposition dose rate in the ionization cell
    """
    colors = ("#e86f83", "#94247d", "#3e0c5f")
    i = 0
    plt.figure(figsize=(7,4))
    for filename in sorted(os.listdir(ionization_dir)):
        if ".csv" in filename:
            data = np.loadtxt(ionization_dir + filename, delimiter=',', skiprows=1)
            U_cell = data[:,0]
            U_out = (data[:,1] + data[:,2])/2 # average of min and max value 
            U_error = 0.1 + np.abs(data[:,1] - data[:,2])/2  # half of difference between max and min value, plus 0.1 error [V]
            I = voltage_to_current(U_out)
            I_error = get_I_error(U_out, U_error) 
            dose_rate = get_dose_rate(1e6*I)  # convert A to uA
            dose_rate_error = 1e6*get_dose_rate_error(I, I_error)  # convert from A/kg to uA/kg
            plt.errorbar(U_cell, dose_rate, xerr=0.1, yerr=dose_rate_error, color=colors[i], marker='o',linestyle='--', label=filename.replace("ionization-","").replace(".csv"," kV"))
            i = i + 1
    plt.xlabel("Applied Cell voltage [V]")
    plt.ylabel("Estimate Dose Rate [$\mu$A/kg]")
    plt.title("Dose Rates in Ionization Cell", fontsize=16)
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(figures_dir + "dose-rate_.png", dpi=200)
    plt.show()


def get_dose_rate(I_ion):
    """
    Calculates exposition dose rate in the ionization cell 
    as a function of ionization current
    Uses the dimensions of the cell and density of air
    """
    return I_ion / (rho_air*V_cell)


def get_I_error(U, u_U):
    """
    Error in ionization cell current
    Input output voltage and its error in volts
    """
    u_R = 0.15 * R  # 15 percent error for lack of more knowledge
    c_U = 1/R
    c_R = U/(R**2)
    return np.sqrt((u_U*c_U)**2 + (u_R*c_R)**2)  # output in amperes


def get_dose_rate_error(I, u_I):
    """
    Error in exposition dose rate in ionization cell
    Input current and its error in amperes
    """
    c_I = 1/(rho_air*V_cell)
    c_rho = I/((rho_air**2)*V_cell)
    c_V = I/(rho_air*(V_cell**2))
    u_rho = 0.01  # kg/m^3
    u_V = 0.1*V_cell  # m^3
    return np.sqrt((u_I*c_I)**2 + (u_rho*c_rho)**2 + (u_V*c_V)**2)

def calc_polarization_averages():
    counts_table = np.zeros((4,7))  # rows are each orientation, columns are runs 1-5, avg and std
    i = 0
    for filename in os.listdir(polarization_dir):
        if ".txt" in filename:
            counts = np.loadtxt(polarization_dir + filename, skiprows=1)
            avg = np.mean(counts)
            std = np.std(counts, ddof=1)
            counts_table[i] = np.hstack([counts,avg,std])
            print("\n" + filename)
            print("Average counts: {:.2f}".format(np.mean(counts)))
            i = i + 1
    np.savetxt(measurement_directory + "counts_table.csv", counts_table, delimiter=',', fmt='%.2f') 


def calc_polarization_error():
    Np = 5639
    Nm = 293
    u = 16
    cp = Nm/(Np**2)
    cm = 1 / Np
    error = np.sqrt((u*cp)**2 + (u*cm)**2)
    print("Error: {:.3f}".format(error))

    Np = 8.38
    Nm = 5.22
    u = 0.4
    cp = Nm/(Np**2)
    cm = 1 / Np
    error = np.sqrt((u*cp)**2 + (u*cm)**2)
    print("Error: {:.3f}".format(error))


# plot_ionization()
# plot_dose_rate()
# calc_polarization_averages()
calc_polarization_error()
