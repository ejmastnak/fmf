import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

measurement_directory = "measurements/"
figures_directory = "../figures/"
sca_directory = measurement_directory + "sca/"
mca_directory = measurement_directory + "mca/"
peak_directory = measurement_directory + "peaks/"
save_figures = False

# Energies of known sodium spectral lines, used for calibration
E1 = 0.511  # [MeV]
E2 = 1.277  # [MeV]
level1 = 1.55  # pm 0.2
level2 = 3.8  # pm 0.2
channel1 = 264  # pm 1
channel2 = 636  # pm 1

def plot_sca():
    markers = ("o", "s", "d")
    colors = ("#bb2205", "#ffa62b", "#16697a")
    fig = plt.figure(figsize=(7,4))
    for i, filename in enumerate(os.listdir(sca_directory)):
        if ".csv" in filename:
            data = np.loadtxt(sca_directory + filename, delimiter=',', skiprows=1)
            lower_level = data[:, 0]
            counts = data[:, 1]
            plt.plot(lower_level, counts, color=colors[i], marker=markers[i], linestyle='--', label=filename.replace(".csv",""))

    plt.xlabel("Lower level (uncalibrated)")
    plt.ylabel("Counts")
    plt.title("Single-Channel Analyzer")
    plt.legend(loc="best")
    plt.tight_layout()
    if save_figures: plt.savefig(figures_directory + "sca_.png", dpi=200)
    plt.show()


def plot_sca_energy():
    markers = ("o", "s", "d")
    colors = ("#bb2205", "#ffa62b", "#16697a")
    fig = plt.figure(figsize=(7,4))
    for i, filename in enumerate(os.listdir(sca_directory)):
        if ".csv" in filename:
            data = np.loadtxt(sca_directory + filename, delimiter=',', skiprows=1)
            lower_level = data[:, 0]
            counts = data[:, 1]
            plt.plot(lower_level, counts, color=colors[i], marker=markers[i], linestyle='--', label=filename.replace(".csv",""))

    plt.xlabel("Lower level (uncalibrated)")
    plt.ylabel("Counts")
    plt.title("Single-Channel Analyzer")
    plt.legend(loc="best")
    plt.tight_layout()
    if save_figures: plt.savefig(figures_directory + "sca_.png", dpi=200)
    plt.show()


def E_sca(lower_level):
    """
    Interpolates energy as a function of lower-level on SCA
    using the known energy of Na22's spectrum peaks
    """
    k = (E2-E1)/(level2-level1)
    return E1 + k*(lower_level - level1)


def get_E_sca_error(level):
    Delta_l1 = level2 - level1
    Delta_l2 = level - level1
    u_l = 0.4
    c1 = Delta_l2 * (E2-E1) / (Delta_l2**2)
    c2 = (E2-E1) / Delta_l1
    return np.sqrt((u_l*c1)**2 + (u_l*c2)**2)


def print_E_sca_errors():
    for level in (2.4, 4.4, 5.0):  # cesium, cobal, cobalt
        print("Energy: {:.3f} +/- {:.3f} MeV".format(E_sca(level), get_E_sca_error(level)))

def E_mca(lower_level):
    """
    Interpolates energy as a function of lower-level on MCA
    using the known energy of Na22's spectrum peaks
    """
    E2 = 1.277  # [MeV]
    level2 = 636  # pm 1
    E1 = 0.511  # [MeV]
    level1 = 264  # pm 1
    k = (E2-E1)/(level2-level1)
    return E1 + k*(lower_level - level1)


def get_E_mca_error(channel):
    Delta_c1 = channel2 - channel1
    Delta_c2 = channel - channel1
    u_c = 2  # error \pm 1 on each channel, difference has error 2
    c1 = Delta_c2 * (E2-E1) / (Delta_c2**2)
    c2 = (E2-E1) / Delta_c1
    return np.sqrt((u_c*c1)**2 + (u_c*c2)**2)


def print_E_mca_errors():
    for channel in (339,589,667):  # cesium, cobal, cobalt
        print("Energy: {:.3f} +/- {:.3f} MeV".format(E_mca(channel), get_E_mca_error(channel)))


def get_background_activity():
    data = np.loadtxt(mca_directory + "background.txt", skiprows=12)
    live_time, real_time = data[0], data[1]
    counts = data[2:-1]  # rows 3 to end
    return counts/live_time  # returns activity


def plot_mca():
    background = get_background_activity()  # get background activity
    plt.figure(figsize=(7,4))
    colors = ("#bb2205", "#ffa62b", "#16697a")
    i = 0
    for filename in os.listdir(mca_directory):
        if "background" in filename:
            continue
        if ".txt" in filename:
            data = np.loadtxt(mca_directory + filename, skiprows=12)
            live_time, real_time = data[0], data[1]
            counts = data[2:-1]  # rows 3 to end
            channel = np.linspace(1, len(counts), len(counts))
            activity = counts/live_time - background
            # np.savetxt(mca_directory + "hi-" + filename, np.column_stack((channel, activity)))
            plt.plot(channel, activity, color=colors[i], marker=".", linestyle='--', label=filename.replace(".txt", "")+"\nlive time: {:.2f} s".format(live_time))
            i = i+1

    plt.title("Multi-Channel Analyzer")
    plt.xlabel("Channel")
    plt.ylabel("Activity [counts per second]")
    plt.legend(loc="best")
    plt.tight_layout()
    if save_figures: plt.savefig(figures_directory + "mca_.png", dpi=200)
    plt.show()


def plot_cesium_mca():
    background = get_background_activity()  # get background activity
    plt.figure(figsize=(7,4))
    data = np.loadtxt(mca_directory + "cesium.txt", skiprows=12)
    live_time, real_time = data[0], data[1]
    counts = data[2:-1]  # rows 3 to end
    channel = np.linspace(1, len(counts), len(counts))
    activity = counts/live_time - background
    energy = E_mca(channel)
    start_channel = 50
    end_channel = 400
    energy = energy[start_channel: end_channel]
    activity = activity[start_channel: end_channel]

    avg_pts = 5  # moving average of noisy activity signal
    box = np.ones(avg_pts) / avg_pts
    activity_avg = np.convolve(activity, box, mode='same')

    plt.plot(energy, activity, color='#999999', marker='.', linestyle='none', alpha=0.5, label="raw data")
    plt.plot(energy, activity_avg, color='black', linestyle='-', label="{}-point moving avg".format(avg_pts))

    plt.annotate('Backscatter', xy=(0.19, 4), xytext=(0.15, 8), fontsize=12, arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Compton edge', xy=(0.48, 2.1), xytext=(0.44, 8), fontsize=12, arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Photopeak', xy=(0.65, 18.7), xytext=(0.43, 16.5), fontsize=12, arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title("Cesium Spectrum with MCA", fontsize=16)
    plt.xlabel("Energy [Mev]", fontsize=11)
    plt.ylabel("Activity [counts per second]", fontsize=11)
    plt.legend(loc="best")
    plt.tight_layout()
    if save_figures: plt.savefig(figures_directory + "cesium_mca.png", dpi=200)
    plt.show()


def gauss_model(x,a,x0,sigma):
    """
    Model for Gauss curve used with curve_fit
     a is coefficient
     x0 is mean
     sigma is standard deviation
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def plot_resolutions():
    """
    Fits a bell (gauss) curve to each photopeak. FWHM is used to estimate energy resolution
    """
    i = 0
    colors = ("#112d4e", "#53354a", "#903749", "#e84545")
    plt.figure(figsize=(7,4))
    for filename in os.listdir(peak_directory):
        if ".txt" in filename:
            data = np.loadtxt(peak_directory + filename, delimiter=' ')
            channel = data[:,0]
            energy = E_mca(channel)
            activity = data[:,1]
            plt.plot(energy, activity, color=colors[i], alpha=0.35, marker='o', mew=0.0, linestyle='none')

            line_energy = np.mean(energy)
            u_E = get_line_energy_error_for_source(filename.replace(".txt",""))

            opt,pcov = curve_fit(gauss_model,energy,activity,p0=[10,line_energy,1])
            a, x0, sigma = np.abs(opt)
            u_sigma = np.sqrt(pcov[2,2]) # sigma is third diagonal element of covariance matrix
            fwhm = 2.355 * sigma 
            u_fwhm = 2.355 * u_sigma
            u_R = get_resolution_error(line_energy, u_E, fwhm, u_fwhm)
            print("\n" + filename)
            print("Resolution: {:.3f} +/- {:.3f}".format(fwhm/line_energy, u_R))
            print(u_fwhm)
            x = np.linspace(np.min(energy), np.max(energy), 250)
            y = gauss_model(x,a, x0, sigma)
            plt.plot(x,y, linewidth=3, color=colors[i],label=filename.replace(".txt", " MeV").replace("-", " ")+"\nFWHM={:.3f} $\pm$ {:.3f} MeV".format(fwhm, u_fwhm))
            i = i+1

    plt.title("Photopeaks with MCA", fontsize=16)
    plt.xlabel("Energy [Mev]", fontsize=11)
    plt.ylabel("Activity [counts per second]", fontsize=11)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()
    if save_figures: plt.savefig(figures_directory + "resolutions_.png", dpi=200)
    plt.show()


def get_line_energy_error_for_source(source_name):
    """
    Returns line energy uncertainty in MeV for each source. Uses values calculated earlier to
    avoid recalculating from scratch
    """
    if "sodium" in source_name:
        return 0.001
    elif "cobalt-1.332" in source_name:
        return 0.006
    elif "cobalt-1.173" in source_name:
        return 0.006
    elif "cesium" in source_name:
        return 0.021
    else: return 0.001


def get_resolution_error(E, u_E, DE, u_DE):
    c_DE = 1/E
    c_E = DE/(E**2)
    return np.sqrt((u_DE*c_DE)**2 + (u_E*c_E)**2)

 # print_E_mca_errors()
plot_mca()
# plot_cesium_mca()
# print(E_sca(2.4))
# plot_resolutions()
