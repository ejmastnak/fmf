import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os
import re

data_dir = "measurements/"
figures_dir = "figures/"
save_figures = True
skip_rows = 11  # number of rows to skip for data files (11-line header)


def gauss_model(x,a,x0,sigma):
    """
    Model for Gauss curve used with curve_fit
     a is coefficient
     x0 is mean
     sigma is standard deviation
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_background():
    """ Estimate of background activity for gamma rays near sodium's 0.551 MeV peak"""
    data = np.loadtxt(data_dir + "background.txt", skiprows=1)  # small window around 0.551 MeV
    T = 790  # [s] measurement time
    return np.mean(data)/T

def tdc_resolution():
    data = np.loadtxt(data_dir + "tdc-resolution/tdc-resolution.txt", skiprows=skip_rows)
    time = data[:,0]
    counts = data[:,1]
    tmin = 185
    tmax = 375
    tmin_index = np.argmin(np.abs(time-tmin))
    tmax_index = np.argmin(np.abs(time-tmax))
    time = time[tmin_index:tmax_index]
    counts = counts[tmin_index:tmax_index]
    count_err = np.sqrt(counts)  # for radiation counts estimate error as square root of count number
    
    plt.figure(figsize=(7,4))
    plt.xlabel("Time $t$ [ps]")
    plt.ylabel("Counts $N$")
    # plt.xlim(185,375)
    plt.bar(time, counts, label="Count Data")

    a_guess = 2000  # [counts]
    mean_guess = np.mean(time)  # [ps]
    var_guess = 25  # [ps]

    opt,pcov = curve_fit(gauss_model,time,counts,p0=[a_guess,mean_guess,var_guess])
    # opt,pcov = curve_fit(gauss_model,time,counts,p0=[a_guess,mean_guess,var_guess], sigma=count_err, absolute_sigma=True)
    a, x0, sigma = np.abs(opt)
    u_sigma = np.sqrt(pcov[2,2]) # sigma is third diagonal element of covariance matrix
    x = np.linspace(np.min(time), np.max(time), 250)
    y = gauss_model(x,a, x0, sigma)
    plt.plot(x,y, linewidth=3, label="Fit: $N=a\exp[-(t-t_0)^2/2\sigma^2)]$\n$\sigma^2$={:.0f} $\pm$ {:.0f} ps".format(sigma, u_sigma))
    plt.legend()

    plt.title("TDC Resolution", fontsize=16)
    plt.tight_layout()
    if save_figures: plt.savefig(figures_dir + "tdc-resolution_.png", dpi=200)
    plt.show()


def plot_poisson():
    """
    Plots the Poisson distribution of single-channel radioactive decay measurements
    """
    data = np.loadtxt(data_dir + "single-channel/poisson1.txt", skiprows=skip_rows)
    time = data[:,0]/1e9  # convert picoseconds to milliseconds
    counts = data[:,1]
    plt.figure(figsize=(7,4))
    plt.xlabel("Time Interval $t$ [ms]")
    plt.ylabel("Counts $N$")

    (markers, stemlines, baseline) =  plt.stem(time, counts)
    plt.setp(markers, marker='o', markerfacecolor="C0", markeredgecolor="none", markersize=6)
    plt.setp(baseline, linestyle="-", color="C0")
    plt.setp(stemlines, linestyle="-", color="C0", linewidth=1.5)

    plt.title("Distribution of Radioactive Decays", fontsize=16)
    plt.tight_layout()
    if save_figures: plt.savefig(figures_dir + "poisson-histogram_.png", dpi=200)
    plt.show()


def linear_model(t, k, b):
    """Model for fitting y(t) = kt + b"""
    return t*k + b


def fit_poisson():
    """
    Fits a Poisson distribution to the single-channel radioactive decay measurements
    """
    data = np.loadtxt(data_dir + "single-channel/poisson1.txt", skiprows=skip_rows)
    time = data[:,0]/1e9  # convert picoseconds to milliseconds
    counts = data[:,1]
    first_zero_index = np.where(counts < 10)[0][0]  # to cut off data where counts become very small (avoid log0)
    time = time[0:first_zero_index]
    log_counts = np.log(counts[0:first_zero_index])
    log_count_err = np.sqrt(counts[0:first_zero_index])/log_counts  # error propagated through ln(N)

    guess = (2.5, 8)
    opt, p_cov = curve_fit(linear_model, time, log_counts, guess, sigma=log_count_err, absolute_sigma=True)
    k,b = opt # unpack fitted parameters
    k_err = np.sqrt(p_cov[0][0])
    b_err = np.sqrt(p_cov[1][1])

    x2 = np.linspace(np.min(time), np.max(time), 100)
    y2 = linear_model(x2, k, b)

    plt.figure(figsize=(7,4))
    plt.xlabel("Time Interval $t$ [ms]")
    plt.ylabel("Logarithm of Counts log$(N)$")
    plt.errorbar(time, log_counts, yerr=log_count_err, color="#AAAAAA", linestyle='--', marker='o', markersize=7, label='data', zorder=-1)
    plt.plot(x2, y2, linestyle='-', linewidth=4, color="C0", label='Fit: log$(N) = -Rt + b$\n$R = {:.2f} \pm {:.2f}$ [ms$^{{-1}}$]\n$b = {:.2f} \pm {:.2f}$ '.format(-1*k, k_err, b, b_err), zorder=1)
    plt.title("Linearized Distribution of Radioactive Decays", fontsize=16)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save_figures: plt.savefig(figures_dir + "poisson-fit_.png", dpi=200)
    plt.show()


def get_count_rate_error(N, n):
    """ Error in R_measured = N_total / t_measurement """
    u_N = np.sqrt(N)/np.sqrt(n)
    t = 30  # [s] all measurements performed over 30 seconds
    u_t = 0.05  # [s]
    c_N = 1/t
    c_t = N/(t**2)
    return np.sqrt((c_N*u_N)**2 + (c_t*u_t)**2)


def do_random():
    """ Compares measured and theoretically predicted random coincidence rates """
    for filename in os.listdir(data_dir + "random/"):
        if ".txt" in filename:
            data = np.loadtxt(data_dir + "random/" + filename, skiprows=skip_rows)
            times = data[:,0]*1e-12  # [ps to s]  bin times
            counts = sum(data[:,1])
            T = 30.1  # measurement time [s]
            N1, N2 = 80735, 53970
            R1 = N1/T
            R2 = N2/T
            R_rand_measured = counts/T - get_background()  # measured random count rate
            R_meas_err = get_count_rate_error(counts, 5)
            tau = times[-1] - times[0]  # [s] width of measurement window
            R_rand_theoretical = R1*R2*tau
            c1, c2 = R2*tau, R1 * tau
            u1 = get_count_rate_error(N1, 5)
            u2 = get_count_rate_error(N2, 1)
            R_theor_err = np.sqrt((c1*u1)**2 + (c2*u2)**2)

            print("File: {}\tWindow: {:.3f} [ns]".format(filename, tau*1e9))  # convert time to ns
            print("Measured random count rate: {:.3f} +/- {:.3f}[/s]\t Theoretical rate: {:.3f} +/- {:.3f}[/s]".format(R_rand_measured, R_meas_err, R_rand_theoretical, R_theor_err))


def zero_angle_resolution():
    plt.figure(figsize=(7,4))
    plt.xlabel("Time $t$ [ns]")
    plt.xlim((-200, 200))
    plt.ylabel("Counts $N$")

    data = pd.read_html(data_dir + "ang-cor-25/0.html", skiprows=0)[0].to_numpy()  # load HTML table as numpy array
    time = data[:,0]  # [ns] (even though simulator says [ps])
    counts = data[:,1]
    plt.bar(time, counts, label="Count Data")

    a_guess = 8000  # [counts]
    mean_guess = np.mean(time)  # [ns]
    var_guess = 35  # [ns]

    opt,pcov = curve_fit(gauss_model,time,counts,p0=[a_guess,mean_guess,var_guess])
    a, x0, sigma = np.abs(opt)
    u_sigma = np.sqrt(pcov[2,2]) # sigma is third diagonal element of covariance matrix
    x = np.linspace(np.min(time), np.max(time), 250)
    y = gauss_model(x,a, x0, sigma)
    plt.plot(x,y, linewidth=3, label="Fit: $N=a\exp[-(t-t_0)^2/2\sigma^2)]$\n$\sigma^2$={:.0f} $\pm$ {:.0f} ns".format(sigma, u_sigma))
    plt.legend()

    plt.title("Resolution for Colinear Scintillators ($\phi = 0$)", fontsize=16)
    plt.tight_layout()
    if save_figures: plt.savefig(figures_dir + "ang-zero-resolution_.png", dpi=200)
    plt.show()


def plot_ang_correlation():
    plt.figure(figsize=(7, 4))
    plt.xlabel("Angular Displacement [degrees]")
    plt.ylabel("Coincidence Rate [s$^-1$]")
    colors = ("#2e6fa7", "#161d63") 
    markers = ("o", "d")
    for j, separation in enumerate((15, 25)): 
        file_dir = data_dir + "ang-cor-{}/".format(separation)
        angles = np.zeros(17)  # there are 17 measurements
        coincidences = np.zeros(17)
        i = 0
        for filename in natural_sort(os.listdir(file_dir)):
            if ".html" in filename:
                data = pd.read_html(file_dir + filename)[0].to_numpy()  # load HTML table as numpy array
                angles[i] = int(filename.replace(".html", ""))
                coincidences[i] = np.sum(data[:,1]) / 30  # convert count to count rate [s^-1]
                i += 1
        plt.plot(angles, coincidences, color=colors[j], marker=markers[j], linewidth=2, linestyle='--', label="separation: {} cm".format(separation))
        plt.legend()
    
    plt.grid()
    plt.title("Angular Correlation of Annihilation Gamma Rays")
    if save_figures: plt.savefig(figures_dir + "ang-correlation.png", dpi=200)
    plt.show()



def natural_sort(l):
    """For sorting filenames in natural order"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


def practice():
    a = np.array([1, 3, 4, 2, 4, 3, 5])
    print(np.where(a==3)[0][0])


# practice()
# tdc_resolution()
# plot_poisson()
# fit_poisson()
do_random()
# zero_angle_resolution()
# plot_ang_correlation()
