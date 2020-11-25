import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.optimize import curve_fit

save_figures = True
data_dir = "measurements/"
fig_dir = "figures/"

color_re = "#244d90"  # darker teal / blue
color_im = "#91331f"  # dark orange


def linear_model_int_cf(t, k, b):
    """
    Linear model with y intercept for fitting with curve_fit
    :param t: independent variable
    :param k: slope
    :param b: y intercept
    :return:
    """
    return k*t + b


def plot_response():
    """Plots real and imaginary components of electro-Optical response
        as functions of word signal frequency"""
    for filename in sorted(os.listdir(data_dir)):
        if ".txt" in filename:  # ignore hidden system files
            analyzer_angle = filename.replace(".txt", "")
            data = np.loadtxt(data_dir + filename, skiprows=1)
            # start, stop = 30, data.shape[0] - 6
            start, stop = 1, data.shape[0] - 1
            w = 2 * np.pi * data[:,0][start:stop]
            psi_re = data[:,1][start:stop]
            psi_im = data[:,2][start:stop]

            plt.figure(figsize=(7,3.5))
            plt.xlabel("Angular Frequency $\omega$ [Hz]")
            plt.ylabel("Electro-Optical Response")
            plt.plot(w, psi_re, color=color_re, ls='--', marker='o', label="real component")
            plt.plot(w, psi_im, ls='--', color=color_im, marker='d', label="imaginary component")
            plt.legend()
            plt.grid()
            plt.title("Real and Imaginary Components of Electro-Optical Response $\phi={}$".format(analyzer_angle))
            plt.tight_layout()
            if save_figures: plt.savefig(fig_dir + "real-im-plot-{}_.png".format(analyzer_angle), dpi=200)
            plt.show()


def find_tau_analytic():
    """ Finds crystal's relaxation time constant tau"""
    for filename in sorted(os.listdir(data_dir)):
        if ".txt" in filename:  # ignore hidden system files
            analyzer_angle = filename.replace(".txt", "")
            data = np.loadtxt(data_dir + filename, skiprows=1)
            # start, stop = 30, data.shape[0] - 6
            start, stop = 27, data.shape[0] - 1
            w = 2 * np.pi * data[:,0][start:stop]
            psi_re = data[:,1][start:stop]
            psi_im = data[:,2][start:stop]

            psi0 = 2 * np.max(np.abs(psi_im))
            psi0_index = np.argmin(np.abs(psi_im) - psi0)
            psi0 *= np.sign(psi_im[psi0_index])
            tau = np.sqrt((psi0/psi_re) - 1)/w

            w_err = 0.02  # assume 2 percent error since no other information is given
            psi_re_err = 0.03 * psi_re
            psi0_err = 0.05 * psi0  # assume 5 percent error for extremum value
            tau_err = get_tau_analytic_error(w, w_err, psi0, psi0_err, psi_re, psi_re_err)

            tau_mean = np.mean(tau) * 1e3  # convert s to ms
            tau_err_mean = np.mean(tau_err) * 1e3  # convert s to ms
            print(tau_err_mean)

            plt.figure(figsize=(7, 4))
            plt.errorbar(w, tau, yerr=tau_err, color=color_re, ls='--', marker='.')
            plt.annotate(r"Average: $\tau = {:.2f} \pm {:.2f}$ ms".format(tau_mean, tau_err_mean), xy=(0.5, 0.75),
                         xycoords='axes fraction', fontsize=14, va="center", ha="center",
                         bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

            plt.xlabel(r"Angular Frequency $\omega$ [Hz]")
            plt.ylabel(r"Relaxation Time $\tau$")
            plt.grid()
            plt.title(r"Finding Relaxation Time Analytically $\phi={}$".format(analyzer_angle), fontsize=16)
            if save_figures: plt.savefig(fig_dir + "tau-analytic-{}_.png".format(analyzer_angle), dpi=200)

            plt.show()


def get_tau_analytic_error(w, u_w, psi0, u_psi0, psi_re, u_psi_re):
    """ Sensitivity coefficient estimate for relaxation time using the analytic approach via psi0 """
    c_w = np.sqrt((psi0/psi_re) - 1)/(w**2)
    c_psi0 = 1/(np.sqrt((psi0/psi_re) - 1)*2*w*psi_re)
    c_re = psi0/(np.sqrt((psi0/psi_re) - 1)*2*w*(psi_re**2))
    return np.sqrt((c_w*u_w)**2 + (c_psi0*u_psi0)**2 + (c_re*u_psi_re)**2)


def find_tau_linear_fit():
    """Find's liquid crystal relaxation time tau with a linear fit of the ratio
        of the real and imaginary components of the electro-optical response"""
    for filename in sorted(os.listdir(data_dir)):
        if ".txt" in filename:  # ignore hidden system files
            analyzer_angle = filename.replace(".txt", "")
            data = np.loadtxt(data_dir + filename, skiprows=1)
            skip_index = np.shape(data)[1] - 15  # how many of the last few points to skip; ratio grows unstable
            w = 2*np.pi * data[:,0][0:skip_index]  # convert frequency to angular frequency
            psi_re = data[:,1][0:skip_index]
            psi_im = data[:,2][0:skip_index]
            ratio = psi_im / psi_re
            psi_accuracy = 0.03  # assume psi data is accurate to 2 percent
            re_err = psi_re * psi_accuracy
            im_err = psi_im * psi_accuracy
            c_re = psi_im / (psi_re**2)
            c_im = 1/psi_re
            ratio_err = np.sqrt((c_im * im_err)**2 + (c_re * re_err)**2)

            guess = (0.02, 0)  # guess for slope and y intercept
            opt, cov = curve_fit(linear_model_int_cf, w, ratio, guess, sigma=ratio_err, absolute_sigma=True)
            tau, b = opt  # unpack slope and y intercept
            tau_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
            f_fit = np.linspace(np.min(w), np.max(w), 200)
            ratio_fit = linear_model_int_cf(f_fit, tau, b)
            tau, tau_err = tau * 1e3, tau_err * 1e3  # convert s to ms

            plt.figure(figsize=(7,4))
            plt.xlabel("Angular Frequency $\omega$ [Hz]")
            plt.ylabel(r"Ratio $\psi_{Im} / \psi_{Re}$")
            plt.errorbar(w, ratio, yerr=ratio_err, color=color_re, ls='--', marker='.', label="data")
            plt.plot(f_fit, ratio_fit, color=color_im, ls='-', label='Fit: y = $\\tau\omega$ + b\n$\\tau$ = {:.2f} $\pm$ {:.2f} ms'.format(tau, tau_err))
            plt.legend()
            plt.grid()
            plt.title("Finding $\\tau$ with a Linear Fit $\phi={}$".format(analyzer_angle), fontsize=16)
            plt.tight_layout()
            if save_figures: plt.savefig(fig_dir + "linear-fit-{}_.png".format(analyzer_angle), dpi=200)
            plt.show()


def find_tau_circle():
    """Plots real and imaginary components of electro-optical response 
        in the complex plane with frequency as a parameter.
        The circle's center is used to find crystal's relaxation time tau"""
    for filename in sorted(os.listdir(data_dir)):
        if ".txt" in filename:  # ignore hidden system files
            analyzer_angle = filename.replace(".txt", "")
            data = np.loadtxt(data_dir + filename, skiprows=1)
            w = 2 * np.pi * data[:,0]  # convert frequency to angular frequency
            psi_re = data[:,1]
            psi_im = data[:,2]

            # center at Re[psi] corresponding to min(Im[psi])
            center_index = np.argmax(np.abs(psi_im))
            w_center = w[center_index]
            print(1/w_center)
            print(psi_re[center_index])
            print(w_center/(2*np.pi))

            plt.figure(figsize=(7,4))
            plt.xlabel(r"Real Component Re[$\psi$]")
            plt.ylabel(r"Imaginary Component Im[$\psi$]")
            plt.plot(psi_re, psi_im, ls='--', color=color_re, marker='o')
            plt.grid()
            plt.title(r"Electro-Optical Response in the Complex Plane $\phi={}$".format(analyzer_angle))
            if save_figures: plt.savefig(fig_dir + "circle-{}_.png".format(analyzer_angle), dpi=200)
            plt.show()


# plot_response()
find_tau_analytic()
# find_tau_linear_fit()
# find_tau_circle()
