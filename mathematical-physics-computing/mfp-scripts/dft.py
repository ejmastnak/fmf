import numpy as np
import os
import re
import time
import soundfile as sf
from numpy.fft import fftshift
from numpy.fft import ifftshift
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

color_signal = "#244d90"  # darker teal / blue
color_idft = "#3997bf"  # lighter teal/blue

color_real = "#91331f"  # dark orange
color_im = "#f5c450"  # light orange
color_dft = "db4823" # vivid orange
color_ref = "#AAAAAA"  # light grey
color_raw = "#5e2c21"
color_zp = "#c23a1f"
marker_re = 'o'
marker_im = 'd'

note_cmap = LinearSegmentedColormap.from_list("test", ["#feec91", "#82001c"], N=6)
note_colors = ("#fde961", "#f2b549", "#e78136", "#de5126", "#a7361f", "#562b22")

data_dir = "../4-dft/data/"
txt_dir = data_dir + "bach-txt/"
time_dir = data_dir + "times/"
first_note_dir = data_dir + "bach-first-note/"
figure_dir = "../4-dft/figures/"
note_fig_dir = figure_dir + "first-note/"
save_figures = True


# START DFT FUNCTIONS

def dft_loop(signal, shift=True):
    """Calculates DFT of np array signal with a double loop"""
    N = signal.shape[0]  # number of samples
    dft = np.zeros(N, dtype=complex)  # preallocate
    for k in range(N):  # k indexes the DFT
        f_k = 0.0
        for n in range(N):  # n indexes the signal
            f_k += signal[n] * np.exp(-2j * np.pi * k * n / N)
        dft[k] = f_k
    if shift: return fftshift(dft)
    else: return dft


def dft_matrix(signal, shift=True):
    """Calculates DFT of np array signal with a matrix approach"""
    N = signal.shape[0]  # number of samples
    indices_row = np.arange(N)  # indeces as a row
    indices_col = indices_row.reshape((N, 1))  # row to column transformation
    dft_mat = np.exp(-2j * np.pi * indices_col * indices_row / N)  # nxn matrix
    if shift: return fftshift(np.dot(dft_mat, signal))
    else: return np.dot(dft_mat, signal)


def dft_np(signal, N=None, shift=True):
    if N is None:
        N = len(signal)
    if shift:
        return fftshift(np.fft.fft(signal, N))
    else:
        return np.fft.fft(signal, N)


def idft_loop(dft, shift=True):
    """Calculates DFT of np array signal with a double loop"""
    N = dft.shape[0]  # number of samples
    idft = np.zeros(N, dtype=complex)  # preallocate
    for n in range(N):  # n indexes the IDFT
        idft_n = 0.0
        for k in range(N):  # k indexes the DFT
            idft_n += (dft[k]/N) * np.exp(2j * np.pi * k * n / N)
        idft[n] = idft_n

    if shift: return ifftshift(idft)
    else: return idft


def idft_matrix(dft, shift=True):
    N = dft.shape[0]
    indices_row = np.arange(N)  # indeces as a row
    indices_col = indices_row.reshape((N, 1))  # row to column transformation
    idft_mat = (1/N)*np.exp(2j * np.pi * indices_col * indices_row / N)  # nxn matrix
    if shift: return ifftshift(np.dot(idft_mat, dft))
    else: return np.dot(idft_mat, dft)


def idft_np(dft, shift=True):
    if shift:
        return ifftshift(np.fft.ifft(dft))
    else:
        return np.fft.ifft(dft)

# END DFT FUNCTIONS


def get_sine(t, a, f, phase):
    """ Simple helper function to generate a sine wave on the points t """
    return a * np.sin(2 * np.pi * f * t + phase)


def get_cosine(t, a, f, phase):
    """ Simple helper function to generate a sine wave on the points t """
    return a * np.cos(2 * np.pi * f * t + phase)


def get_gauss(t, a, b, t0):
    """ Simple helper function to generate a gauss bell curve on the points t """
    return a * np.exp(-b*((t - t0)**2))


def plot_gauss(f, dft, t, signal, idft, fs, ref_signal=None, ref_idft=None, f_lim=None, suptitle=None):
    """
        Used when investigated a Gauss curve
        Plots signal on first axes
        Im[DFT] and Re[DFT] together on the second axes
        Inverse FT on the third axes
        """
    fig, axes = plt.subplots(3, 1, figsize=(6, 4.5)) #, constrained_layout=True)

    # Plot signal
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("Signal $h(t)$")

    if ref_signal is not None:
        ax.plot(t, signal, color=color_signal, linewidth=4, label="shifted signal")
        ax.plot(t, ref_signal, linewidth=2, linestyle='--', color=color_ref, label='raw signal', zorder=-1)
    else:
        ax.plot(t, signal, color=color_signal, linewidth=4, label="signal")
    ax.text(0.02, 0.9, "$N={}$\n$f_{{s}}={}$".format(len(signal), fs), transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))

    ax.legend()
    ax.grid()

    # Plot DFT
    ax = axes[1]
    ax.set_ylabel('DFT $H(f)$')
    ax.set_xlabel("Freq. $f$")
    ax.xaxis.set_label_coords(-0.05, -0.08)
    ax.set_xlim(f_lim)  # restrict to a small interval around the bell curve for better viewing
    dft_min, dft_max = np.min(np.real(dft)), np.max(np.real(dft))
    ax.set_ylim((1.2*dft_min, 1.2*dft_max))  # slightly roomier y-axis spacing
    ax.grid()
    # Real DFT
    (markers, stemlines, baseline) = ax.stem(f, np.real(dft), label="Re[DFT]")
    plt.setp(markers, marker=marker_re, markerfacecolor=color_real, markeredgecolor="none", markersize=6)
    plt.setp(baseline, linestyle="-", color=color_real)
    plt.setp(stemlines, linestyle="-", color=color_real, linewidth=0.5)
    # Imaginary DFT
    (markers, stemlines, baseline) = ax.stem(f, np.imag(dft), label="Im[DFT]")
    plt.setp(markers, marker=marker_im, markerfacecolor=color_im, markeredgecolor="none", markersize=6)
    plt.setp(baseline, linestyle="-", color=color_im)
    plt.setp(stemlines, linestyle="-", color=color_im, linewidth=0.5)
    # ax.text(0.02, 0.9, "$f_{{s}}={}$".format(fs), transform=ax.transAxes, ha="left", va="top",
    #         bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.margins(y=0)
    ax.legend(loc="upper right", framealpha=1.0)

    # Plot inverse FT
    ax = axes[2]
    if ref_idft is not None:
        ax.plot(t, np.real(idft), color=color_idft, linewidth=4, label="shifted IDFT")
        ax.plot(t, np.real(ref_idft), linewidth=2, linestyle='--', color=color_ref, label='raw IDFT', zorder=-1)
    else:
        ax.plot(t, np.real(idft), color=color_idft, linewidth=4, label="IDFT")

    # ax.set_xlabel("Time $t$", labelpad=-10)
    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("IDFT $h(t)$")
    ax.legend()
    ax.grid()

    if suptitle is not None: plt.suptitle(suptitle)
    plt.subplots_adjust(top=0.93, hspace=0.3)
    if save_figures: plt.savefig(figure_dir + "gauss_.png", dpi=200)
    plt.show()


def plot_sinusoids(f, dft, t, signal, fs, f_low, f_high, f_lim=None, suptitle=None):
    """
        Used when investigating sinusoidal signals
        Plots signal on first axes
        Im[DFT] and Re[DFT] together on the second axes
        """
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5))

    # Plot signal
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.06)
    ax.set_ylabel("Signal $h(t)$")

    ax.plot(t, signal, color=color_signal, linewidth=2.5, label="signal")
    ax.text(0.99, 0.05, "$N={}$\n$f_{{s}}={}$\n$f_{{1}}={}$\n$f_{{2}}={}$".format(len(signal), fs, f_low, f_high), transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.grid()

    # Plot abs(DFT)
    ax = axes[1]
    ax.set_ylabel('DFT $H(f)$', labelpad=-5)
    ax.set_xlabel("Freq. $f$")
    ax.xaxis.set_label_coords(-0.05, -0.05)
    if f_lim is not None: ax.set_xlim(f_lim)  #
    dft_min = min(np.min(np.imag(dft)), np.min(np.real(dft)))
    dft_max = max(np.max(np.imag(dft)), np.max(np.real(dft)))
    ax.set_ylim((1.15*dft_min, 1.15*dft_max))  # slightly roomier y-axis spacing
    ax.grid()
    # Real DFT
    (markers, stemlines, baseline) = ax.stem(f, np.real(dft), label="Re[DFT]")
    plt.setp(markers, marker=marker_re, markerfacecolor=color_real, markeredgecolor="none", markersize=8)
    plt.setp(baseline, linestyle="-", color=color_real)
    plt.setp(stemlines, linestyle="-", color=color_real, linewidth=2)
    # Imaginary DFT
    (markers, stemlines, baseline) = ax.stem(f, np.imag(dft), label="Im[DFT]")
    plt.setp(markers, marker=marker_im, markerfacecolor=color_im, markeredgecolor="none", markersize=8)
    plt.setp(baseline, linestyle="-", color=color_im)
    plt.setp(stemlines, linestyle="-", color=color_im, linewidth=2)

    ax.margins(y=0)
    ax.legend(loc="upper right")

    if suptitle is not None: plt.suptitle(suptitle)
    plt.subplots_adjust(top=0.93, hspace=0.3)
    if save_figures: plt.savefig(figure_dir + "sinusoids_.png", dpi=200)
    plt.show()


def plot_spectral_leakage(f, dft, t, signal, idft, T, f_low, f_high, f_lim=None, suptitle=None):
    """
        Used when investigating sinusoidal signals
        Plots signal on first axes
        Im[DFT] and Re[DFT] together on the second axes
        """
    fig, axes = plt.subplots(3, 1, figsize=(6, 4.5))

    # Plot signal
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("Signal $h(t)$")

    ax.plot(t, signal, color=color_signal, linewidth=2)
    ax.text(0.01, 0.04, "$T_{{s}}={}T_{{0}}$\n$f_{{1}}={}$\n$f_{{2}}={}$".format(T/2, f_low, f_high), transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.grid()

    # Plot abs(DFT)
    ax = axes[1]
    ax.set_ylabel('DFT $H(f)$', labelpad=-3)
    ax.set_xlabel("Freq. $f$")
    ax.xaxis.set_label_coords(-0.05, -0.07)

    if f_lim is not None: ax.set_xlim(f_lim)  #
    dft_min = min(np.min(np.imag(dft)), np.min(np.real(dft)))
    dft_max = max(np.max(np.imag(dft)), np.max(np.real(dft)))
    ax.set_ylim((1.15*dft_min, 1.15*dft_max))  # slightly roomier y-axis spacing
    ax.grid()
    # Real DFT
    (markers, stemlines, baseline) = ax.stem(f, np.real(dft), label="Re[DFT]")
    plt.setp(markers, marker=marker_re, markerfacecolor=color_real, markeredgecolor="none", markersize=6)
    plt.setp(baseline, linestyle="-", color=color_real)
    plt.setp(stemlines, linestyle="-", color=color_real, linewidth=2)
    # Imaginary DFT
    (markers, stemlines, baseline) = ax.stem(f, np.imag(dft), label="Im[DFT]")
    plt.setp(markers, marker=marker_im, markerfacecolor=color_im, markeredgecolor="none", markersize=6)
    plt.setp(baseline, linestyle="-", color=color_im)
    plt.setp(stemlines, linestyle="-", color=color_im, linewidth=2)

    ax.margins(y=0)
    ax.legend(loc="upper right", framealpha=1.0)

    # plot IDFT
    ax = axes[2]

    ax.plot(t, np.real(idft), color=color_idft, linewidth=2, label="IDFT")
    ax.plot(t, signal, linewidth=1.5, linestyle='--', color=color_ref, label='reference', zorder=-1)

    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("IDFT $h(t)$")
    ax.legend(framealpha=1.0)
    ax.grid()

    if suptitle is not None: plt.suptitle(suptitle)
    plt.subplots_adjust(top=0.93, hspace=0.3)
    if save_figures: plt.savefig(figure_dir + "spectral-leakage_.png", dpi=200)
    plt.show()


def plot_aliasing(f, dft, t, signal, idft, fs, f_low, f_high, t_ref=None, ref_signal=None, f_lim=None):
    fig, axes = plt.subplots(3, 1, figsize=(6, 4.5))

    # Plot signal
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("Signal $h(t)$")

    if ref_signal is not None:
        ax.plot(t, signal, color=color_signal, linestyle='--', marker='o', linewidth=2.5, label="samples")
        ax.plot(t_ref, ref_signal, linewidth=1.5, linestyle='-', color=color_ref, label='reference', zorder=-1)
    else:
        ax.plot(t, signal, color=color_signal, linewidth=1, label="signal")

    # ax.text(0.02, 0.9, "$N={}$\n$f_{{s}}={}$\n$f_{{1}}={}$\n$f_{{2}}={}$".format(len(signal), fs, f_low, f_high), transform=ax.transAxes, ha="left", va="top",
    #         bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.text(0.99, 0.05, "$f_{{c}}={}$\n$f_{{1}}={}$\n$f_{{2}}={}$".format(fs/2, f_low, f_high), transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))

    ax.legend(loc="lower left", framealpha=1.0)
    ax.grid()

    # Plot DFT
    ax = axes[1]
    ax.set_ylabel('DFT $H(f)$')
    ax.set_xlabel("Freq. $f$")
    ax.xaxis.set_label_coords(-0.05, -0.07)

    if f_lim is not None: ax.set_xlim(f_lim)  #
    dft_min = min(np.min(np.imag(dft)), np.min(np.real(dft)))
    dft_max = max(np.max(np.imag(dft)), np.max(np.real(dft)))
    ax.set_ylim((1.15*dft_min, 1.15*dft_max))  # slightly roomier y-axis spacing
    ax.grid()
    # Real DFT
    (markers, stemlines, baseline) = ax.stem(f, np.real(dft), label="Re[DFT]")
    plt.setp(markers, marker=marker_re, markerfacecolor=color_real, markeredgecolor="none", markersize=8)
    plt.setp(baseline, linestyle="-", color=color_real)
    plt.setp(stemlines, linestyle="-", color=color_real, linewidth=2)
    # Imaginary DFT
    (markers, stemlines, baseline) = ax.stem(f, np.imag(dft), label="Im[DFT]")
    plt.setp(markers, marker=marker_im, markerfacecolor=color_im, markeredgecolor="none", markersize=6)
    plt.setp(baseline, linestyle="-", color=color_im)
    plt.setp(stemlines, linestyle="-", color=color_im, linewidth=2)

    ax.margins(y=0)
    # ax.legend(loc="upper right")
    ax.legend(loc="best")


    # plot IDFT
    ax = axes[2]

    ax.plot(t, np.real(idft), color=color_idft, linestyle='--', marker='o', linewidth=2.5, label="IDFT")
    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("IDFT $h(t)$")
    ax.legend()
    ax.grid()

    plt.suptitle("Aliasing")
    plt.subplots_adjust(top=0.93, hspace=0.3)
    if save_figures: plt.savefig(figure_dir + "aliasing_.png", dpi=200)
    plt.show()


def plot_zero_padding(f, dft, f_zp, dft_zp, t, signal, idft, idft_zp, T, f_low, f_high, f_lim=None):
    fig, axes = plt.subplots(3, 1, figsize=(6, 4.5))

    # Plot signal
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("Signal $h(t)$")

    ax.plot(t, signal, color=color_signal, linewidth=2)
    ax.text(0.01, 0.04, "$T_{{s}}={}T$\n$f_{{1}}={}$\n$f_{{2}}={}$".format(T, f_low, f_high), transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.grid()

    # Plot abs(DFT)
    ax = axes[1]
    ax.set_ylabel('DFT $|H(f)|$')
    ax.set_xlabel("Freq. $f$")
    ax.xaxis.set_label_coords(-0.05, -0.07)

    if f_lim is not None: ax.set_xlim(f_lim)  #
    dft_min = min(np.min(np.abs(dft)), np.min(np.abs(dft_zp)))
    dft_max = max(np.max(np.abs(dft)), np.max(np.abs(dft_zp)))
    ax.set_ylim((1.15 * dft_min, 1.15 * dft_max))  # slightly roomier y-axis spacing
    ax.grid()
    # Raw DFT
    (markers, stemlines, baseline) = ax.stem(f, np.abs(dft), label="raw")
    plt.setp(markers, marker=marker_re, markerfacecolor=color_raw, markeredgecolor="none", markersize=8)
    plt.setp(baseline, linestyle="-", color=color_raw)
    plt.setp(stemlines, linestyle="-", color=color_raw, linewidth=2.5)
    # Zero-padded DFT
    ax.plot(f_zp, np.abs(dft_zp), color=color_zp, linestyle='--', label="padded", zorder=-1)

    ax.margins(y=0)
    ax.legend(loc="upper right")

    # plot IDFT
    ax = axes[2]

    ax.plot(t, np.real(idft), color=color_raw, linewidth=2, label="raw")
    ax.plot(t, np.real(idft_zp), linewidth=1, linestyle='--', color=color_zp, label='padded', zorder=-1)

    ax.set_xlabel("Time $t$")
    ax.xaxis.set_label_coords(-0.05, -0.07)
    ax.set_ylabel("IDFT $h(t)$")
    ax.legend()
    ax.grid()

    plt.suptitle("Zero Padding")
    plt.subplots_adjust(top=0.93, hspace=0.3)
    if save_figures: plt.savefig(figure_dir + "zero-padding_.png", dpi=200)
    plt.show()


def plot_times():
    fig, ax = plt.subplots(figsize=(8, 4))

    # n = 1000 and below include all
    # n 10000 and below include matrix

    loop_files, matrix_files, np_files, npr_files = 0, 0, 0, 0

    # find how many files there are for each implementation
    for filename in sorted((os.listdir(time_dir))):
        if ".csv" in filename and "_" not in filename:
            n = int(filename.replace("dft-times-", "").replace("_", ""). replace(".csv", ""))
            np_files += 1 # increment in any case---all n include np fft measurements
            npr_files += 1
            if n <= 1000:
                loop_files += 1
                matrix_files += 1
            elif n <= 10000:
                matrix_files += 1

    loop_n, loop_times = np.zeros(loop_files), np.zeros(loop_files)
    matrix_n, matrix_times = np.zeros(matrix_files), np.zeros(matrix_files)
    np_n, np_times = np.zeros(np_files), np.zeros(np_files)
    npr_n, npr_times = np.zeros(npr_files), np.zeros(npr_files)

    i_loop = 0
    i_mat = 0
    i_np = 0
    for filename in natural_sort((os.listdir(time_dir))):
        if ".csv" in filename and "_" not in filename:
            n = int(filename.replace("dft-times-", "").replace("_", ""). replace(".csv", ""))
            data = np.loadtxt(time_dir + filename, delimiter=',', skiprows=1)

            np_time = np.mean(data[:, 2])
            npr_time = np.mean(data[:, 3])
            np_n[i_np], npr_n[i_np] = n, n
            np_times[i_np] = np_time
            npr_times[i_np] = npr_time
            i_np += 1

            if n <= 1000:
                loop_time = np.mean(data[:, 0])
                matrix_time = np.mean(data[:, 1])
                loop_n[i_loop] = n
                matrix_n[i_mat] = n
                loop_times[i_loop] = loop_time
                matrix_times[i_mat] = matrix_time
                i_loop += 1
                i_mat += 1
            elif n <= 10000:
                matrix_time = np.mean(data[:, 1])
                matrix_n[i_mat] = n
                matrix_times[i_mat] = matrix_time
                i_mat += 1

    plt.subplot(1, 2, 1)
    plt.xlabel("Samples in Signal $N$")
    plt.ylabel("Computation Time [s]")
    plt.xlim(0, 15000)
    plt.plot(loop_n, loop_times, color="#f05454", linestyle='--', marker='d', linewidth=2, label="loop")
    plt.plot(matrix_n, matrix_times, color="#ce6262", linestyle='--', marker='.', linewidth=2, label="matrix")
    plt.plot(np_n, np_times, color="#af2d2d", linestyle='--', marker='s', linewidth=2, label="np.fft")
    plt.plot(npr_n, npr_times, color="#16697a", linestyle='--', marker='o', linewidth=2, label="np.fftr")
    plt.legend()
    plt.title("Linear Scale", fontsize=10)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.xlabel("Samples in Signal $N$")
    plt.xlim(0, 60000)
    # plt.ylabel("Computation Time [s]")
    plt.yscale("log")
    plt.plot(loop_n, loop_times, color="#f05454", linestyle='--', marker='d', linewidth=2, label="loop")
    plt.plot(matrix_n, matrix_times, color="#ce6262", linestyle='--', marker='.', linewidth=2, label="matrix")
    plt.plot(np_n, np_times, color="#af2d2d", linestyle='--', marker='s', linewidth=2, label="np.fft")
    plt.plot(npr_n, npr_times, color="#16697a", linestyle='--', marker='o', linewidth=2, label="np.fftr")
    plt.legend()
    plt.title("Logarithmic Scale", fontsize=10)
    plt.grid()

    plt.suptitle("Time Comparison of DFT Implementations", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    if save_figures: plt.savefig(figure_dir + "times_.png", dpi=200)
    plt.show()


def natural_sort(l):
    """For sorting filenames in natural order"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


def do_gauss():
    T = 10
    t_start = 0
    t_end = t_start + T
    fs = 100  # sample rate
    fc = 0.5 * fs  # Nyquist (critical) frequency
    N = fs * T  # total number of samples
    pad_factor = 1  # for zero padding
    N = int(N * pad_factor)

    t = np.arange(t_start, t_end, 1 / fs)  # generate time samples
    f = np.arange(-fc, fc, fs / N)  # create frequency samples
    signal = get_gauss(t, 1, 10, T/2)  # generate Guass curve
    # signal = fftshift(signal)  # shift signal to make Gauss ''periodic''

    # Testing different DFT implementations
    dft = dft_matrix(signal)
    # dft = dft_loop(signal)
    # dft = dft_np(signal, N=N, shift=True)

    # Testing different IDFT implementations
    idft = idft_matrix(fftshift(dft), shift=True)
    # idft = idft_loop(fftshift(dft), shift=True)
    # idft = idft_np(fftshift(dft), shift=True)

    # plot_gauss(f, dft, t, signal, idft, fs, ref_signal=fftshift(signal), ref_idft=idft_np(fftshift(dft), shift=False), f_lim=(-2.3, 2.3), suptitle="Gauss Curve With Pre-Shift")  # for shifted, unmodulated guass
    plot_gauss(f, dft, t, signal, idft, fs, ref_idft=idft_np(fftshift(dft), shift=False), f_lim=(-2.3, 2.3), suptitle="Gauss Curve Without Pre-Shift")  # for moduulated gauss


def do_sinusoids():
    fs = 100  # sample rate
    fc = 0.5 * fs  # Nyquist (critical) frequency
    f_low = 2.
    f_high = 5.

    periods = 2  # number of periods of the wave to sample
    T = periods / f_low  # time of signal
    N = int(fs*T)
    t_start = 0
    t_end = t_start + T

    f_lim_scale = 1.3
    f_limits = (-f_lim_scale*max(f_high, f_low), f_lim_scale*max(f_high, f_low))

    t = np.linspace(t_start, t_end, N, endpoint=False)
    f = np.linspace(-fc, fc, N, endpoint=False)  # create frequency points

    signal = get_cosine(t, 1, f_low, 0) + 0.5*get_sine(t, 1, f_high, 0)  # time samples, ampitude, frequency, phase

    # t = np.arange(t_start, t_end, 1 / fs)  # generate time samples
    # f = np.arange(-fc, fc, fs / N)  # create frequency points

    # Testing different DFT implementations
    dft = dft_matrix(signal)
    # dft = dft_loop(signal)
    # dft = dft_np(signal, N=N, shift=True)

    # Testing different IDFT implementations
    idft = idft_matrix(fftshift(dft), shift=True)
    # idft = idft_loop(fftshift(dft), shift=True)
    # idft = idft_np(fftshift(dft), shift=True)

    plot_sinusoids(f, dft, t, signal, fs, f_low, f_high, f_lim=f_limits, suptitle="Sinusoids")


def do_spectral_leakage():
    fs = 100  # sample rate
    fc = 0.5 * fs  # Nyquist (critical) frequency
    f_low = 2.
    f_high = 5.

    periods = 2.7  # number of periods of the wave to sample
    T = periods / f_low  # time of signal
    N = int(fs*T)
    t_start = 0
    t_end = t_start + T

    f_lim_scale = 1.3
    f_limits = (-f_lim_scale*max(f_high, f_low), f_lim_scale*max(f_high, f_low))

    t = np.linspace(t_start, t_end, N, endpoint=False)
    t_ref = np.linspace(t_start, t_end, 10*N, endpoint=False)
    f = np.linspace(-fc, fc, N, endpoint=False)  # create frequency points

    signal = get_cosine(t, 1, f_low, 0) + 0.5*get_sine(t, 1, f_high, 0)  # time samples, ampitude, frequency, phase
    ref_signal = get_cosine(t_ref, 1, f_low, 0) + 0.5*get_sine(t_ref, 1, f_high, 0)  # time samples, ampitude, frequency, phase

    # t = np.arange(t_start, t_end, 1 / fs)  # generate time samples
    # f = np.arange(-fc, fc, fs / N)  # create frequency points

    # Testing different DFT implementations
    dft = dft_matrix(signal)
    # dft = dft_loop(signal)
    # dft = dft_np(signal, N=N, shift=True)

    # Testing different IDFT implementations
    idft = idft_matrix(fftshift(dft), shift=True)
    # idft = idft_loop(fftshift(dft), shift=True)
    # idft = idft_np(fftshift(dft), shift=True)

    plot_spectral_leakage(f, dft, t, signal, idft, periods, f_low, f_high, f_lim=f_limits, suptitle="Spectral Leakage")


def do_aliasing():
    fs = 8  # sample rate
    fc = 0.5 * fs  # Nyquist (critical) frequency
    f_low = 2.
    f_high = 5.

    periods = 10  # number of periods of the wave to sample
    T = periods / f_low  # time of signal
    N = int(fs * T)
    t_start = 0
    t_end = t_start + T

    f_lim_scale = 1.3
    # f_limits = (-f_lim_scale * max(f_high, f_low), f_lim_scale * max(f_high, f_low))
    f_limits = (- 1.1*fc, 1.1*fc)

    t = np.linspace(t_start, t_end, N, endpoint=False)
    t_ref = np.linspace(t_start, t_end, 10 * N, endpoint=False)
    f = np.linspace(-fc, fc, N, endpoint=False)  # create frequency points

    signal = get_cosine(t, 1, f_low, 0) + 0.5 * get_sine(t, 1, f_high, 0)  # time samples, ampitude, frequency, phase
    ref_signal = get_cosine(t_ref, 1, f_low, 0) + 0.5 * get_sine(t_ref, 1, f_high,
                                                                 0)  # time samples, ampitude, frequency, phase

    # t = np.arange(t_start, t_end, 1 / fs)  # generate time samples
    # f = np.arange(-fc, fc, fs / N)  # create frequency points

    # Testing different DFT implementations
    dft = dft_matrix(signal)
    # dft = dft_loop(signal)
    # dft = dft_np(signal, N=N, shift=True)

    # Testing different IDFT implementations
    idft = idft_matrix(fftshift(dft), shift=True)
    # idft = idft_loop(fftshift(dft), shift=True)
    # idft = idft_np(fftshift(dft), shift=True)

    plot_aliasing(f, dft, t, signal, idft, fs, f_low, f_high, t_ref=t_ref, ref_signal=ref_signal, f_lim=f_limits)


def do_zero_padding():
    fs = 100  # sample rate
    T = 0.9
    fc = 0.5 * fs  # Nyquist (critical) frequency
    f_low = 2.
    f_high = 5.

    N = int(fs*T)

    t_start = 0
    t_end = t_start + T

    f_lim_scale = 1.3
    f_limits = (-f_lim_scale * max(f_high, f_low), f_lim_scale * max(f_high, f_low))

    t = np.linspace(t_start, t_end, N, endpoint=False)
    f = np.linspace(-fc, fc, N, endpoint=False)  # create frequency points

    signal = get_cosine(t, 1, f_low, 0) + 0.5 * get_sine(t, 1, f_high, 0)  # time samples, ampitude, frequency, phase

    dft = dft_matrix(signal)
    idft = idft_matrix(fftshift(dft), shift=True)

    # ZERO PADDING
    zp_factor = 10
    N_zp = N * zp_factor

    f_zp = np.linspace(-fc, fc, N_zp, endpoint=False)  # create zero-padded frequency points

    dft_zp = dft_np(signal, N=N_zp, shift=True)
    idft_zp = idft_np(fftshift(dft_zp), shift=True)[int((zp_factor*N)/2):int((0.5 + 1/zp_factor)*N*zp_factor)]

    plot_zero_padding(f, dft, f_zp, dft_zp, t, signal, idft, idft_zp, T, f_low, f_high, f_lim=f_limits)


def plot_first_note():
    fig, axes = plt.subplots(3, 2, figsize=(6, 7))

    # Plot left column
    frequencies = (882, 1378, 2756)
    for i, fs in enumerate(frequencies):
        samples, _ = sf.read(first_note_dir + "{}.wav".format(fs))  # read wav samples (mono)
        fc = 0.5 * fs  # Nyquist frequency
        N = len(samples)  # number of samples
        f = np.linspace(-fc, fc, N, endpoint=False)
        dft = np.abs(dft_np(samples, shift=True))

        ax = axes[i][0]
        ax.set_xlim((0, 800))
        ax.set_ylabel('DFT $|H(f)|$')
        if i == 2: ax.set_xlabel("Frequency $f$ [Hz]")
        ax.plot(f, dft, linestyle='--', marker='.', linewidth=2, color=note_colors[i], label="$f_{{s}} = {}$".format(fs))
        ax.legend(loc="best", fontsize=11)

    # Plot right column
    frequencies = (5512, 11025, 44100)
    note_markers = get_B4_markers(5)
    y_scale=1.15
    for i, fs in enumerate(frequencies):
        samples, _ = sf.read(first_note_dir + "{}.wav".format(fs))  # read wav samples (mono)
        fc = 0.5 * fs  # Nyquist frequency
        N = len(samples)  # number of samples
        f = np.linspace(-fc, fc, N, endpoint=False)
        dft = np.abs(dft_np(samples, shift=True))
        max_dft = np.max(dft)

        ax = axes[i][1]
        ax.set_xlim((0, 3200))
        ax.set_ylim(top=y_scale * max_dft)
        if i == 2: ax.set_xlabel("Frequency $f$ [Hz]")
        ax.plot(f, dft, linestyle='--', marker='.', linewidth=2, color=note_colors[i+3], label="$f_{{s}} = {}$".format(fs))
        ax.legend(loc="best", fontsize=11)

        for k, marker in enumerate(note_markers):
            ax.axvline(marker[0], ymin=0, ymax=((6-k)/6)/y_scale, color='#888888', linestyle='--')
            ax.annotate(marker[1], (marker[0], ((6-k)/6)*max_dft), ha="center", va="bottom")

    plt.suptitle("The Partita's First Note", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    if save_figures: plt.savefig(figure_dir + "first-note_.png", dpi=200)
    plt.show()


def plot_partita():
    fig, axes = plt.subplots(3, 2, figsize=(6, 7))

    # Plot left column
    frequencies = (882, 1378, 2756)
    for i, fs in enumerate(frequencies):
        samples = np.loadtxt(txt_dir + "{}.txt".format(fs))
        fc = 0.5 * fs  # Nyquist frequency
        N = len(samples)  # number of samples
        f = np.linspace(-fc, fc, N, endpoint=False)
        dft = np.abs(dft_np(samples, shift=True))

        ax = axes[i][0]
        ax.set_xlim((0, 4000))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.set_ylabel('DFT $|H(f)|$')
        if i == 2: ax.set_xlabel("Frequency $f$ [Hz]")
        ax.plot(f, dft, linestyle='--', marker='.', linewidth=2, color=note_colors[i], label="$f_{{s}} = {}$".format(fs))
        ax.legend(loc="best", fontsize=11)

    # Plot right column
    frequencies = (5512, 11025, 44100)
    for i, fs in enumerate(frequencies):
        samples = np.loadtxt(txt_dir + "{}.txt".format(fs))
        fc = 0.5 * fs  # Nyquist frequency
        N = len(samples)  # number of samples
        f = np.linspace(-fc, fc, N, endpoint=False)
        dft = np.abs(dft_np(samples, shift=True))
        max_dft = np.max(dft)

        ax = axes[i][1]
        ax.set_xlim((0, 4000))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        if i == 2: ax.set_xlabel("Frequency $f$ [Hz]")
        ax.plot(f, dft, linestyle='--', marker='.', linewidth=2, color=note_colors[i+3], label="$f_{{s}} = {}$".format(fs))
        ax.legend(loc="best", fontsize=11)

    plt.suptitle("The Partita's Spectrum", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.925)
    if save_figures: plt.savefig(figure_dir + "partita-spectrum_.png", dpi=200)
    plt.show()


def get_B4_markers(n_harmonics):
    """
    Returns (frequency, note name) tuple markers for first n_harmonics harmonics of B4, the first note in the partita
    """
    fundamental = 493.88
    markers = []
    for i in range(1, n_harmonics+1):
        markers.append((fundamental * i, "B{}".format(4+i-1)))
    return markers


def dft_times(N):
    T = 10
    t_start = 0
    t_end = t_start + T
    fs = N/T  # sample rate
    fc = fs/2

    f_sin = 10  # frequency of sine wave

    t = np.arange(t_start, t_end, 1 / fs)  # generate time samples
    signal = get_sine(t, 1, f_sin, 0)

    runs = 5
    dft_loop_times = np.zeros(runs)
    dft_matrix_times = np.zeros(runs)
    dft_np_times = np.zeros(runs)
    dft_npr_times = np.zeros(runs)

    for i in range(runs):
        print(i)
        # t = time.time()
        # dft_loop(signal)
        # t = time.time() - t
        # dft_loop_times[i] = t

        # t = time.time()
        # dft_matrix(signal)
        # t = time.time() - t
        # dft_matrix_times[i] = t

        t = time.time()
        dft_np(signal)
        t = time.time() - t
        dft_np_times[i] = t

        t = time.time()
        fftshift(np.fft.rfft(signal))
        t = time.time() - t
        dft_npr_times[i] = t

    times = np.column_stack([dft_loop_times, dft_matrix_times, dft_np_times, dft_npr_times])
    header = "Loop, Matrix, Numpy FFT, Numpy RFFT"
    np.savetxt(time_dir + "dft-times-{}_.csv".format(int(N)), times, delimiter=',', header=header)
    print(times)


def run_times():
    """Wrapper method to test timing"""
    for N in range(2, 11):
        dft_times(N*1e4)


def run():
    # do_gauss()
    # do_sinusoids()
    # do_aliasing()
    # do_spectral_leakage()
    # do_zero_padding()
    plot_times()
    plot_first_note()
    # plot_partita()
    # dft_times()
    # run_times()

run()
