import numpy as np
import os
import re
import time
import soundfile as sf
from numpy.fft import fft, ifft, fftshift
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

data_dir = "../5-fft-correlation/data/"
image_dir = data_dir + "images/"
data_txt_dir = data_dir + "txt/"
data_wav_dir = data_dir + "wav/"
time_dir = "../5-fft/data/times/"
figure_dir = "../5-fft/figures/"
save_figures = False

f_bubo1 = 380  # fundamental frequency of first owl
f_bubo2 = 334.5  # fundamental frequency of first owl


# START CORRELATION FUNCTIONS
def autocov_sum_biased(signal):
    """
    Direct implementation of basic 1D autocovariance using the sum definition
    Uses autocor(n) = (1/N)*sum_{k=0}^{N}sig[k]*sig[(k+n) mod N]  (assumes periodicity of input signal)
    :param signal: 1D numpy array holding inputted signal
    """
    N = np.shape(signal)[0]  # number of samples in inputted signal
    autocor = np.zeros(N)
    for n in range(N):  # n indexes the autocorrelation
        autocor_n = 0.0  # n-th autocor value
        for k in range(N - n - 1):  # k is summation index
            autocor_n += signal[k]*signal[(k+n) % N]  # (k+n) % N wraps around to start (assumes periodicity of N)
        autocor[n] = autocor_n/N  # normalize with N
    return autocor


def autocov_sum_unbiased(signal):
    """
    Direct implementation of 1D autocovariance for periodic signals using the sum definition
    Uses autocor(n) = (1/(N-n))*sum_{k=0}^{N-n-1}sig[k]*sig[k+n]
    :param signal: 1D numpy array holding inputted signal
    """
    N = np.shape(signal)[0]  # number of samples in inputted signal
    autocor = np.zeros(N)
    for n in range(N):  # n indexes the autocorrelation
        autocor_n = 0.0  # n-th autocor value
        for k in range(N - n - 1):  # k is summation index
            autocor_n += signal[k]*signal[k+n]
        autocor[n] = autocor_n/(N - n)  # normalize with (N_signal - n)
    return autocor


def autocov_fft_biased(signal, center=True):
    """
    Finds autocovarianiance using FFT.
    Assumes input signal is periodic with period N
    Does not use zero-padding
    :param signal: 1D numpy array holding inputted signal
    :param center: control subtracting signal's average from signal
    """
    N = np.shape(signal)[0]  # number of samples in inputted signal
    if center: signal = signal - np.mean(signal)  # center the signal (subtract average)
    return np.real(ifft(np.abs(fft(signal)) ** 2) / N)  # return normalized autocovariance


def autocor_fft_biased(signal):
    """
    Returns autocorrelation, i.e. the normalized version of autocovariance
    Assumes input signal is periodic with period N
    :param signal: 1D numpy array holding inputted signal
    """
    autocov = autocov_fft_biased(signal)
    variance = autocov[0]  # variance is first element (assuming signal is centered around zero)
    if variance == 0.:  # special case for a constant signal, avoids divide by zero errors
        return np.zeros(autocov.shape)
    else:  # signals with non-zero variance
        return autocov / variance  # normalize


def autocov_fft_unbiased(signal):
    """
    Efficient implementation of autocov_sum using the FFT
    First uses zero-padding to double input signal's length
    Safe for use with potentially periodic signals (even when input signal is not one representative period of some parent signal)

    :param signal: 1D numpy array holding inputted signal
    """
    N = np.shape(signal)[0]  # number of samples in inputted signal
    signal_zp = np.zeros(2*N)
    signal_zp[0:N] = signal
    auto_cov = ifft(np.abs(fft(signal_zp))**2)[0:N]  # inverse Fourier transform of zero-padded power-spectral-density. Take only N samples

    for n in range(N):  # normalize with N - n
        auto_cov[n] = auto_cov[n]/(N-n)

    return np.real(auto_cov)


def autocov_fft_unbiased_corrected(signal, center=True):
    """
    # Gives identical results to autocov_fft_cyclic_H
    Finds autocovarianiance using FFT. Uses zero-padding
    :param signal: 1D numpy array holding inputted signal
    """
    N = np.shape(signal)[0]  # number of samples in inputted signal
    if center: signal = signal - np.mean(signal)  # center the signal (subtract average)
    signal_zp = np.zeros(2*N)
    signal_zp[0:N] = signal
    autocov = ifft(np.abs(fft(signal_zp))**2)

    # correction for error from zero-padding
    mask = np.zeros(2*N)  # length 2N array
    mask[0:N] = np.ones(N)  # equal to one for 0 to N - 1 and zero for N to 2N - 1
    ft_mask = np.fft.fft(mask)
    correction = np.fft.ifft(np.abs(ft_mask) ** 2)
    autocov[0:N] = autocov[0:N] / correction[0:N]  # element-wise correction to raw autocov (only correct the useful values from 0 to N)

    return np.real(autocov[0:N])  # return only N samples 0 to N-1 corresponding to autocov of original signal


def autocor_fft_unbiased(signal):
    """
    Returns autocorrelation, i.e. the normalized version of autocovariance
    :param signal: 1D numpy array holding inputted signal
    """
    autocov = autocov_fft_unbiased_corrected(signal)
    variance = autocov[0]  # variance is first element (assuming signal is centered around zero)
    if variance == 0.:  # special case for a constant signal, avoids divide by zero errors
        return np.zeros(autocov.shape)
    else:  # signals with non-zero variance
        return autocov / variance  # normalize


def crosscor_fft_basic(f, g):
    """
    Efficient implementation of cross-correlation (covariance) using the FFT
    Assumes input signal is periodic with period N
    Does not use zero-padding
    In test-probe convention f is probe and g is test
    """
    N_f = np.shape(f)[0]  # number of samples in inputted signal
    N_g = np.shape(g)[0]  # number of samples in inputted signal
    if N_f == N_g:  # signals are the same length
        return np.real(ifft(np.conj(fft(f)) * fft(g)))/N_f
    elif N_f > N_g:
        g_padded = np.zeros(N_f)
        g_padded[0:N_g] = g
        return np.real(ifft(np.conj(fft(f)) * fft(g_padded))) / N_f
    else:  # if N_g > N_g:
        f_padded = np.zeros(N_g)
        f_padded[0:N_f] = f
        return np.real(ifft(np.conj(fft(f_padded)) * fft(g))) / N_g


def crosscor_2d_sum(probe, test):
    """
    Input a small KxL pixel probe image and a larger MXN pixel test image where K<=M and L<=N
    Finds cross-correlation of the two image
    """
    M, N = test.shape[0], test.shape[1]
    K, L = probe.shape[0], probe.shape[1]
    test = test - np.mean(test)  # center signals
    probe = probe - np.mean(probe)  # center signals
    crosscor = np.zeros((M, N))  # preallocate
    test_padded = np.zeros((M + K, N + L))  # pad test image edges
    test_padded[K//2 : K//2 + M, L//2 : L//2 + N] = test

    for cor_row in range(M):  # loop through rows in correlation
        print(cor_row)
        for cor_col in range(N):  # loop through columns in correlation
            crosscor_row_col = 0
            for probe_row in range(K):
                for probe_col in range(L):
                    crosscor_row_col += probe[probe_row, probe_col]*test_padded[cor_row + probe_row, cor_col + probe_col]
            crosscor[cor_row, cor_col] = crosscor_row_col

    return crosscor

    # END CORRELATION FUNCTIONS


def get_bubof0_markers(fundamental, n_harmonics):
    """
    Returns (frequency) markers for first n_harmonics harmonics of fundamental, the owl's fudamental frequency
    """
    markers = []
    for i in range(1, n_harmonics+1):
        markers.append(fundamental * i)
    return markers


def plot_probe_spectrum():
    """
    Plots spectra of the probe signals bubo1.wav and bubo2.wav and notes characteristic frequencies
    """
    for filename in ("bubo1.wav", "bubo2.wav"):
        samples, fs = sf.read(data_wav_dir + filename)  # read wav samples (mono)
        fc = 0.5 * fs  # Nyquist frequency
        N = len(samples)  # number of samples

        dft = np.abs(fftshift(fft(samples)))[N//2 - 1:N]  # postive frequencies only
        f = np.linspace(0, fc, N//2, endpoint=False)  # frequency space, positive frequencies only

        dft_max = np.max(dft)  # fundamental frequency -> largest value in spectrum
        f0 = f[np.where(dft == dft_max)[0][0]]  # search for index of f0
        n_harmonics = 5  # show first 5 harmonics
        fmax = n_harmonics*f0  # 5 times fundamental
        fmax_index = np.argmin(np.abs(f - fmax))  # index of max frequency
        dft = dft[0:fmax_index]
        f = f[0:fmax_index]

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        # Linear scale axis
        ax = axes[0]
        ax.set_xlabel("Frequency $f$ [Hz]")
        ax.set_ylabel("Spectrum |DFT|")
        ax.set_yscale("linear")
        ax.plot(f, dft, color=color_blue, label="Abs. DFT |H(f)|")
        ax.annotate('Fundamental frequency\n$f_0$ = {:.0f} Hz'.format(f0-1), xy=(1.1*f0, 0.99*dft_max), xytext=(1.8*f0, 0.75*dft_max), fontsize=11,
                    arrowprops=dict(facecolor='black', width=1, headwidth=10),
                    bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

        ax.set_title("Linear")

        # Logarithmic scale axis
        f0_markers = get_bubof0_markers(f0, n_harmonics)
        ax = axes[1]
        ax.set_xlabel("Frequency $f$ [Hz]")
        ax.set_yscale("log")
        ax.plot(f, dft, color=color_blue, label="Abs. DFT |H(f)|")
        for k, f_harmonic in enumerate(f0_markers):
            f_harmonic_index = np.argmin(np.abs(f - f_harmonic))
            ax.vlines(f_harmonic, ymin=dft[f_harmonic_index], ymax=5*dft[f_harmonic_index], color='#888888', linestyle='--')
            if k == 0:
                ax.annotate("$f_0$ = {:.0f} Hz".format(f0_markers[k]-1), (f_harmonic, 6 * dft[f_harmonic_index]), ha="center", va="bottom",
                                    bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.2'))
            else:
                ax.annotate("${}f_0$".format(k), (f_harmonic, 6 * dft[f_harmonic_index]),
                            ha="center", va="bottom",
                            bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.2'))

        ax.set_ylim(top=30*dft_max)  # so frequency markers fit in frame
        ax.set_xlim(right=1.08*f[-1])
        ax.set_title("Logarithmic")

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.rc('text', usetex=True)
        plt.suptitle(r'Characteristic Harmonic Frequencies in $\texttt{{{}}}$'.format(filename), fontsize=20)
        plt.rc('text', usetex=False)

        if save_figures: plt.savefig(figure_dir + "spectra/" + filename.replace(".wav", "") + "-spectrum_.png", dpi=200)
        plt.show()


def plot_wav_spectrum(filename):
    """
    Import the name of a wav file. Plots wav file's spectrum
    :return:
    """
    samples, fs = sf.read(data_wav_dir + filename)  # read wav samples (mono)
    fc = 0.5 * fs  # Nyquist frequency
    N = len(samples)  # number of samples
    f = np.linspace(-fc, fc, N, endpoint=False)  # frequency space
    dft = np.abs(fftshift(fft(samples)))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    ax.set_xlim(0, 2000)
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.set_ylabel("Abs. DFT $H(f)$")
    ax.set_yscale("linear")
    ax.plot(f, dft, label="Abs. DFT |H(f)|")
    ax.set_title("Linear")
    ax.legend()

    ax = axes[1]
    ax.set_xlim(0, 2000)
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.set_ylabel("Abs. DFT $H(f)$")
    ax.set_yscale("log")
    ax.plot(f, dft, label="Abs. DFT |H(f)|")
    ax.set_title("Logarithmic")
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.rc('text', usetex=True)
    plt.suptitle(r'Spectrum of $\texttt{{{}}}$'.format(filename), fontsize=20)
    plt.rc('text', usetex=False)

    if save_figures: plt.savefig(figure_dir + "spectra/" + filename.replace(".wav", "") + "-spectrum_.png", dpi=200)
    plt.show()


def plot_compared_spectra(filename, makeguess=True):
    """
    Input the name of a wav file.
    Compares the signal's direct spectrum and the spectrum of the signal's autocorrelation function. Takeaways are:
     Autocorrelation spectrum removes noise and improves visibility of fundamental frequency
     Direct signal spectrum is noisier but also shows fundamental's higher harmonics, which autocorrelation removes

    Guesses the spectra's corresponding owl based on height of spectral peaks at each owl's frequency
    """
    samples, fs = sf.read(data_wav_dir + filename)  # read wav samples (mono)
    N = np.shape(samples)[0]  # number of samples in inputted signal
    # N = int(0.99*N)  # don't take all sample because autocov diverges for large n

    autocov = autocov_fft_unbiased(samples)[0:N]

    f_end = 2000
    n_start = int(N / 2) - 1
    n_end = n_start + int(N * f_end / fs)
    f = np.linspace(-fs / 2, fs / 2, N)[n_start: n_end]
    dft_signal = np.abs(fftshift(np.real(fft(samples)))[n_start: n_end])  # spectrum of signal
    dft_autocov = np.abs(fftshift(np.real(fft(autocov)))[n_start: n_end])  # spectrum of autocorrelation

    f1_index, f2_index, f1_strength, f2_strength = 0, 0, 0, 0
    dft_signal_peak, dft_autocov_peak = 0, 0
    if makeguess:
        f1_index = np.argmin(np.abs(f - f_bubo1))  # index of bubo1's frequency
        f2_index = np.argmin(np.abs(f - f_bubo2))  # index of bubo2's frequency
        f1_strength = np.mean(dft_autocov[f1_index-2:f1_index+3])  # determine spectrum strength in 5-point window around f1
        f2_strength = np.mean(dft_autocov[f2_index - 2:f2_index + 3])  # determine spectrum strength in 5-point window around f1
        if f1_strength >= f2_strength:
            dft_signal_peak = np.max(dft_signal[f1_index-5:f1_index+6])
            dft_autocov_peak = np.max(dft_autocov[f1_index-5:f1_index+6])
        else:  # f2_strength > f1_strength:
            dft_signal_peak = np.max(dft_signal[f2_index - 5:f2_index + 6])
            dft_autocov_peak = np.max(dft_autocov[f2_index - 5:f2_index + 6])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot signal's spectrum
    ax = axes[0]
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.set_ylabel("Spectrum |DFT|")
    ax.plot(f, dft_signal, color=color_orange_mid, marker='.', linestyle='-', label="signal")
    if makeguess:
        if np.max(dft_signal) == dft_signal_peak:  # adjust y limits to fit annotation
            ax.set_ylim(top=1.13*dft_signal_peak)

        f_mark, guess = 0, ""
        if f1_strength >= f2_strength: f_mark, guess = f_bubo1, "bubo1"
        else: f_mark, guess = f_bubo2, "bubo2"

        ax.annotate("$f$ ≈ {:.0f} Hz".format(f_mark), xy=(1.1 * f_mark, 0.99 * dft_signal_peak),
                    xytext=(1.8 * f_mark, 0.75 * dft_signal_peak), fontsize=11,
                    arrowprops=dict(facecolor='black', width=1, headwidth=10),
                    bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

        ax.text(0.95, 0.2, "Best guess: {}".format(guess), transform=ax.transAxes, ha="right", va="top", fontsize=12,
                bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))

    ax.set_title("Signal Spectrum")

    # Plot autocorrelation's spectrum
    ax = axes[1]
    ax.set_xlabel("Frequency $f$ [Hz]")

    ax.plot(f, dft_autocov, color=color_orange_dark, marker='.', linestyle='-', label="autocorrelation")
    if makeguess:
        if np.max(dft_autocov) == dft_autocov_peak:  # adjust y limits to fit annotation
            ax.set_ylim(top=1.13*dft_autocov_peak)

        f_mark, guess = 0, ""
        if f1_strength >= f2_strength: f_mark, guess = f_bubo1, "bubo1"
        else: f_mark, guess = f_bubo2, "bubo2"

        ax.annotate("$f$ ≈ {:.0f} Hz".format(f_mark), xy=(1.1 * f_mark, 0.99 * dft_autocov_peak),
                    xytext=(1.8 * f_mark, 0.75 * dft_autocov_peak), fontsize=11,
                    arrowprops=dict(facecolor='black', width=1, headwidth=10),
                    bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
        ax.text(0.95, 0.2, "Best guess: {}".format(guess), transform=ax.transAxes, ha="right", va="top", fontsize=12,
                bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))

        # if f1_strength >= f2_strength:
        #     ax.annotate("$f$ ≈ {:.0f} Hz".format(f_bubo1), (f_bubo1, 1.04*dft_autocov_peak), ha="center", va="bottom",
        #                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.2'))
        #     ax.text(0.95, 0.3, "Best guess: bubo1", transform=ax.transAxes, ha="right", va="top", fontsize=12,
        #             bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
        # else:
        #     ax.annotate("$f$ ≈ {:.0f} Hz".format(f_bubo2), (f_bubo2, 1.06*dft_autocov_peak), ha="center", va="bottom",
        #                 bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.2'))
        #     ax.text(0.95, 0.3, "Best guess: bubo2", transform=ax.transAxes, ha="right", va="top", fontsize=12,
        #             bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))

    ax.set_title("Autocorrelation Spectrum")

    plt.tight_layout()
    plt.rc('text', usetex=True)
    plt.suptitle(r'Compared Spectra of $\texttt{{{}}}$'.format(filename), fontsize=20)
    plt.rc('text', usetex=False)
    plt.subplots_adjust(top=0.85)

    if save_figures: plt.savefig(figure_dir + "spectra/" + filename.replace(".wav", "") + "-compared-spectra_.png", dpi=200)
    plt.show()


def confirm_cyclic():
    N = 300
    t = np.linspace(0, 1, N)
    signal = np.sin(10*2*np.pi*t)
    N = int(0.8*N)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Time $t$ [s]")
    ax.set_ylabel("Autocovariance")
    ax.set_yscale("linear")

    autocov_cyclic_sum = autocov_sum_unbiased(signal)
    autocov_cyclicH = autocov_fft_unbiased(signal)
    print(autocov_cyclicH)

    t = t[0:N]
    signal = signal[0:N]
    autocov_cyclic_sum = autocov_cyclic_sum[0:N]
    autocov_cyclicH = autocov_cyclicH[0:N]

    # ax.plot(t, signal, marker='.', linestyle='-', label="signal")
    ax.plot(t, autocov_cyclic_sum, marker='o', linestyle='-', label="autocov cyclic sum")
    ax.plot(t, autocov_cyclicH, marker='.', linestyle='-', label="autocov cyclic Horvat")

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_autcor(filename):
    """
    Import the name of a wav file.
    Plots sound signal's autocorrelation
    And a small, inset zoomed-in version of the autocorrelation to show high-frequency periodicity
    :return:
    """
    samples, fs = sf.read(data_wav_dir + filename)  # read wav samples (mono)
    N = np.shape(samples)[0]  # number of samples in inputted signal
    N = int(0.99*N)  # don't take all sample because autocov diverges for large n
    T = N/fs
    t = np.linspace(0, T, N, endpoint=False)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.set_xlabel("Time $t$ [s]", fontsize=11, labelpad=-3)
    ax.set_ylabel("Autocorrelation", fontsize=11, labelpad=-5)
    ax.set_yscale("linear")

    autocor = autocor_fft_unbiased(samples)[0:N]
    ax.plot(t, autocor, marker='.', linestyle='-', zorder=-1)

    # Configuring x and y range of zoomed region
    y_scale = 1.1
    xmin = 0.06
    xmax = 0.12
    xmin_index = np.argmin(np.abs(t - xmin))
    xmax_index = np.argmin(np.abs(t - xmax))
    t_zoomed = t[xmin_index:xmax_index]
    autocor_zoomed = autocor[xmin_index:xmax_index]
    ymin = y_scale * np.min(autocor[xmin_index:xmax_index])
    ymax = y_scale * np.max(autocor[xmin_index:xmax_index])

    # Make the zoom-in plot:
    axins = inset_axes(ax, width='60%', height="40%", loc="upper right", bbox_to_anchor=(0, -0.08, 1, 1), bbox_transform=ax.transAxes)

    axins.plot(t_zoomed, autocor_zoomed)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.set_title("Zoomed Autocorrelation".format(xmin, xmax))

    # disable ticks on inset axis
    axins.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    mark_inset(ax, axins, loc1=2, loc2=4, linewidth=2)

    plt.rc('text', usetex=True)
    plt.suptitle(r'Autocorrelation of $\texttt{{{}}}$'.format(filename), fontsize=20)
    plt.subplots_adjust(top=0.91)
    plt.rc('text', usetex=False)

    if save_figures: plt.savefig(figure_dir + "autocor/" + filename.replace(".wav", "") + "-autocor_.png", dpi=200)
    plt.show()


def plot_autocor_pure_noise():
    """
    Used to investigate autocorrelation function of a noisy signal
    Basically shows autocorrelation is maximum at zero shift and zero elsewhere
    And that an uncentered autocorrelation decayse to the square of the noisy signal's mean
    """

    # Pure noise
    N = 1000
    tmin = 0
    tmax = 1
    t = np.linspace(tmin, tmax, N, endpoint=False)
    np.random.seed(6)  # 6 seed decays nicely to <signal>^2 for np.random.normal(0.5, 1, size=N)
    mu = 0.5
    var = 1
    noisy_signal = np.random.normal(mu, var, size=N)
    autocov = autocov_fft_biased(noisy_signal, center=False)
    autocor = autocor_fft_biased(noisy_signal)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Amplitude")
    ax.plot(t, noisy_signal, color="#AAAAAA", linestyle="-")
    ax.set_title("Guassian Noise $\sim \mathcal{{N}} ({{{}}}, {{{}}})$".format(mu, var))

    ax = axes[1]
    ax.set_xlabel("Time $t$")
    ax.plot(t, autocov, color_orange_dark, linestyle="-", label="raw")
    ax.plot(t, autocor, color=color_blue, linestyle="-", label="normalized")
    ax.annotate("Decay to $\mu^2 ≈ {}$".format(mu**2), xy=(tmax/1.1, 1.1*(mu**2)), xytext=(tmax/3, 2*(mu**2)), fontsize=11,
                arrowprops=dict(facecolor='black', width=1, headwidth=10),
                bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    ax.legend()
    ax.set_title("Autocorrelation")

    plt.tight_layout()
    plt.suptitle("Autocorrelation of Gaussian Noise", fontsize=16)
    plt.subplots_adjust(top=0.86)
    if save_figures: plt.savefig(figure_dir + "samples/autocor-noise_.png", dpi=200)
    plt.show()


def plot_autocor_noisy_signal():
    """
    Used to investigate autocorrelation function of a noisy signal

    Plots a decaying sinusoid superimposed on increasing, uniformly distributed noise
    The resulting autocorrelation is periodic
    """
    N = 1000
    tmin = 0
    tmax = 1.2
    damp_factor = 3.5
    t = np.linspace(tmin, tmax, N, endpoint=False)
    sine = np.sin(2*np.pi*5*t) * np.exp(-damp_factor*t)  # damp sine wave---large at 0 and weak at one
    noise = np.random.uniform(-1, 1, size=N) * (1-np.exp(-damp_factor*t))  # reverse-damp noise---weak at 0 and strong at one
    signal = sine + noise
    autocor = autocor_fft_unbiased(signal)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Amplitude")
    stop_index = np.argmin(np.abs(t - 1))  # cut off last few points where autocorrelation diverges
    ax.plot(t[1:stop_index], signal[1:stop_index], color="#AAAAAA", linestyle="-", label="noisy signal")
    ax.plot(t[1:stop_index], sine[1:stop_index], color=color_blue, linestyle="--", alpha=0.85, label="pure signal (for reference)")
    ax.set_title("Test Signal")
    ax.legend()

    ax = axes[1]
    ax.set_xlabel("Time $t$")
    ax.plot(t[1:stop_index], autocor[1:stop_index], color=color_blue,  linestyle="-")
    ax.set_title("Autocorrelation")

    plt.tight_layout()
    plt.suptitle("Autocorrelation a Noisy Signal", fontsize=16)
    plt.subplots_adjust(top=0.86)
    if save_figures: plt.savefig(figure_dir + "samples/autocor-noisy_.png", dpi=200)

    plt.show()


def plot_crosscor_test():
    """
    Shows how cross-correlation with a known periodic probe
    can detect the probe's presence even in a very noisy signal

    Plots a small sinewave superimposed on loud Gaussian noise
    and dectects the sine wave with cross-correlation
    """
    N_probe = 100
    t_min = 0
    t_max = 1
    T_probe = t_max - t_min
    t_probe = np.linspace(t_min, t_max, N_probe, endpoint=False)
    probe = 0.95 * np.sin(2*np.pi*3*t_probe)

    test_probe_factor = 10
    N_test = test_probe_factor*N_probe
    t_test = np.linspace(test_probe_factor*t_min, test_probe_factor*t_max, N_test, endpoint=False)
    np.random.seed(3)
    test = np.random.normal(0, 1, size=N_test)

    start_factor = 7
    start_time = start_factor * T_probe
    probe_start_index = start_factor*N_probe  # where to insert probe signal inside test signal
    test[probe_start_index: probe_start_index + N_probe] = test[probe_start_index: probe_start_index + N_probe] + probe

    crosscor = crosscor_fft_basic(probe, test)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Amplitude")
    ax.plot(t_test, test, color="#AAAAAA", linestyle="-", label="noisy signal")
    ax.plot(t_probe+start_time, probe, color=color_blue, linestyle=":", linewidth=1.5, label="probe (for reference)")

    ax.annotate("Signal onset: $t$ = {:.2f}".format(start_time), xy=(start_time, 0),
                xytext=(3, -1.5), fontsize=11, va="top", ha="center",
                arrowprops=dict(facecolor='black', width=1, headwidth=10),
                bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    ax.set_title("Test Signal")
    ax.legend()

    ax = axes[1]
    ax.set_xlabel("Time $t$")
    ax.plot(t_test, crosscor, color=color_blue, linestyle="-")

    max_index = np.argmax(crosscor)
    crosscor_max = crosscor[max_index]
    max_time = t_test[max_index]

    ax.annotate("Onset guess: $t$ = {:.2f}".format(max_time), xy=(0.98*max_time, 0.98*crosscor_max),
                xytext=(3, 0.03), fontsize=11, va="top", ha="center",
                arrowprops=dict(facecolor='black', width=1, headwidth=10),
                bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    ax.set_title("Cross-Correlation")

    plt.tight_layout()
    plt.suptitle("Detecting a Hidden Periodic Signal with Cross-Correlation", fontsize=16)
    plt.subplots_adjust(top=0.86)
    if save_figures: plt.savefig(figure_dir + "samples/crosscor-noisy_.png", dpi=200)

    plt.show()


def plot_bubo_crosscor():
    """
    Determines which owl in a noisy clip is which with cross-correlation, using the short noiseless clips as probes
    Each noisy mix signal is cross-correlated against both probes
    The cross-correlation with a larger max is deemed the matchj
    """
    bubo1, fs = sf.read(data_wav_dir + "bubo1.wav")  # read wav samples (mono)
    bubo2, fs = sf.read(data_wav_dir + "bubo2.wav")  # read wav samples (mono)

    for filename in sorted(os.listdir(data_wav_dir)):
        if "mix" in filename:
            signal, fs = sf.read(data_wav_dir + filename)
            N = len(signal)
            T = N/fs
            t = np.linspace(0, T, N, endpoint=False)

            cross_cor1 = crosscor_fft_basic(bubo1, signal)
            cross_cor2 = crosscor_fft_basic(bubo2, signal)
            max1 = np.max(cross_cor1)
            max2 = np.max(cross_cor2)
            min1 = np.min(cross_cor1)
            min2 = np.min(cross_cor2)
            norm = max(abs(min(min1, min2)), max(max1, max2))  # for normalization to [-1, 1]

            tmatch = 0
            if max1 >= max2:
                tmatch = t[np.where(cross_cor1 == norm)][0]
            elif max2 > max1:
                tmatch = t[np.where(cross_cor2 == norm)][0]

            cross_cor1 = crosscor_fft_basic(bubo1, signal)/norm
            cross_cor2 = crosscor_fft_basic(bubo2, signal)/norm

            ylim = (-1.1, 1.1)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            ax = axes[0]
            ax.set_xlabel("Time $t$ [s]")
            ax.set_ylim(ylim)
            ax.set_ylabel("Normalized Cross-Correllation")
            if max1 >= max2:
                ax.plot(t, cross_cor1, color=color_orange_dark, linewidth=3)
                ax.text(0.95, 0.9, "Best match: bubo1", transform=ax.transAxes, ha="right", va="top", fontsize=12,
                        bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
                ax.text(tmatch, -0.5, "Match time: {:.3f} s".format(tmatch), ha="left", va="top", fontsize=11,
                        bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
            else:
                ax.plot(t, cross_cor1, color=color_blue, linewidth=3)

            ax.set_title("Bubo 1", fontsize=13)


            ax = axes[1]
            ax.set_xlabel("Time $t$ [s]")
            ax.set_ylim(ylim)
            # remove y axis ticks
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off

            if max2 > max1:
                ax.plot(t, cross_cor2, color=color_orange_dark, linewidth=3)
                ax.text(0.95, 0.9, "Best match: bubo2", transform=ax.transAxes, ha="right", va="top", fontsize=12,
                        bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
                ax.text(tmatch, -0.5, "Match time: {:.3f} s".format(tmatch), ha="left", va="top", fontsize=11,
                        bbox=dict(facecolor='#FFFFFF', edgecolor='black', boxstyle='round,pad=0.2'))
            else:
                ax.plot(t, cross_cor2, color=color_blue, linewidth=3)
            ax.set_title("Bubo 2", fontsize=13)

            plt.tight_layout()
            plt.rc('text', usetex=True)
            plt.suptitle(r'Cross-Correlation Test for $\texttt{{{}}}$'.format(filename), fontsize=20)
            plt.rc('text', usetex=False)
            plt.subplots_adjust(top=0.85, hspace=0.3)

            if save_figures: plt.savefig(figure_dir + "cross-cor/" + filename.replace(".wav", "") + "-crosscor_.png", dpi=200)
            plt.show()


def plot_guitar_crosscor(probefile):
    """
    Determines when a specific-frequency guitar harmonic is played in a noisy clip
    with cross-correlation, using noiseless samples of the harmonics clips as probes
    The cross-correlation with a larger max is deemed the matchj
    """
    wav_data_dir = data_dir + "harmonics/"
    probe, fs = sf.read(wav_data_dir + probefile)  # read wav samples (mono)
    true_onset_time = get_onset_time(probefile.replace(".wav", ""))

    for filename in sorted(os.listdir(wav_data_dir)):
        if "sequence" in filename and "clean" not in filename:
            signal, fs = sf.read(wav_data_dir + filename)
            N = len(signal)
            T = N/fs
            t = np.linspace(0, T, N, endpoint=False)

            cross_cor = crosscor_fft_basic(probe, signal)
            cross_cor = cross_cor / max(np.abs(cross_cor))  # normalize

            fig, ax = plt.subplots(figsize=(8, 4))

            plt.plot(t, cross_cor, color=color_blue, linewidth=3)

            max_index = np.argmax(cross_cor)
            crosscor_max = cross_cor[max_index]
            max_time = t[max_index]

            ax.annotate("Onset guess: $t$ = {:.3f} s\nTrue onset: $t$ = {:.3f} s".format(max_time, true_onset_time), xy=(0.98 * max_time, 0.98 * crosscor_max),
                        xytext=(0.1*max_time, 0.7), fontsize=11, va="top", ha="left",
                        arrowprops=dict(facecolor='black', width=1, headwidth=10),
                        bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

            ax.set_xlabel("Time $t$ [s]", labelpad=0)
            ax.set_ylabel("Normalized Cross-Correllation")

            plt.tight_layout()
            plt.rc('text', usetex=True)
            plt.suptitle(r'Test for $\texttt{{{}}}$ in $\texttt{{{}}}$'.format(probefile.replace(".wav", ""), filename), fontsize=20)
            plt.rc('text', usetex=False)
            plt.subplots_adjust(top=0.91)

            if save_figures: plt.savefig(figure_dir + "cross-cor/" + probefile.replace(".wav", "-") + filename.replace(".wav", "") + ".png", dpi=200)
            plt.show()

def get_onset_time(note):
    """Helper method to map each guitar note to its corresponding start time in the recording sequence-clean.wav"""
    if note == "E3": return 0.912
    elif note == "G4": return 3.345
    elif note == "A3": return 5.473
    else: return 0.0  # should never occur

def test_cross_cor2d():
    test = np.zeros((64, 64))
    probe_size = 16
    probe = np.ones((probe_size, probe_size))
    test[4:4+probe_size, 3:3+probe_size] = probe
    crosscor = crosscor_2d_sum(probe, test)


def plot_2d_crosscor():
    probe_image = Image.open(image_dir + "ear-tuft.png")
    probe = np.asarray(probe_image, dtype=np.uint8)
    probe_title = "Probe (Primary Feathers)"

    test_image = Image.open(image_dir + "bubo-body-cropped.png")
    test = np.asarray(test_image, dtype=np.uint8)

    crosscor = crosscor_2d_sum(probe, test)**2  # compute cross-correlation

    f, axes = plt.subplots(1, 3)
    ax = axes[0]
    ax.imshow(probe, cmap=plt.cm.Greys_r)
    ax.set_title(probe_title)

    ax = axes[1]
    ax.imshow(test, cmap=plt.cm.Greys_r)
    ax.set_title("Test Image")

    ax = axes[2]
    ax.imshow(crosscor, cmap=plt.cm.Greys_r)
    ax.set_title("Squared Cross-Correlation")


    plt.show()


def plot_times():
    """
    Plots computation times of various autocorrelation implementations
    :return:
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # n = 1000 and below include all
    # n 10000 and below include matrix
    
    sum_cutoff = 6000  # sum measurements not included beyond this point---to expensive!

    sum_basic_files, sum_padded_files, fft_basic_files, fft_padded_files = 0, 0, 0, 0

    # find how many files there are for each implementation
    for filename in sorted((os.listdir(time_dir))):
        if ".csv" in filename and "_" not in filename:
            n = int(filename.replace("autocov-times-", "").replace("_", ""). replace(".csv", ""))
            fft_basic_files += 1  # increment in any case---all files include fft measurements
            fft_padded_files += 1
            if n <= sum_cutoff:
                sum_basic_files += 1
                sum_padded_files += 1

    sum_basic_n, sum_basic_times = np.zeros(sum_basic_files), np.zeros(sum_basic_files)
    sum_padded_n, sum_padded_times = np.zeros(sum_padded_files), np.zeros(sum_padded_files)
    fft_basic_n, fft_basic_times = np.zeros(fft_basic_files), np.zeros(fft_basic_files)
    fft_padded_n, fft_padded_times = np.zeros(fft_padded_files), np.zeros(fft_padded_files)

    i_sum = 0
    i_fft = 0
    for filename in natural_sort((os.listdir(time_dir))):
        if ".csv" in filename and "_" not in filename:
            n = int(filename.replace("autocov-times-", "").replace("_", ""). replace(".csv", ""))
            data = np.loadtxt(time_dir + filename, delimiter=',', skiprows=1)

            fft_basic_time = np.mean(data[:, 2])
            fft_padded_time = np.mean(data[:, 3])
            fft_basic_n[i_fft], fft_padded_n[i_fft] = n, n
            fft_basic_times[i_fft] = fft_basic_time
            fft_padded_times[i_fft] = fft_padded_time
            i_fft += 1

            if n <= sum_cutoff:
                sum_basic_time = np.mean(data[:, 0])
                sum_padded_time = np.mean(data[:, 1])
                sum_basic_n[i_sum] = n
                sum_padded_n[i_sum] = n
                sum_basic_times[i_sum] = sum_basic_time
                sum_padded_times[i_sum] = sum_padded_time
                i_sum += 1

    plt.subplot(1, 2, 1)
    plt.xlabel("Samples in Signal $N$")
    plt.ylabel("Computation Time [s]")
    plt.plot(sum_basic_n, sum_basic_times, color="#f05454", linestyle='--', marker='d', linewidth=2, label="sum biased")
    plt.plot(sum_padded_n, sum_padded_times, color="#ce6262", linestyle='--', marker='.', linewidth=2, label="sum unbiased")
    plt.plot(fft_basic_n, fft_basic_times, color="#af2d2d", linestyle='--', marker='s', linewidth=2, label="fft biased")
    plt.plot(fft_padded_n, fft_padded_times, color="#16697a", linestyle='--', marker='o', linewidth=2, label="fft unbiased")
    plt.legend()
    plt.title("Linear Scale", fontsize=10)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.xlabel("Samples in Signal $N$")
    plt.yscale("log")
    plt.plot(sum_basic_n, sum_basic_times, color="#f05454", linestyle='--', marker='d', linewidth=2, label="sum biased")
    plt.plot(sum_padded_n, sum_padded_times, color="#ce6262", linestyle='--', marker='.', linewidth=2, label="sum unbiased")
    plt.plot(fft_basic_n, fft_basic_times, color="#af2d2d", linestyle='--', marker='s', linewidth=2, label="fft unbiased")
    plt.plot(fft_padded_n, fft_padded_times, color="#16697a", linestyle='--', marker='o', linewidth=2, label="fft unbiased")
    plt.legend()
    plt.title("Logarithmic Scale", fontsize=10)
    plt.grid()

    plt.suptitle("Time Comparison of Autocorrelation Implementations", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    if save_figures: plt.savefig(figure_dir + "times_.png", dpi=200)
    plt.show()


def natural_sort(l):
    """For sorting filenames in natural order"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


def run_autocor_times(N):
    """
    Tests timing of various autocorrelation methods
    """
    T = 10
    t_start = 0
    t_end = t_start + T
    fs = N/T  # sample rate

    f_sin = 10  # frequency of sine wave

    t = np.arange(t_start, t_end, 1 / fs)  # generate time samples
    signal = np.sin(2*np.pi*t)

    runs = 5
    sum_basic_times = np.zeros(runs)
    sum_padded_times = np.zeros(runs)
    fft_basic_times = np.zeros(runs)
    fft_padded_times = np.zeros(runs)

    for i in range(runs):
        # print(i)
        # t = time.time()
        # autocov_sum_basic(signal)
        # t = time.time() - t
        # sum_basic_times[i] = t
        #
        # t = time.time()
        # autocov_sum(signal)
        # t = time.time() - t
        # sum_padded_times[i] = t

        t = time.time()
        autocov_fft_biased(signal, center=False)
        t = time.time() - t
        fft_basic_times[i] = t

        t = time.time()
        autocov_fft_unbiased(signal)
        t = time.time() - t
        fft_padded_times[i] = t

    times = np.column_stack([sum_basic_times, sum_padded_times, fft_basic_times, fft_padded_times])
    header = "Sum Basic, Sum Padded, FFT Basic, FFT Padded"
    np.savetxt(time_dir + "autocov-times-{}_.csv".format(int(N)), times, delimiter=',', header=header)
    print(times)


def do_times():
    """Wrapper method to test timing"""
    for N in range(2, 11):
        run_autocor_times(N * 1e5)


def do_compared_spectra():
    for filename in ("mix.wav", "mix1.wav", "mix2.wav", "mix22.wav"):
        plot_compared_spectra(filename, makeguess=True)


def do_autocor():
    for filename in ("mix.wav", "mix1.wav", "mix2.wav", "mix22.wav"):
        plot_autcor(filename)


def do_guitar_cross_cor():
    for probenote in ("E3.wav", "A3.wav", "G4.wav"):
        plot_guitar_crosscor(probenote)


def run():
    # plot_autocor_pure_noise()
    # plot_autocor_noisy_signal()
    # plot_crosscor_test()
    # practice()
    # plot_2d_crosscor()
    # plot_probe_spectrum()
    # do_autocor()
    plot_autcor("mix.wav")

    # do_compared_spectra()
    # confirm_cyclic()
    # plot_bubo_crosscor()
    # do_guitar_cross_cor()
    # do_times()
    # plot_times()

run()
