import numpy as np
from numpy.fft import fft
from numpy.fft import fftshift
from matplotlib import pyplot as plt
import os
from scipy.optimize import curve_fit

save_figures = True
data_dir = "measurements/"
fig_dir = "figures/"

color_re = "#244d90"  # darker teal / blue
color_im = "#91331f"  # dark orange


def clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_spectra():
    for filter in ("bartlett", "box", "cosine", "gauss"):
        filter_dir = data_dir + filter + "/"

        backgrounds = ("#111111", "white", "white")
        line_colors = ("white", "#48ffc8", "red")
        fig, axes = plt.subplots(3, 2, figsize=(7, 9))
        for row, source in enumerate(("bulb", "hg", "laser")):

            data = np.loadtxt(filter_dir + source + ".txt", skiprows=1)
            data_fft = np.loadtxt(filter_dir + source + "-fft.txt", skiprows=1)
            x = data[:,0]  # [um] mirror displacement
            I = data[:,2]  # apodized intensity
            k = data_fft[:,0]  # [1/um]  wave number
            dft = data_fft[:,1]  # abs[fft(I)]

            # first column - spectra
            ax = axes[row][0]
            clean_axis(ax)
            ax.set_facecolor(backgrounds[row])
            ax.plot(x, I, color=line_colors[row])
            ax.set_ylabel("Intensity $I$")
            if row == 2:
                ax.set_xlabel("Mirror Displacement $x$ [um]")

            # second column - fft
            ax = axes[row][1]
            clean_axis(ax)
            ax.set_facecolor(backgrounds[row])
            ax.plot(k, dft, color=line_colors[row])
            ax.set_ylabel("|FFT[I(x)]|")
            if row == 2:
                ax.set_xlabel("Wave Number $2\pi k$ [1/um]")

        center = 0.55
        plt.figtext(center, 0.315, "Helium-Neon Laser", va="bottom", ha="center", size=15)
        plt.figtext(center, 0.615, "Mercury Lamp", va="bottom", ha="center", size=15)
        plt.figtext(center, 0.915, "White Light", va="bottom", ha="center", size=14)

        plt.suptitle("      " + filter.capitalize() + " Apodization", fontsize=18)
        plt.tight_layout()
        plt.subplots_adjust(top=0.91, hspace=0.3)
        if save_figures: plt.savefig(fig_dir + filter + "_.png", dpi=200)
        plt.show()


def print_hg_lines():
    for k in (1.78, 1.83, 2.30, 2.48):  # [1/um]
        print("\nWavelength: {:.0f} [nm]".format(1000/k))  # convert to nm
        print("Frequency: {:.2e}".format(k * 1.0e6 * 3.0e8))  # convert to Hz

# plot_spectra()
print_hg_lines()
