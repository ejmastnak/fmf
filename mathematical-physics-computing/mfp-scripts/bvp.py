import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import re
import time
from ivp_methods import *

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange

color_gray_ref = "#AAAAAA"  # light gray for reference quantities

# data_dir = "data/"
data_dir = "/Users/ejmastnak/Documents/Media/academics/fmf-media-winter-3/mafiprak/bvp/"
error_dir = data_dir + "error-step/"
time_dir = data_dir + "times/"
step_global_error_dir = data_dir + "stepsize-vs-global-error/"
step_local_error_dir = data_dir + "stepsize-vs-local-error/"

figure_dir = "../8-boundary-value-problem/figures/"
save_figures = True


# -----------------------------------------------------------------------------
# START
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------


def practice():
    # x = np.linspace(0, 1, 10)
    # y = np.linspace(10, 12.2342, 10)
    # data = np.column_stack([x, y])
    # print(data)
    # data = np.column_stack([x, y, x])
    # filename = data_dir + "test.csv"
    # header = "Max error:,{}\nTime [h], Temp [C], Abs Error [C]".format(np.max(y))
    # np.savetxt(filename, data, delimiter=',',  header=header)

    a = (1, 2, 3, 4)
    b = np.asarray(a)
    print(b)


def run():
    practice()


run()
