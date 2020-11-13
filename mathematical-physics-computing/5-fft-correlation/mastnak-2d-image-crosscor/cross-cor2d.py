import numpy as np
from numpy.fft import fft, ifft, fftshift
from PIL import Image
import matplotlib.pyplot as plt

# data_dir = "../5-dft/data/"
data_dir = "/Users/ejmastnak/Documents/Media/academics/fmf-media-winter-3/mafiprak/fft/"
image_dir = data_dir + "images/"
data_txt_dir = data_dir + "txt/"
data_wav_dir = data_dir + "wav/"
time_dir = "../5-fft/data/times/"
figure_dir = "../5-fft/figures/"
save_figures = True

def crosscor_2d_sum(probe, test):
    """
    Input a small KxL pixel probe image and a larger MXN pixel test image where K<=M and L<=N
    Finds cross-correlation of the two image
    Zero-pads the test image's edges
    Note that images do note have to be square or powers of two.
    """
    M, N = test.shape[0], test.shape[1]
    K, L = probe.shape[0], probe.shape[1]
    test = test - np.mean(test)  # center signals
    probe = probe - np.mean(probe)  # center signals
    crosscor = np.zeros((M, N))  # preallocate
    test_padded = np.zeros((M + K, N + L))  # pad test image edges
    test_padded[K//2 : K//2 + M, L//2 : L//2 + N] = test

    for cor_row in range(M):  # loop through rows in correlation
        print("Row {} of {}".format(cor_row, M))
        for cor_col in range(N):  # loop through columns in correlation
            crosscor_row_col = 0
            for probe_row in range(K):
                for probe_col in range(L):
                    crosscor_row_col += probe[probe_row, probe_col]*test_padded[cor_row + probe_row, cor_col + probe_col]
            crosscor[cor_row, cor_col] = crosscor_row_col

    return crosscor


def plot_2d_crosscor():
    image_dir = "image-samples/"
    probe_file = "eye-left.png"
    test_file = "bubo-body-cropped.png"
    probe_image = Image.open(image_dir + probe_file)  # load file (requires Pillow library)
    probe = np.asarray(probe_image, dtype=np.uint8)  # convert to 8-bit int per grayscale convention
    probe_title = "Probe (Left Eye)"

    test_image = Image.open(image_dir + test_file)  # load file (requires Pillow library)
    test = np.asarray(test_image, dtype=np.uint8)  # convert to 8-bit int per grayscale convention

    crosscor = crosscor_2d_sum(probe, test)  # compute cross-correlation

    # Plot regular cross-correlation
    f, axes = plt.subplots(1, 3)
    ax = axes[0]
    ax.imshow(probe, cmap=plt.cm.Greys_r)
    ax.set_title(probe_title)
    ax = axes[1]
    ax.imshow(test, cmap=plt.cm.Greys_r)
    ax.set_title("Test Image")
    ax = axes[2]
    ax.imshow(crosscor, cmap=plt.cm.Greys_r)
    ax.set_title("Cross-Correlation")
    plt.show()

    # Plot squared cross-correlation to highlight agreements
    f, axes = plt.subplots(1, 3)
    ax = axes[0]
    ax.imshow(probe, cmap=plt.cm.Greys_r)
    ax.set_title(probe_title)
    ax = axes[1]
    ax.imshow(test, cmap=plt.cm.Greys_r)
    ax.set_title("Test Image")
    ax = axes[2]
    ax.imshow(crosscor**2, cmap=plt.cm.Greys_r)
    ax.set_title("Squared Cross-Correlation")
    plt.show()

plot_2d_crosscor()
