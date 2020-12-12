import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.sparse as sparse

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

color_blue = "#244d90"  # darker teal / blue
color_teal = "#3997bf"  # lighter teal/blue

color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange
color_orange_light = "#f5c450"  # light orange
color_gray_ref = "#AAAAAA"  # light gray for reference quantities

l, sigma, amp, D = 1.0, 0.2, 1.0, 2.0
dt = 1e-12
speed = int(1e8)


def gauss(x, amp, x0, sigma):
    return amp * np.exp(-(x-x0)**2/sigma**2)


def spectral():
    """
    Fourier method solution for heat diffusion in a rod of length l
    using homogeneous Dirichlet boundary conditions
    T(0, 0) = T(0, l) = 0  i.e. zero temperature at rod endpounts
    """
    N = 2**10  # 1024 number points
    x = np.linspace(0, l, N)  # x interval spanning rod length

    T = gauss(x, amp, l/2, sigma)  # initial temperature distribution
    T -= gauss(0, amp, l/2, sigma)  # subtract initial value

    T = np.concatenate((T, -T[::-1]))  # concatenate T and - T for Dirichlet boundary conditions
    c = fftpack.fft(T)

    r = 2
    fmin = -(len(T)/(r*l))/2  # basically -N/2
    fmax = (len(T)/(r*l))/2  # N/2
    frange = np.arange(fmin, fmax, 1/(r*l))  # 2048 points from fmin to fmax
    # f = np.roll(frange, len(T)//2)
    f = np.fft.fftshift(frange)  # 0 to N/2 followed by -N/2 to 0
    step = np.power(np.ones(len(f)) - 4*np.pi**2*D*dt*np.multiply(f, f), speed)  # step for numerical integration

    plt.plot(np.linspace(-l, l, int(2*N)), T)
    plt.show()


def data_gen_spectral(T, N, c, step, t):
    """

    :param T: Temperature function
    :param N: number of points used in the x position grid spanning rod length
    :param c: coefficients c_k(t)
    :param step: TOOD see report
    :param t: float value giving current tie
    :return:
    """
    while True:
        t += speed*dt  # increment time
        c *= step  # increment coefficients
        T = np.real(fftpack.ifft(c))
        yield T[:N]  # first N elements of T


def collocation():
    N = 500
    x = np.linspace(0, l, N + 1)
    dx = x[1] - x[0]

    T = gauss(x, amp, l/2, sigma)  # initial condition
    T -= gauss(0, amp, l/2, sigma)

    A = sparse.diags([np.ones(N-2), 4*np.ones(N-1), np.ones(N-2)], [-1, 0, 1]).toarray()
    A_inv = np.linalg.inv(A)
    B = 6.0 * D / (dt**2) * sparse.diags([np.ones(N-2), -2.0*np.ones(N-1), np.ones(N-2)], [-1, 0, 1]).toarray()
    step = np.linalg.matrix_power(np.diag(np.ones(N-1)) + dt*np.dot(A_inv, B), speed)
    a = np.dot(A_inv, T[1:N])


def data_gen_collocation(T, N, a, step, A, t):
    """
    :param T: initial condition 1D numpy array
    :param N: number of points in x position grid spanning rod length
    :param a: coefficient TODO see report
    :param step: matrix used calculate a
    :param A: another matrix
    :param t: float value giving current time
    :return:
    """
    while True:
        if int(t/(speed*dt))%100 ==0:
            # axarr[0].plot(x, T)
            print("plot here!")

        # time_text.set_text("t: {:.4f}".format(t))
        t += speed * dt
        a = step.dot(a)
        T = np.concatenate(([0], A.dot(a), [0]))
        yield T


# -----------------------------------------------------------------------------
# END ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
