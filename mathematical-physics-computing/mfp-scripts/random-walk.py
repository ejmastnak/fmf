from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
import os

# Start global constants
figure_directory = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-winter-3/mafiprak/2-random-walk/figures/"
file_directory = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-winter-3/mafiprak/2-random-walk/data/"

log_to_console = False      # controls whether to log outputs of series calculations
save_figures = True      # controls whether to save figures to avoid accidental overwrites
save_files = True
rest_time = False

# ranges for random variables
phi_min = 0         # azimuthal angle
phi_max = 2*np.pi
l_min = 1e-5        # min step size
# l_max = 1e25        # max step size
t_min = 1e-10        # min waiting time
t_max = 1e10        # max waiting time

mu = 3.5            # parameter for step length power distribution
nu = 1.2            # parameter for rest time power distribution
N_walks = int(5e3)       # number of walks/flights to simulate

# walk parameters
v0 = 1              # the fixed speed of a step in Levy walks
walk_time = 1.8e-2     # how long a walk should take (set walk time instead of number of steps)
median_factor = 1   # basically how many times the median number of samples to take when analyzing walks; see below

# flight parameters
t0 = 1              # the fixed time of a step in Levy flights. Set to 1 so Levy flight steps and time are equal.
flight_time = 1e1*t0   # how long a flight should take.
flight_steps = int(flight_time/t0)  # number of steps in a levy flight


random_seed = 24  # seed for random number generation
np.random.seed(random_seed)


light_blue = "#b3e5fc"  # lighter blue
dark_blue = "#01579b"  # darker blue

light_orange = "#fdd062"  # light orange
dark_orange = "#a1001d"   # dark orange

light_color2 = "#cccccc"
light_color3 = "#aaaaaa"

blue_cmap = LinearSegmentedColormap.from_list("test", [light_blue, dark_blue], N=100)
orange_cmap = LinearSegmentedColormap.from_list("test", [light_orange, dark_orange], N=100)


marker_circle = 'o'   # circle
marker_triangle = '^'   # triangle
marker_square = 's'   # square
# End global constants

def linear_model_int_cf(t, k, b):
    """
    Linear model with y intercept for fitting with curve_fit
    :param t: independent variable
    :param k: slope
    :param b: y intercept
    :return:
    """
    return k*t + b


def get_gamma_flight(mu):
    """
    Returns theoretically predicted gamma as a function of mu for Levy flights
    :param mu:
    :return:
    """
    if 1 < mu < 3:
        return 2 / (mu - 1)
    elif mu >= 3:
        return 1
    else:  # for mu outside of simulated range
        return -1


def get_gamma_walk(mu):
    """
    Returns theoretically predicted gamma as a function of mu for Levy walk
    :param mu:
    :return:
    """
    if 1 < mu < 2:
        return 2
    elif 2 < mu < 3:
        return 4 - mu
    elif mu >= 3:
        return 1
    else:  # for mu outside of simulated range
        return -1


def get_rand_phi():
    """
    Returns a random float uniformly distributed on [phi_min, phi_max] (usually 0, 2*pi)
    Represents azimuthal angle; used to determine direction of a single random step in the plane/space
    :return:
    """
    return np.random.uniform(phi_min, phi_max)


def get_rand_l():
    """
    Returns a random float distributed on [l_min, l_max] (usually an approximation of 0, \infty)
    according to the power distribution f(l) \propto l^{-\mu}
    Represents the size/length/distance of a single random step (just step size, not step direction!)
    in an anomalous diffusion simulation
    :return:
    """
    rho = np.random.uniform(0, 1)   # generate random float on (0, 1)
    return l_min * (1 - rho) ** (1/(1 - mu))


def get_rand_t():
    """
    Returns a random float distributed on [t_min, t_max] (usually an approximation of 0, \infty)
    according to the power distribution f(t) \propto l^{-\nu}
    Represents the resting time of a single random step in an anomalous diffusion simulation
    :return:
    """
    rho = np.random.uniform(0, 1)   # generate random float on (0, 1)
    return t_min * (1 - rho) ** (1/(1 - nu))


def get_next_flight_step(x, y, t, n):
    """
    Calculates relevant information for the next step for a Levy flight.

    :param x: previous x position
    :param y: previous x position
    :param t: time at step start
    :param n: number step in the walk
    :return: next (x, y) position, time at start of next step, next step number
    """
    phi = get_rand_phi()
    l = get_rand_l()
    x_step = l * np.cos(phi)    # calculate increments to x and y
    y_step = l * np.sin(phi)
    x = x + x_step              # update x and y positions
    y = y + y_step
    t += t0     # increment time by fixed step time
    if rest_time:
        t += get_rand_t()
    n += 1
    return x, y, t, n


def get_next_walk_step(x, y, t, n):
    """
    Calculates relevant information for the next step for a Levy walk.

    :param x: previous x position
    :param y: previous x position
    :param t: time at step start
    :param n: number step in the walk
    :return: next (x, y) position, time at start of next step, next step number
    """
    phi = get_rand_phi()
    l = get_rand_l()
    x_step = l * np.cos(phi)    # calculate increments to x and y
    y_step = l * np.sin(phi)
    x = x + x_step              # update x and y positions
    y = y + y_step
    t += l / v0     # increment time using step distance and fixed step speed
    if rest_time:
        t += get_rand_t()
    n += 1
    return x, y, t, n


def get_levy_walk_history(walk_time):
    """
    Returns the step history of a simulated Levy walk, i.e. every position over time
    Walk length specified by total time
    :param walk_time: total walk time
    :return: arrays holding the x, y, r and time history of the walk and total number of steps
    """
    x, y, t, n = 0, 0, 0, 0  # initialize position, time and steps
    x_list, y_list, r_list, t_list = [], [], [], []  # initialize arrays to hold the flight history
    while t < walk_time:
        x, y, t, n = get_next_walk_step(x, y, t, n)  # generate steps
        # if log_to_console: print("Time: {:.5f}\t Step: {}".format(t, n))
        x_list.append(x)
        y_list.append(y)
        r_list.append(np.sqrt(x ** 2 + y ** 2))
        t_list.append(t)

    return x_list, y_list, r_list, t_list, n


def get_levy_walk_history_step_input(walk_steps):
    """
    Returns the step history of a simulated Levy walk, i.e. every position over time
    Walk length specified by number of steps
    :param walk_steps: total steps in walk
    :return: arrays holding the x, y, r and time history of the walk and total number of steps
    """
    x, y, t, n = 0, 0, 0, 0  # initialize position, time and steps
    x_list, y_list, r_list, t_list = np.zeros(walk_steps), np.zeros(walk_steps), np.zeros(walk_steps), \
                                     np.zeros(walk_steps)  # initialize arrays to hold the flight history
    for i in range(0, walk_steps):
        x, y, t, n = get_next_walk_step(x, y, t, n)  # generate steps
        x_list[i] = x
        y_list[i] = y
        r_list[i] = np.sqrt(x ** 2 + y ** 2)
        t_list[i] = t

    if save_files:
        np.savetxt(file_directory + "walk-mu-{:.1f}-steps-{}.txt".format(mu, walk_steps),
                   np.column_stack((t_list, x_list, y_list)),
                   header="t, x, y for {}-step Levy walk with mu = {:.1f}".format(walk_steps, mu))

    return x_list, y_list, r_list, t_list, n


def get_levy_flight_history(flight_time):
    """
    Returns the step history of a simulated Levy flight, i.e. every position over time
    Used to create a 2D plot of the positions parameterized by time
    :param flight_time: total flight time
    :return: numpy arrays holding the x, y, r, and time history of the walk and total number of steps
    """
    x, y, t, n = 0, 0, 0, 0  # initialize position, time and steps
    x_list, y_list, r_list, t_list = np.zeros(flight_steps), np.zeros(flight_steps), np.zeros(flight_steps), \
        np.zeros(flight_steps)  # initialize arrays to hold the flight history
    i = 0
    while t < flight_time:
        x, y, t, n = get_next_flight_step(x, y, t, n)  # generate steps
        x_list[i] = x
        y_list[i] = y
        r_list[i] = np.sqrt(x ** 2 + y ** 2)
        t_list[i] = t
        i += 1

    if save_files: np.savetxt(file_directory + "flight-mu-{:.1f}-steps-{}.txt".format(mu, flight_steps),
                              np.column_stack((t_list, x_list, y_list)),
                              header="t, x, y for {}-step Levy flight with mu = {:.1f}".format(flight_steps, mu))

    return x_list, y_list, r_list, t_list


def get_walk_simulation(n_walks, walk_time):
    """
    Simulates a large number of Levy walks.
    The code is little more involved than for flights because of the variable number of steps in a Levy
     walk of a given time

    :param n_walks: Number of walks to simulate
    :param walk_time: Time of the walk
    :return: vector of time t and MAD of distance from origin squared at time t
    """

    n_list = np.zeros(n_walks)  # 1D array to hold number of steps in each walk
    r_list, t_list = [], []   # effectively 2D lists: store the R and t lists for each walk (a list of lists)
    for i in range(0, n_walks):
        _, _, r_walk, t_walk, n_walk = get_levy_walk_history(walk_time)  # x, y, r, time and steps for each walk.
        r_list.append(r_walk)
        t_list.append(t_walk)
        n_list[i] = n_walk  # record number of steps in ith walk
        print(i)

    # calculate median number of steps in each walk and round to an int value.
    # use median and not mean to avoid disruption from outliers
    n_median = int(np.median(n_list)) * median_factor
    #
    # mxn 2D matrix holding distance from origin squared (R=r_list^2) of each walker over time.
    # m is the median of steps per walk
    # n is the number of walks
    # Columns hold R of a single walker over the course of the entire walk
    # Rows hold R of every walker at a specific time
    #
    r_matrix = np.zeros((n_median, n_walks))  # matrix of distances from origin.
    t_median = np.linspace(0, walk_time, n_median)   # splits time of walk into n_median samples

    # NOTES:
    # t_median and columns in t_list both grow monotonically from the same initial to the same final value
    #  i.e. from 0 to walk_time but have a different number of elements
    # A column in t_list and r_list at a given walk number have the same number of elements
    #  i.e. the number of steps in the walk

    # print("t-tist length: {}".format(len(t_list)))
    #
    # print("Full t-list")
    # print(t_list)
    #
    # print("\nLengths Incoming")
    # for i in range(0, len(t_list)):
    #     print(len(t_list[i]))
    #
    # print("\nT-List Elements Incoming")
    # for i in range(0, len(t_list)):
    #     print(t_list[i])
    # print("\nList of steps per walk:")
    # print(n_list)
    #


    for walk_counter in range(0, n_walks):  # loop through number of walks

        loop_start = 1  # where to start looping through t_list[walk_counter] (see below)

        for i_median in range(0, len(t_median)):  # loop through elements in t_median

            # find index i_match of element in t_list that best matches t_median[i_median]
            # set r_matrix[i_median][walk_counter] = r_list[walk_counter][i_match]

            this_t_median = t_median[i_median]  # value of t_median at index i_median

            prev_t, this_t = 0, t_list[walk_counter][0]  # successive elements of t_list[walk_counter]

            for i in range(loop_start, len(t_list[walk_counter])):  # loop through t_list[walk_counter].

                prev_t = this_t
                this_t = t_list[walk_counter][i]

                if prev_t < this_t_median < this_t:  # this_t_median falls between two elements of t_list[walk_counter]
                    if abs(this_t_median - prev_t) < abs(this_t_median - this_t):  # if prev_t is closest to this_t_median
                        r_matrix[i_median][walk_counter] = r_list[walk_counter][i-1]  # fill element of r_matrix with r_list value corresponding to prev_t
                        loop_start = i - 1  # start next loop through t_list[walk_counter] at i-1
                    else:
                        r_matrix[i_median][walk_counter] = r_list[walk_counter][i]  # fill element of r_matrix with r_list value corresponding to this_t
                        loop_start = i  # start next loop through t_list[walk_counter] at i
                    break

                # Reaching end of loop through t_list[walk_counter] without hitting if statement
                #  means this_t_median is larger than all elements in t_list[walk_counter]. In this case
                #  fill element of r_matrix with r_list value corresponding last time value of t_list[walk_counter]
                if i == len(t_list[walk_counter]) - 1: r_matrix[i_median][walk_counter] = \
                    r_list[walk_counter][len(t_list[walk_counter]) - 1]

    r_mad = stats.median_abs_deviation(r_matrix, axis=1)  # takes MAD of each row, i.e. MAD of r at a fixed time t

    t_median += walk_time / n_median   # shift all values up by one interval to avoid zero at the first element to avoid log(0)

    print("Mu: {:.2f}\t Walk time{:2e}\t Steps: {}".format(mu, walk_time, n_median))

    if save_files:
        np.savetxt(file_directory + "walk-MADr-mu-{:.1f}.txt".format(mu), np.column_stack((t_median, r_mad)), header="Time, MAD(r) for {} flights; {} steps per flight; seed: {}; mu: {:.2f}".format(n_walks, n_median, random_seed, mu))

    return t_median, r_mad


def get_flight_simulation(n_walks, flight_time):
    """
    Simulates a large number of Levy flights
    :param n_walks: Number of flights to simulate
    :param flight_time: Time of the flight
    :return: vector of time t and MAD of distance from origin squared at time t
    """
    #
    # 2D matrix holding distance from origin r of each walker over time.
    # Columns hold r of a single walker over the course of the entire walk
    # Rows hold r of every walker at a specific time
    #
    r_matrix = np.zeros((flight_steps, n_walks))  # matrix of distances from origin
    t = np.zeros(flight_steps)  # vector of times
    for i in range(0, n_walks):
        # the if/else is just a hacky way to use only the first time vector for t,
        #  since the time vector is the same in all the simulated Levy flights
        if i == 0:
            _, _, r, t = get_levy_flight_history(flight_time)  # list of x, y, r, time for each walk. Only r is used
        else:
            _, _, r, _ = get_levy_flight_history(flight_time)  # list of x, y, r, time for each walk. Only r is used

        r_matrix[:, i] = r  # sets the walk r as the r_matrix's ith column

    r_mad = stats.median_abs_deviation(r_matrix, axis=1)  # takes MAD of each row, i.e. MAD of r at a fixed time t

    if save_files:
        np.savetxt(file_directory + "flight-MADr-mu-{:.1f}.txt".format(mu), np.column_stack((t, r_mad)), header="Time, MAD(r) for {} flights; {} steps per flight; seed: {}; mu: {:.2f}".format(n_walks, flight_steps, random_seed, mu))
    return t, r_mad


def generate_flight_error_samples(n_samples, n_walks, flight_time):
    """
    Generates a list of files holding randomly generated data or r vs t

    :param n_samples:
    :param n_walks:
    :param flight_time:
    :return:
    """
    for i in range(0, n_samples):  # loop through desired number of MAD samples
        if i > 42: continue
        random_seed = i  # update seed
        np.random.seed(random_seed)  # reseed

        r_matrix = np.zeros((flight_steps, n_walks))  # matrix of distances from origin
        t = np.zeros(flight_steps)  # vector of times

        for j in range(0, n_walks):  # simulate n_walk walks
            # _, _, r, _ = get_levy_flight_history(flight_time)  # list of x, y, r, time for each walk. Only r is used
            r_matrix[:, j] = get_levy_flight_history(flight_time)[2]  # sets the walk r as the r_matrix's ith column

        r_mad = stats.median_abs_deviation(r_matrix, axis=1)  # takes MAD of each row, i.e. MAD of r at a fixed time t
        mad_log = 2 * np.log(r_mad)  # the quantity that's actually plotted to determine gamma parameter
        np.savetxt(file_directory + "flight-error-mu-{:.2f}-sample-{}.txt".format(mu, i+1), mad_log.T, header="2*np.log[MAD(r)] for {} flights; {} steps per flight; run {} of {}; mu: {:.2f}; seed: {}".format(n_walks, flight_steps, i+1, n_samples, mu, i))

    _, _, _, t = get_levy_flight_history(flight_time)  # list of x, y, r, time for each walk. Only r is used
    np.savetxt(file_directory + "flight-error-mu-{:.2f}-times.txt".format(mu), np.log(t).T, header="Logarithm of times for {} flights; {} steps per flight; {} samples; mu: {:.2f}".format(n_walks, flight_steps, n_samples, mu))


def calc_flight_MAD_error():
    """
    Calculates standard deviation of MAD as an error estimate

    :return:
    """
    n_samples = 50  # I made 50 sample files
    steps_per_flight = int(1e3)  # samples were made with 1000 steps per flight
    mad_r_matrix = np.zeros((steps_per_flight, n_samples))

    for local_mu in (1.5, 2.0, 2.2, 2.5, 3.0, 3.5):
        i = 0
        local_file_directory = file_directory + "flight-mu-{:.1f}/".format(local_mu)
        for filename in sorted(os.listdir(local_file_directory)):
            if "sample" in filename:
                this_mad_r = np.loadtxt(local_file_directory + filename)
                mad_r_matrix[:, i] = this_mad_r  # sets this this_mad_r as the r_matrix's ith column
                i += 1

        print("Value of i: {}\t(should be {})".format(i, n_samples))

        mad_r_std = np.std(mad_r_matrix, axis=1)  # takes std of MAD at each row, i.e. std of MAD at a fixed time t

        np.savetxt(file_directory + "flight-std/flight-std-mu-{:.2f}.txt".format(local_mu), mad_r_std.T, header="Standard deviation of 2*ln[MAD(r)] for mu value of {:.2f}".format(mu))


def plot_flight_error_samples():
    plt.xlabel("$\ln(t)$")
    plt.ylabel("Standard Deviation of $2\ln[$MAD$(r)]$")

    local_file_directory = file_directory + "flight-std/"

    t = np.loadtxt(local_file_directory+"flight-times.txt")

    for filename in sorted(os.listdir(local_file_directory)):
        if "std" in filename:
            std = np.loadtxt(local_file_directory + filename)
            print(len(std))
            plt.plot(t, std, linestyle='--', marker='o', label=filename.replace("flight-std-", ""))

    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_walk():
    """
    Plots the result of a single levy walk
    Color changes from light to dark with increasing time
    :return:
    """
    x_list, y_list, _, t_list, n = get_levy_walk_history(walk_time)
    plt.xlabel("x Position")
    plt.ylabel("y Position")
    plt.plot(x_list, y_list, color='#444444', linestyle='-', linewidth=1,
             label="Levy walk\n$\mu = {:.2f}$\nTime:  {:.2e}\nSteps: {:.2e}".format(mu, t_list[-1], n), zorder=2)
    plt.scatter(x_list, y_list, c=t_list, cmap=blue_cmap, s=30, zorder=1)
    plt.gca().set_aspect('equal')
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_flight():
    """
    Plots the result of a single levy walk
    Color changes from light to dark with increasing time
    :return:
    """
    x_list, y_list, _, t_list = get_levy_flight_history(flight_time)
    plt.xlabel("x Position")
    plt.ylabel("y Position")
    plt.plot(x_list, y_list, color='#444444', linestyle='-', linewidth=1,
             label="Levy flight $\mu = {:.2f}$\nTime:  {:.0e}\nSteps: {:.0e}".format(mu, t_list[-1], flight_steps), zorder=2)
    plt.scatter(x_list, y_list, c=t_list, cmap=blue_cmap, s=30, zorder=1)
    plt.gca().set_aspect('equal')
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.show()


def generate_2dflight_graphs():
    """
    Plots report-ready graphs of flights with 10, 100, 1000 and 10000 steps
    Color changes from light to dark with increasing time
    :return:
    """
    local_file_directory = file_directory + "flight-graph-data-2d/"
    titles = ("Levy Flight Ballistic Diffusion: $\mu=2$", "Levy Flight Super-Diffusion: $\mu=2.5$", "Levy Flight Normal Diffusion: $\mu=3.5$" )
    title_counter = 0
    for local_mu in (2.0, 2.5, 3.5):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        i = 1

        for filename in sorted(os.listdir(local_file_directory)):
            if "mu-{:.1f}".format(local_mu) in filename:
                plt.subplot(2,2, i)
                n_steps = filename.replace("flight-mu-{:.1f}-steps-".format(local_mu), "").replace(".txt", "")

                data = np.loadtxt(local_file_directory + filename)
                t, x, y = data[:,0], data[:,1], data[:,2]

                plt.gca().ticklabel_format(axis='both', style='scientific', scilimits=(0, 0))

                if i >= 3: plt.xlabel("x Position")
                if i % 2 == 1: plt.ylabel("y Position")
                plt.plot(x, y, color='#444444', linestyle='-', linewidth=1,
                         label=n_steps+"-step flight", zorder=2)
                plt.scatter(x, y, c=t, cmap=blue_cmap, s=30, zorder=1)
                # plt.gca().set_aspect('equal')
                if i <= 2: plt.legend(loc="best")
                elif i == 3: plt.legend(loc="upper left")
                else: plt.legend(loc="upper right")
                plt.grid()
                plt.tight_layout()
                i+=1

        plt.suptitle(titles[title_counter])
        plt.subplots_adjust(top=0.92)
        if save_figures: plt.savefig(figure_directory + "2dflight-mu-{:.1f}_.png".format(local_mu), dpi=250)
        plt.show()
        title_counter += 1


def generate_3dflight_graphs():
    """
    Plots report-ready graphs of flights with 10, 100, 1000 and 10000 steps
    Color changes from light to dark with increasing time
    :return:
    """
    local_file_directory = file_directory + "flight-graph-data-3d/"

    title_counter = 0
    titles = ("Levy Flight Ballistic Diffusion: $\mu=2$", "Levy Flight Super-Diffusion: $\mu=2.5$", "Levy Flight Normal Diffusion: $\mu=3.5$" )
    for local_mu in (2.0, 2.5, 3.5):
        fig = plt.figure(figsize=(8, 6.5))

        i = 1
        for filename in sorted(os.listdir(local_file_directory)):
            if "mu-{:.1f}".format(local_mu) in filename:
                ax = fig.add_subplot(2, 2, i, projection='3d')

                n_steps = filename.replace("flight3d-mu-{:.1f}-steps-".format(local_mu), "").replace(".txt", "")

                data = np.loadtxt(local_file_directory + filename)
                t, x, y, z = data[:,0], data[:,1], data[:,2], data[:,3]

                ax.ticklabel_format(axis='both', style='scientific', scilimits=(0, 0))

                ax.plot(x, y, z, color='#444444', linestyle='-', linewidth=1,
                         label=n_steps+"-step flight", zorder=2)
                ax.scatter(x, y, z, c=t, cmap=blue_cmap, s=30, zorder=1)
                ax.legend(loc="upper right")
                i+=1

        fig.tight_layout()
        plt.suptitle(titles[title_counter])
        plt.subplots_adjust(top=0.95)
        if save_figures: plt.savefig(figure_directory + "3dflight-mu-{:.1f}_.png".format(local_mu), dpi=250)
        plt.show()
        title_counter += 1


def generate_2dwalk_graphs():
    """
    Plots report-ready graphs of walks with 10, 100, 1000 and 10000 steps
    Color changes from light to dark with increasing time
    :return:
    """
    local_file_directory = file_directory + "walk-graph-data-2d/"
    title_counter = 0
    titles = ("Levy Walk Ballistic Diffusion: $\mu=1.5$", "Levy Walk Super-Diffusion: $\mu=2.5$", "Levy Walk Normal Diffusion: $\mu=3.5$" )
    for local_mu in (1.5, 2.5, 3.5):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        i = 1
        for filename in sorted(os.listdir(local_file_directory)):
            if "mu-{:.1f}".format(local_mu) in filename:
                plt.subplot(2,2, i)
                n_steps = filename.replace("walk-mu-{:.1f}-steps-".format(local_mu), "").replace(".txt", "")

                data = np.loadtxt(local_file_directory + filename)
                t, x, y = data[:,0], data[:,1], data[:,2]

                plt.gca().ticklabel_format(axis='both', style='scientific', scilimits=(0, 0))

                if i >= 3: plt.xlabel("x Position")
                if i % 2 == 1: plt.ylabel("y Position")
                plt.plot(x, y, color='#444444', linestyle='-', linewidth=1,
                         label=n_steps+"-step walk", zorder=2)
                plt.scatter(x, y, c=range(0, int(n_steps)), cmap=orange_cmap, s=30, zorder=1)
                if i <= 3: plt.legend(loc="best")
                # elif i == 3: plt.legend(loc="lower left")
                else: plt.legend(loc="upper right")
                plt.grid()
                plt.tight_layout()
                i+=1

        plt.suptitle(titles[title_counter])
        plt.subplots_adjust(top=0.92)
        if save_figures: plt.savefig(figure_directory + "2dwalk-mu-{:.1f}_.png".format(local_mu), dpi=250)
        plt.show()
        title_counter += 1


def generate_3dwalk_graphs():
    """
    Plots report-ready graphs of flights with 10, 100, 1000 and 10000 steps
    Color changes from light to dark with increasing time
    :return:
    """
    local_file_directory = file_directory + "walk-graph-data-3d/"

    title_counter = 0
    titles = ("Levy Walk Ballistic Diffusion: $\mu=1.5$", "Levy Walk Super-Diffusion: $\mu=2.5$", "Levy Walk Normal Diffusion: $\mu=3.5$" )
    for local_mu in (1.5, 2.5, 3.5):
        fig = plt.figure(figsize=(8, 6.5))

        i = 1
        for filename in sorted(os.listdir(local_file_directory)):
            if "mu-{:.1f}".format(local_mu) in filename:
                ax = fig.add_subplot(2, 2, i, projection='3d')

                n_steps = filename.replace("walk3d-mu-{:.1f}-steps-".format(local_mu), "").replace(".txt", "")

                data = np.loadtxt(local_file_directory + filename)
                t, x, y, z = data[:,0], data[:,1], data[:,2], data[:,3]

                ax.ticklabel_format(axis='both', style='scientific', scilimits=(0, 0))

                ax.plot(x, y, z, color='#444444', linestyle='-', linewidth=1,
                         label=n_steps+"-step walk", zorder=2)
                ax.scatter(x, y, z, c=range(0, int(n_steps)), cmap=orange_cmap, s=30, zorder=1)
                ax.legend(loc="upper right")
                i+=1

        fig.tight_layout()
        plt.suptitle(titles[title_counter])
        plt.subplots_adjust(top=0.95)
        if save_figures: plt.savefig(figure_directory + "3dwalk-mu-{:.1f}_.png".format(local_mu), dpi=250)
        plt.show()
        title_counter += 1


def plot_walk_simulation():
    """
    Plots a simulation of many Levy walks; used to determine parameter gamma
    Plots 2ln(MAD(r)) vs ln(t); parameter gamma is slope of fitted line
    :return:
    """
    local_mu = mu

    local_file_directory = file_directory + "walk-data/"
    filename = "walk-MADr-mu-{:.1f}.txt".format(local_mu)
    # data = np.loadtxt(local_file_directory + filename)
    # t, r_mad = data[:, 0], data[:, 1]  # unpack time and MADr

    t, r_mad = get_walk_simulation(N_walks, walk_time)

    t_log = np.log(t)[1:-1]  # drops first elements which behave strangely
    mad_log = 2*np.log(r_mad)[1:-1]  # drops first elements which behave strangely

    guess = (local_mu, 10)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, t_log, mad_log, guess, sigma=0.02*mad_log, absolute_sigma=True) #, sigma=0.01*mad_log, absolute_sigma=False)
    gamma_fit, b = opt  # unpack slope and y intercept
    gamma_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element

    t_fit = np.linspace(np.min(t_log), np.max(t_log), 100)
    mad_fit = linear_model_int_cf(t_fit, gamma_fit, b)

    plt.xlabel("$\ln(t)$")
    plt.ylabel("$\ln[$MAD$(r)]$")

    plt.plot(t_log, mad_log, linestyle='--', marker='o', label="Levy walk $\mu$={:.2f}\nExpected $\gamma$: {:.2f}".format(local_mu, get_gamma_walk(local_mu)))
    plt.plot(t_fit, mad_fit, linestyle='--', label='Fit: y = $\gamma$x + b\n  $\gamma$ = {:.2f} $\pm$ {:.2f}'.format(gamma_fit, gamma_err))
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_flight_simulation():
    """
    Plots a simulation of many Levy flights; used to determine parameter gamma

    Plots 2ln(MAD(r)) vs ln(t); parameter gamma is slope of fitted line
    :return:
    """
    # t, r_mad = get_flight_simulation(N_walks, flight_time)

    data = np.loadtxt(file_directory + "flight-data/flight-MADr-mu-{:.1f}.txt".format(mu))
    t, mad_r = data[:, 0], data[:, 1]  # unpack time and MADr
    t_log = np.log(t)
    mad_log = 2*np.log(mad_r)

    # scale error by sqrt(N), noting error was performed with N_walks = 100
    errors = np.loadtxt(file_directory + "flight-std/flight-std-mu-{:.2f}.txt".format(mu), skiprows=1) / (np.sqrt(N_walks/100))
    guess = (mu, -25)  # guess for slope and y intercept
    opt, cov = curve_fit(linear_model_int_cf, t_log, mad_log, guess, sigma=errors, absolute_sigma=True)
    gamma_fit, b = opt  # unpack slope and y intercept
    gamma_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
    print("Fit: {:.3f}\t Error:  {:.3f}".format(gamma_fit, gamma_err))
    t_fit = np.linspace(np.min(t_log), np.max(t_log), 100)
    mad_fit = linear_model_int_cf(t_fit, gamma_fit, b)

    plt.xlabel("$\ln(t)$\t{} flights at {} steps per flight".format(N_walks, flight_steps))
    plt.ylabel("$2\ln[$MAD$(r)]$")

    # plt.plot(t, r_mad, label="Flight")
    plt.plot(t_log, mad_log, linestyle='--', marker='o', label="Levy flight $\mu$={:.2f}\nExpected $\gamma$: {:.2f}".format(mu, get_gamma_flight(mu)))
    plt.plot(t_fit, mad_fit, linestyle='--', label='Fit: y = $\gamma$x + b\n  $\gamma$ = {:.2f} $\pm$ {:.2f}'.format(gamma_fit, gamma_err))
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()


def generate_flight_fits():
    local_file_directory = file_directory + "flight-data/"
    N_walks = 10000

    fig = plt.figure(figsize=(7, 4))

    i = 0
    colors = ("#75c1c5", "#2e70a7", "#161e6e")  # blues from light to dark
    for filename in sorted(os.listdir(local_file_directory)):
        if "flight-MADr" in filename:
            data = np.loadtxt(local_file_directory + filename)
            t, mad_r = data[:, 0], data[:, 1]  # unpack time and MADr
            t_log = np.log(t)
            mad_log = 2 * np.log(mad_r)
            mu = float(filename.replace("flight-MADr-mu-", "").replace(".txt", ""))  # extracts mu from file name


            # scale error by sqrt(10000/100) = 10, noting error was performed with N_walks = 100
            errors = np.loadtxt(file_directory + "flight-std/flight-std-mu-{:.2f}.txt".format(mu), skiprows=1)
            guess = (mu, -25)  # guess for slope and y intercept
            opt, cov = curve_fit(linear_model_int_cf, t_log, mad_log, guess, sigma=errors, absolute_sigma=True)
            gamma_fit, b = opt  # unpack slope and y intercept
            gamma_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
            t_fit = np.linspace(np.min(t_log), np.max(t_log), 100)
            mad_fit = linear_model_int_cf(t_fit, gamma_fit, b)

            # print("Fit: {:.3f}\t Error:  {:.2f}".format(gamma_fit, gamma_err))

            plt.plot(t_log, mad_log, linestyle='none', marker='o', color=colors[i],
                     label="$\mu$={:.2f} | $\gamma$: {:.2f}\nFit $\gamma$: {:.2f} $\pm$ {:.2f}".format(mu, get_gamma_flight(mu), gamma_fit, gamma_err))
            plt.plot(t_fit, mad_fit, color="#efa444", linestyle='--')  # label='Fit: y = $\gamma$x + b\n  $\gamma$ = {:.2f} $\pm$ {:.2f}'.format(gamma_fit, gamma_err)
            i += 1

    plt.title("10000 Levy flights at 1000 steps per flight")
    plt.xlabel("$\ln(t)$")
    plt.ylabel("$2\ln[$MAD$(t)]$")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()

    if save_figures: plt.savefig(figure_directory + "flight-fit_.png", dpi=250)

    plt.show()


def generate_walk_fits():
    local_file_directory = file_directory + "walk-data/"

    fig = plt.figure(figsize=(7, 4))

    i = 0
    colors = ("#f5c450", "#e1692e", "#91331f")  # oranges from light to dark

    for filename in sorted(os.listdir(local_file_directory)):
        if "walk-MADr" in filename:
            data = np.loadtxt(local_file_directory + filename)
            t, mad_r = data[:, 0], data[:, 1]  # unpack time and MADr
            t_log = np.log(t)[2:-1]
            mad_log = 2 * np.log(mad_r)[2:-1]
            mu = float(filename.replace("walk-MADr-mu-", "").replace(".txt", ""))  # extracts mu from file name

            t_log -= t_log[0]  # shift times to start at 0
            mad_log -= 20 + mu + mad_log[0]  # shift mad to roughly same positions

            # scale error by sqrt(10000/100) = 10, noting error was performed with N_walks = 100
            errors = np.loadtxt(file_directory + "flight-std/flight-std-mu-{:.2f}.txt".format(mu), skiprows=1)/10
            guess = (mu, -25)  # guess for slope and y intercept
            opt, cov = curve_fit(linear_model_int_cf, t_log, mad_log, guess, sigma=0.08*mad_log, absolute_sigma=True)
            gamma_fit, b = opt  # unpack slope and y intercept
            gamma_err = np.power(cov[0][0], 0.5)  # error is square root of corresponding covariance matrix element
            t_fit = np.linspace(np.min(t_log), np.max(t_log), 100)
            mad_fit = linear_model_int_cf(t_fit, gamma_fit, b)

            plt.plot(t_log, mad_log, linestyle='none', marker='o', color=colors[i],
                     label="$\mu$={:.2f} | $\gamma$: {:.2f}\nFit $\gamma$: {:.2f} $\pm$ {:.2f}".format(mu, get_gamma_walk(mu), gamma_fit, gamma_err))
            plt.plot(t_fit, mad_fit, color="#1d3580", linestyle='--')  # label='Fit: y = $\gamma$x + b\n  $\gamma$ = {:.2f} $\pm$ {:.2f}'.format(gamma_fit, gamma_err)
            i += 1

    plt.title("10000 Levy walks at$\sim$1000 steps per walks")
    plt.xlabel("$\ln(t)$")
    plt.ylabel("$2\ln[$MAD$(t)]$")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()

    if save_figures: plt.savefig(figure_directory + "walk-fit_.png", dpi=250)

    plt.show()


def numpy_test():
    i = 0
    for i in range(0, 5):
        print(i)

    print(i)

    m, n = 3, 5

    A = np.zeros((m, n))

    col = np.zeros(m)  # column is three elements long
    for i in range(0, m): col[i] = i+1

    row = np.zeros(n)  # column is three elements long
    for i in range(0, n): row[i] = i + 10

    A[1] = row
    A[2] = row
    A[:, 1] = col
    A[:, 3] = col
    print(A)
    print("\nColumn:")
    print(col)

    A = np.column_stack((col, A))
    print(A)
    # np.savetxt(np.append)

    # print(stats.median_abs_deviation(A, axis=0))  # returns a 1D matrix whose elements are MAD of each column
    # print(stats.median_abs_deviation(A, axis=1))  # returns a 1D matrix whose elements are MAD of each row

def run():
    # plot_walk()
    # plot_flight()
    # for i in (10, 100, 1000, 10000):
    #     get_levy_walk_history_step_input(i)
    # get_levy_flight_history(flight_time)

    # generate_2dflight_graphs()
    # generate_2dwalk_graphs()
    # generate_3dflight_graphs()
    # generate_3dwalk_graphs()
    # generate_flight_fits()
    generate_walk_fits()
    # plot_walk_simulation()

    # plot_flight_simulation()
    # get_walk_simulation(N_walks, walk_time)
    # numpy_test()
    # generate_flight_error_samples(50, N_walks, flight_time)
    # plot_flight_error_samples()
    # calc_flight_MAD_error()

run()


# Yo walks should be stopped after a given time, not a number of steps!
# for flights it doesn't really matter because time and steps are directly proportional