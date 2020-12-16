from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Start global constants
figure_directory = "../2-random-walks/figures/"
data_directory = "../2-random-walks/data/"

log_to_console = True      # controls whether to log outputs of series calculations
save_figures = False      # controls whether to save figures to avoid accidental overwrites
save_files = True      # controls whether to save figures to avoid accidental overwrites

# ranges for random variables
phi_min = 0         # azimuthal angle
phi_max = 2*np.pi
theta_min = 0       # polar angle
theta_max = np.pi
l_min = 1e-5        # min step size
l_max = 1e15        # max step size

mu = 1.5           # parameter for power distribution
N_walks = int(1e2)       # number of walks/flights to simulate

# walk parameters
v0 = 1              # the fixed speed of a step in Levy walks

# flight parameters
t0 = 1              # the fixed time of a step in Levy flights. Set to 1 so Levy flight steps and time are equal.


random_seed = 24  # seed for random number generation
np.random.seed(random_seed)

color1 = "#b3e5fc"  # lighter blue
color2 = "#01579b"  # darker blue
my_color_map = LinearSegmentedColormap.from_list("test", [color1, color2], N=100)


# End global constants

def get_rand_phi():
    """
    Returns a random float uniformly distributed on [phi_min, phi_max] (usually 0, 2*pi)
    Represents azimuthal angle; used to determine direction of a single random step in the plane/space
    :return:
    """
    return np.random.uniform(phi_min, phi_max)

def get_rand_theta():
    """
    Returns a random float uniformly distributed on [theta_min, theta_max] (usually 0, pi)
    Represents polar angle; used to determine direction of a single random step in space
    :return:
    """
    return np.random.uniform(theta_min, theta_max)


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


def get_next_flight_step(x, y, z, t, n):
    """
    Calculates relevant information for the next step for a Levy flight.

    :param x: previous x position
    :param y: previous x position
    :param z: previous x position
    :param t: time at step start
    :param n: number step in the walk
    :return: next (x, y, z) position, time at start of next step, next step number
    """
    phi = get_rand_phi()
    theta = get_rand_theta()
    l = get_rand_l()
    x_step = l * np.sin(theta) * np.cos(phi)    # calculate increments to x y and z
    y_step = l * np.sin(theta) * np.sin(phi)
    z_step = l * np.cos(theta)
    x = x + x_step              # update x y and z positions
    y = y + y_step
    z = z + z_step
    t += t0     # increment time by fixed step time
    n += 1
    return x, y, z, t, n


def get_next_walk_step(x, y, z, t, n):
    """
    Calculates relevant information for the next step for a Levy walk.

    :param x: previous x position
    :param y: previous x position
    :param z: previous x position
    :param t: time at step start
    :param n: number step in the walk
    :return: next (x, y, z) position, time at start of next step, next step number
    """
    phi = get_rand_phi()
    theta = get_rand_theta()
    l = get_rand_l()
    x_step = l * np.sin(theta) * np.cos(phi)  # calculate increments to x y and z
    y_step = l * np.sin(theta) * np.sin(phi)
    z_step = l * np.cos(theta)
    x = x + x_step  # update x y and z positions
    y = y + y_step
    z = z + z_step
    t += l / v0     # increment time using step distance and fixed step speed
    n += 1
    return x, y, z, t, n


def get_levy_walk_history(walk_steps):
    """
    Returns the step history of a simulated Levy walk, i.e. every position over time
    Used to create a 3D plot of the positions parameterized by time
    :param walk_time: total walk time
    :return: arrays holding the x, y, z and time history of the walk and total number of steps
    """
    x, y, z, t, n = 0, 0, 0, 0, 0  # initialize position, time and steps
    x_list, y_list, z_list, t_list = np.zeros(walk_steps), np.zeros(walk_steps), np.zeros(walk_steps), \
        np.zeros(walk_steps)  # initialize arrays to hold the flight history
    for i in range(0, walk_steps):
        x, y, z, t, n = get_next_walk_step(x, y, z, t, n)  # generate steps
        x_list[i] = x
        y_list[i] = y
        z_list[i] = z
        t_list[i] = t

    if save_files: np.savetxt(data_directory + "walk3d-mu-{:.1f}-steps-{}.txt".format(mu, walk_steps),
                              np.column_stack((t_list, x_list, y_list, z_list)),
                              header="t, x, y, z for {}-step 3D Levy walk with mu = {:.1f}".format(walk_steps, mu))
    return x_list, y_list, z_list, t_list, n


def get_levy_flight_history(flight_steps):
    """
    Returns the step history of a simulated Levy flight, i.e. every position over time
    Used to create a 3D plot of the positions parameterized by time
    :param flight_steps: steps in flight
    :return: numpy arrays holding the x, y, z and time history of the walk and total number of steps
    """
    x, y, z, t, n = 0, 0, 0, 0, 0  # initialize position, time and steps
    x_list, y_list, z_list, t_list = np.zeros(flight_steps), np.zeros(flight_steps), np.zeros(flight_steps), \
        np.zeros(flight_steps)  # initialize arrays to hold the flight history
    for i in range(0, flight_steps):
        x, y, z, t, n = get_next_flight_step(x, y, z, t, n)  # generate steps
        x_list[i] = x
        y_list[i] = y
        z_list[i] = z
        t_list[i] = t

    if save_files: np.savetxt(data_directory + "flight3d-mu-{:.1f}-steps-{}.txt".format(mu, flight_steps),
                              np.column_stack((t_list, x_list, y_list, z_list)),
                              header="t, x, y, z for {}-step 3D Levy flight with mu = {:.1f}".format(flight_steps, mu))
    return x_list, y_list, z_list, t_list


def plot_walk():
    """
    Plots the result of a single levy walk
    Color changes from light to dark with increasing time
    :return:
    """
    walk_steps = 1000
    x_list, y_list, z_list, t_list, n = get_levy_walk_history(walk_steps)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_list, y_list, z_list, color='#444444', linestyle='-', linewidth=1,
             label="Levy walk\n$\mu = {:.2f}$\nSteps: {}\nTime: {:.2e}".format(mu, n, t_list[-1]), zorder=-1)
    ax.scatter(x_list, y_list, z_list, c=t_list, cmap=my_color_map, s=30, zorder=-1)

    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_flight():
    """
    Plots the result of a single levy walk
    Color changes from light to dark with increasing time
    :return:
    """
    flight_steps = 1000
    x_list, y_list, z_list, t_list = get_levy_flight_history(flight_steps)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.plot(x_list, y_list, z_list, color='#444444', linestyle='-', linewidth=1,
             label="Levy flight $\mu = {:.2f}$\nSteps: {}".format(mu, flight_steps), zorder=2)
    ax.scatter(x_list, y_list, z_list, c=t_list, cmap=my_color_map, s=30, zorder=1)

    ax.legend()

    plt.tight_layout()
    plt.show()

mu = 3.5
def run():
    for i in (10, 100, 1000, 10000):
        get_levy_walk_history(i)
    # plot_walk()
    # plot_flight()


run()
