import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from ivp_methods import *

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# -----------------------------------------------------------------------------
# START GLOBAL CONSTANTS AND PARAMETERS
# -----------------------------------------------------------------------------
G = 6.672e-11  # universal gravitational constant in SI units
m_E = 5.972e24  # mass of the earth [kg]
m_S = 1.989e30  # mass of the sun [kg]
m_M = 7.348e22  # mass of the moon [kg]

earth_to_sun = 1.50e11  # mean distance from the earth to the sun in meters
earth_orbital_velocity = 2.98e4  # earth's mean orbital velocity around the sun in m/s
moon_to_earth = 3.75e8
moon_inclination = 0.09  # inclination of the moon to the ecliptic in radians (5.2 degrees)
moon_orbital_velocity = 1.02e3  # the moon's mean orbital velocity around the earth in m/s

blue_light, blue_dark = "#b3e5fc", "#01579b"  # lighter and darker blue
blue_cmap = LinearSegmentedColormap.from_list("blues", [blue_light, blue_dark], N=100)
gray_light, gray_dark = "#BBBBBB", "#333333"
gray_cmap = LinearSegmentedColormap.from_list("grays", [gray_light, gray_dark], N=100)

figure_dir = "/Users/ejmastnak/Documents/Dropbox/academics/fmf-local/fmf-winter-3/mafiprak/7-newton/figures/"
save_figures = True
# -----------------------------------------------------------------------------
# END GLOBAL CONSTANTS AND PARAMETERS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START EMS DE EQUATION FUNCTIONS
# -----------------------------------------------------------------------------
def get_ems_state(state, t):
    """
    Differential equations for a simple model for the Earth-Moon-Sun system
    Assumes the Earth, Moon, and Sun are point masses
    Assumes the Sun is stationary
    Takes the sun as the center of an inertial coordinate system

    (x,y,z), v, and a denote position, velocity and acceleration, respectively
    e and m subscripts denote Earth and Moon, respectively

    :param state: position and velocity of earth and moon [xe, ye, ze, xm, ym, zm, vex, vey, vez, vmx, vmy, vmz]
    :param t: time
    :return: velocity and acceleration of earth and moon [vex, vey, vez, vmx, vmy, vmz, aex, aey, aez, amx, amy, amz]
    """

    # vex = state[6]
    # vey = state[7]
    # vez = state[8]
    # vmx = state[9]
    # vmy = state[10]
    # vmz = state[11]

    r_x = state[3] - state[0]
    r_y = state[4] - state[1]
    r_z = state[5] - state[2]

    aex = -G * ((m_S * state[0] / (vector_magnitude([state[0], state[1], state[2]]) ** 3)) -
                (m_M * r_x / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    aey = -G * ((m_S * state[1] / (vector_magnitude([state[0], state[1], state[2]]) ** 3)) -
                (m_M * r_y / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    aez = -G * ((m_S * state[2] / (vector_magnitude([state[0], state[1], state[2]]) ** 3)) -
                (m_M * r_z / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    amx = -G * ((m_S * state[3] / (vector_magnitude([state[3], state[4], state[5]]) ** 3)) +
                (m_E * r_x / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    amy = -G * ((m_S * state[4] / (vector_magnitude([state[3], state[4], state[5]]) ** 3)) +
                (m_E * r_y / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    amz = -G * ((m_S * state[5] / (vector_magnitude([state[3], state[4], state[5]]) ** 3)) +
                (m_E * r_z / (vector_magnitude([r_x, r_y, r_z]) ** 3)))

    return_state = np.zeros(np.shape(state))
    for i in range(6, 12):
        return_state[i-6] = state[i]
    return_state[6], return_state[7], return_state[8] = aex, aey, aez
    return_state[9], return_state[10], return_state[11] = amx, amy, amz

    return return_state
    # return [vex, vey, vez, vmx, vmy, vmz, aex, aey, aez, amx, amy, amz]


def get_ems_state_symp(coordinates):
    """
    Differential equations for a simple model for the Earth-Moon-Sun system
    Written for use with symplectic integration methods

    Assumes the Earth, Moon, and Sun are point masses
    Assumes the Sun is stationary
    Takes the sun as the center of an inertial coordinate system

    (x,y,z), v, and a denote position, velocity and acceleration, respectively
    e and m subscripts denote Earth and Moon, respectively
    
    :param coordinates: positions of earth and moon [xe, ye, ze, xm, ym, zm]
    :return: acceleration of earth and moon [aex, aey, aez, amx, amy, amz]
    """

    # calculate relative position of earth and moon
    r_x = coordinates[3] - coordinates[0]
    r_y = coordinates[4] - coordinates[1]
    r_z = coordinates[5] - coordinates[2]

    aex = -G * ((m_S * coordinates[0] / (vector_magnitude([coordinates[0], coordinates[1], coordinates[2]]) ** 3)) -
                (m_M * r_x / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    aey = -G * ((m_S * coordinates[1] / (vector_magnitude([coordinates[0], coordinates[1], coordinates[2]]) ** 3)) -
                (m_M * r_y / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    aez = -G * ((m_S * coordinates[2] / (vector_magnitude([coordinates[0], coordinates[1], coordinates[2]]) ** 3)) -
                (m_M * r_z / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    amx = -G * ((m_S * coordinates[3] / (vector_magnitude([coordinates[3], coordinates[4], coordinates[5]]) ** 3)) +
                (m_E * r_x / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    amy = -G * ((m_S * coordinates[4] / (vector_magnitude([coordinates[3], coordinates[4], coordinates[5]]) ** 3)) +
                (m_E * r_y / (vector_magnitude([r_x, r_y, r_z]) ** 3)))
    amz = -G * ((m_S * coordinates[5] / (vector_magnitude([coordinates[3], coordinates[4], coordinates[5]]) ** 3)) +
                (m_E * r_z / (vector_magnitude([r_x, r_y, r_z]) ** 3)))

    accelerations = np.zeros(np.shape(coordinates))
    accelerations[0], accelerations[1], accelerations[2] = aex, aey, aez
    accelerations[3], accelerations[4], accelerations[5] = amx, amy, amz

    return accelerations


def vector_magnitude(r):
    """ Helper funciton to return the length (magnitude) of a 3D vector """
    return np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)


def get_initial_conditions():
    """ Auxiliary function to pack initial conditions into an array """
    
    # initial positions
    xe0 = earth_to_sun  # initial x position of earth in m
    ye0 = 0  # initial y position of earth in m
    ze0 = 0  # initial z position of earth in m
    xm0 = moon_to_earth * np.cos(moon_inclination) + earth_to_sun  # initial x position of the moon in m
    ym0 = 0  # initial y position of the moon
    zm0 = moon_to_earth * np.sin(moon_inclination)  # initial z position of the moon

    # initial velocities
    vex0 = 0  # initial x velocity of earth
    vey0 = earth_orbital_velocity  # initial y velocity of earth
    vez0 = 0  # initial z velocity of earth
    vmx0 = 0  # initial x velocity of the moon
    vmy0 = moon_orbital_velocity + earth_orbital_velocity  # initial y velocity of the moon
    vmz0 = 0  # initial z velocity of the moon

    # pack initial conditions into an array and return values
    return [xe0, ye0, ze0, xm0, ym0, zm0, vex0, vey0, vez0, vmx0, vmy0, vmz0]


def get_ems_motion(sim_years=1.0, dt=1e5):
    """
    Solves the motion of the Earth-Moon-Sun system using the model in get_ems_state on the time values t
    :param sim_years: number of years over which to run the simulation
    :param dt: time step, in seconds
    :return: position and velocity of the earth and moon over the time t
    """
    t_min = 0  # start time
    t_max = (3600. * 24. * 365. * sim_years)  # convert years to second to maintain SI units
    t = np.linspace(t_min, t_max, int((t_max - t_min) / dt))

    # get the array of initial conditions from the auxiliary function
    initial_state = get_initial_conditions()
    x0 = initial_state[0:6]
    v0 = initial_state[6:12]

    # return odeint(get_ems_state, initial_state, t)
    # return rk4(get_ems_state, initial_state, t)
    return pefrl(get_ems_state_symp, x0, v0, t)


def get_em_motion(sim_years=1.0, dt=1e5):
    """
    Solves the motion of the Earth-Moon-Sun system using the model in get_ems_state on the time values t
    :param sim_years: number of years over which to run the simulation
    :param dt: time step, in seconds
    :return: position and velocity of the earth and moon over the time t
    """
    t_min = 0  # start time
    t_max = (3600. * 24. * 365. * sim_years)  # convert years to second to maintain SI units
    t = np.linspace(t_min, t_max, int((t_max - t_min) / dt))

    # get the array of initial conditions from the auxiliary function
    initial_state = get_initial_conditions()
    x0 = initial_state[0:6]
    v0 = initial_state[6:12]

    ems_positions = pefrl(get_ems_state_symp, x0, v0, t)[0]

    # get the moon's position relative to the earth through the simulation
    moon_positions = np.empty([ems_positions.shape[0], 3])  # 2-d np array (i.e. matrix) holding moon's positions relative to earth
    for i in range(ems_positions.shape[0]):
        moon_positions[i] = ems_positions[i][3:6] - ems_positions[i][0:3]  # moon xyz - earth xyz to get moon relative to earth
        i += 1
    return moon_positions

# -----------------------------------------------------------------------------
# END EMS DE EQUATION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START GRAPHING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_ems_orbits():
    """ Plots the orbit of the earth and moon around the sun in 3 dimensions """
    sim_years = 1.0
    ems_orbits = get_ems_motion(sim_years=sim_years)[0]
    indeces = np.arange(0, np.shape(ems_orbits)[0], 1)  # for use with color map

    ax = plt.axes(projection="3d")
    xlim, ylim, zlim = 1.5e11, 1.5e11, 1e8
    ax.set_xlim3d([-xlim, xlim])
    ax.set_ylim3d([-ylim, ylim])
    ax.set_zlim3d([-zlim, zlim])
    ax.view_init(18, -60)  # set view angle

    # time_label = ax.text(-0.8*xlim, ylim, 0.7*zlim, 'Hi!',bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    ax.scatter(0, 0, 0, c='orange', s=400, label="Sun")  # the sun
    ax.scatter(ems_orbits[:, 0], ems_orbits[:, 1], ems_orbits[:, 2], c=indeces, cmap=blue_cmap, s=30, label="earth")
    ax.scatter(ems_orbits[:, 3], ems_orbits[:, 4], ems_orbits[:, 5], c=indeces, cmap=gray_cmap, s=10, alpha=0.75, label="moon")

    # plt.legend(framealpha=0.9)
    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "ems-orbits_.png", dpi=200)
    plt.show()


def plot_em_orbits():
    """ Plots the orbit of the moon around the earth in 3 dimensions """

    sim_years = 16.0
    moon_positions = get_em_motion(sim_years=sim_years)
    indeces = np.arange(0, np.shape(moon_positions)[0], 1)  # for use with color map

    ax = plt.axes(projection="3d")
    xlim, ylim, zlim = 4e8, 4e8, 4e7
    ax.set_xlim3d([-xlim, xlim])
    ax.set_ylim3d([-ylim, ylim])
    ax.set_zlim3d([-zlim, zlim])
    ax.view_init(18, -60)  # set view angle

    ax.scatter(0, 0, 0, c=blue_dark, s=400)  # big blue earth
    ax.text(-1.1*xlim, ylim, 0.88*zlim, 'Years: 18.6', bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'), zorder=10)
    ax.scatter(moon_positions[:, 0], moon_positions[:, 1], moon_positions[:, 2], c=indeces, cmap=gray_cmap, alpha=0.85)

    plt.tight_layout()
    if save_figures: plt.savefig("em-orbit-{:.2f}_.png".format(sim_years))
    plt.show()


def plot_nodal_precession():
    """ Plots the orbit of the moon around the earth in 3 dimensions """

    fig = plt.figure(figsize=(7, 6))
    for i, sim_years in enumerate((1.0, 6.0, 12.0, 18.6)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        moon_positions = get_em_motion(sim_years=sim_years)

        xlim, ylim, zlim = 4e8, 4e8, 4e7
        ax.set_xlim3d([-xlim, xlim])
        ax.set_ylim3d([-ylim, ylim])
        ax.set_zlim3d([-zlim, zlim])
        ax.view_init(18, -60)  # set view angle

        start, stop = -100, -1
        indeces = np.arange(0, stop-start, 1)

        ax.text(-1.1*xlim, ylim, 0.88*zlim, 'Years: {:.1f}'.format(sim_years), bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'), zorder=10)
        ax.scatter(moon_positions[start:stop, 0], moon_positions[start:stop, 1], moon_positions[start:stop, 2], c=indeces, cmap=gray_cmap, alpha=0.85)
        ax.scatter(0, 0, 0, c=blue_dark, s=300, zorder=10)  # big blue earth

    plt.tight_layout()
    if save_figures: plt.savefig(figure_dir + "em-precesion_.png", dpi=200)
    plt.show()
# -----------------------------------------------------------------------------
# END GRAPHING FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------
def animate_ems_scatters(i, ems, earth_scatter, moon_scatter, time_label, time_label_template, N_points, sim_years):
    """
    Update the data held by the scatter plot and therefore animates it.
    """
    # earth_scatter._offsets3d = (ems[i, 0:1], ems[i, 1:2], ems[i, 2:3])
    # moon_scatter._offsets3d = (ems[i, 3:4], ems[i, 4:5], ems[i, 5:6])

    print("{} of {}".format(i, N_points))
    offset = min(i, N_points)
    earth_scatter._offsets3d = (ems[i-offset:i, 0], ems[i-offset:i, 1], ems[i-offset:i, 2])
    moon_scatter._offsets3d = (ems[i-offset:i, 3], ems[i-offset:i, 4], ems[i-offset:i, 5])
    time_label.set_text(time_label_template.format(365*i*sim_years/N_points))

    return earth_scatter, moon_scatter


def create_ems_animation():
    """ Creates a simulation of the earth and moon orbiting the sun """

    sim_years = 1.003
    ems = get_ems_motion(sim_years=sim_years)[0]
    iterations = np.shape(ems)[0]

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    xlim, ylim, zlim = 1.5e11, 1.5e11, 1e8
    ax.set_xlim3d([-xlim, xlim])
    ax.set_ylim3d([-ylim, ylim])
    ax.set_zlim3d([-zlim, zlim])
    ax.view_init(18, -60)  # set view angle

    ax.scatter(0, 0, 0, c='orange', s=400)  # big orange sun

    time_label_template = "Days: {:.0f}"
    time_label = ax.text(-0.8*xlim, ylim, 0.7*zlim, '',bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    earth_scatter = ax.scatter(ems[0, 0:1], ems[0, 1:2], ems[0, 2:3], color=blue_dark, s=30)
    moon_scatter = ax.scatter(ems[0, 3:4], ems[0, 3:5], ems[0, 5:6], c="#999999", s=15)
    ems_animation = animation.FuncAnimation(fig, animate_ems_scatters, iterations,
                                            fargs=(ems, earth_scatter, moon_scatter, time_label, time_label_template, iterations, sim_years),
                                            interval=50, blit=True, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    ems_animation.save(figure_dir + 'ems-animated.mp4', writer=writer)


def animate_em_scatters(i, moon_positions, moon_scatter, time_label, time_label_template, N_points, sim_years):
    """
    Update the data held by the scatter plot and therefore animates it.
    """
    print("{} of {}".format(i, N_points))
    offset = min(i, 100)
    moon_scatter._offsets3d = (moon_positions[i - offset:i, 0], moon_positions[i - offset:i, 1], moon_positions[i - offset:i, 2])
    time_label.set_text(time_label_template.format(i*sim_years/N_points))

    return moon_scatter


def create_em_animation():
    """ Creates a simulation of the moon orbitting the earth """

    sim_years = 18.6
    moon_positions = get_em_motion(sim_years=sim_years)
    iterations = np.shape(moon_positions)[0]

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    xlim, ylim, zlim = 4e8, 4e8, 4e7
    ax.set_xlim3d([-xlim, xlim])
    ax.set_ylim3d([-ylim, ylim])
    ax.set_zlim3d([-zlim, zlim])
    ax.view_init(18, -60)  # set view angle

    ax.scatter(0, 0, 0, c=blue_dark, s=400)  # big blue earth
    time_label_template = "Years: {:.1f}"
    time_label = ax.text(-1.1*xlim, ylim, 0.88*zlim, '', bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'), zorder=10)

    moon_scatter = ax.scatter(moon_positions[0, 0:1], moon_positions[0, 1:2], moon_positions[0, 2:3], c="#999999", s=15)
    em_animation = animation.FuncAnimation(fig, animate_em_scatters, iterations,
                                           fargs=(moon_positions, moon_scatter, time_label, time_label_template, iterations, sim_years),
                                           interval=1, blit=False, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    em_animation.save(figure_dir + 'em-animated-{:.1f}_.mp4'.format(sim_years), writer=writer)
# -----------------------------------------------------------------------------
# END ANIMATION FUNCTIONS
# -----------------------------------------------------------------------------


def practice():
    # rows = 10
    # cols = 6
    # a = np.zeros((10, 6))
    # for r in range(rows):
    #     for c in range(cols):
    #         a[r][c] = c + 1
    # print(a)
    # print(a[0:4, 2:3])

    a = np.arange(1, 10, 1)
    print(a)
    print(a[-4:-1])


# practice()
# create_ems_animation()
# plot_ems_orbits()
# plot_em_orbits()
# plot_nodal_precession()
create_ems_animation()
# create_em_animation()

