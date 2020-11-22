import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plot
# from mpl_toolkits.mplot3d import Axes3D
from numerical_methods_odes import *


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

save_figures = False
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

    aex = -G * ((m_S * state[0] / (vec_mag([state[0], state[1], state[2]]) ** 3)) -
                (m_M * r_x / (vec_mag([r_x, r_y, r_z]) ** 3)))
    aey = -G * ((m_S * state[1] / (vec_mag([state[0], state[1], state[2]]) ** 3)) -
                (m_M * r_y / (vec_mag([r_x, r_y, r_z]) ** 3)))
    aez = -G * ((m_S * state[2] / (vec_mag([state[0], state[1], state[2]]) ** 3)) -
                (m_M * r_z / (vec_mag([r_x, r_y, r_z]) ** 3)))
    amx = -G * ((m_S * state[3] / (vec_mag([state[3], state[4], state[5]]) ** 3)) +
                (m_E * r_x / (vec_mag([r_x, r_y, r_z]) ** 3)))
    amy = -G * ((m_S * state[4] / (vec_mag([state[3], state[4], state[5]]) ** 3)) +
                (m_E * r_y / (vec_mag([r_x, r_y, r_z]) ** 3)))
    amz = -G * ((m_S * state[5] / (vec_mag([state[3], state[4], state[5]]) ** 3)) +
                (m_E * r_z / (vec_mag([r_x, r_y, r_z]) ** 3)))

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

    aex = -G * ((m_S * coordinates[0] / (vec_mag([coordinates[0], coordinates[1], coordinates[2]]) ** 3)) -
                (m_M * r_x / (vec_mag([r_x, r_y, r_z]) ** 3)))
    aey = -G * ((m_S * coordinates[1] / (vec_mag([coordinates[0], coordinates[1], coordinates[2]]) ** 3)) -
                (m_M * r_y / (vec_mag([r_x, r_y, r_z]) ** 3)))
    aez = -G * ((m_S * coordinates[2] / (vec_mag([coordinates[0], coordinates[1], coordinates[2]]) ** 3)) -
                (m_M * r_z / (vec_mag([r_x, r_y, r_z]) ** 3)))
    amx = -G * ((m_S * coordinates[3] / (vec_mag([coordinates[3], coordinates[4], coordinates[5]]) ** 3)) +
                (m_E * r_x / (vec_mag([r_x, r_y, r_z]) ** 3)))
    amy = -G * ((m_S * coordinates[4] / (vec_mag([coordinates[3], coordinates[4], coordinates[5]]) ** 3)) +
                (m_E * r_y / (vec_mag([r_x, r_y, r_z]) ** 3)))
    amz = -G * ((m_S * coordinates[5] / (vec_mag([coordinates[3], coordinates[4], coordinates[5]]) ** 3)) +
                (m_E * r_z / (vec_mag([r_x, r_y, r_z]) ** 3)))

    accelerations = np.zeros(np.shape(coordinates))
    accelerations[0], accelerations[1], accelerations[2] = aex, aey, aez
    accelerations[3], accelerations[4], accelerations[5] = amx, amy, amz

    return accelerations


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


def solve_ems_motion(t):
    """
    Solves the motion of the Earth-Moon-Sun system using the model in get_ems_state on the time values t
    :param t: 1D array of time values on which to find the solution
    :return: position and velocity of the earth and moon over the time t
    """
    # get the array of initial conditions from the auxiliary function
    initial_state = get_initial_conditions()
    x0 = initial_state[0:6]
    v0 = initial_state[6:12]

    # # time parameters
    # sim_years = 10  # number of years over which to run the simulation; depracated
    # t_min = 0 # start time
    # t_max = (3600. * 24. * 365. * sim_years) # seconds hours days years
    # dt = 10000 # time step
    # time_values = np.linspace(t_min, t_max, int((t_max - t_min) / dt))

    # call odeint and get solutions
    # return odeint(get_ems_state, initial_state, t)
    return rk4(get_ems_state, initial_state, t)
    # return pefrl(get_ems_state_symp, x0, v0, t)[0]

# -----------------------------------------------------------------------------
# END EMS DE EQUATION FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START GRAPHING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_ems_orbits(ems_orbits, sim_years):
    """ Plots the orbit of the earth and moon around the sun in 3 dimensions """

    ax = plot.axes(projection="3d")
    ax.plot([], [], ' ', label="Simulation: {:.2f} years".format(sim_years))  # hacked blank label for simulation time
    ax.scatter(0, 0, 0, c='orange', s=350, label="Sun")  # the sun
    ax.plot3D(ems_orbits[:, 3], ems_orbits[:, 4], ems_orbits[:, 5], 'b', alpha=0.75, label="Moon orbiting sun")
    ax.plot3D(ems_orbits[:, 0], ems_orbits[:, 1], ems_orbits[:, 2], 'g', label="Earth orbiting sun")
    ax.set_zlim([-1e8, 1e8])
    plot.legend(framealpha=0.9)
    if save_figures: plot.savefig("ems-orbits-{:.2f}_.png".format(sim_years))
    plot.show()


def plot_em_orbits(ems_orbits, sim_years):
    """ Plots the orbit of the moon around the earth in 3 dimensions """

    ax = plot.axes(projection="3d")
    ax.plot([], [], ' ', label="Simulation: {:.2f} years".format(sim_years))  # hacked blank label for simulation time
    ax.scatter(0, 0, 0, c='g', s=100, label="Earth")  # the label for the earth
    ax.scatter(0, 0, 0, c='g', s=500)  # the earth
    ax.plot3D(np.subtract(ems_orbits[:, 3], ems_orbits[:, 0]),
              np.subtract(ems_orbits[:, 4], ems_orbits[:, 1]),
              np.subtract(ems_orbits[:, 5], ems_orbits[:, 2]),
              'b', alpha=0.85, label="Moon orbiting Earth")
    ax.set_zlim([-5e7, 5e7])
    plot.legend(framealpha=0.9)
    if save_figures: plot.savefig("moon-earth-orbit-{:.2f}_.png".format(sim_years))
    plot.show()


def extract_apsidal_precession(moon_positions, sim_years):
    # calculate earth-moon distance throughout the simulation
    distances = np.empty(moon_positions.shape[0]) # empty array to hold distances
    i = 0
    while i < moon_positions.shape[0]:
        distances[i] = vec_mag(moon_positions[i])
        i += 1

    # Plots distances to moon during the orbit
    plot_2d(np.linspace(0, sim_years, distances.shape[0]), distances, "Earth-Moon distance", marker="-",
            xlabel="Time [years]", ylabel="Distance [m]", save_figures=save_figures,
            extra_label="Simulation: %.2f years" % sim_years)

    # TODO clean this part up. Consider changing all while loops

    # create an empty array to hold perigee and apogee positions
    perigee_positions = np.empty([0, 3])
    apogee_positions = np.empty([0, 3])
    cut_off_distance = 0.035e8 # hacky auxiliary variable

    before_last_position = moon_positions[0]  # temporary variable
    last_position = moon_positions[1] # temporary variable

    # loop through the moon's positions find the perigees and apogees,
    # store the perigees and apogees in arrays
    i = 2
    while i < moon_positions.shape[0]:

        # condition for a perigee position (have to do checking for little double mins and maxes)
        if (vec_mag(before_last_position) > vec_mag(last_position)) and (vec_mag(last_position) < vec_mag(moon_positions[i])):
            # Essentially if the very first extrema
            if apogee_positions.shape[0] == 0:
                perigee_positions = np.append(perigee_positions, [last_position], axis=0)

            # To avoid double maxes (if abs(last_max - current_min) > cut_off_distance)
            elif np.abs(vec_mag(apogee_positions[apogee_positions.shape[0] - 1]) - vec_mag(last_position)) > cut_off_distance:
                perigee_positions = np.append(perigee_positions, [last_position], axis=0)

        # conditions for an apogee position (have to do checking for little double mins and maxes)
        elif (vec_mag(before_last_position) < vec_mag(last_position)) and (vec_mag(last_position) > vec_mag(moon_positions[i])):

            # Essentially if the very first extrema
            if perigee_positions.shape[0] == 0:
                apogee_positions = np.append(apogee_positions, [last_position], axis=0)

            # To avoid double maxes (if abs(current_max - last_min) > cut_off_distance)
            elif np.abs(vec_mag(last_position) - vec_mag(perigee_positions[perigee_positions.shape[0] - 1])) > cut_off_distance:
                apogee_positions = np.append(apogee_positions, [last_position], axis=0)

        # update before-last and last positions
        before_last_position = last_position
        last_position = moon_positions[i]

        i += 1

    # Plot perigee and apogee positions
    fig = plot.figure()
    ax = plot.axes(projection="3d")
    ax.plot([], [], ' ', label="Simulation: %.2f years" % sim_years)  # blank label for simulation time
    ax.scatter(perigee_positions[:, 0], perigee_positions[:, 1], perigee_positions[:, 2], label="Perigee Positions")
    ax.scatter(apogee_positions[:, 0], apogee_positions[:, 1], apogee_positions[:, 2], label="Apogee Positions")
    ax.scatter(0, 0, 0, c='g', s=250, label="Earth")  # the earth
    ax.set_zlim([-1e8, 1e8])  # adjust z axis scale
    plot.legend()
    if save_figures: plot.savefig("Aps_Preces_Visual_%.f.png" % sim_years)
    plot.show()

    # Declare an empty array to hold perigee and apogee angles
    perigee_angles = np.empty(perigee_positions.shape[0]-1)
    apogee_angles = np.empty(apogee_positions.shape[0]-1)

    # Calculate perigee angles
    for i in range(0, perigee_angles.shape[0]): # loop through the empty angles array
        perigee_angles[i] = angle_btwn(perigee_positions[0], perigee_positions[i + 1])

    # TODO for loop example here
    # Calculate apogee angles
    for i in range(0, apogee_angles.shape[0]):  # loop through the empty angles array
        apogee_angles[i] = angle_btwn(apogee_positions[0], apogee_positions[i + 1])

    # plot angle between sucessive perigees over time
    plot_2d(np.linspace(0, sim_years, perigee_angles.shape[0]), perigee_angles, "Angular displacement of perigee", xlabel="Time [years]", ylabel="Angle [degrees]", marker="--", save_figures=save_figures, extra_label="Simulation: %.2f years" % sim_years)

    # fit a sine curve to the perigee angle over time
    fit_sin_curve(np.linspace(0, sim_years, perigee_angles.shape[0]), perigee_angles, a_guess=90, w_guess=2*np.pi/8.85, k_guess=90, data_label="Angular displacement of perigee", save_figures=save_figures)
    fit_sin_curve(np.linspace(0, sim_years, apogee_angles.shape[0]), apogee_angles, a_guess=90, w_guess=2*np.pi/8.85, k_guess=90, data_label="Angular displacement of apogee", save_figures=save_figures)


def extract_nodal_precession(moon_positions, sim_years):
    asc_node_positions = np.empty([0, 3]) # array to hold ascending node positions
    desc_node_positions = np.empty([0, 3]) # array to hold desending node positions
    last_position = moon_positions[0]  # tempory variable

    # loop through the moon's positions, find asc and desc nodes, store them in arrays
    i = 1
    while i < moon_positions.shape[0]:

        # condition for an ascending node position
        if (last_position[2] < 0.0) and (moon_positions[i][2] >= 0.0):
            asc_node_positions = np.append(asc_node_positions, [last_position], axis=0)

        # condition for an descending node position
        if (last_position[2] > 0.0) and (moon_positions[i][2] <= 0.0):
            desc_node_positions = np.append(desc_node_positions, [last_position], axis=0)

        last_position = moon_positions[i] # update last positions

        i += 1

    # plots positions of ascending and descending nodes
    fig = plot.figure()
    ax = plot.axes(projection="3d")
    ax.plot([], [], ' ', label="Simulation: %.2f years" % sim_years)  # blank label for simulation time

    ax.scatter(asc_node_positions[:, 0], asc_node_positions[:, 1], asc_node_positions[:, 2],
               label="Ascending Node Positions")
    ax.scatter(desc_node_positions[:, 0], desc_node_positions[:, 1], desc_node_positions[:, 2],
               label="Descending Node Positions")
    ax.scatter(0, 0, 0, c='g', s=250, label="Earth")  # the earth
    ax.set_zlim([-1e8, 1e8])
    plot.legend()
    if save_figures: plot.savefig("Nodal_Preces_Visual_%.f.png" % sim_years)
    plot.show()

    # Declare an empty array to hold ascending and descending node angles
    asc_node_angles = np.empty(asc_node_positions.shape[0] - 1)
    desc_node_angles = np.empty(desc_node_positions.shape[0] - 1)

    # Calculate ascending node angles
    for i in range(0, asc_node_angles.shape[0]):  # loop through the empty angles array
        asc_node_angles[i] = angle_btwn(asc_node_positions[0], asc_node_positions[i + 1])

    # Calculate descending node angles
    for i in range(0, desc_node_angles.shape[0]):  # loop through the empty angles array
        desc_node_angles[i] = angle_btwn(desc_node_positions[0], desc_node_positions[i + 1])

    """
    # plot angle between sucessive ascending angles over time
    plot_2d(np.linspace(0, sim_years, asc_node_angles.shape[0]), asc_node_angles,
            "Angular displacement of ascending node", xlabel="Time [years]", ylabel="Angle [degrees]",
            save_figures=save_figures, extra_label="Simulation: %.2f years" % sim_years)
    """

    # fit a sine curve to the ascending and descending angles over time
    fit_sin_curve(np.linspace(0, sim_years, asc_node_angles.shape[0]), asc_node_angles, a_guess=90, w_guess=2 * np.pi / 18.6,
                  k_guess=90, data_label="Angular displacement of ascending node")
    fit_sin_curve(np.linspace(0, sim_years, desc_node_angles.shape[0]), desc_node_angles, a_guess=90, w_guess=2 * np.pi / 18.6,
                  k_guess=90, data_label="Angular displacement of descending node")


def plot_2d(x_values, y_values, label, xlabel="", ylabel="", marker="--", save_figures=False, extra_label=None):
    """
    A simple function to plot a 2-D array
    """
    if extra_label is not None:
        plot.plot([], [], ' ', label=extra_label)

    plot.plot(x_values, y_values, marker, color="b", label=label)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)

    plot.legend(framealpha=0.85)
    if save_figures: plot.savefig(label + ".png")
    plot.show()
    plot.close()


def vec_mag(r):
    """
    Returns the length (magnitude) of a 3D vector
    :param r: A 3D vector e.g. r = [r1, r2, r3], generally position r = [x, y, z]
    :return:
    """
    return np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)


def dot(r1, r2):
    """
    Returns the dot product of two 3D vectors represented by length 3 arrays of the form [x, y, z]
    """
    return r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]


def angle_btwn(r1, r2):
    """
    Returns the angle in degrees between two 3D vectors
    """
    return np.arccos(dot(r1, r2) / (vec_mag(r1) * vec_mag(r2))) * 180.0 / np.pi


def sin_model(t, a, w, k):
    return a * np.sin(w * t) + k


def fit_sin_curve(x_values, y_values, a_guess=1., w_guess=1., k_guess=1., data_label="", save_figures=False):
    """
    Fits a sin curve to the x-y data points x_values and y_values
    using the parameters a (amplitude) w (angular frequency) and k (vertical shift) and plots
    the fitted curve
    """
    p0 = (a_guess, w_guess, k_guess) # guesses for coefficients a and k
    opt, p_cov = curve_fit(sin_model, x_values, y_values, p0)

    a, w, k = opt

    x2 = np.linspace(x_values[0], x_values[x_values.shape[0]-1], 100)
    y2 = sin_model(x2, a, w, k)

    plot.plot(x_values, y_values, 'o', c="orange", alpha=0.5, label=data_label)
    plot.plot(x2, y2, c="blue", linewidth=2.0, label='Fit: f(x) = %.f sin(%.3f $t$) + %.f' % (a, w, k))
    plot.plot([], [], ' ', label="Precession period: %.2f years" % (2 * np.pi / w))
    plot.xlabel("Time [years]")
    plot.ylabel("Angle [degrees]")
    plot.legend(framealpha=1.0)
    if save_figures: plot.savefig(data_label + ".png")
    plot.show()


def run_simulation():
    """ Wrapper function to run the simulation """

    sim_years = 1  # number of years over which to run the simulation; depracated
    t_min = 0  # start time
    t_max = (3600. * 24. * 365. * sim_years)  # seconds hours days years
    dt = 10000  # time step
    time_values = np.linspace(t_min, t_max, int((t_max - t_min) / dt))

    ems_system_solution = solve_ems_motion(time_values)

    # call function to graph the orbits of the earth moon
    plot_ems_orbits(ems_system_solution, sim_years)

    # # get the moon's position relative to the earth through the simulation
    # moon_positions = np.empty([np.shape(ems_system_solution)[0], 3])  # 2-d np array (i.e. matrix) holding moon's positions relative to earth
    # i = 0
    # while i < np.shape(ems_system_solution)[0]:
    #     moon_positions[i] = ems_system_solution[i][3:6] - ems_system_solution[i][0:3]  # moon xyz - earth xyz to get moon relative to earth
    #     i += 1
    #
    # # call functions to extract apsidal precession
    # extract_apsidal_precession(moon_positions, sim_years)
    #
    # # call functions to extract nodal precession
    # extract_nodal_precession(moon_positions, sim_years)


run_simulation()
