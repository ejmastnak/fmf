import numpy as np

"""
Collection of differential equations of motion to model the various pendulums and oscillators
 encountered in the Newton report.
 
There are three similar classes of equations:
 f(state, t) used with the numerical_methods_odes.py funcitons
 f(t, state) used with scipy's new solve_ivp API
 f(coordinates) used with symplectic methods
"""


# -----------------------------------------------------------------------------
# START (STATE, TIME) DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------
def get_linear_pendulum_state(state, t):
    """
    Dimensionless differential equation of motion for a linear pendulum
    Time t is included for compatability with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = (x, v)
    :param t: time; not used, but needed to work with numerical ODE functions
    :return: 2-element array holding velocity and acceleration i.e. [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = -state[0]
    return return_state  # i.e. v, -x


def get_simple_pendulum_state(state, t):
    """
    Dimensionless differential equation of motion for a simple (mathematical) pendulum
    Time t is included for compatability with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time; not used, but needed to work with numerical ODE functions
    :return: 2-element array holding velocity and acceleration i.e. [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = -np.sin(state[0])
    return return_state


def get_simple_pendulum_damped_driven_state(state, t, b=0.5, w_d=(2/3), a_d=1.0):
    """
    Dimensionless differential equation of motion for a damped and driven simple pendulum
     written to work with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time
    :param b: damping coefficient
    :param w_d: driving (angular) frequency
    :param a_d: driving amplitude
    :return: 2-element array holding velocity and acceleration i.e. return [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = a_d*np.cos(w_d*t) - b*state[1] - np.sin(state[0])  # a_d*cos(w_d*t) - b*v - sin(x)
    return return_state


def get_van_der_pol_state(state, t, mu=1.0):
    """
    Dimensionless differential equation of motion for a van der Pol oscillator
     written to work with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time; not used, but needed to work with numerical ODE functions
    :param mu: damping coefficient
    :return: 2-element array holding velocity and acceleration i.e. return [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = mu*state[1]*(1-state[0]**2) - state[0]  # mu*v*(1-x^2) - x
    return return_state


def get_van_der_pol_driven_state(state, t, mu=1.0, w_d=1.0, a_d=1.0):
    """
    Dimensionless differential equation of motion for a driven van der Pol oscillator
     written to work with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time
    :param mu: damping coefficient
    :param w_d: driving (angular) frequency
    :param a_d: driving amplitude
    :return: 2-element array holding velocity and acceleration i.e. return [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = a_d*np.cos(w_d*t) + mu*state[1]*(1-state[0]**2) - state[0]  # a_d*cos(w_d*t) + mu*v*(1-x^2) - x
    return return_state
# -----------------------------------------------------------------------------
# END (STATE, TIME) DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START (TIME, STATE) DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------
def get_linear_pendulum_state_tfirst(t, state):
    """
    Dimensionless differential equation of motion for a linear pendulum
    Time t is included for compatability with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = (x, v)
    :param t: time; not used, but needed to work with numerical ODE functions
    :return: 2-element array holding velocity and acceleration i.e. [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = -state[0]
    return return_state  # i.e. v, -x


def get_simple_pendulum_state_tfirst(t, state):
    """
    Dimensionless differential equation of motion for a simple (mathematical) pendulum
    Time t is included for compatability with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time; not used, but needed to work with numerical ODE functions
    :return: 2-element array holding velocity and acceleration i.e. [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = -np.sin(state[0])
    return return_state


def get_simple_pendulum_damped_driven_state_tfirst(t, state, b=0.5, w_d=(2/3), a_d=1.0):
    """
    Dimensionless differential equation of motion for a damped and driven simple pendulum
     written to work with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time
    :param b: damping coefficient
    :param w_d: driving (angular) frequency
    :param a_d: driving amplitude
    :return: 2-element array holding velocity and acceleration i.e. return [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = a_d*np.cos(w_d*t) - b*state[1] - np.sin(state[0])  # a_d*cos(w_d*t) - b*v - sin(x)
    return return_state


def get_van_der_pol_state_tfirst(t, state, mu=1.0):
    """
    Dimensionless differential equation of motion for a van der Pol oscillator
     written to work with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time; not used, but needed to work with numerical ODE functions
    :param mu: damping coefficient
    :return: 2-element array holding velocity and acceleration i.e. return [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = mu*state[1]*(1-state[0]**2) - state[0]  # mu*v*(1-x^2) - x
    return return_state


def get_van_der_pol_driven_state_tfirst(t, state, mu=1.0, w_d=1.0, a_d=1.0):
    """
    Dimensionless differential equation of motion for a driven van der Pol oscillator
     written to work with numerical methods for ODEs

    :param state: 2-element array holding position and velocity i.e. state = [x, v]
    :param t: time
    :param mu: damping coefficient
    :param w_d: driving (angular) frequency
    :param a_d: driving amplitude
    :return: 2-element array holding velocity and acceleration i.e. return [v, a]
    """
    return_state = np.zeros(np.shape(state))
    return_state[0] = state[1]
    return_state[1] = a_d*np.cos(w_d*t) + mu*state[1]*(1-state[0]**2) - state[0]  # a_d*cos(w_d*t) + mu*v*(1-x^2) - x
    return return_state
# -----------------------------------------------------------------------------
# END (TIME, STATE) DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# START SYMPLECTIC DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------
def get_linear_pendulum_state_symp(x):
    """
    Dimensionless differential equation of motion for a linear pendulum
    Written (without time dependence) for compatibility with symplectic integrators

    :param x: the pendulum's angular displacement from equilibrium
    :return: the pendulum's angular acceleration
    """
    return -x


def get_simple_pendulum_state_symp(x):
    """
    Dimensionless differential equation of motion for a simple (mathematical) pendulum
    Written (without time dependence) for compatibility with symplectic integrators

    :param x: the pendulum's angular displacement from equilibrium
    :return: the pendulum's angular acceleration
    """
    return -np.sin(x)
# -----------------------------------------------------------------------------
# END SYMPLECTIC DIFFERENTIAL EQUATION AND PENDULUM FUNCTIONS
# -----------------------------------------------------------------------------
