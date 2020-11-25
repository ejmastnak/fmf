from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib
import matplotlib.animation as animation
from diffeq_2 import *

#-----------------------------------------------------------------------------
# global variable as frequency

omega2=0.1

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def forcel(y):
    return -omega2*y

#-----------------------------------------------------------------------------
# sine force (e.g. math. pendulum)
def forcesin(y):
    return -omega2*sin(y)


#-----------------------------------------------------------------------------
# a simple pendulum

if __name__ == "__main__":

    #import diffeq
    from pylab import *

    # create a time array from 0..100 sampled at 0.1 second steps
    dt =  0.2
    t = np.arange(0.0, 100, dt)

    #initial conditions
    x0=1.
    v0=0.

    L=1.2
    x_pefrl=pefrl(forcel,x0,v0,t)
    xc = L*sin(x_pefrl[0,:])
    yc = -L*cos(x_pefrl[0,:])

    L=1.2
    xs_pefrl=pefrl(forcesin,x0,v0,t)
    xc2 = L*sin(xs_pefrl[0,:])
    yc2 = -L*cos(xs_pefrl[0,:])

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()

    line, = ax.plot([], [], 'o-g', lw=2)
    line2, = ax.plot([], [], 'o-r', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        return line,line2, time_text

    def animate(i):
        thisx = [0, xc[i]]
        thisy = [0, yc[i]]
        line.set_data(thisx, thisy)
        thisx2 = [0, xc2[i]]
        thisy2 = [0, yc2[i]]
        line2.set_data(thisx2, thisy2)

        time_text.set_text(time_template%(i*dt))
        return line,line2, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
        interval=25, blit=False, init_func=init)

    #ani.save('double_pendulum.mp4', fps=15, clear_temp=True)
    ax.legend(("linearno","matematično"),loc='lower right')
    plt.show()

