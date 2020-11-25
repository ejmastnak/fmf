from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib
import matplotlib.animation as animation

from diffeq_2 import *

#-----------------------------------------------------------------------------
# radial gravitational force in 2D
def forceg(r):
    G=1
    M1=1
    M2=1
    x=r[0]
    y=r[1]
    d2=x*x+y*y
    d=np.sqrt(d2)
    f=-G*M1/M2/d2
    fx=x*f/d
    fy=y*f/d
    return np.array([fx,fy])

#-----------------------------------------------------------------------------
# a simple rock y''= F(y)
def rock(state,t):

    dydt=np.zeros_like(state)
    dydt[0]=state[2]
    dydt[1]=state[3]
    f=forceg([state[0],state[1]])
    dydt[2]=f[0]
    dydt[3]=f[1]

    return dydt

#-----------------------------------------------------------------------------
# a simple satellite

if __name__ == "__main__":

    #import diffeq
    from pylab import *

    # create a time array from 0..100 sampled at 0.1 second steps
    dt =  0.05
    t = np.arange(0.0, 100.0, dt)

    #initial conditions
    x0=1.
    y0=0.
    vx0=0.
    vy0=1.

    iconds=np.array([x0,y0,vx0,vy0])

    x_euler=euler(rock,iconds,t)
    xc2=x_euler[:,0]
    yc2=x_euler[:,1]
    res=pefrl(forceg,[x0,y0],[vx0,vy0],t)
    x_pefrl=res[0,:]
    xc=x_pefrl[:,0]
    yc=x_pefrl[:,1]



    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()
 
    line, = ax.plot([], [], 'og', lw=2) # o-g if connecting line
    line2, = ax.plot([], [], 'or', lw=2)
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
    ax.plot(0,0,'ob',lw=2)
    ax.legend(("PEFRL","Euler"),loc='lower right')
    plt.show()

