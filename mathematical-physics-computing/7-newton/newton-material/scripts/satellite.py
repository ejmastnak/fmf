from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib
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
# a simple rock y''= F(y) -> sistem 1. reda (x,y,vx,vy)
def rock(state,t):

    dydt=np.zeros_like(state)
    dydt[0]=state[2] # x' = v_x
    dydt[1]=state[3] # y' = v_y 
    f=forceg([state[0],state[1]])
    dydt[2]=f[0]  # v_x' = F_x
    dydt[3]=f[1]  # v_y' = F_y

    return dydt

#-----------------------------------------------------------------------------
# a simple satellite

if __name__ == "__main__":

    #import diffeq
    from pylab import *

    ion()
    # create a time array from 0..100 sampled at 0.1 second steps
    dt =  0.2
    t = np.arange(0.0, 10.0, dt)

    #initial conditions
    x0=1.
    y0=0.
    vx0=0.
    vy0=1.

    iconds=np.array([x0,y0,vx0,vy0])

    x_scipy=integrate.odeint(rock,iconds,t)

    x_euler=euler(rock,iconds,t)
    x_er_eu=x_euler[:,0]-x_scipy[:,0]
    y_er_eu=x_euler[:,1]-x_scipy[:,1]

    x_rk4=rku4(rock,iconds,t)
    x_er_rk4=x_rk4[:,0]-x_scipy[:,0]
    y_er_rk4=x_rk4[:,1]-x_scipy[:,1]

    res=verlet(forceg,[x0,y0],[vx0,vy0],t)
    x_verlet=res[0,:]
    x_er_ver=x_verlet[:,0]-x_scipy[:,0]
    y_er_ver=x_verlet[:,1]-x_scipy[:,1]

    res=pefrl(forceg,[x0,y0],[vx0,vy0],t)
    x_pefrl=res[0,:]
    x_er_pef=x_pefrl[:,0]-x_scipy[:,0]
    y_er_pef=x_pefrl[:,1]-x_scipy[:,1]

    plot(x_scipy[:,0],x_scipy[:,1],'r-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Scipy',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_euler[:,0],x_euler[:,1],'b-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Euler',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_verlet[:,0],x_verlet[:,1],'g-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Verlet',),loc='lower left')
    draw()


    input( "Press Enter to continue... " )
    cla()
    plot(x_euler[:,0],x_euler[:,1],'b-o',x_verlet[:,0],x_verlet[:,1],'g-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Euler','Verlet'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_scipy[:,0],x_scipy[:,1],'r-o',x_verlet[:,0],x_verlet[:,1],'g-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Scipy','Verlet'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_er_ver,y_er_ver,'g-o')
    title('Solutions-differences of satellite flight')
    legend(('Verlet-Scipy',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_scipy[:,0],x_scipy[:,1],'r-o',x_pefrl[:,0],x_pefrl[:,1],'b-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Scipy','PEFRL'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_er_pef,y_er_pef,'b-o')
    title('Solutions-differences of satellite flight')
    legend(('PEFRL-Scipy',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_er_ver,y_er_ver,'g-o',x_er_pef,y_er_pef,'b-o')
    title('Solutions-differences of satellite flight')
    legend(('Verlet-Scipy','PEFRL-Scipy',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_scipy[:,0],x_scipy[:,1],'r-o',x_euler[:,0],x_euler[:,1],'b-o',x_rk4[:,0],x_rk4[:,1],'g-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Scipy','Euler','RK4'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_er_eu,y_er_eu,'b-o',x_er_rk4,y_er_rk4,'g-o')
    title('Solutions-differences of satellite flight')
    legend(('Euler-Scipy','RK4-Scipy',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_scipy[:,0],x_scipy[:,1],'r-o',x_verlet[:,0],x_verlet[:,1],'b-o',x_rk4[:,0],x_rk4[:,1],'g-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Scipy','Verlet','RK4'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_er_ver,y_er_ver,'b-o',x_er_rk4,y_er_rk4,'g-o')
    title('Solutions-differences of satellite flight')
    legend(('Verlet-Scipy','RK4-Scipy',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_scipy[:,0],x_scipy[:,1],'r-o',x_pefrl[:,0],x_pefrl[:,1],'b-o',x_rk4[:,0],x_rk4[:,1],'g-o')
    axis([-1.8,1.8,-1.8,1.8])
    title('Solutions of satellite flight')
    legend(('Scipy','PEFRL','RK4'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(x_er_pef,y_er_pef,'b-o',x_er_rk4,y_er_rk4,'g-o')
    title('Solutions-differences of satellite flight')
    legend(('PEFRL-Scipy','RK4-Scipy',),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
