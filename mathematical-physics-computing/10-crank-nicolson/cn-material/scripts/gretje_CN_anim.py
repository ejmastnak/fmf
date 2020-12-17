#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Jonathan Senning <jonathan.senning@gordon.edu>
# Gordon College
# April 22, 1999
# Converted to Python November 2008
#
# $Id$
#
# Use Crank-Nicolson scheme to solve the heat equation in a thin rod.
# The equation solved is
#
#       du     d  du
#       -- = K -- --
#       dt     dx dx
#
# along with boundary conditions
#
#       u(xmin,t) = a(t)
#       u(xmax,t) = b(t)
#
# and initial conditions
#
#       u(x,tmin) = f(x)
#
#-----------------------------------------------------------------------------

import matplotlib

from pylab import *
import time
import matplotlib.animation as animation


# Determine the desired plot types.  A value of 1 means show the plot and
# a value of 0 means don't show the plot.

individual_curves = 1          # Show curves during computation
contour_plot = 1                # Show contour plot of solution

# Set the value of the thermal diffusivity.

K = 0.125

# Set size of image.  Also, these values are combined with interval sizes to
# compute h (spatial stepsize) and k (temporal stepsize).  Define array
# to hold all time-dependent data.

n = 640                  # Number of spatial intervals
m = 1280                  # Number of temporal intervals
u = zeros( ( n+1, m+1 ), float )

# X position of left and right endpoints

xmin, xmax = ( 0, 1 )

# Interval of time: tmin should probably be left at zero

tmin, tmax = ( 0, 2 )

# Generate x and t values.  These aren't really needed to solve the PDE but
# they are useful for computing boundary/initial conditions and graphing.

x = linspace( xmin, xmax, n+1 )
t = linspace( tmin, tmax, m+1 )

# Initial condition f(x)

u[:,0] = zeros(n+1, float)
for i in range(n+1):
    if (x[i]>=0.2 and x[i]<=0.4):
        u[i,0]=50
# print(u[:,0])

# Boundary conditions: left a(t) and right b(t)

u[0,:] = zeros( m+1, float )                    # Left
#u[n,:] = 60 * ( ( 1 - cos( pi * t ) ) / 2.0 )   # Right
u[n,:] = zeros( m+1, float )                    # Right


# We are using a Crank-Nicolson scheme, and can vary the weighting of the
# u(x,t+k) values with the u(x,t) values using the parameter c (called gamma
# in my notes).
#
# If c == 0 then the iteration reduces to the fully explicit marching scheme
# which is not stable unless k <= (h^2)/2.
#
# If c == 1 then the iteration is fully implicit and is stable for any choice
# of k.
#
# The method should be unconditionally stable if c >= 0.5.  However, when
# c == 0.5 the solution can have undesirable oscillation.

c = 1

#-----------------------------------------------------------------------------
#       Should not need to make changes below this point :)
#-----------------------------------------------------------------------------

# Compute step sizes.  Since we also need the square of the spatial step
# size, we go ahead and compute it now as well.

h = ( xmax - xmin ) / float( n )
k = ( tmax - tmin ) / float( m )
h2 = h * h

# Find likely extremes for u

umin, umax = ( u.min(), u.max() )

# Now we are basically done with the setup.  The discritation used leads to
# a tridiagonal linear system.  Here we construct the diagonals of tridiagonal
# matrix.

D = ( 2 * k * K * c + h2 ) * ones( n - 1, float )
A = -k * K * c * ones( n-2, float )
C = -k * K * c * ones( n-2, float )

AB=np.array([pad(A,(1,0),'constant'),D,pad(C,(0,1),'constant')])

#tridiagonal.factor( A, D, C )

# Plot initial condition curve.  The "sleep()" is used to allow time for the
# plot to appear on the screen before actually starting to solve the problem
# for t > 0.

ion()

if individual_curves != 0:
    plot( x, u[:,0], '-' )
    axis( [xmin, xmax, umin, 1.2*umax] )
    xlabel( 'x' )
    ylabel( 'Temperature' )
    title( 'step = %3d; t = %f' % ( 0, 0.0 ) )
    input("Continue...")


# Main loop.  This consists of computing the appropriate right-hand-side
# vector and then solving the linear system.  We also plot the solution at
# each time step.

from scipy.linalg import solve_banded

for j in range( m ):
    B = zeros( n - 1, float )
    B = k * K * ( 1 - c ) * ( u[0:-2,j] + u[2:,j] ) \
                - ( 2 * k * K * ( 1 - c ) - h2 ) * u[1:-1,j]
    B[0]  = B[0]  + k * K * c * u[0,j+1]
    B[-1] = B[-1] + k * K * c * u[n,j+1]

    uu=solve_banded((1,1),AB,B)
    u[1:-1,j+1]=uu
    #u[1:-1,j+1] = tridiagonal.solve( A, D, C, B )


figa = plt.figure()
ax = figa.add_subplot(111, autoscale_on=False, xlim=(xmin,xmax), ylim=(umin,1.2*umax))
ax.grid()

temp, = ax.plot([], [], 'b-', lw=2)
time_template = 'scaled time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    temp.set_data([], [])
    time_text.set_text('')
    return temp, time_text

def animate(j):
    thisx = x
    thisy = u[:,j]
    temp.set_data(thisx, thisy)
    time_text.set_text(time_template%(j*k))
    return temp, time_text

ani = animation.FuncAnimation(figa, animate, np.arange(1, m+1),
        interval=200, blit=False, init_func=init)

xlabel( 'x' )
ylabel( 't' )
title( 'Evolution of Temperature in a Thin Rod' )

plt.show()

input("Continue...")

# All done computing solution, now show the desired plots...


if contour_plot != 0:
    umin, umax = ( u.min(), u.max() )
    levels = linspace( umin, umax, 21 )
    cla()
    contour( x, t, u.transpose(), levels )
    xlabel( 'x' )
    ylabel( 't' )
    title( 'Evolution of Temperature in a Thin Rod' )
    draw()
    input( "Press Enter to continue... " )

# End of file
