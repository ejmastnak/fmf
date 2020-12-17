import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import *
import matplotlib.animation as animation

'''Geometry and physics'''
a = 1.  # m
c = 1000. 
wavelength = 2 * a
omega = 2 * math.pi * c/wavelength


'''Spatial grid'''
dx = 0.001  # m
nx = int(a / dx) + 1
x = np.linspace(0, a, nx)

print ("x=",x)
'''Time grid'''
Cstab = 0.9  # koeficient stabilnosti, garantira, da je (dt*c/dx) = Cstab < 1 
r=Cstab*Cstab
dt = Cstab * dx / c
t_sim = 2 * math.pi / omega # = t_0 = c//wavelength  
nt = int(round(t_sim / dt)) + 1

print("c = {0}".format(c))
print("dx = {0}".format(dx))
print("nx = {0}".format(nx))
print("a = {0}".format(a))
print("dt = {0}".format(dt))
print("nt = {0}".format(nt))
print("t_sim = {0}".format(t_sim))

print("Cstab=dt*c/dx = {0}".format(Cstab))
print("omega = {0}".format(omega))

'''Define Results Parameter'''

#struna - odmik T
T = np.zeros((nx,nt),float)

'''Initial and boundary conditions'''


# lega
n1x = int(nx * 0.25)
n2x = 2*int(nx * 0.25)
y0 = 0.005  # vrh strune
x0 = a * 0.25  # v tem x-u doseže yx

kx = y0 / x0
for i in range(n1x + 1):
    T[i,0] = i * dx * kx

kx = - kx
for i in range(n1x, n1x+n2x+1):
    T[i,0] = y0 + (i * dx - x0) * kx

kx = - kx
for i in range(n1x+n2x,nx):
    T[i,0] = -y0 + (i * dx - 3*x0) * kx

# robni pogoji:

T[0,:]=0.
T[nx-1,:]=0.

# začetna hitrost = 0 . Potem dobimo:

T[1:-1,1]= T[1:-1,0] + 0.5*r*(T[2:,0]-2.*T[1:-1,0]+T[0:-2,0])

'''Simulation'''

for n in range(2, nt):
    T[1:-1,n]= (2*T[1:-1,n-1] - T[1:-1,n-2]) + r*(T[2:,n-1]-2.*T[1:-1,n-1]+T[0:-2,n-1])

ion()

plt.figure()
tx = 0
n = 0
xx = omega * tx / math.pi
plt.plot(x, T[:,0], label=f'n = {n:.0f}, t = {tx:.5f}, w*t/pi = {xx:.2f}')
plt.title('Struna pri nihanju - začetno stanje ob t=0')
plt.xlabel('x')
plt.ylabel('y')

input("Continue...")

plt.figure()
plt.plot(x, T[:,0], label=f'n = {n:.0f}, t = {tx:.5f}, w*t/pi = {xx:.2f}')
plt.plot(x, T[:,1], label=f'n = {n:.0f}, t = {tx:.5f}, w*t/pi = {xx:.2f}')
for n in range(2, nt):
    tx = n * dt
    xx = omega * tx / math.pi
    if (n % 100 == 0):  
        plt.plot(x, T[:,n], label=f'n = {n:.0f}, t = {tx:.5f}, w*t/pi = {xx:.2f}')
    # plt.xlabel('x')
    #	plt.ylabel('z')
    #	plt.legend()


input("Continue...")

figa = plt.figure()
ax = figa.add_subplot(111, autoscale_on=False, xlim=(0., 1.), ylim=(-1.2*y0,1.2*y0))
ax.grid()

wav, = ax.plot([], [], 'b-', lw=2)
time_template = 'scaled time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    wav.set_data([], [])
    time_text.set_text('')
    return wav, time_text

def animate(n):
    thisx = x
    thisy = T[:,n]
    wav.set_data(thisx, thisy)
    time_text.set_text(time_template%(n*omega*dt/pi))
    return wav, time_text

ani = animation.FuncAnimation(figa, animate, np.arange(1, nt),
        interval=5, blit=False, init_func=init)


plt.title('Struna pri nihanju - časovni razvoj')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

input("Continue...")


