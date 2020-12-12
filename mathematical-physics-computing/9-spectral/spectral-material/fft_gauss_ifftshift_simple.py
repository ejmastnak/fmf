from numpy import fft
import numpy as np
import matplotlib.pyplot as plt

N = 2**8 # Number of data points
L = 1. # interval
delta=L/N
print ("Number of sampling points:",N," on interval length:",L)
print ("Sampling interval:",delta)

print ("Sampling freq (or k):",1./delta)
nuc=0.5/delta
print ("Critical freq (or k):",nuc)
# asymmetric [0,L]
x=np.linspace(0,L,N,endpoint=False) # x coordinates
# symmetric [-L/2,L/2]
#x=np.linspace(-0.5*L,0.5*L,N,endpoint=False)
#print x

#signal functions
gauss = lambda x, a, sig, amp: amp*np.exp(-(x-a)**2/sig**2)
f = lambda x, k, amp: amp*np.sin(2*np.pi*x*k)

amp = 1 # amplitude
sigma = 0.02 # Gauss width
k1 = 50.0 # equivalent of frequency in distance units
D = 2 # diffusion constant

#FFT
#fx=f(x,k1,amp)
# asymmetric [0,L]
fx=gauss(x,0.5*L,sigma,amp)
# symmetric [-L/2,L/2]
#fx=gauss(x,0.*L,sigma,amp)

Fk = fft.fft(fft.ifftshift(fx))/N # Fourier coefficients (divided by n), correctly shifted (blue)
nu = fft.fftfreq(N,delta) # Natural frequencies
Fk = fft.fftshift(Fk) # Shift zero freq to center
nu = fft.fftshift(nu) # Shift zero freq to center
Fk_noshift = fft.fftshift(fft.fft(fx)/N) # Fourier coefficients (divided by n), no shift, oscillating (orange)
shiftcorr=np.exp(-1j*nu*L*np.pi)
Fk_corr=fft.fftshift(fft.fft(fx)/N*shiftcorr) # Fourier coefficients (divided by n), explicitly corrected for shift (green)
f, ax = plt.subplots(3,1)
ax[0].plot(x, fx)
ax[0].set_ylabel(r'$f(x)$', size = 'x-large')
ax[0].set_xlabel(r'$x$')
# Plot FFT Real Cosine terms
ax[1].plot(nu, np.real(Fk),label="shifted")
ax[1].plot(nu, np.real(Fk_noshift),label="not shifted")
ax[1].plot(nu, np.real(Fk_corr),label="explicit correction")
ax[1].set_ylabel(r'$Re[F(k)]$', size = 'x-large')
ax[1].set_xlabel(r'$k$')
ax[1].legend(loc="upper right")
# Plot FFT Imag Sine terms
ax[2].plot(nu, np.imag(Fk))
ax[2].plot(nu, np.imag(Fk_noshift))
ax[2].plot(nu, np.imag(Fk_corr))
ax[2].set_ylabel(r'$Im[F(k)]$', size = 'x-large')
ax[2].set_xlabel(r'$k$')
plt.subplots_adjust(top=0.9, hspace=0.5, wspace=0.5)
# Plot spectral power
#ax[2].plot(nu, np.absolute(Fk)**2)
#ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size = 'x-large')
#ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large')
plt.show()
