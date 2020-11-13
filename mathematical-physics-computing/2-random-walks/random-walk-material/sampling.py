import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 10001

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#sampling
N=100000
x=np.random.normal(size=N,loc=1.,scale=0.5) # loc and scale specify mean and standard deviation

def plot_gauss():
    #gauss
    r1=np.random.random_sample(N)
    r2=np.random.random_sample(N)
    y=1.0 + 0.5*np.sin(2*np.pi*r1)*np.sqrt(-2*np.log(r2))

    #plot  histogram!
    plt.figure(figsize=(6,3))
    plt.hist(x,density=True,bins=100,histtype='step',color='r',label="Gauss")
    plt.hist(y,density=True,bins=100,histtype='step',color='b',label="Sampled")
    plt.xlabel('x')
    plt.ylabel('1/N * dN/dx')
    plt.legend()
    #plt.yscale('log')
    plt.show()


def plot_cauchy():
    r1=np.random.random_sample(N)
    r2=np.random.random_sample(N)
    y=np.tan(np.pi*(r1-0.5))

    #plot  histogram!
    plt.figure(figsize=(6,3))
    plt.hist(x,density=True,bins=100,histtype='step',color='r',label="Cauchy")
    plt.hist(y,density=True,bins=100,histtype='step',color='b',label="Sampled")
    plt.xlabel('x')
    plt.ylabel('1/N * dN/dx')
    plt.legend()
    #plt.yscale('log')
    plt.show()


def plot_paretto():
    #pareto distribution, pazi na (a+1) potenco!
    a, m = 3., 2.  # shape and mode
    s = (np.random.pareto(a, 1000) + 1) * m
    #Display the histogram of the samples, along with the probability density function:
    plt.figure(figsize=(6,3))
    count, binEdges, _ = plt.hist(s, 100, density=True,color='b',label="Sampled")
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    fit = a*m**a / bincenters**(a+1)
    plt.plot(bincenters, max(count)*fit/max(fit), linewidth=2, color='r',label="Pareto")
    plt.xlabel('x')
    plt.ylabel('1/N * dN/dx')
    #plt.yscale('log')
    plt.legend()
    plt.show()


def plot_error_bars():
    #drawing error bars example - kot dodatek...
    data       = np.array(np.random.rand(1000))
    y,binEdges = np.histogram(data,bins=10)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    yerrs    = np.sqrt(y)
    width      = 0.05
    xerrs = 0.5*(binEdges[1:]-binEdges[:-1]) # or None
    plt.errorbar(bincenters, y, yerrs, xerrs, fmt="ob",)
    plt.show()

plot_gauss()
