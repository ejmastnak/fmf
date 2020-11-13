import numpy as np
import sys,math

x=2.3
print("Initial value: x = {}".format(x))

def model():
    max_interations=100
    epsilon=1e-6
    result=1.
    val=1.
    for n in range(1,max_interations+1):
        print("Iteration {}".format(n))
        val = val*x/n
        old = result
        result += val
        print ("Step/term {:.20e}".format(val))
        print ("Result: old {:.20f}, new {:.20f} step {:.20f}".format(result,old,val))
        # koncaj prej: 
        # razlicne opcije
        #if val < epsilon:
        #if abs(result-old) == 0.:
        if result == old:
            print("Koncam z rezultatom {:.20f}, tocno {:.20f}, razlika {:.20f}".format(result,math.exp(x),result-math.exp(x)))
            break    

def exp(x):
    max_iterations=100 # controls when to exit loop
    epsilon=1e-6 # controls tolerance
    result = 1. # start at one to avoid calculating first Taylor series term:update|!python3 
    step = 1.
    for n in range(1,max_iterations):
        print("Iteration {}".format(n))
        step = step*x/n # efficient way to recursively calculate factorial and power terms in exp Taylor series
        previous = result # store previous result (to calculate difference between subsequent terms)
        result += step # append most recent term to result
        print("Step: {:.20e}".format(step))
        print("New Result: {:.20f}\t Previous: {:20f}\t Step: {:20f}".format(result,previous,step))

        # if abs(result-previous) == 0.: # no difference between current and previous step
        if result==previous: # if current and previous are ==
        # if(step < epsilon):
            print("\nExiting after {} iterations with result {:.20f}\t Exact value: {:.20f}\t Difference: {:.20f}".format(n, result,math.exp(x),result-math.exp(x)))
            break

exp(x)
