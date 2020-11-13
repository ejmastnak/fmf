import sys,math
import numpy as np
from matplotlib import pyplot as plt

def cos_print(x):
    max_iterations=100
    eps = 1e-6
    result = 1.
    step = 1.
    for n in range(1,max_iterations):
        print("Iteration: {}".format(n))
        step *= (-1**(n-1))*(x**2)/(2*n)/((2*n)-1)
        previous = result
        result += step

        if abs(result-previous) < eps or n == 99:
            print("Stopping at {} iterations\t Result: {:.20f} \t Exact: {:.20f} \t Difference: {:.20f}".format(n, result, math.cos(x), abs(math.cos(x) - result)))
            break

def cos(x):
    max_iterations=100
    eps = 1e-2
    result = 1.
    step = 1.
    for n in range(1,max_iterations):
        step *= (-1**(n-1))*(x**2)/(2*n)/((2*n)-1)
        previous = result
        result += step

        if abs(result-previous) < eps:
            break

    return result

def plot_comparison():
    pi = math.pi
    X = np.linspace(-pi, pi, 100)
    my_cos = []
    for i in range(100):
        my_cos.append(cos(X[i]))
        
    true_cos = np.cos(X)
    
    plt.xlabel("x")
    plt.ylabel("cos(x)")
    plt.plot(X, my_cos,label="My cosine")
    plt.plot(X, true_cos,label="True cosine")
    plt.legend()
    plt.show()

def plot_error():
    X = np.linspace(-math.pi, math.pi, 100)
    my_cos = []
    for i in range(100):
        my_cos.append(cos(X[i]))

    true_cos = np.cos(X)
    # error = np.abs(true_cos - my_cos)
    error = []
    for i in range(100):
        error.append(abs(true_cos[i] - my_cos[i]))
    plt.xlabel("x")
    plt.ylabel("Absoute error")
    plt.plot(X, error, marker='.', label="My cosine vs true cos")
    plt.legend()
    plt.show()

# plot_comparison()
plot_error()
