import numpy as np
import matplotlib.pyplot as plt

def ode1(n):
    h = 1/n
    x = np.linspace(0, 1, n+1)
    A = np.diag([2+h**2]*(n-1)) + np.diag([-1]*(n-2), -1) + np.diag([-1]*(n-2), 1)
    b = np.zeros([n-1])
    b[0] = 1
    b[-1] = np.exp(1)
    y = np.linalg.solve(A, b)
    return x, np.concatenate((np.array([1]), y, np.array([np.exp(1)])))

def ode2(n):
    h = 2/n
    x = np.linspace(-1, 1, n+1)
    A = np.diag([2 * (1-h**2)]*(n-1)) + np.diag([-(1-3*h/2)]*(n-2), -1) + np.diag([-(1+3*h/2)]*(n-2), 1)
    b = np.zeros([n-1])
    b[0] = (1 - 3*h/2) * (1 + np.exp(4))
    b[-1] = (1 + 3*h/2) * (1 + np.exp(-2))
    y = np.linalg.solve(A, b)
    return x, np.concatenate((np.array([1+np.exp(4)]), y, np.array([1+np.exp(-2)])))

if __name__ == "__main__":
    n = 8
    x, y = ode1(n)
    for i in range(n+1):
        print("(x_i, y_i) = (%s, %s)" % (x[i], y[i]))
    print()
    plt.plot(x, y)
    plt.savefig("ode1.png")
    plt.close()

    n = 8
    x, y = ode2(n)
    for i in range(n+1):
        print("(x_i, y_i) = (%s, %s)" % (x[i], y[i]))
    x_2, y_2 = ode2(1024)
    plt.plot(x, y)
    plt.plot(x_2, y_2)
    x_i = np.linspace(-1, 1, n+1)
    plt.plot(x_i, np.exp(-x_i-1) + np.exp(-2*x_i+2))
    plt.savefig("ode2.png")