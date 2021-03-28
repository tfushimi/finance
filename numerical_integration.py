import numpy as np

# Midpoint method
def midpoint(f, a, b, n):
    h = (b - a) / n
    I_midpoint = 0
    for i in range(1, n+1):
        I_midpoint += f(a + (i - 0.5) * h)
    return I_midpoint * h

# Trapezoidal method
def trapezoidal(f, a, b, n):
    h = (b - a) / n
    I_trap = (f(a) + f(b)) / 2
    for i in range(1, n+1):
        I_trap += f(a + i * h)
    return I_trap * h

# Simpson's method
def simpson(f, a, b, n):
    h = (b - a) / n
    I_simpson = (f(a) + f(b)) / 6
    for i in range(1, n):
        I_simpson += f(a + i * h) / 3
    for i in range(1, n+1):
        I_simpson += 2 * f(a + (i - 0.5) * h) / 3
    return I_simpson * h

# numerical integration
def integration(f, a, b, tol, method):
    n = 4
    I_old = method(f, a, b, n)
    n *= 2
    I_new = method(f, a, b, n)
    while np.abs(I_new - I_old) > tol:
        I_old = I_new
        n *= 2
        I_new = method(f, a, b, n)
    return I_new

if __name__ == "__main__":
    # an integrand
    def f1(x):
        return np.exp(-x ** 2)

    a = 0
    b = 2
    print("Numerical integration:", integration(f1, a, b, 10e-6, midpoint))
    print("Numerical integration:", integration(f1, a, b, 10e-6, trapezoidal))
    print("Numerical integration:", integration(f1, a, b, 10e-6, simpson))

    # Exercise 3 in chapter 2
    def f2(x):
        return np.sqrt(x) * np.exp(-x)
    print("Exercise 3 in chapter2")
    print("Numerical integration:", integration(f2, 1, 3, 10e-6, midpoint))
    print("Numerical integration:", integration(f2, 1, 3, 10e-6, trapezoidal))
    print("Numerical integration:", integration(f2, 1, 3, 10e-6, simpson))

    # Exercise 4 in chapter 2
    def f3(x):
        return x**(5/2) / (1 + x**2)
    print("Exercise 4 in chapter2")
    print("Numerical integration:", integration(f3, 0, 1, 10e-6, midpoint))
    print("Numerical integration:", integration(f3, 0, 1, 10e-6, simpson))
