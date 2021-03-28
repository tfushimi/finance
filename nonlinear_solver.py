import numpy as np

# Bisection method
def bisection(f, a, b, tol_int = 10e-6, tol_approx = 10e-9):
    xL = a
    xR = b
    while np.max([np.abs(f(xL)), np.abs(f(xR))]) > tol_approx or xR - xL > tol_int:
        xM = (xL + xR) / 2
        if f(xL) * f(xM) < 0:
            xR = xM
        else:
            xL = xM
    return xM

# Newton's method
def newton(f, f_prime, init_guess, tol_consec = 10e-6, tol_approx = 10e-9):
    x_new = init_guess
    x_old = x_new - 1
    while np.abs(f(x_new)) > tol_approx or np.abs(x_new - x_old) > tol_consec:
        x_old = x_new
        x_new = x_old - f(x_old) / f_prime(x_old)
    return x_new

# Secant method
def secant(f, init_guess, tol_consec = 10e-6, tol_approx = 10e-9):
    x_new = init_guess
    x_old = x_new - 1
    while np.abs(f(x_new)) > tol_approx or np.abs(x_new - x_old) > tol_consec:
        x_oldest = x_old
        x_old = x_new
        x_new = x_old - f(x_old) * (x_old - x_oldest) / (f(x_old) - f(x_oldest))
    return x_new

# N-dimensional Newton's method
def Ndim_newton(f, grad, init_guess, tol_consec = 10e-6, tol_approx = 10e-9):
    x_new = init_guess
    x_old = x_new - 1
    while np.linalg.norm(f(x_new), 2) > tol_approx or np.linalg.norm(x_new - x_old, 2) > tol_consec:
        x_old = x_new
        F = f(x_old)
        DF = grad(x_old)
        x_new = x_old - np.linalg.solve(DF, F)
    return x_new

# Approximate N-dimensional Newton's method
def approx_Ndim_newton(f, init_guess, h = 10e-6, tol_consec = 10e-6, tol_approx = 10e-9):
    N = len(init_guess)
    x_new = init_guess
    x_old = x_new - 1
    while np.linalg.norm(f(x_new), 2) > tol_approx or np.linalg.norm(x_new - x_old, 2) > tol_consec:
        x_old = x_new
        F = f(x_old)
        DF = np.zeros([N, N])
        f_old = f(x_old)
        for i in range(N):
            h_vec = np.zeros([N])
            h_vec[i] = h
            x_old_h = x_old + h_vec
            DF[i] = (f(x_old_h) - f_old) / h
        x_new = x_old - np.linalg.solve(DF.T, F)
    return x_new

if __name__ == "__main__":
    # Example on page 137
    def function1(x):
        return x**4 -5 * x**2 + 4 - 1 / (1 + np.exp(x**3))
    print("(Bisection method) The root: ", bisection(function1, -2, 3))
    print()

    # Example on page 140
    def gradient1(x):
        return 4 * x**3 - 10 * x + (3 * x**2 * np.exp(x**3)) / (1 + np.exp(x**3))**2
    print("(Newton's method) The root: ", newton(function1, gradient1, 3))
    print()

    # Example on page 143
    print("(Secant method) The root: ", secant(function1, 3))
    print()

    # Example on page 146
    def function2(x):
        x1 = x[0]**3 + 2 * x[0] * x[1] + x[2]**2 - x[1] * x[2] + 9
        x2 = 2 * x[0]**2 + 2 * x[0] * x[1]**2 + x[1]**3 * x[2]**2 - x[1]**2 * x[2] - 2
        x3 = x[0] * x[1] * x[2] + x[0]**3 - x[2]**2 - x[0] * x[1]**2 - 4
        return np.array([x1, x2, x3])

    def gradient2(x):
        x11 = 3 * x[0]**2 + 2 * x[1]
        x12 = 2 * x[0] - x[2]
        x13 = 2 * x[2] - x[1]
        x21 = 4 * x[0] + 2 * x[1]**2
        x22 = 4 * x[0] * x[1] + 3 * x[1]**2 + x[2]**2 - 2 * x[1] * x[2]
        x23 = 2 * x[1]**3 * x[2] - x[1]**2
        x31 = x[1] * x[2] + 3 * x[0]**2 - x[1]**2
        x32 = x[0] * x[2] - 2 * x[0] * x[1]
        x33 = x[0] * x[1] - 2 * x[2]
        return np.array([[x11, x12, x13], [x21, x22, x23], [x31, x32, x33]])

    print("(N-dim Newton's method) The root: ", Ndim_newton(function2, gradient2, np.array([1, 2, 3])))
    print("(N-dim Newton's method) The root: ", Ndim_newton(function2, gradient2, np.array([2, 2, 2])))
    print()

    # Example on page 148
    def function3(x):
        x1 = x[0]**3 + 2 * x[0] * x[1] + x[2]**2 - x[1] * x[2] + 9
        x2 = 2 * x[0]**2 + 2 * x[0] * x[1]**2 + x[1]**3 * x[2]**2 - x[1]**2 * x[2] - 2
        x3 = x[0] * x[1] * x[2] + x[0]**3 - x[2]**2 - x[0] * x[1]**2 - 4
        return np.array([x1, x2, x3])
    print("(Approx N-dim Newton's method) The root: ", approx_Ndim_newton(function3, np.array([1, 2, 3])))
    print("(Approx N-dim Newton's method) The root: ", approx_Ndim_newton(function3, np.array([2, 2, 2])))
    print()
