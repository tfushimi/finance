import numpy as np
from numerical_integration import *
from nonlinear_solver import *

class bond(object):
    def __init__(self, t_cash_flow, v_cash_flow):
        self.n = len(t_cash_flow)
        self.time = t_cash_flow
        self.cash_flow = v_cash_flow

    def price_zero_rate(self, r_zero):
        B = 0
        for i in range(self.n):
            discount = np.exp(-self.time[i] * r_zero(self.time[i]))
            B += self.cash_flow[i] * discount
        return B

    def price_inst_rate(self, r_inst, tol, method):
        B = 0
        for i in range(self.n):
            I_numerical = integration(r_inst, 0, self.time[i], tol[i], method)
            discount = np.exp(-I_numerical)
            B += self.cash_flow[i] * discount
        return B

    def convexity_and_duration(self, yield_rate):
        B = D = C = 0
        for i in range(self.n):
            discount = np.exp(-self.time[i] * yield_rate)
            B += self.cash_flow[i] * discount
            D += self.time[i] * self.cash_flow[i] * discount
            C += self.time[i] ** 2 * self.cash_flow[i] * discount
        return B, D / B, C / B

    # bond price as a function of yield
    def price_yield(self, yield_rate):
        B = 0
        for i in range(self.n):
            B += self.cash_flow[i] * np.exp(-yield_rate * self.time[i])
        return B

    # gradient of bond price with respect to yield
    def grad_price_yield(self, yield_rate):
        B = 0
        for i in range(self.n):
            B -= self.time[i] * self.cash_flow[i] * np.exp(-yield_rate * self.time[i])
        return B

    # compute yield rate
    def yield_rate(self, price, init_guess=0.1):
        # bond_obj = self.__init__(t_cash_flow=self.time, v_cash_flow=self.cash_flow)
        # return newton(f=lambda x: bond_obj.price_yield(x) - price, f_prime=lambda x: bond_obj.grad_price_yield(x), init_guess=0.1)
        return newton(f=lambda x: self.price_yield(x) - price, f_prime=lambda x: self.grad_price_yield(x), init_guess=init_guess)

# main
if __name__ == "__main__":
    t_cash_flow = [2/12, 8/12, 14/12, 20/12]
    v_cash_flow = [3, 3, 3, 103]
    tol = [10e-4, 10e-4, 10e-4, 10e-6]
    bond1 = bond(t_cash_flow, v_cash_flow)

    # zero rate
    print("(i) Bond price given the zero rate:", bond1.price_zero_rate(lambda x: 0.0525 + np.log(1 + 2 * x) / 200))

    # instantaneous rate
    print("(ii) Bond price given the instantaneous rate:", bond1.price_inst_rate(lambda x: 0.0525 + 1 / (100 * (1 + np.exp(-x ** 2))), tol, simpson))


    # istantaneous rate2
    tol2 = [10e-4, 10e-4, 10e-4, 10e-4]
    print("(iii) Bond price given the instantaneous rate:", bond1.price_inst_rate(lambda x: 0.0525 + np.log(1 + 2 * x) / 200 + x / (100 * (1 + 2 * x)), tol, simpson))
    print("(iii) Bond price given the instantaneous rate:", bond1.price_inst_rate(lambda x: 0.0525 + np.log(1 + 2 * x) / 200 + x / (100 * (1 + 2 * x)), tol2, simpson))

    # Duration and Convexity
    bond2 = bond([2/12, 8/12, 14/12, 20/12], [3, 3, 3, 103])
    B, D, C = bond2.convexity_and_duration(6.5 / 100)
    print("(iv) Price: %s, Duration: %s and Convexity: %s" % (B, D, C))
    print()

    # Exercise 10 in chapter 2
    print("Exercise 10 in chapter 2")
    def zero_rate2(t):
        if t == 6/12:
            return 0.05
        elif t == 1:
            return 0.0525
        elif t == 18/12:
            return 0.0535
        else:
            return 0.055
    bond3 = bond([0.5, 1, 1.5, 2], [2.5, 2.5, 2.5, 102.5])
    print("Bond price:", bond3.price_zero_rate(zero_rate2))
    print()

    # Exercise 11 in chapter 2
    print("Exercise 11 in chapter 2")
    bond4 = bond([0.5, 1, 1.5, 2], [2.5, 2.5, 2.5, 102.5])
    print("the price of a two year semianual coupon bond with coupon rate 5%:", bond4.price_zero_rate(lambda x: 0.045 + 0.005 * (1 + x) * np.log(1 + x) / x))
    print()

    # Exercise 13
    print("Exercise 13 in chapter 2")
    print("1-year discount factor:", np.exp(-integration(lambda x: 0.05 / (1 + np.exp(-(1 + x)**2)), 0, 1, 10e-6, simpson)))
    print("2-year discount factor:", np.exp(-integration(lambda x: 0.05 / (1 + np.exp(-(1 + x)**2)), 0, 2, 10e-6, simpson)))
    print("1-year discount factor:", np.exp(-integration(lambda x: 0.05 / (1 + np.exp(-(1 + x)**2)), 0, 3, 10e-8, simpson)))
    tol = [10e-6, 10e-6, 10e-8]
    bond5 = bond([1, 2, 3], [5, 5, 105])
    print("the price of a three year annual coupon bond with coupon rate 5%:", bond5.price_inst_rate(lambda x: 0.05 / (1 + np.exp(-(1 + x)**2)), tol, simpson))
    print()

    # Exercise 14 in chapter 2
    print("Exercise 14 in chapter 2")
    bond6 = bond([0.5, 1, 1.5, 2, 2.5, 3], [3, 3, 3, 3, 3, 103])
    B, D, C = bond6.convexity_and_duration(0.09)
    print("The price: %s, the duration: %s, and the convexity: %s" % (B, D, C))
    print()

    # Exercise 15 in chapter 2
    print("Exercise 14 in chapter 2")
    bond7 = bond([2/12, 5/12, 8/12, 11/12, 14/12], [2, 2, 2, 2, 102])
    B, D, C = bond7.convexity_and_duration(0.07)
    print("The price: %s, the duration: %s, and the convexity: %s" % (B, D, C))
    print()

    # Example on page page 149
    B = 105
    bond8 = bond([4/12, 10/12, 16/12, 22/12, 28/12, 34/12], [4, 4, 4, 4, 4, 104])
    print("(bisection method) The yield of a bond: ", bisection(lambda x: bond8.price_yield(x) - B, 0, 0.1))
    print("(Newton's method) The yield of a bond: ", newton(f=lambda x: bond8.price_yield(x) - B, f_prime=lambda x: bond8.grad_price_yield(x), init_guess=0.1))
    print("(secant method) The yield of a bond: ", secant(lambda x: bond8.price_yield(x) - B, init_guess=0.1))
    print()

    # Exercise 1 in chapter 5
    c = 0.03375 / 2
    bond9 = bond([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], \
                 [100 * c, 100 * c, 100 * c, 100 * c, 100 * c, 100 * c, 100 * c, 100 * c, 100 * c, 100 * (1 + c)])
    yield_rate = bond9.yield_rate(100 + 1/32)
    _, D, C = bond9.convexity_and_duration(yield_rate)
    print("Exercise 1 in chapter 5")
    print("(Newton's method) The yield of a bond: ", yield_rate)
    print("The duration: ", D)
    print("The convexity: ", C)
    print()

    # Exercise 2 in chapter 5
    bond10 = bond([0.5, 1, 1.5, 2], [4, 4, 4, 104])
    price = bond10.price_zero_rate(lambda x: 0.05 + 0.01 * np.log(1 + x/2))
    yield_rate = bisection(lambda x: bond10.price_yield(x) - price, 0, 0.1)
    _, D, C = bond10.convexity_and_duration(yield_rate)
    print("Exercise 2 in chapter 5")
    print("The bond price: ", price)
    print("The yield: ", yield_rate)
    print("The duration: ", D)
    print("The convexity: ", C)