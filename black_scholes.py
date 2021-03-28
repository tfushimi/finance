import numpy as np
import scipy.stats as st
from nonlinear_solver import newton, bisection, secant

def cum_dist_normal(t):
    z = np.abs(t)
    y = 1 / (1 + 0.2316419 * z)
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    m = 1 - np.exp(- t**2 / 2) * (a1 * y + a2 * y**2 + a3 * y**3 + a4 * y**4 + a5 * y**5) / np.sqrt(2 * np.pi)
    if t > 0:
        return m
    else:
        return 1 - m

class callOption(object):
    def __init__(self, t, S, K, T, sigma, r, q):
        self.spot = S
        self.strike = K
        self.time_to_maturity = T - t
        self.volatility = sigma
        self.rate = r
        self.dividend = q

        # private variables
        self.d1 = (np.log(self.spot / self.strike) + (self.rate - self.dividend + self.volatility ** 2 / 2) * (self.time_to_maturity)) / (self.volatility * np.sqrt(self.time_to_maturity))
        self.d2 = self.d1 - self.volatility * np.sqrt(self.time_to_maturity)

    # Call option price
    def price(self):
        return self.spot * np.exp(-self.dividend * self.time_to_maturity) * cum_dist_normal(self.d1) - self.strike * np.exp(-self.rate * self.time_to_maturity) * cum_dist_normal(self.d2)

    # Greeks
    def delta(self):
        return np.exp(-self.dividend * self.time_to_maturity) * cum_dist_normal(self.d1)

    def gamma(self):
        return np.exp(-self.dividend * self.time_to_maturity - self.d1 ** 2 / 2) / (self.spot * self.volatility * np.sqrt(2 * np.pi * self.time_to_maturity))

    def vega(self):
        return self.spot * np.exp(-self.dividend * self.time_to_maturity - self.d1 ** 2 / 2) * np.sqrt(self.time_to_maturity) / np.sqrt(2 * np.pi)

    # Implied volatility
    def implied_volatility(self, price):
        return newton(f=lambda x: callOption(0, self.spot, self.strike, self.time_to_maturity, x, self.rate, self.dividend).price() - price, \
                      f_prime=lambda x: callOption(0, self.spot, self.strike, self.time_to_maturity, x, self.rate, self.dividend).vega(), init_guess=0.25)

class putOption(callOption):
    def price(self):
        return self.strike * np.exp(-self.rate * self.time_to_maturity) * cum_dist_normal(-self.d2) - self.spot * np.exp(-self.dividend * self.time_to_maturity) * cum_dist_normal(-self.d1)

    def delta(self):
        return np.exp(-self.dividend * self.time_to_maturity) * cum_dist_normal(-self.d1)

    # Implied volatility
    def implied_volatility(self, price):
        return newton(f=lambda x: putOption(0, self.spot, self.strike, self.time_to_maturity, x, self.rate, self.dividend).price() - price, \
                      f_prime=lambda x: putOption(0, self.spot, self.strike, self.time_to_maturity, x, self.rate, self.dividend).vega(), init_guess=0.25)

if __name__ == "__main__":
    # Test cum_dist_normal function
    print("Test cum_dist_normal function")
    candidate_t = [-1, -0.5, 0, 0.5, 1]
    for t in candidate_t:
        if np.allclose(cum_dist_normal(t), st.norm(0, 1).cdf(t), rtol=1e-05, atol=1e-06):
            print("Pass at t = %s" % (t))
    print()

    # Example on page 103
    print("Example on page 103")
    call = callOption(t=0, S=42, K=40, T=0.5, sigma=0.3, r=0.05, q=0.03)
    call_price = call.price()
    put = putOption(t=0, S=42, K=40, T=0.5, sigma=0.3, r=0.05, q=0.03)
    put_price = put.price()
    print("the price of call option", call_price)
    print("the price of put option", put_price)
    if np.allclose(put_price + 42 * np.exp(-0.03*0.5) - call_price, 40 * np.exp(-0.05*0.5), rtol=1e-5, atol=1e-6):
        print("Put-Call parity holds")
    print()

    # Exercise 12 in chapter 3
    print("Exercise 12 in chapter 3")
    call = callOption(t=0, S=50, K=45, T=0.5, sigma=0.2, r=0.06, q=0.02)
    put = putOption(t=0, S=50, K=45, T=0.5, sigma=0.2, r=0.06, q=0.02)
    print("the price of call option:", call.price())
    print("the price of put option:", put.price())
    print()

    # Exercise 22 in chapter 3
    print("Exercise 22 in chapter 3")
    call_15days = callOption(0, 50, 50, 1/24, 0.3, 0.05, 0)
    call_3months = callOption(0, 50, 50, 1/4, 0.3, 0.05, 0)
    call_1year = callOption(0, 50, 50, 1, 0.3, 0.05, 0)
    print("Gamma with 15 days maturity:", call_15days.gamma())
    print("Gamma with 3 months maturity:", call_3months.gamma())
    print("Gamma with 1 year maturity:", call_1year.gamma())
    print()

    # Exercise 23 in chapter 3
    print("Exercise 23 in chapter 3")
    call_15days = callOption(0, 50, 50, 1 / 24, 0.3, 0, 0)
    call_3months = callOption(0, 50, 50, 1 / 4, 0.3, 0, 0)
    call_1year = callOption(0, 50, 50, 1, 0.3, 0, 0)
    print("Vega with 15 days maturity:", call_15days.vega())
    print("Vega with 3 months maturity:", call_3months.vega())
    print("Vega with 1 year maturity:", call_1year.vega())
    print()

    # Exercise 26 in chapter 3
    print("Exercise 26 in chapter 3")
    call = callOption(0, 92, 90, 1/4, 0.2, 0.05, 0)
    print("Delta:", 1000 * call.delta())
    print()

    # Exercise 27 in chapter 3
    print("Exercise 27 in chapter 3")
    call = callOption(0, 100, 100, 0.5, 0.3, 0.05, 0)
    print("the price of a call option:", call.price() * 1000)
    delta = np.round(call.delta() * 1000)
    print("Delta:", delta )
    new_call = callOption(0, 98, 100, 125/252, 0.3, 0.05, 0)
    print("the new price of a call option:", new_call.price() * 1000)
    initial_position = 1000 * call.price() - delta * 100
    print("the initial value of your position:", initial_position)
    new_position = 1000 * new_call.price() - delta * 98
    print("The value of your position on the next day:", new_position)
    print("The loss without delta-hedging:", 1000 * (new_call.price() - call.price()))
    print("The loss with delta-hedging:", new_position - initial_position)
    print()

    # Exercise 28 in chapter 3
    print("Exercise 28 in chapter 3")
    put = putOption(0, 20, 25, 0.5, 0.3, 0.04, 0)
    print("the price of put:", put.price())
    initial_position = 1000 * put.price() + 400 * 20 + 10000
    print("the initial position:", initial_position)
    delta = np.round(1000 * (put.delta() - 1))
    print("Delta:", delta)
    hedged_position = 1000 * put.price() - delta * 20 + 10000
    print("the hedged position:", hedged_position)
    new_put = putOption(0, 24, 25, 7/12, 0.3, 0.04, 0)
    print("the new price of put:", new_put.price())
    new_position = 1000 * new_put.price() - delta * 24 + 10000
    print("the new position:", new_position)
    new_delta = np.round(1000 * (new_put.delta() - 1))
    print("the new delta:", new_delta)
    print()

    # Example on page 149
    print("Example on page 149")
    print("Implied volatility of a call option: ", callOption(t=0, S=25, K=20, T=1, sigma=0.1, r=0.05, q=0).implied_volatility(price=7))
    print()

    # Exercise 4 in chapter 5
    print("Exercise 4 in chapter 5")
    c = 2.5
    print("(Bisection method)Implied volatility of a call optioin: ",\
          bisection(f=lambda x: callOption(t=0, S=30, K=30, T=1 / 4, sigma=x, r=0.06, q=0.02).price() - c, a = 0.0001, b = 1))
    print("(Secant method)Implied volatility of a call optioin: ", \
          secant(f=lambda x: callOption(t=0, S=30, K=30, T=1 / 4, sigma=x, r=0.06, q=0.02).price() - c, init_guess = 0.5))
    print("(Netwon's method)Implied volatility of a call optioin: ", \
          newton(f=lambda x: callOption(t=0, S=30, K=30, T=1/4, sigma=x, r=0.06, q=0.02).price() - c, \
                 f_prime=lambda x: callOption(t=0, S=30, K=30, T=1/4, sigma=x, r=0.06, q=0.02).vega(), init_guess=0.5))
    print()