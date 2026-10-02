import matplotlib.pyplot as plt
import optimal_consumption
import numpy as np

R = 2
rho = -0.84


def delta(y):
    return np.full_like(y, 0.02)


def r(y):
    return np.full_like(y, 0.013)


def lmbda(y):
    return 1.66 * np.sqrt(y)


def sigma(y):
    return np.sqrt(y)


def a(y):
    return -0.088 * (y - 0.035)


def b(y):
    return 0.031 * np.sqrt(y)


grid, optimal_consumption_rate = optimal_consumption.optimal_consumption_rate(
    R, r, delta, lmbda, rho, a, b, 1 / 10000, np.sqrt(10000), 10000
)
optimal_investment_fraction = optimal_consumption.optimal_investment_fraction(
    R, sigma, lmbda, rho, b, grid, optimal_consumption_rate
)

fig, axs = plt.subplots(1, 2)

axs[0].plot(
    Y := np.linspace(0.01, 0.06, 100),
    100 * np.interp(Y, grid, optimal_consumption_rate),
)
axs[0].set_xlabel("$y$")
axs[0].set_ylabel("Optimal Consumption Rate (%)")

axs[1].plot(
    Y := np.linspace(0.01, 0.06, 100),
    100 * np.interp(Y, grid, optimal_investment_fraction),
)
axs[1].set_xlabel("$y$")
axs[1].set_ylabel("Optimal Risky Asset Allocation (%)")

fig.tight_layout()

fig.savefig("heston.png", bbox_inches="tight")
