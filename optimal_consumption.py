import math
from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt

import tridiagonal

Array = npt.NDArray[np.floating]
Coefficient = Callable[[Array], Array]


def _construct_diagonals(
    myopic_consumption_rate: npt.ArrayLike,
    adjusted_drift: npt.ArrayLike,
    factor_vol: npt.ArrayLike,
    grid_spacing: float,
    correlation_adjusted_risk_aversion: float,
) -> tuple[Array, Array, Array]:
    r"""Construct the tridiagonal discretisation matrix.

    Parameters
    ----------
    myopic_consumption_rate : array_like of shape (N,)
        Myopic consumption rate evaluated on the factor grid.
    adjusted_drift : array_like of shape (N,)
        Correlation-adjusted factor drift evaluated on the grid.
    factor_vol : array_like of shape (N,)
        Factor volatility evaluated on the grid.
    grid_spacing : float
        Distance between adjacent points of the uniform grid.
    correlation_adjusted_risk_aversion : float
        Correlation-adjusted risk-aversion parameter :math:`\tilde R`.

    Returns
    -------
    lower : ndarray of shape (N - 1,)
        Lower diagonal of the discretisation matrix.
    main : ndarray of shape (N,)
        Main diagonal of the discretisation matrix.
    upper : ndarray of shape (N - 1,)
        Upper diagonal of the discretisation matrix.
    """
    myopic_consumption_rate = np.asarray(myopic_consumption_rate, dtype=float)
    adjusted_drift = np.asarray(adjusted_drift, dtype=float)
    factor_vol = np.asarray(factor_vol, dtype=float)

    factor_diffusion = 0.5 * factor_vol**2 / grid_spacing**2
    factor_drift = adjusted_drift / grid_spacing

    main = myopic_consumption_rate + (2.0 * factor_diffusion + np.abs(factor_drift)) / (
        correlation_adjusted_risk_aversion
    )
    upper = (
        -(factor_diffusion[:-1] + np.maximum(factor_drift[:-1], 0.0))
        / correlation_adjusted_risk_aversion
    )
    lower = (
        -(factor_diffusion[1:] + np.maximum(-factor_drift[1:], 0.0))
        / correlation_adjusted_risk_aversion
    )

    # Match Neumann boundary conditions
    upper[0] -= factor_diffusion[0] / correlation_adjusted_risk_aversion
    lower[-1] -= factor_diffusion[-1] / correlation_adjusted_risk_aversion
    main[0] = myopic_consumption_rate[0] - upper[0]
    main[-1] = myopic_consumption_rate[-1] - lower[-1]

    return lower, main, upper


@numba.njit(cache=True)
def check_wellposedness(lower: Array, main: Array, upper: Array) -> bool:
    """Check the M-Matrix condition for the discretised problem.

    Parameters
    ----------
    lower : ndarray of shape (N - 1,)
        Lower diagonal of the tridiagonal matrix.
    main : ndarray of shape (N,)
        Main diagonal of the tridiagonal matrix.
    upper : ndarray of shape (N - 1,)
        Upper diagonal of the tridiagonal matrix.

    Returns
    -------
    bool
        ``True`` if every pivot in the tridiagonal factorisation is finite
        and positive; otherwise ``False``.

    Notes
    -----
    For the tridiagonal Z-matrices produced by this scheme, the positive-pivot
    condition identifies the required nonsingular M-matrix structure.
    """
    r = main[0]

    if r <= 0:
        return False

    for i in range(1, len(main)):
        r = main[i] - lower[i-1] * upper[i-1] / r

        if r <= 0:
            return False
    
    return True



def _compute_bounds(c: Array, p: float) -> tuple[float, float]:
    r"""Compute lower and upper bounds for the fixed point.

    Parameters
    ----------
    c : ndarray of shape (N,)
        Positive vector :math:`A^{-1}\mathbf{1}`.
    p : float
        Fixed-point exponent in ``(-1, 1)``.

    Returns
    -------
    m : float
        Componentwise lower bound for the positive fixed point.
    M : float
        Componentwise upper bound for the positive fixed point.
    """
    C_min = np.min(c)
    C_max = np.max(c)

    if p < 0.0:
        m = (C_min * C_max**p) ** (1 / (1 - p**2))
        M = (C_min**p * C_max) ** (1 / (1 - p**2))
    else:
        m = C_min ** (1 / (1 - p))
        M = C_max ** (1 / (1 - p))

    return float(m), float(M)


def max_n_iterations(m: float, M: float, p: float, abs_tol: float) -> int:
    """Compute a conservative fixed-point iteration count.

    Parameters
    ----------
    m : float
        Positive lower bound for the fixed point.
    M : float
        Upper bound satisfying ``M >= m``.
    p : float
        Contraction exponent satisfying ``0 < abs(p) < 1``.
    abs_tol : float
        Desired positive absolute-error bound for the transformed fixed point.

    Returns
    -------
    int
        Nonnegative upper bound on the required number of iterations.
    """
    initial_error_bound = M * (M / m - 1.0)
    if initial_error_bound <= abs_tol or p == 0.0:
        return 0

    return max(
        0,
        math.ceil(
            (math.log(abs_tol) - math.log(initial_error_bound)) / math.log(abs(p))
        ),
    )


def _initial_guess(
    myopic_consumption_rate: Array, correlation_adjusted_risk_aversion: float, m: float, M: float
) -> Array:
    r"""Construct an initial guess for the fixed point.

    Parameters
    ----------
    myopic_consumption_rate : ndarray of shape (N,)
        Myopic consumption rate evaluated on the factor grid.
    correlation_adjusted_risk_aversion : float
        Correlation-adjusted risk-aversion parameter :math:`\tilde R`.
    m : float
        Positive lower bound for the fixed point.
    M : float
        Upper bound satisfying ``M >= m``.

    Returns
    -------
    ndarray of shape (N,)
        Initial guess for :math:`x = u^{-\tilde R}`, clipped to ``[m, M]``.
    """
    return np.clip(
        myopic_consumption_rate,
        M ** (-1.0 / correlation_adjusted_risk_aversion),
        m ** (-1.0 / correlation_adjusted_risk_aversion),
    ) ** (-correlation_adjusted_risk_aversion)


def _fixed_point_error_bound(x: Array, x_new: Array, p: float) -> float:
    r"""Bound the absolute error of ``x_new`` using the contraction map.
    """
    log_step = np.max(np.abs(np.log(x_new) - np.log(x)))
    log_error_bound = abs(p) / (1.0 - abs(p)) * log_step

    try:
        relative_error_bound = math.expm1(log_error_bound)
    except OverflowError:
        return np.inf
    return float(np.max(x_new) * relative_error_bound)


def _compute_fixed_point_iteration(
    correlation_adjusted_risk_aversion: float,
    myopic_consumption_rate: Array,
    adjusted_drift: Array,
    factor_vol: Array,
    grid: Array,
    abs_tol: float,
) -> Array:
    r"""Solve the discretised HJB equation by fixed-point iteration.

    Parameters
    ----------
    correlation_adjusted_risk_aversion : float
        Correlation-adjusted risk-aversion parameter :math:`\tilde R`.
    myopic_consumption_rate : ndarray of shape (N,)
        Myopic consumption rate evaluated on the factor grid.
    adjusted_drift : ndarray of shape (N,)
        Correlation-adjusted factor drift evaluated on the grid.
    factor_vol : ndarray of shape (N,)
        Factor volatility evaluated on the grid.
    grid : ndarray of shape (N,)
        Uniform, strictly increasing factor grid with at least two points.
    abs_tol : float
        Desired absolute-error bound for the transformed fixed point.

    Returns
    -------
    ndarray of shape (N,)
        Approximate optimal consumption rate on ``grid``.
    """
    p = (correlation_adjusted_risk_aversion - 1) / correlation_adjusted_risk_aversion
    if not -1.0 < p < 1.0:
        raise ValueError("The fixed-point iteration requires -1 < p < 1.")

    lower, main, upper = _construct_diagonals(
        myopic_consumption_rate,
        adjusted_drift,
        factor_vol,
        grid[1] - grid[0],
        correlation_adjusted_risk_aversion,
    )
    factors = tridiagonal.factor(lower, main, upper)
    c = tridiagonal.solve_factored(lower, factors, np.ones(grid.size))

    if p == 0.0:
        x = c
    else:
        m, M = _compute_bounds(c, p)
        x = _initial_guess(myopic_consumption_rate, correlation_adjusted_risk_aversion, m, M)
        num_iter = max_n_iterations(m, M, p, abs_tol)

        rhs = np.empty_like(x)
        x_new = np.empty_like(x)
        for _ in range(num_iter):
            np.power(x, p, out=rhs)
            tridiagonal.solve_factored_into(lower, factors, rhs, x_new)

            if _fixed_point_error_bound(x, x_new, p) <= abs_tol:
                x = x_new
                break

            x, x_new = x_new, x

    return x ** (-1.0 / correlation_adjusted_risk_aversion)


def optimal_consumption_rate(
    risk_aversion: float,
    interest_rate: Coefficient,
    impatience_rate: Coefficient,
    market_price_of_risk: Coefficient,
    correlation: float,
    factor_drift: Coefficient,
    factor_vol: Coefficient,
    y_min: float,
    y_max: float,
    grid_density: float,
    abs_tol: float = 1e-8,
) -> tuple[Array, Array]:
    r"""Approximate the optimal consumption rate.

    Parameters
    ----------
    risk_aversion : float
        Relative risk-aversion parameter :math:`R`. The implemented
        iteration requires the correlation-adjusted risk aversion
        :math:`\tilde R=(1-\rho^2)R+\rho^2` to exceed ``0.5``.
    interest_rate : Coefficient
        State-dependent risk-free rate :math:`r(y)`.
    impatience_rate : Coefficient
        State-dependent impatience rate :math:`\delta(y)`.
    market_price_of_risk : Coefficient
        State-dependent market price of risk :math:`\lambda(y)`.
    correlation : float
        Correlation :math:`\rho` between the Brownian motions driving the
        risky asset and the stochastic factor.
    factor_drift : Coefficient
        Drift :math:`a(y)` of the stochastic factor.
    factor_vol : Coefficient
        Volatility :math:`b(y)` of the stochastic factor.
    y_min : float
        Lower endpoint of the factor interval.
    y_max : float
        Upper endpoint of the factor interval.
    grid_density : float
        Requested number of grid intervals per unit of factor-state space.
    abs_tol : float, default=1e-8
        Desired absolute-error bound for the fixed point.

    Returns
    -------
    grid : ndarray of shape (N + 1,)
        Uniform factor grid including both endpoints.
    consumption_rate : ndarray of shape (N + 1,)
        Approximate optimal consumption rate as a fraction of wealth.
    """
    N = int(grid_density * (y_max - y_min))
    grid = np.linspace(y_min, y_max, N + 1, dtype=np.float64)

    interest = np.asarray(interest_rate(grid), dtype=float)
    impatience = np.asarray(impatience_rate(grid), dtype=float)
    market_price = np.asarray(market_price_of_risk(grid), dtype=float)
    drift = np.asarray(factor_drift(grid), dtype=float)
    factor_volatility = np.asarray(factor_vol(grid), dtype=float)

    eta = (
        impatience
        - (1.0 - risk_aversion) * (interest + market_price**2 / (2.0 * risk_aversion))
    ) / risk_aversion
    effective_risk_aversion = (1.0 - correlation**2) * risk_aversion + correlation**2
    adjusted_drift = drift + (
        (1.0 - risk_aversion)
        / risk_aversion
        * correlation
        * market_price
        * factor_volatility
    )

    return grid, _compute_fixed_point_iteration(
        effective_risk_aversion,
        eta,
        adjusted_drift,
        factor_volatility,
        grid,
        abs_tol,
    )


def myopic_consumption_rate(
    risk_aversion: float,
    interest_rate: Coefficient,
    impatience_rate: Coefficient,
    market_price_of_risk: Coefficient,
) -> Coefficient:
    r"""Construct the myopic consumption-rate function.

    Parameters
    ----------
    risk_aversion : float
        Relative risk-aversion parameter :math:`R`.
    interest_rate : Coefficient
        State-dependent risk-free rate :math:`r(y)`.
    impatience_rate : Coefficient
        State-dependent impatience rate :math:`\delta(y)`.
    market_price_of_risk : Coefficient
        State-dependent market price of risk :math:`\lambda(y)`.

    Returns
    -------
    Coefficient
        Function that evaluates the myopic consumption rate :math:`\eta(y)`
        elementwise on an array of factor states.
    """

    def evaluate(y: Array) -> Array:
        return (
            impatience_rate(y)
            - (1.0 - risk_aversion)
            * (interest_rate(y) + market_price_of_risk(y) ** 2 / (2.0 * risk_aversion))
        ) / risk_aversion

    return evaluate


def value_function(
    wealth: float, risk_aversion: float, optimal_consumption_rate: npt.ArrayLike
) -> Array:
    r"""Compute the value function.

    Parameters
    ----------
    wealth : float
        Positive initial wealth :math:`x`.
    risk_aversion : float
        Relative risk-aversion parameter :math:`R`, with ``R != 1``.
    optimal_consumption_rate : array_like
        Optimal consumption-to-wealth rate :math:`u(y)`.

    Returns
    -------
    ndarray
        Value function evaluated at ``wealth`` and at each factor state
        represented by ``optimal_consumption_rate``.
    """
    return (
        wealth ** (1.0 - risk_aversion)
        / (1.0 - risk_aversion)
        * np.asarray(optimal_consumption_rate) ** (-risk_aversion)
    )


def myopic_investment_fraction(
    risk_aversion: float,
    vol: Coefficient,
    market_price_of_risk: Coefficient,
) -> Coefficient:
    r"""Construct the myopic risky-asset fraction function.

    Parameters
    ----------
    risk_aversion : float
        Relative risk-aversion parameter :math:`R`.
    vol : Coefficient
        Risky-asset volatility :math:`\sigma(y)`.
    market_price_of_risk : Coefficient
        State-dependent market price of risk :math:`\lambda(y)`.

    Returns
    -------
    Coefficient
        Function that evaluates the myopic fraction of wealth invested in the
        risky asset elementwise on an array of factor states.
    """

    def evaluate(y: Array) -> Array:
        return market_price_of_risk(y) / (risk_aversion * vol(y))

    return evaluate


def optimal_investment_fraction(
    risk_aversion: float,
    vol: Coefficient,
    market_price_of_risk: Coefficient,
    correlation: float,
    factor_vol: Coefficient,
    grid: Array,
    optimal_consumption_rate: Array,
) -> Array:
    r"""Approximate the optimal risky-asset fraction on a uniform grid.

    Parameters
    ----------
    risk_aversion : float
        Relative risk-aversion parameter :math:`R`.
    vol : Coefficient
        Risky-asset volatility :math:`\sigma(y)`.
    market_price_of_risk : Coefficient
        State-dependent market price of risk :math:`\lambda(y)`.
    correlation : float
        Correlation :math:`\rho` between the Brownian motions driving the
        risky asset and the stochastic factor.
    factor_vol : Coefficient
        Factor volatility :math:`b(y)`.
    grid : ndarray of shape (N,)
        Uniform, strictly increasing factor grid with at least two points.
    optimal_consumption_rate : ndarray of shape (N,)
        Optimal consumption rate :math:`u(y)` on ``grid``.

    Returns
    -------
    ndarray of shape (N,)
        Fraction of wealth invested in the risky asset at each grid point.
    """
    h = grid[1] - grid[0]
    if grid.size >= 3:
        derivative = np.gradient(optimal_consumption_rate, h, edge_order=2)
    else:
        derivative = np.gradient(optimal_consumption_rate, h, edge_order=1)

    return (
        market_price_of_risk(grid)
        - risk_aversion
        * correlation
        * factor_vol(grid)
        * derivative
        / optimal_consumption_rate
    ) / (risk_aversion * vol(grid))
