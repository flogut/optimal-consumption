import sys

import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]
Factors = tuple[Array, Array]


@numba.njit(cache=True)
def factor(lower: Array, main: Array, upper: Array) -> Factors:
    """Factor a tridiagonal matrix with positive Thomas pivots.

    Parameters
    ----------
    lower : ndarray of shape (N - 1,)
        Lower diagonal of the matrix.
    main : ndarray of shape (N,)
        Main diagonal of the matrix. 
    upper : ndarray of shape (N - 1,)
        Upper diagonal of the matrix.

    Returns
    -------
    modified_upper : ndarray of shape (N - 1,)
        Upper diagonal after forward elimination, divided elementwise by the
        corresponding pivots.
    inverse_pivot : ndarray of shape (N,)
        Reciprocals of the Thomas-algorithm pivots.
    """
    n = main.size
    modified_upper = np.empty(n - 1)
    inverse_pivot = np.empty(n)

    pivot = main[0]
    if not np.isfinite(pivot) or pivot <= 0.0:
        raise ValueError("Tridiagonal matrix has a non-positive pivot.")
    inverse_pivot[0] = 1.0 / pivot

    if n == 1:
        return modified_upper, inverse_pivot

    modified_upper[0] = upper[0] * inverse_pivot[0]
    for i in range(1, n - 1):
        pivot = main[i] - lower[i - 1] * modified_upper[i - 1]
        if not np.isfinite(pivot) or pivot <= 0.0:
            raise ValueError("Tridiagonal matrix has a non-positive pivot.")
        inverse_pivot[i] = 1.0 / pivot
        modified_upper[i] = upper[i] * inverse_pivot[i]

    pivot = main[-1] - lower[-1] * modified_upper[-1]
    if not np.isfinite(pivot) or pivot <= 0.0:
        raise ValueError("Tridiagonal matrix has a non-positive pivot.")
    inverse_pivot[-1] = 1.0 / pivot

    return modified_upper, inverse_pivot


@numba.njit(cache=True)
def solve_factored_into(
    lower: Array,
    factors: Factors,
    rhs: Array,
    out: Array,
) -> None:
    """Solve a factored tridiagonal system into an output array.

    Parameters
    ----------
    lower : ndarray of shape (N - 1,)
        Lower diagonal used to construct the factorisation.
    factors : tuple of ndarray
        ``(modified_upper, inverse_pivot)`` returned by :func:`factor`, with
        shapes ``(N - 1,)`` and ``(N,)`` respectively.
    rhs : ndarray of shape (N,)
        Right-hand side of the linear system.
    out : ndarray of shape (N,)
        Output buffer.

    Returns
    -------
    None
    """
    modified_upper, inverse_pivot = factors
    n = inverse_pivot.size

    out[0] = rhs[0] * inverse_pivot[0]
    for i in range(1, n):
        out[i] = (rhs[i] - lower[i - 1] * out[i - 1]) * inverse_pivot[i]
    for i in range(n - 2, -1, -1):
        out[i] -= modified_upper[i] * out[i + 1]


@numba.njit(cache=True)
def solve_factored(lower: Array, factors: Factors, rhs: Array) -> Array:
    """Solve a system using a precomputed tridiagonal factorisation.

    Parameters
    ----------
    lower : ndarray of shape (N - 1,)
        Lower diagonal used to construct the factorisation.
    factors : tuple of ndarray
        ``(modified_upper, inverse_pivot)`` returned by :func:`factor`, with
        shapes ``(N - 1,)`` and ``(N,)`` respectively.
    rhs : ndarray of shape (N,)
        Right-hand side of the linear system.

    Returns
    -------
    ndarray of shape (N,)
    """
    out = np.empty(rhs.size)
    solve_factored_into(lower, factors, rhs, out)
    return out


@numba.njit(cache=True)
def solve(lower: Array, main: Array, upper: Array, rhs: Array) -> Array:
    """Factor and solve one tridiagonal system.

    Parameters
    ----------
    lower : ndarray of shape (N - 1,)
        Lower diagonal of the matrix.
    main : ndarray of shape (N,)
        Main diagonal of the matrix. The system must contain at least one
        row.
    upper : ndarray of shape (N - 1,)
        Upper diagonal of the matrix.
    rhs : ndarray of shape (N,)
        Right-hand side of the linear system.

    Returns
    -------
    ndarray of shape (N,)

    Examples
    --------
    Solve ``[[2, -1, 0], [-1, 2, -1], [0, -1, 2]] x = [1, 0, 1]``:

    >>> solve(
    ...     np.array([-1.0, -1.0]),
    ...     np.array([2.0, 2.0, 2.0]),
    ...     np.array([-1.0, -1.0]),
    ...     np.array([1.0, 0.0, 1.0]),
    ... )
    array([1., 1., 1.])
    """
    factors = factor(lower, main, upper)
    return solve_factored(lower, factors, rhs)


if __name__ == "__main__":
    import doctest

    result = doctest.testmod()
    sys.exit(1 if result.failed else 0)
