"""
swatmf.metrics
==============
Objective functions for comparing simulated and observed time series.

All functions accept 1-D NumPy arrays (or anything convertible via
``numpy.asarray``).  They mirror the ``ObjFns`` class in
``pyfolder/utils.py`` but are exposed as plain module-level functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def nse(sims: ArrayLike, obds: ArrayLike) -> float:
    """Nash-Sutcliffe Efficiency (NSE).

    .. math::

        E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}(e_i - s_i)^2}
        {\\sum_{i=1}^{N}(e_i - \\bar{e})^2}

    Parameters
    ----------
    sims : array-like
        Simulated values.
    obds : array-like
        Observed values.

    Returns
    -------
    float
    """
    s = np.asarray(sims, dtype=np.float64)
    o = np.asarray(obds, dtype=np.float64)
    return float(
        1.0
        - np.sum((o - s) ** 2)
        / np.sum((o - np.mean(o)) ** 2)
    )


def rmse(sims: ArrayLike, obds: ArrayLike) -> float:
    """Root Mean Square Error (RMSE).

    .. math::

        E_{\\text{RMSE}} = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_i - s_i)^2}

    Parameters
    ----------
    sims : array-like
        Simulated values.
    obds : array-like
        Observed values.

    Returns
    -------
    float
    """
    s = np.asarray(sims, dtype=np.float64)
    o = np.asarray(obds, dtype=np.float64)
    return float(np.sqrt(np.mean((o - s) ** 2)))


def pbias(sims: ArrayLike, obds: ArrayLike) -> float:
    """Percent Bias (PBias).

    .. math::

        E_{\\text{PBias}} = 100 \\times
        \\frac{\\sum_{i=1}^{N}(e_i - s_i)}{\\sum_{i=1}^{N} e_i}

    Parameters
    ----------
    sims : array-like
        Simulated values.
    obds : array-like
        Observed values.

    Returns
    -------
    float
    """
    s = np.asarray(sims, dtype=np.float64)
    o = np.asarray(obds, dtype=np.float64)
    return float(100.0 * np.sum(o - s) / np.sum(o))


def rsq(sims: ArrayLike, obds: ArrayLike) -> float:
    """Coefficient of determination (R²).

    Parameters
    ----------
    sims : array-like
        Simulated values.
    obds : array-like
        Observed values.

    Returns
    -------
    float
    """
    s = np.asarray(sims, dtype=np.float64)
    o = np.asarray(obds, dtype=np.float64)
    numerator = np.sum((o - o.mean()) * (s - s.mean())) ** 2
    denominator = np.sum((o - o.mean()) ** 2) * np.sum((s - s.mean()) ** 2)
    return float(numerator / denominator)


def all_metrics(sims: ArrayLike, obds: ArrayLike) -> dict[str, float]:
    """Compute NSE, RMSE, PBias and R² in one call.

    Parameters
    ----------
    sims : array-like
        Simulated values.
    obds : array-like
        Observed values.

    Returns
    -------
    dict with keys ``nse``, ``rmse``, ``pbias``, ``rsq``.
    """
    return {
        "nse": nse(sims, obds),
        "rmse": rmse(sims, obds),
        "pbias": pbias(sims, obds),
        "rsq": rsq(sims, obds),
    }
