"""
swatmf.epm
===========
Exponential Piston Flow Model (EPM) for groundwater transit-time lag correction.

Based on Maloszewski & Zuber (1982).  Parameterised with tritium-derived
catchment data for the Lake Rotorua catchment (Puarenga and neighbouring
streams).

Typical use
-----------
Apply to RT3D concentration outputs so that simulated stream concentrations
reflect the full travel-time distribution through both the vadose zone and
the saturated zone — rather than the instantaneous recharge-to-stream
assumption implicit in RT3D.

Because the EPM is a *linear* filter, applying it to the output is
mathematically equivalent to applying it to the source term.  The result is
the concentration a sampler would actually observe at the stream gauge.

Quick start
-----------
>>> from swatmf.epm import ROTORUA_EPM, apply_epm_lag
>>> params = ROTORUA_EPM["Puarenga"]
>>> lagged = apply_epm_lag(simulated_ts, params, dt_yr=1/365.25)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EPMParams:
    """Parameters for a (binary) Exponential Piston Flow Model.

    Attributes
    ----------
    T1, f1 : float
        Mean residence time [years] and exponential fraction for component 1.
    frac1 : float
        Fraction of total discharge from component 1.  ``1 - frac1`` goes to
        component 2.  Set to 1.0 for a single-component model.
    T2, f2 : float or None
        Parameters for the optional second (fast / shallow) component.
    stream : str
        Label — stream name used for the calibration.
    avg_mrt : float or None
        Weighted average MRT [years] (informational only).
    """
    T1:       float
    f1:       float
    frac1:    float      = 1.0
    T2:       Optional[float] = None
    f2:       Optional[float] = None
    stream:   str        = ""
    avg_mrt:  Optional[float] = None

    @property
    def is_binary(self) -> bool:
        return self.T2 is not None and self.frac1 < 1.0

    @property
    def weighted_mrt(self) -> float:
        if self.is_binary:
            return self.frac1 * self.T1 + (1 - self.frac1) * self.T2
        return self.T1


# ---------------------------------------------------------------------------
# Rotorua catchment — Table 2 parameters (tritium-derived)
# Puarenga is the example stream for this repository.
# ---------------------------------------------------------------------------

ROTORUA_EPM: dict[str, EPMParams] = {
    "Hamurana":   EPMParams(T1=185, f1=0.82, frac1=0.65, T2=12, f2=0.77,
                            stream="Hamurana",   avg_mrt=125),
    "Awahou":     EPMParams(T1=80,  f1=1.00, frac1=0.92, T2=6,  f2=0.91,
                            stream="Awahou",     avg_mrt=75),
    "Waiteti":    EPMParams(T1=60,  f1=1.00, frac1=0.78, T2=3,  f2=0.90,
                            stream="Waiteti",    avg_mrt=45),
    "Ngongotaha": EPMParams(T1=35,  f1=1.00, frac1=0.82, T2=1,  f2=0.91,
                            stream="Ngongotaha", avg_mrt=30),
    "Waiowhiro":  EPMParams(T1=40,  f1=0.63, frac1=1.00,
                            stream="Waiowhiro",  avg_mrt=40),
    "Utuhina":    EPMParams(T1=85,  f1=0.60, frac1=0.70, T2=1,  f2=1.00,
                            stream="Utuhina",    avg_mrt=60),
    "Puarenga":   EPMParams(T1=44,  f1=1.00, frac1=0.95, T2=2,  f2=1.00,
                            stream="Puarenga",   avg_mrt=42),
    "Waingaehe":  EPMParams(T1=160, f1=0.94, frac1=0.90, T2=3,  f2=1.00,
                            stream="Waingaehe",  avg_mrt=145),
    "Waiohewa":   EPMParams(T1=55,  f1=1.00, frac1=0.75, T2=1,  f2=1.00,
                            stream="Waiohewa",   avg_mrt=40),
}


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------

def epm_kernel(
    T: float,
    f: float,
    tau_max_yr: Optional[float] = None,
    dt_yr: float = 1 / 365.25,
    lambda_decay: float = 0.0,
) -> np.ndarray:
    """Compute a discrete EPM transit-time kernel g(τ).

    The kernel is normalised so it sums to 1 (after trapezoidal integration).
    Multiply by ``exp(-lambda * tau)`` for a decaying tracer (e.g. tritium).

    Parameters
    ----------
    T : float
        Mean residence time [years].
    f : float
        Exponential fraction (0 < f ≤ 1).  ``f=1`` → pure exponential;
        ``f→0`` → pure piston flow.
    tau_max_yr : float, optional
        Upper integration limit [years].  Defaults to ``5 * T``.
    dt_yr : float, optional
        Time step [years].  For daily data use ``1/365.25`` (default).
    lambda_decay : float, optional
        Radioactive decay constant [1/year].  Zero for conservative solutes.

    Returns
    -------
    np.ndarray
        1-D array of kernel weights, length ``ceil(tau_max_yr / dt_yr) + 1``.
    """
    if tau_max_yr is None:
        tau_max_yr = 5.0 * T

    tau = np.arange(0.0, tau_max_yr + dt_yr, dt_yr)
    tau_min = T * (1.0 - f)
    Tf = T * f

    g = np.where(
        tau < tau_min,
        0.0,
        (1.0 / Tf) * np.exp(-(tau / Tf) + (1.0 / f) - 1.0),
    )

    if lambda_decay > 0:
        g = g * np.exp(-lambda_decay * tau)

    # Normalise via trapezoidal rule so weights sum to 1
    norm = np.trapezoid(g, tau)
    if norm > 0:
        g /= norm

    return g


def binary_epm_kernel(
    params: EPMParams,
    tau_max_yr: Optional[float] = None,
    dt_yr: float = 1 / 365.25,
    lambda_decay: float = 0.0,
) -> np.ndarray:
    """Compute a combined binary EPM kernel from an :class:`EPMParams` object.

    For a single-component model (``frac1 == 1.0`` or ``T2 is None``), this
    is identical to :func:`epm_kernel` with component-1 parameters.

    Returns
    -------
    np.ndarray
        Combined, normalised kernel weights.
    """
    if tau_max_yr is None:
        tau_max_yr = 5.0 * (params.T2 or params.T1)
        tau_max_yr = max(tau_max_yr, 5.0 * params.T1)

    k1 = epm_kernel(params.T1, params.f1, tau_max_yr, dt_yr, lambda_decay)

    if not params.is_binary:
        return k1

    k2 = epm_kernel(params.T2, params.f2, tau_max_yr, dt_yr, lambda_decay)

    # Pad shorter kernel to equal length
    n = max(len(k1), len(k2))
    k1 = np.pad(k1, (0, n - len(k1)))
    k2 = np.pad(k2, (0, n - len(k2)))

    combined = params.frac1 * k1 + (1.0 - params.frac1) * k2

    # Re-normalise
    dt = dt_yr
    tau = np.arange(len(combined)) * dt
    norm = np.trapezoid(combined, tau)
    if norm > 0:
        combined /= norm

    return combined


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

def apply_epm_lag(
    series: pd.Series,
    params: EPMParams,
    dt_yr: float = 1 / 365.25,
    tau_max_yr: Optional[float] = None,
    lambda_decay: float = 0.0,
    pad_mode: str = "edge",
) -> pd.Series:
    """Apply an EPM lag filter to a single concentration time series.

    Parameters
    ----------
    series : pd.Series
        Daily (or other regular) concentration values.  Index is preserved.
    params : EPMParams
        EPM parameters for this location.
    dt_yr : float, optional
        Time step in years.  Default ``1/365.25`` (daily).
    tau_max_yr : float, optional
        Truncation lag in years.  Defaults to ``5 * max(T1, T2 or T1)``.
    lambda_decay : float, optional
        Decay constant [yr⁻¹].  Zero for conservative solutes (default).
    pad_mode : str, optional
        How to fill values before the start of the record when the kernel
        extends back further than the series.  ``"edge"`` (default) repeats
        the first value; ``"mean"`` uses the series mean.

    Returns
    -------
    pd.Series
        Lagged concentration series, same length and index as *series*.
    """
    kernel = binary_epm_kernel(params, tau_max_yr=tau_max_yr,
                                dt_yr=dt_yr, lambda_decay=lambda_decay)
    n_pad = len(kernel) - 1
    values = series.values.astype(float)

    if pad_mode == "mean":
        pad_val = np.nanmean(values[:min(365, len(values))])
    else:
        pad_val = values[0] if len(values) > 0 else 0.0

    padded = np.concatenate([np.full(n_pad, pad_val), values])
    lagged = fftconvolve(padded, kernel, mode="full")[n_pad: n_pad + len(values)]

    return pd.Series(lagged, index=series.index, name=series.name)


def apply_epm_lag_df(
    df: pd.DataFrame,
    params: EPMParams,
    dt_yr: float = 1 / 365.25,
    tau_max_yr: Optional[float] = None,
    lambda_decay: float = 0.0,
    pad_mode: str = "edge",
) -> pd.DataFrame:
    """Apply EPM lag to every column of a concentration DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = time steps, columns = spatial units (cells, subbasins, HRUs).
    params : EPMParams
        Single set of EPM parameters applied uniformly to all columns.
        For per-column parameters use :func:`apply_epm_lag_spatial`.

    Returns
    -------
    pd.DataFrame
        Same shape as *df*.
    """
    kernel = binary_epm_kernel(params, tau_max_yr=tau_max_yr,
                                dt_yr=dt_yr, lambda_decay=lambda_decay)
    n_pad = len(kernel) - 1
    arr = df.values.astype(float)

    if pad_mode == "mean":
        pad_row = np.nanmean(arr[:min(365, len(arr))], axis=0)
    else:
        pad_row = arr[0].copy()

    pad_block = np.tile(pad_row, (n_pad, 1))
    padded = np.vstack([pad_block, arr])

    # FFT-convolve all columns at once: convolve each column by broadcasting
    # the kernel in frequency space — O((N+M)·log(N+M)) vs O(N·M) per column.
    n_fft = len(padded) + len(kernel) - 1
    padded_fft = np.fft.rfft(padded, n=n_fft, axis=0)
    kernel_fft = np.fft.rfft(kernel, n=n_fft)
    conv_all   = np.fft.irfft(padded_fft * kernel_fft[:, None], n=n_fft, axis=0)
    out = conv_all[n_pad: n_pad + len(arr)]

    return pd.DataFrame(out, index=df.index, columns=df.columns)


def apply_epm_lag_spatial(
    df: pd.DataFrame,
    col_params: dict[str, EPMParams],
    dt_yr: float = 1 / 365.25,
    tau_max_yr: Optional[float] = None,
    default_params: Optional[EPMParams] = None,
) -> pd.DataFrame:
    """Apply per-column EPM parameters to a concentration DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = time steps; columns are spatial unit identifiers.
    col_params : dict
        Mapping of column name → :class:`EPMParams`.
    default_params : EPMParams, optional
        Fallback parameters for columns not in *col_params*.  If ``None``,
        columns without an explicit mapping are left unchanged.

    Returns
    -------
    pd.DataFrame
        Same shape as *df*, each column lagged with its own kernel.
    """
    out = df.copy()
    for col in df.columns:
        p = col_params.get(col, default_params)
        if p is None:
            continue
        out[col] = apply_epm_lag(df[col], p, dt_yr=dt_yr,
                                  tau_max_yr=tau_max_yr)
    return out


# ---------------------------------------------------------------------------
# Spatial assignment: stream EPM params → MODFLOW subbasin columns
# ---------------------------------------------------------------------------

def assign_stream_params(
    subbasin_stream_map: dict[int, str],
    epm_table: dict[str, EPMParams] = ROTORUA_EPM,
    col_prefix: str = "sub_",
) -> dict[str, EPMParams]:
    """Map EPM stream parameters to DataFrame column names.

    Parameters
    ----------
    subbasin_stream_map : dict
        Mapping of subbasin ID (int) → stream name (str) from *epm_table*.
    epm_table : dict, optional
        Lookup table of stream → EPMParams.  Defaults to ``ROTORUA_EPM``.
    col_prefix : str, optional
        Column prefix used in the concentration DataFrame (e.g. ``"sub_"``).

    Returns
    -------
    dict
        Column name (e.g. ``"sub_3"``) → :class:`EPMParams`.

    Examples
    --------
    >>> mapping = {1: "Puarenga", 2: "Puarenga", 3: "Utuhina"}
    >>> col_params = assign_stream_params(mapping)
    """
    col_params: dict[str, EPMParams] = {}
    missing: list[int] = []
    for sub_id, stream in subbasin_stream_map.items():
        col = f"{col_prefix}{sub_id}"
        if stream in epm_table:
            col_params[col] = epm_table[stream]
        else:
            missing.append(sub_id)
    if missing:
        warnings.warn(
            f"No EPM parameters found for subbasins: {missing}. "
            "Those columns will not be lagged.",
            stacklevel=2,
        )
    return col_params


# ---------------------------------------------------------------------------
# Convenience: lag RT3D rivflux output (the most common use case)
# ---------------------------------------------------------------------------

def lag_rivflux(
    flux_df: pd.DataFrame,
    params: EPMParams,
    dt_yr: float = 1 / 365.25,
) -> pd.DataFrame:
    """Apply EPM lag to a GW-SW flux DataFrame from :func:`read_rt3d_rivflux`.

    This adjusts the *timing* of nutrient arrival at the stream to reflect
    the transit-time distribution through the groundwater system.  Magnitudes
    are preserved; only the temporal distribution is shifted.

    Parameters
    ----------
    flux_df : pd.DataFrame
        Daily GW-SW flux (output of ``read_rt3d_rivflux``).
    params : EPMParams
        EPM parameters for the catchment / stream.

    Returns
    -------
    pd.DataFrame
        Same shape; DatetimeIndex preserved.
    """
    return apply_epm_lag_df(flux_df, params, dt_yr=dt_yr)


def lag_recharge_conc(
    rech_df: pd.DataFrame,
    params: EPMParams,
    dt_yr: float = 1 / 365.25,
) -> pd.DataFrame:
    """Apply EPM lag to recharge concentration output from RT3D.

    Equivalent to :func:`lag_rivflux` but named for the recharge context.
    """
    return apply_epm_lag_df(rech_df, params, dt_yr=dt_yr)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def kernel_summary(params: EPMParams, dt_yr: float = 1 / 365.25) -> dict:
    """Return summary statistics for the combined EPM kernel.

    Returns
    -------
    dict with keys ``peak_lag_yr``, ``median_lag_yr``, ``mean_lag_yr``,
    ``kernel_length_yr``, ``weighted_mrt_yr``.
    """
    kernel = binary_epm_kernel(params, dt_yr=dt_yr)
    tau = np.arange(len(kernel)) * dt_yr
    cum = np.cumsum(kernel) * dt_yr

    peak_lag   = tau[np.argmax(kernel)]
    median_lag = tau[np.searchsorted(cum / cum[-1], 0.5)]
    mean_lag   = np.sum(tau * kernel) * dt_yr

    return {
        "peak_lag_yr":      round(float(peak_lag), 2),
        "median_lag_yr":    round(float(median_lag), 2),
        "mean_lag_yr":      round(float(mean_lag), 2),
        "kernel_length_yr": round(float(tau[-1]), 1),
        "weighted_mrt_yr":  round(params.weighted_mrt, 1),
    }


def plot_kernel(
    params: EPMParams,
    dt_yr: float = 1 / 365.25,
    ax=None,
    label: Optional[str] = None,
    **plot_kwargs,
):
    """Plot the combined EPM transit-time kernel.

    Parameters
    ----------
    params : EPMParams
    ax : matplotlib Axes, optional
    label : str, optional
        Legend label.  Defaults to ``params.stream`` or ``"EPM kernel"``.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    kernel = binary_epm_kernel(params, dt_yr=dt_yr)
    tau = np.arange(len(kernel)) * dt_yr

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    lbl = label or params.stream or "EPM kernel"
    ax.plot(tau, kernel / kernel.max(), label=lbl, **plot_kwargs)

    stats = kernel_summary(params, dt_yr)
    ax.axvline(stats["mean_lag_yr"],   ls="--", lw=0.8, color="grey",
               label=f"mean lag {stats['mean_lag_yr']:.0f} yr")
    ax.axvline(stats["median_lag_yr"], ls=":",  lw=0.8, color="grey",
               label=f"median lag {stats['median_lag_yr']:.0f} yr")

    ax.set_xlabel("Transit time (years)")
    ax.set_ylabel("Normalised g(τ)")
    ax.set_title(f"EPM transit-time distribution — {lbl}")
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Comparison: RT3D output vs EPM-corrected signal
# ---------------------------------------------------------------------------

def compare_epm_vs_rt3d(
    flux_df: pd.DataFrame,
    params: EPMParams,
    freq: str = "ME",
    subbasins: Optional[list] = None,
    figsize: tuple = (13, 7),
) -> tuple:
    """Compare raw RT3D GW-SW flux against the EPM-lag-corrected prediction.

    The raw RT3D output assumes instantaneous vadose-zone transit.  The
    EPM-corrected signal spreads each day's recharge over the transit-time
    distribution, showing when that N actually arrives at the stream.

    Parameters
    ----------
    flux_df : pd.DataFrame
        Daily GW-SW flux from ``read_rt3d_rivflux(..., by='subbasin')``.
        Columns are ``sub_1``, ``sub_2``, … ; DatetimeIndex.
    params : EPMParams
        EPM parameters for the catchment.
    freq : str, optional
        Pandas resample frequency for plotting.  ``"ME"`` (monthly, default)
        or ``"YE"`` (annual).
    subbasins : list of str, optional
        Column names to include in the catchment total.  ``None`` = all.
    figsize : tuple, optional

    Returns
    -------
    (fig, axes, summary_df)
        *summary_df* has columns ``rt3d``, ``epm_corrected``, ``difference``
        at the requested frequency.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    cols = subbasins if subbasins is not None else flux_df.columns.tolist()
    cols = [c for c in cols if c in flux_df.columns]

    # Catchment totals (sum across subbasins, resample to requested freq)
    raw_total   = flux_df[cols].sum(axis=1).resample(freq).mean()
    lagged_df   = apply_epm_lag_df(flux_df[cols], params)
    lag_total   = lagged_df.sum(axis=1).resample(freq).mean()
    difference  = lag_total - raw_total

    stats = kernel_summary(params)
    label = params.stream or "catchment"

    fig, axes = plt.subplots(3, 1, figsize=figsize,
                             sharex=True, constrained_layout=True)

    freq_label = "Monthly mean" if freq == "ME" else "Annual mean"

    # ── Panel 1: raw vs EPM-corrected ────────────────────────────────────────
    ax = axes[0]
    ax.plot(raw_total.index,  raw_total.values,  lw=1.2,
            color="steelblue", label="RT3D (no vadose lag)")
    ax.plot(lag_total.index,  lag_total.values,  lw=1.2,
            color="darkorange", label=f"EPM-corrected (MRT={stats['weighted_mrt_yr']:.0f} yr)")
    ax.set_ylabel("GW→SW NO₃ flux (kg/day)")
    ax.set_title(f"{label} — RT3D vs EPM-corrected GW-SW flux  [{freq_label}]")
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls="--")

    # ── Panel 2: difference (EPM - raw) ──────────────────────────────────────
    ax = axes[1]
    ax.fill_between(difference.index, difference.values, 0,
                    where=difference.values >= 0,
                    color="darkorange", alpha=0.5, label="EPM > RT3D (delayed arrival)")
    ax.fill_between(difference.index, difference.values, 0,
                    where=difference.values < 0,
                    color="steelblue",  alpha=0.5, label="EPM < RT3D (signal not yet arrived)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Difference (kg/day)")
    ax.set_title("EPM-corrected minus RT3D  (positive = more N arriving than RT3D predicts)")
    ax.legend(fontsize=9)

    # ── Panel 3: EPM kernel for reference ────────────────────────────────────
    ax = axes[2]
    kernel = binary_epm_kernel(params)
    tau    = np.arange(len(kernel)) / 365.25
    ax.fill_between(tau, kernel / kernel.max(), alpha=0.35, color="green")
    ax.plot(tau, kernel / kernel.max(), lw=1, color="green")
    ax.axvline(stats["mean_lag_yr"],   ls="--", lw=1, color="grey",
               label=f"mean {stats['mean_lag_yr']:.0f} yr")
    ax.axvline(stats["median_lag_yr"], ls=":",  lw=1, color="grey",
               label=f"median {stats['median_lag_yr']:.0f} yr")
    sim_yrs = (flux_df.index[-1] - flux_df.index[0]).days / 365.25
    ax.axvline(sim_yrs, ls="-", lw=1.2, color="red",
               label=f"simulation length {sim_yrs:.0f} yr")
    ax.set_xlabel("Transit time (years)")
    ax.set_ylabel("Normalised g(τ)")
    ax.set_title("EPM transit-time distribution  "
                 "(fraction of kernel covered by simulation shown in red)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(tau[-1], max(stats["mean_lag_yr"] * 4, sim_yrs * 1.2)))

    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))

    summary_df = pd.DataFrame({
        "rt3d":          raw_total,
        "epm_corrected": lag_total,
        "difference":    difference,
    })

    return fig, axes, summary_df


def epm_coverage_fraction(params: EPMParams, sim_years: float) -> float:
    """Return the fraction of the EPM kernel covered by the simulation length.

    A value < 1 means the simulation is too short to capture the full
    transit-time distribution — concentrations at the end of the run are
    still building up toward their equilibrium value.

    Parameters
    ----------
    params : EPMParams
    sim_years : float
        Length of simulation in years.

    Returns
    -------
    float
        Fraction of kernel mass (0–1) within *sim_years*.
    """
    dt = 1 / 365.25
    kernel = binary_epm_kernel(params, dt_yr=dt)
    tau    = np.arange(len(kernel)) * dt
    cum    = np.cumsum(kernel) * dt
    norm   = cum[-1]
    idx    = np.searchsorted(tau, sim_years)
    return float(cum[min(idx, len(cum) - 1)] / norm)
