"""
swatmf.outputs.rt3d
====================
Read and visualise RT3D concentration output produced by the SWAT-MODFLOW-RT3D
coupled model.

The coupled executable writes several result files that contain spatially
distributed concentrations and time-series of nutrient mass exchange between
the groundwater and surface-water systems.

Key output files
----------------
``swatmf_out_RT_cno3_monthly``
    Monthly averaged NO₃ concentration per grid cell.
``swatmf_out_RT_cno3_yearly``
    Yearly averaged NO₃ concentration per grid cell.
``swatmf_out_RT_cp_monthly``
    Monthly averaged P concentration per grid cell.
``swatmf_out_RT_cp_yearly``
    Yearly averaged P concentration per grid cell.
``swatmf_out_RT_rivno3``
    NO₃ mass flux [kg/day] at each GW-SW exchange cell (by MF grid cell).
``swatmf_out_SWAT_rivno3``
    NO₃ mass flux aggregated by SWAT subbasin.
``swatmf_out_RT_rivP``
    P mass flux at each GW-SW exchange cell.
``swatmf_out_SWAT_rivP``
    P mass flux aggregated by SWAT subbasin.
``swatmf_out_RT_rechno3``
    NO₃ recharge concentration at each grid cell.
``swatmf_out_SWAT_rechno3``
    NO₃ recharge concentration aggregated by SWAT subbasin.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_rt3d_conc(
    swatmf_folder: str | os.PathLike,
    species: str = "no3",
    freq: str = "monthly",
    start_date: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Read spatially-distributed concentration output from RT3D.

    Parameters
    ----------
    swatmf_folder : path-like
        SWAT-MODFLOW working directory.
    species : {'no3', 'p'}
        Which species to read.
    freq : {'monthly', 'yearly'}
        Temporal resolution of output file.
    start_date : str, optional
        Simulation start date (e.g. ``"2000-01-01"``).  If provided, the
        returned dictionary keys will be formatted dates; otherwise, the raw
        time labels from the file are used.

    Returns
    -------
    dict
        Mapping of ``{date_label: np.ndarray}``.  Each array has shape
        ``(nrow, ncol)`` (inferred from the file).
    """
    species_map = {"no3": "cno3", "p": "cp"}
    sp_key = species_map.get(species.lower(), species.lower())
    filename = f"swatmf_out_RT_{sp_key}_{freq}"

    path = os.path.join(str(swatmf_folder), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"RT3D concentration file not found: {path}"
        )

    # Determine header keyword based on frequency
    if freq == "monthly":
        skip_prefix = "Monthly"
        date_prefix = "month:"
    else:
        skip_prefix = "Yearly"
        date_prefix = "year:"

    # Parse the file
    dates: list[str] = []
    grids: list[list[list[float]]] = []
    current_grid: list[list[float]] = []

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(skip_prefix):
                continue
            if stripped.startswith(date_prefix):
                # Save previous grid if it has data
                if current_grid:
                    grids.append(current_grid)
                current_grid = []
                dates.append(stripped.split()[1])
            else:
                # Grid data row
                vals = [float(v) for v in stripped.split()]
                current_grid.append(vals)

    # Don't forget the last grid
    if current_grid:
        grids.append(current_grid)

    # Build date labels
    if start_date and dates:
        sd = pd.Timestamp(start_date)
        freq_code = "ME" if freq == "monthly" else "YE"
        date_labels = (
            pd.date_range(sd, periods=len(dates), freq=freq_code)
            .strftime("%Y-%m" if freq == "monthly" else "%Y")
            .tolist()
        )
    else:
        date_labels = dates

    # Convert to numpy arrays
    result = {}
    for lbl, grid in zip(date_labels, grids):
        result[lbl] = np.array(grid)

    return result


def read_rt3d_rivflux(
    swatmf_folder: str | os.PathLike,
    species: str = "no3",
    by: str = "grid",
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """Read GW-SW nutrient mass flux time series from RT3D output.

    Parameters
    ----------
    swatmf_folder : path-like
        SWAT-MODFLOW working directory.
    species : {'no3', 'p'}
        Which species to read.
    by : {'grid', 'subbasin'}
        Whether to read per-cell fluxes (``swatmf_out_RT_riv*``) or
        per-subbasin aggregated fluxes (``swatmf_out_SWAT_riv*``).
    start_date : str, optional
        If provided, used as the start for a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns correspond to cells/subbasins; rows correspond to time
        steps.
    """
    sp_suffix = "no3" if species.lower() == "no3" else "P"
    if by == "grid":
        filename = f"swatmf_out_RT_riv{sp_suffix}"
    else:
        filename = f"swatmf_out_SWAT_riv{sp_suffix}"

    path = os.path.join(str(swatmf_folder), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"RT3D river flux file not found: {path}"
        )

    # Read all numeric rows (skip header lines starting with text)
    rows = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Try parsing as numbers
            try:
                vals = [float(v) for v in stripped.split()]
                rows.append(vals)
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    if start_date:
        df.index = pd.date_range(start_date, periods=len(df), freq="D")
    df.columns = [f"cell_{i+1}" if by == "grid" else f"sub_{i+1}"
                  for i in range(df.shape[1])]
    return df


def read_rt3d_recharge_conc(
    swatmf_folder: str | os.PathLike,
    species: str = "no3",
    by: str = "grid",
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """Read recharge nutrient concentration output from RT3D.

    Parameters
    ----------
    swatmf_folder : path-like
        SWAT-MODFLOW working directory.
    species : {'no3', 'p'}
        Which species.
    by : {'grid', 'subbasin'}
        Per-cell or per-subbasin.
    start_date : str, optional
        Start date for index.

    Returns
    -------
    pd.DataFrame
    """
    sp_suffix = "no3" if species.lower() == "no3" else "P"
    if by == "grid":
        filename = f"swatmf_out_RT_rech{sp_suffix}"
    else:
        filename = f"swatmf_out_SWAT_rech{sp_suffix}"

    path = os.path.join(str(swatmf_folder), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"RT3D recharge file not found: {path}"
        )

    rows = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                vals = [float(v) for v in stripped.split()]
                rows.append(vals)
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    if start_date:
        df.index = pd.date_range(start_date, periods=len(df), freq="D")
    df.columns = [f"cell_{i+1}" if by == "grid" else f"sub_{i+1}"
                  for i in range(df.shape[1])]
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_conc_map(
    conc_array: np.ndarray,
    title: str = "Concentration",
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a 2-D concentration grid as a heat map.

    Parameters
    ----------
    conc_array : np.ndarray
        2-D array (nrow × ncol) of concentration values.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colorbar limits.
    ax : matplotlib Axes, optional
        If provided, plot on this axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(conc_array, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, label="Concentration [mg/L]")
    return ax


def plot_rivflux_ts(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    species: str = "NO₃",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot time series of GW-SW nutrient mass exchange.

    Parameters
    ----------
    df : pd.DataFrame
        Output from :func:`read_rt3d_rivflux`.
    columns : list, optional
        Subset of columns to plot.  Default: first 5.
    species : str
        Species name for the y-axis label.
    ax : matplotlib Axes, optional
        Existing axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    cols = columns or list(df.columns[:5])
    df[cols].plot(ax=ax, linewidth=0.8)
    ax.set_ylabel(f"{species} flux [kg/day]")
    ax.set_xlabel("Date")
    ax.set_title(f"GW-SW {species} Exchange")
    ax.legend(fontsize=8, loc="upper right")
    return ax
