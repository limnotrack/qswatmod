"""
swatmf.outputs.recharge
========================
Read MODFLOW recharge output files and build grid-indexed DataFrames.

Mirrors the gridded-recharge reading logic in ``pyfolder/post_iii_rch.py``
without any QGIS/Qt dependency.  The spatial operations (joining values back
to a shapefile) are out of scope for this package; the functions here return
plain DataFrames that can be used for further analysis or passed to a
geo-library such as GeoPandas.

Output files handled
--------------------
* ``swatmf_out_MF_recharge``         — daily
* ``swatmf_out_MF_recharge_monthly`` — monthly
* ``swatmf_out_MF_recharge_yearly``  — yearly
"""

from __future__ import annotations

import datetime
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_SKIP_DAILY = ("MODFLOW", "--Calculated", "daily")
_SKIP_MONTHLY = ("Monthly",)
_SKIP_YEARLY = ("Yearly",)


def _read_recharge_file(
    path: str, skip_prefixes: tuple[str, ...]
) -> list[list[str]]:
    with open(path, "r") as fh:
        data = [
            x.strip()
            for x in fh
            if x.strip() and not x.strip().startswith(skip_prefixes)
        ]
    return [line.split() for line in data]


def _collect_grid_values(
    data: list[list[str]], start_idx: int, n_cells: int
) -> list[float]:
    """Extract *n_cells* float values starting at *data[start_idx]*."""
    values: list[float] = []
    idx = start_idx
    while len(values) < n_cells:
        for v in data[idx]:
            values.append(float(v))
            if len(values) >= n_cells:
                break
        idx += 1
    return values


# ---------------------------------------------------------------------------
# Date-list builders
# ---------------------------------------------------------------------------

def _daily_date_list(data: list[list[str]], startDate: str) -> list[str]:
    day_nums = [int(row[1]) for row in data if row and row[0] == "Day:"]
    sdate = datetime.datetime.strptime(startDate, "%m-%d-%Y")
    return [
        (sdate + datetime.timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
        for d in day_nums
    ]


def _monthly_date_list(data: list[list[str]], startDate: str) -> list[str]:
    months = [row for row in data if row and row[0] == "month:"]
    return (
        pd.date_range(startDate, periods=len(months), freq="ME")
        .strftime("%b-%Y")
        .tolist()
    )


def _yearly_date_list(data: list[list[str]], startDate: str) -> list[str]:
    years = [row for row in data if row and row[0] == "year:"]
    return (
        pd.date_range(startDate, periods=len(years), freq="YE")
        .strftime("%Y")
        .tolist()
    )


# ---------------------------------------------------------------------------
# Public: read recharge dates
# ---------------------------------------------------------------------------

def read_recharge_dates(
    swatmf_folder: str | os.PathLike,
    timescale: str = "Daily",
    start_date: Optional[str] = None,
) -> list[str]:
    """Return the list of date labels present in the recharge output file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    timescale : str, optional
        ``'Daily'``, ``'Monthly'``, or ``'Annual'``.  Default ``'Daily'``.
    start_date : str, optional
        Simulation start date (format ``"MM-DD-YYYY"``).  Required for
        ``'Monthly'`` and ``'Annual'``; for ``'Daily'`` it is used to convert
        Julian day numbers to calendar dates.

    Returns
    -------
    list of str
        Date label strings matching the timescale format.
    """
    wd = str(swatmf_folder)
    if timescale == "Daily":
        fname = "swatmf_out_MF_recharge"
        data = _read_recharge_file(os.path.join(wd, fname), _SKIP_DAILY)
        return _daily_date_list(data, start_date or "01-01-2000")
    elif timescale == "Monthly":
        fname = "swatmf_out_MF_recharge_monthly"
        data = _read_recharge_file(os.path.join(wd, fname), _SKIP_MONTHLY)
        return _monthly_date_list(data, start_date or "01-01-2000")
    elif timescale == "Annual":
        fname = "swatmf_out_MF_recharge_yearly"
        data = _read_recharge_file(os.path.join(wd, fname), _SKIP_YEARLY)
        return _yearly_date_list(data, start_date or "01-01-2000")
    else:
        raise ValueError(
            f"timescale must be 'Daily', 'Monthly', or 'Annual'; got {timescale!r}"
        )


# ---------------------------------------------------------------------------
# Public: get recharge grid DataFrame
# ---------------------------------------------------------------------------

def get_recharge(
    swatmf_folder: str | os.PathLike,
    n_cells: int,
    start_date: str,
    timescale: str = "Monthly",
    sdate: Optional[str] = None,
    edate: Optional[str] = None,
) -> pd.DataFrame:
    """Read gridded recharge values and return a wide DataFrame.

    Each row represents one MODFLOW grid cell; each column represents one
    time step.  The ``grid_id`` column (1-based) is included as the first
    column.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    n_cells : int
        Total number of MODFLOW grid cells (required to know when each
        time-step's block ends in the output file).
    start_date : str
        Simulation start date as ``"MM-DD-YYYY"``.
    timescale : str, optional
        ``'Daily'``, ``'Monthly'``, or ``'Annual'``.  Default ``'Monthly'``.
    sdate : str, optional
        First date label to include (inclusive).  ``None`` uses the earliest.
    edate : str, optional
        Last date label to include (inclusive).  ``None`` uses the latest.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_cells, n_dates + 1)``; first column is ``grid_id``.
    """
    wd = str(swatmf_folder)

    if timescale == "Daily":
        fname = os.path.join(wd, "swatmf_out_MF_recharge")
        data = _read_recharge_file(fname, _SKIP_DAILY)
        date_labels = _daily_date_list(data, start_date)
        header_prefix = "Day:"
        date_fmt_in = "%Y-%m-%d"
    elif timescale == "Monthly":
        fname = os.path.join(wd, "swatmf_out_MF_recharge_monthly")
        data = _read_recharge_file(fname, _SKIP_MONTHLY)
        date_labels = _monthly_date_list(data, start_date)
        header_prefix = "month:"
        date_fmt_in = "%b-%Y"
    elif timescale == "Annual":
        fname = os.path.join(wd, "swatmf_out_MF_recharge_yearly")
        data = _read_recharge_file(fname, _SKIP_YEARLY)
        date_labels = _yearly_date_list(data, start_date)
        header_prefix = "year:"
        date_fmt_in = "%Y"
    else:
        raise ValueError(
            f"timescale must be 'Daily', 'Monthly', or 'Annual'; got {timescale!r}"
        )

    # Apply date filter
    si = date_labels.index(sdate) if sdate else 0
    ei = date_labels.index(edate) + 1 if edate else len(date_labels)
    selected_dates = date_labels[si:ei]

    big_df = pd.DataFrame()
    for selected in selected_dates:
        # Find the starting line for this date in *data*
        start_idx = _find_block_start(data, header_prefix, selected, date_labels, date_fmt_in)
        values = _collect_grid_values(data, start_idx, n_cells)
        s = pd.Series(values, name=selected)
        big_df = pd.concat([big_df, s], axis=1)

    big_df = big_df.T
    big_df.columns = range(1, n_cells + 1)  # grid_id columns
    return big_df


def _find_block_start(
    data: list[list[str]],
    prefix: str,
    selected_date: str,
    date_labels: list[str],
    date_fmt_in: str,
) -> int:
    """Return the index of the data row *after* the header row for *selected_date*."""
    date_idx = date_labels.index(selected_date)
    # Count header rows seen so far
    header_count = 0
    for i, row in enumerate(data):
        if row and row[0] == prefix:
            if header_count == date_idx:
                return i + 1  # row after the header
            header_count += 1
    raise ValueError(f"Could not find block for {selected_date!r}")


# ---------------------------------------------------------------------------
# Convenience: average monthly recharge
# ---------------------------------------------------------------------------

def average_monthly_recharge(
    big_df: pd.DataFrame,
    date_fmt: str = "%b-%Y",
) -> pd.DataFrame:
    """Compute the mean recharge for each calendar month across all years.

    Parameters
    ----------
    big_df : pd.DataFrame
        Wide DataFrame with monthly date-string index and grid-cell columns.
    date_fmt : str, optional
        Format of the index labels.  Default ``'%b-%Y'``.

    Returns
    -------
    pd.DataFrame
        Shape ``(12, n_cells)``; index 1–12 (calendar month numbers).
    """
    df = big_df.copy()
    df.index = pd.to_datetime(df.index, format=date_fmt)
    return df.groupby(df.index.month).mean()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_recharge_timeseries(
    big_df: pd.DataFrame,
    figsize: tuple[float, float] = (12, 4),
    dark_theme: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the spatial mean recharge over time.

    Parameters
    ----------
    big_df : pd.DataFrame
        Wide DataFrame returned by :func:`get_recharge` (columns = grid cells).
    figsize : tuple, optional
        Figure size.
    dark_theme : bool, optional
        Dark background style.

    Returns
    -------
    (fig, ax)
    """
    plt.style.use("dark_background" if dark_theme else "default")
    fig, ax = plt.subplots(figsize=figsize)

    mean_rch = big_df.mean(axis=1)
    ax.plot(mean_rch.index, mean_rch.values, c="dodgerblue", lw=1.2)
    ax.set_ylabel(r"Mean Recharge $[mm/day]$", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Spatial Mean MODFLOW Recharge", fontsize=10, loc="left")
    fig.tight_layout()
    return fig, ax


def plot_avg_monthly_recharge(
    avg_df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4),
    dark_theme: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of average monthly recharge (spatial mean across all grid cells).

    Parameters
    ----------
    avg_df : pd.DataFrame
        Output of :func:`average_monthly_recharge`.
    figsize : tuple, optional
        Figure size.
    dark_theme : bool, optional
        Dark background style.

    Returns
    -------
    (fig, ax)
    """
    import calendar

    plt.style.use("dark_background" if dark_theme else "default")
    fig, ax = plt.subplots(figsize=figsize)

    monthly_mean = avg_df.mean(axis=1)
    ax.bar(
        [calendar.month_abbr[m] for m in monthly_mean.index],
        monthly_mean.values,
        color="dodgerblue",
        alpha=0.8,
    )
    ax.set_ylabel(r"Average Recharge $[mm/day]$", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Average Monthly MODFLOW Recharge", fontsize=10, loc="left")
    fig.tight_layout()
    return fig, ax
