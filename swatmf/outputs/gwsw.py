"""
swatmf.outputs.gwsw
====================
Read MODFLOW groundwater-surface-water exchange output files.

Mirrors the time-series reading logic in ``pyfolder/post_iv_gwsw.py``
without any QGIS/Qt dependency.  Spatial bar plots that require shapefile
geometry are not reproduced here; instead the functions return DataFrames
that can be visualised with standard Matplotlib or GeoPandas.

Output files handled
--------------------
* ``swatmf_out_MF_gwsw``         — daily
* ``swatmf_out_MF_gwsw_monthly`` — monthly
* ``swatmf_out_MF_gwsw_yearly``  — yearly
"""

from __future__ import annotations

import datetime
import os
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_SKIP = ("for", "Positive:", "Negative:", "Daily", "Monthly", "Annual", "Layer,")


def _read_gwsw_file(path: str) -> list[list[str]]:
    with open(path, "r") as fh:
        data = [
            x.strip()
            for x in fh
            if x.strip() and not x.strip().startswith(_SKIP)
        ]
    return [line.split() for line in data]


# ---------------------------------------------------------------------------
# Date-list builders
# ---------------------------------------------------------------------------

def _daily_date_list(data: list[list[str]], start_date: str) -> list[str]:
    day_nums = [int(row[1]) for row in data if row and row[0] == "Day:"]
    sdate = datetime.datetime.strptime(start_date, "%m-%d-%Y")
    return [
        (sdate + datetime.timedelta(days=int(d) - 1)).strftime("%m-%d-%Y")
        for d in day_nums
    ]


def _monthly_date_list(data: list[list[str]], start_date: str) -> list[str]:
    months = [row for row in data if row and row[0] == "month:"]
    return (
        pd.date_range(start_date, periods=len(months), freq="ME")
        .strftime("%b-%Y")
        .tolist()
    )


def _yearly_date_list(data: list[list[str]], start_date: str) -> list[str]:
    years = [row for row in data if row and row[0] == "year:"]
    return (
        pd.date_range(start_date, periods=len(years), freq="YE")
        .strftime("%Y")
        .tolist()
    )


# ---------------------------------------------------------------------------
# Block locator
# ---------------------------------------------------------------------------

def _find_block_start(
    data: list[list[str]],
    prefix: str,
    date_idx: int,
) -> int:
    count = 0
    for i, row in enumerate(data):
        if row and row[0] == prefix:
            if count == date_idx:
                return i + 1
            count += 1
    raise ValueError(f"Could not find block #{date_idx} for prefix {prefix!r}")


def _detect_n_riv_cells(data: list[list[str]], prefix: str) -> int:
    """Count actual data rows between the first two time-step headers.

    Returns 0 if the file has no data rows (e.g. swatmf_river2grid.txt
    was written with 0 river cells).
    """
    markers = [i for i, row in enumerate(data) if row and row[0] == prefix]
    if len(markers) < 2:
        # Only one block — count rows until end of list
        start = markers[0] + 1 if markers else 0
        return sum(1 for row in data[start:] if row and row[0] != prefix)
    start = markers[0] + 1
    end   = markers[1]
    return end - start


# ---------------------------------------------------------------------------
# Public: read GW-SW exchange dates
# ---------------------------------------------------------------------------

def read_gwsw_dates(
    swatmf_folder: str | os.PathLike,
    timescale: str = "Daily",
    start_date: Optional[str] = None,
) -> list[str]:
    """Return the list of date labels available in the GW-SW exchange file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    timescale : str, optional
        ``'Daily'``, ``'Monthly'``, or ``'Annual'``.
    start_date : str, optional
        Simulation start date as ``"MM-DD-YYYY"``.

    Returns
    -------
    list of str
    """
    wd = str(swatmf_folder)
    sd = start_date or "01-01-2000"
    if timescale == "Daily":
        data = _read_gwsw_file(os.path.join(wd, "swatmf_out_MF_gwsw"))
        return _daily_date_list(data, sd)
    elif timescale == "Monthly":
        data = _read_gwsw_file(os.path.join(wd, "swatmf_out_MF_gwsw_monthly"))
        return _monthly_date_list(data, sd)
    elif timescale == "Annual":
        data = _read_gwsw_file(os.path.join(wd, "swatmf_out_MF_gwsw_yearly"))
        return _yearly_date_list(data, sd)
    else:
        raise ValueError(
            f"timescale must be 'Daily', 'Monthly', or 'Annual'; got {timescale!r}"
        )


# ---------------------------------------------------------------------------
# Public: get GW-SW exchange time series
# ---------------------------------------------------------------------------

def get_gwsw(
    swatmf_folder: str | os.PathLike,
    n_riv_cells: int,
    start_date: str,
    timescale: str = "Daily",
    sdate: Optional[str] = None,
    edate: Optional[str] = None,
) -> pd.DataFrame:
    """Read GW-SW exchange values and return a time-indexed DataFrame.

    Each row is a time step; each column is a river cell (0-based index
    matching the order in the output file).  Positive values indicate
    flux from aquifer to stream; negative values indicate stream losses.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    n_riv_cells : int
        Number of river cells (rows per time-step block in the file).
    start_date : str
        Simulation start date as ``"MM-DD-YYYY"``.
    timescale : str, optional
        ``'Daily'``, ``'Monthly'``, or ``'Annual'``.
    sdate : str, optional
        First date label to include (inclusive).
    edate : str, optional
        Last date label to include (inclusive).

    Returns
    -------
    pd.DataFrame
        Shape ``(n_dates, n_riv_cells)``; index is the date label strings.
    """
    wd = str(swatmf_folder)
    if timescale == "Daily":
        data = _read_gwsw_file(os.path.join(wd, "swatmf_out_MF_gwsw"))
        date_labels = _daily_date_list(data, start_date)
        prefix = "Day:"
    elif timescale == "Monthly":
        data = _read_gwsw_file(os.path.join(wd, "swatmf_out_MF_gwsw_monthly"))
        date_labels = _monthly_date_list(data, start_date)
        prefix = "month:"
    elif timescale == "Annual":
        data = _read_gwsw_file(os.path.join(wd, "swatmf_out_MF_gwsw_yearly"))
        date_labels = _yearly_date_list(data, start_date)
        prefix = "year:"
    else:
        raise ValueError(
            f"timescale must be 'Daily', 'Monthly', or 'Annual'; got {timescale!r}"
        )

    # Auto-detect actual cell count from the file; warn if it differs from the
    # caller-supplied n_riv_cells (which may be 0 when swatmf_river2grid.txt
    # was written with no cells).
    actual_n = _detect_n_riv_cells(data, prefix)
    if actual_n != n_riv_cells:
        import warnings
        warnings.warn(
            f"get_gwsw: n_riv_cells={n_riv_cells} but the output file contains "
            f"{actual_n} river cells per time step.  Using {actual_n}.",
            stacklevel=2,
        )
        n_riv_cells = actual_n

    if n_riv_cells == 0:
        return pd.DataFrame(index=date_labels, dtype=float)

    si = date_labels.index(sdate) if sdate else 0
    ei = date_labels.index(edate) + 1 if edate else len(date_labels)
    selected = date_labels[si:ei]

    rows = []
    for date_str in selected:
        date_idx = date_labels.index(date_str)
        block_start = _find_block_start(data, prefix, date_idx)
        # The exchange value is in column index 3 (0-based) of each data row
        values = [float(data[block_start + i][3]) for i in range(n_riv_cells)]
        rows.append(values)

    df = pd.DataFrame(rows, index=selected, columns=range(1, n_riv_cells + 1))
    return df


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def gwsw_basin_total(df: pd.DataFrame) -> pd.Series:
    """Sum all river cells for each time step.

    Positive total → net groundwater discharge to stream.
    Negative total → net stream loss to aquifer.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_gwsw`.

    Returns
    -------
    pd.Series
    """
    return df.sum(axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gwsw_timeseries(
    df: pd.DataFrame,
    timescale: str = "Daily",
    figsize: tuple[float, float] = (12, 4),
    dark_theme: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the total (basin-wide) GW-SW exchange over time.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_gwsw`.
    timescale : str, optional
        Used for the y-axis label.
    figsize : tuple, optional
        Figure size.
    dark_theme : bool, optional
        Dark background.

    Returns
    -------
    (fig, ax)
    """
    plt.style.use("dark_background" if dark_theme else "default")
    fig, ax = plt.subplots(figsize=figsize)

    total = gwsw_basin_total(df)
    colors = ["dodgerblue" if v >= 0 else "tomato" for v in total.values]
    ax.bar(range(len(total)), total.values, color=colors, alpha=0.8)
    ax.axhline(0, color="grey", lw=0.8, ls="--")

    # Label every N-th tick to avoid overcrowding
    n = max(1, len(total) // 10)
    ax.set_xticks(range(0, len(total), n))
    ax.set_xticklabels(list(total.index)[::n], rotation=45, ha="right", fontsize=7)

    ax.set_ylabel(r"GW-SW Exchange $[m^3/day]$", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(
        f"{timescale} Groundwater–Surface-Water Exchange (basin total)",
        fontsize=10, loc="left",
    )
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_gwsw(
    df: pd.DataFrame,
    out_folder: str | os.PathLike,
    timescale: str = "Daily",
) -> str:
    """Export GW-SW exchange DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_gwsw`.
    out_folder : str or path-like
        Destination directory.
    timescale : str, optional
        Used in the filename.

    Returns
    -------
    str
        Absolute path to the written CSV.
    """
    fname = f"swatmf_gwsw_{timescale.lower()}.csv"
    fpath = os.path.join(str(out_folder), fname)
    df.to_csv(fpath, index_label="Date")
    return fpath
