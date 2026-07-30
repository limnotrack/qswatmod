"""
swatmf.outputs.streamflow
==========================
Read and visualise SWAT stream-flow output (``output.rch``).

Mirrors the functions in ``pyfolder/post_i_sw.py`` without any QGIS/Qt
dependency.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from ..metrics import all_metrics


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_output_rch(swatmf_folder: str | os.PathLike, col: int = 6) -> pd.DataFrame:
    """Read ``output.rch`` and return a tidy DataFrame.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (contains ``output.rch``).
    col : int, optional
        Column index (0-based after selecting cols 1, 3, *col*) to extract.
        Column 6 (0-based in the full file) is ``FLOW_OUTcms`` — stream
        discharge in m³/s.  Default is 6.

    Returns
    -------
    pd.DataFrame
        Index: subbasin number (``RCH``).
        Columns: ``date`` (integer day/month counter), ``stf_sim``.
    """
    path = os.path.join(str(swatmf_folder), "output.rch")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=9,
        usecols=[1, 3, col],
        names=["date", "filter", "stf_sim"],
        index_col=0,
    )
    return df


def read_stf_obd(
    swatmf_folder: str | os.PathLike,
    obd_file: str,
) -> pd.DataFrame:
    """Read a streamflow observation CSV file (``stf*.obd.csv``).

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    obd_file : str
        Filename of the observation CSV (e.g. ``stf_1.obd.csv``).

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame; columns are station/subbasin identifiers.
    """
    return pd.read_csv(
        os.path.join(str(swatmf_folder), obd_file),
        index_col=0,
        header=0,
        parse_dates=True,
        na_values=[-999, ""],
    )


# ---------------------------------------------------------------------------
# Time-step conversion
# ---------------------------------------------------------------------------

_RESAMPLE_CODES = {"Monthly": "ME", "Annual": "YE"}


def _apply_timescale(df: pd.DataFrame, timescale: str) -> pd.DataFrame:
    """Resample *df* to *timescale* (``'Daily'``, ``'Monthly'``, ``'Annual'``)."""
    if timescale == "Daily":
        return df
    code = _RESAMPLE_CODES.get(timescale)
    if code is None:
        raise ValueError(
            f"timescale must be 'Daily', 'Monthly', or 'Annual'; got {timescale!r}"
        )
    return df.resample(code).mean()


def _index_daily(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.date_range(start_date, periods=len(df))
    return df


def _index_monthly(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    df = df[df["filter"] < 13].copy()
    df.index = pd.date_range(start_date, periods=len(df), freq="ME")
    return df


def _index_annual(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.date_range(start_date, periods=len(df), freq="YE")
    return df


# ---------------------------------------------------------------------------
# Main data-access function
# ---------------------------------------------------------------------------

def get_streamflow(
    swatmf_folder: str | os.PathLike,
    subbasin: int,
    start_date: str,
    iprint: int = 1,
    timescale: str = "Daily",
    obd_file: Optional[str] = None,
    obd_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a DataFrame of simulated (and optionally observed) streamflow.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    subbasin : int
        Subbasin number (``RCH`` column in ``output.rch``).
    start_date : str
        Warmup-adjusted start date as ``"MM/DD/YYYY"`` (i.e.
        ``SimPeriod.start_date_warmup.strftime("%m/%d/%Y")``).
    iprint : int, optional
        SWAT print code from ``file.cio`` (0=monthly, 1=daily, 2=annual).
        Determines how the raw output is indexed.  Default 1 (daily).
    timescale : str, optional
        Desired output time-step: ``'Daily'``, ``'Monthly'``, or
        ``'Annual'``.  Default ``'Daily'``.
    obd_file : str, optional
        Observation CSV filename.  If provided, observations are joined.
    obd_col : str, optional
        Column name in the observation file.  Required when *obd_file* is set.

    Returns
    -------
    pd.DataFrame
        Columns: ``stf_sim`` and, if observations supplied, *obd_col*.
    """
    raw = read_output_rch(swatmf_folder).loc[subbasin]

    if iprint == 1:
        df = _index_daily(raw, start_date)
    elif iprint == 0:
        df = _index_monthly(raw, start_date)
    else:
        df = _index_annual(raw, start_date)

    df = _apply_timescale(df, timescale)

    if obd_file is not None:
        if obd_col is None:
            raise ValueError("obd_col must be provided when obd_file is given")
        obd = read_stf_obd(swatmf_folder, obd_file)
        df = pd.concat([df, obd[obd_col]], axis=1).dropna()

    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(
    df: pd.DataFrame,
    sim_col: str = "stf_sim",
    obd_col: Optional[str] = None,
) -> dict[str, float]:
    """Compute objective-function metrics for simulated vs. observed flow.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *sim_col* and, if supplied, *obd_col*.
    sim_col : str, optional
        Column name for simulated values.  Default ``'stf_sim'``.
    obd_col : str, optional
        Column name for observed values.  If ``None``, raises ``ValueError``.

    Returns
    -------
    dict with keys ``nse``, ``rmse``, ``pbias``, ``rsq``.
    """
    if obd_col is None:
        raise ValueError("obd_col is required to compute statistics")
    return all_metrics(df[sim_col].values, df[obd_col].values)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_streamflow(
    df: pd.DataFrame,
    subbasin: int,
    sim_col: str = "stf_sim",
    obd_col: Optional[str] = None,
    timescale: str = "Daily",
    figsize: tuple[float, float] = (12, 4),
    dark_theme: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot simulated (and optionally observed) streamflow.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_streamflow`.
    subbasin : int
        Subbasin number — used only for the plot title.
    sim_col : str, optional
        Column name for simulated discharge.  Default ``'stf_sim'``.
    obd_col : str, optional
        Column name for observed discharge.  When supplied, metrics are
        annotated on the axes.
    timescale : str, optional
        Label used in the title (``'Daily'``, ``'Monthly'``, ``'Annual'``).
    figsize : tuple, optional
        Matplotlib figure size.  Default ``(12, 4)``.
    dark_theme : bool, optional
        Use a dark background style.  Default ``False``.

    Returns
    -------
    (fig, ax)
    """
    plt.style.use("dark_background" if dark_theme else "default")
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df.index, df[sim_col], c="limegreen", lw=1, label="Simulated")
    ax.set_ylabel(r"Stream Discharge $[m^3/s]$", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%Y"))

    if obd_col is not None and obd_col in df.columns:
        ax.plot(
            df.index, df[obd_col], c="m", lw=1.5, alpha=0.6, label="Observed"
        )
        if len(df[obd_col].dropna()) > 1:
            stats = compute_statistics(df, sim_col=sim_col, obd_col=obd_col)
            ax.text(
                0.01, 0.95,
                f"NSE: {stats['nse']:.4f}",
                fontsize=8, ha="left", color="limegreen", transform=ax.transAxes,
            )
            ax.text(
                0.01, 0.90,
                f"R²: {stats['rsq']:.4f}",
                fontsize=8, ha="left", color="limegreen", transform=ax.transAxes,
            )
            ax.text(
                0.99, 0.95,
                f"PBIAS: {stats['pbias']:.4f}",
                fontsize=8, ha="right", color="limegreen", transform=ax.transAxes,
            )

    ax.set_title(
        f"{timescale} Stream Discharge — Subbasin {subbasin}",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_streamflow(
    df: pd.DataFrame,
    out_folder: str | os.PathLike,
    subbasin: int,
    obd_col: str = "",
    timescale: str = "Daily",
    stats: Optional[dict[str, float]] = None,
) -> str:
    """Export the streamflow DataFrame to a tab-delimited text file.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_streamflow`.
    out_folder : str or path-like
        Destination directory.
    subbasin : int
        Subbasin number — used in the filename.
    obd_col : str, optional
        Observed column name — used in the filename.  Default ``''``.
    timescale : str, optional
        Time-step label used in the filename.  Default ``'Daily'``.
    stats : dict, optional
        Pre-computed statistics dict from :func:`compute_statistics`.
        If ``None``, statistics section will show ``---``.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    from datetime import datetime as _dt
    from swatmf import _EXPORT_VERSION

    version = _EXPORT_VERSION
    ctime = _dt.now().strftime("- %m/%d/%y %H:%M:%S -")
    fname = f"swatmf_reach({subbasin})_obd({obd_col})_{timescale.lower()}.txt"
    fpath = os.path.join(str(out_folder), fname)

    export_df = df.drop(columns=["filter"], errors="ignore")
    with open(fpath, "w") as fh:
        fh.write(
            f"# {fname} created by swatmf package {version}{ctime}\n"
        )
        export_df.to_csv(
            fh, index_label="Date", sep="\t",
            float_format="%10.4f", lineterminator="\n", encoding="utf-8",
        )
        fh.write("\n# Statistics\n")
        if stats is not None:
            fh.write(f"Nash-Sutcliffe: {stats['nse']:.4f}\n")
            fh.write(f"R-squared: {stats['rsq']:.4f}\n")
            fh.write(f"PBIAS: {stats['pbias']:.4f}\n")
        else:
            fh.write("Nash-Sutcliffe: ---\n")
            fh.write("R-squared: ---\n")
            fh.write("PBIAS: ---\n")

    return fpath
