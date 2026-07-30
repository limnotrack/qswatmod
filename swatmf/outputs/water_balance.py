"""
swatmf.outputs.water_balance
=============================
Read and visualise the SWAT water-balance output (``output.std``).

Mirrors the functions in ``pyfolder/post_v_wb.py`` without any QGIS/Qt
dependency.

Columns extracted from ``output.std``
--------------------------------------
prec   — Precipitation (mm)
surq   — Surface runoff (mm)
latq   — Lateral flow (mm)
gwq    — Groundwater discharge to stream (mm)
swgw   — Seepage from stream to aquifer (mm)
perco  — Deep percolation to aquifer (mm)  [column index 7 in SWAT-MF output]
tile   — Tile drain flow (mm)
sw     — Soil water (mm)
gw     — Groundwater volume (mm)
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# I/O: parse output.std
# ---------------------------------------------------------------------------

_SKIP_PREFIXES = ("TIME", "UNIT", "SWAT", "(mm)")


def _is_valid_day_row(parts: list[str]) -> bool:
    """Return True if *parts* looks like a daily data row (starts with a day number)."""
    try:
        int(parts[0])
        return len(parts[0]) <= 3  # day numbers are 1-366, not 4-digit years
    except (ValueError, IndexError):
        return False


def read_output_std(
    swatmf_folder: str | os.PathLike,
    end_year: int,
    iprint: int = 1,
) -> pd.DataFrame:
    """Parse ``output.std`` and return a tidy daily DataFrame.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    end_year : int
        Last simulation year (used to stop reading when yearly totals are
        encountered).
    iprint : int, optional
        SWAT print code: 1 = daily output, 0 = monthly.  Default 1.

    Returns
    -------
    pd.DataFrame
        Columns: ``prec``, ``surq``, ``latq``, ``gwq``, ``swgw``,
        ``perco``, ``tile``, ``sw``, ``gw``.
        Index is integer (row number in the filtered file).
    """
    path = os.path.join(str(swatmf_folder), "output.std")
    raw_lines: list[str] = []
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if len(stripped) > 100 and not stripped.startswith(_SKIP_PREFIXES):
                raw_lines.append(line)

    # --- filter out year-header rows and stop at the end year ---------------
    filtered: list[str] = []
    for line in raw_lines:
        try:
            token = line.split()[0]
        except IndexError:
            continue
        if token == str(end_year):
            break
        if len(str(token)) == 4:  # year row (e.g. "2010")
            continue
        filtered.append(line)

    # --- remove duplicate-month artefacts from daily output -----------------
    # Threshold used to distinguish a genuine day-1 (start of a new period)
    # from a duplicate month artefact: if the previous day number is more than
    # this many days before day 1 of the next row, the row is a duplicate.
    _DUP_MONTH_THRESHOLD = 20

    if iprint == 1:
        clean: list[str] = []
        for i, line in enumerate(filtered):
            parts = line.split()
            if not parts:
                continue
            try:
                day = int(parts[0])
            except ValueError:
                continue
            if i > 0 and clean:
                prev_day = int(clean[-1].split()[0])
                if day == 1 and (prev_day - day) > _DUP_MONTH_THRESHOLD:
                    continue
                if day < prev_day and day != 1:
                    continue
            clean.append(line)
    else:
        clean = filtered

    # --- extract columns -----------------------------------------------------
    rows = []
    for line in clean:
        p = line.split()
        try:
            rows.append({
                "prec":  float(p[1]),
                "surq":  float(p[2]),
                "latq":  float(p[3]),
                "gwq":   float(p[4]),
                "swgw":  float(p[5]),
                "perco": float(p[7]),  # column 7 in SWAT-MF (reach column at 6)
                "tile":  float(p[8]),
                "sw":    float(p[10]),
                "gw":    float(p[11]),
            })
        except (IndexError, ValueError):
            continue

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Higher-level: get water balance with date index
# ---------------------------------------------------------------------------

def get_water_balance(
    swatmf_folder: str | os.PathLike,
    start_date: str,
    end_year: int,
    iprint: int = 1,
    timescale: str = "Daily",
    sdate: Optional[str] = None,
    edate: Optional[str] = None,
) -> pd.DataFrame:
    """Return a date-indexed water-balance DataFrame.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    start_date : str or datetime
        Warmup-adjusted start date (``SimPeriod.start_date_warmup``).
    end_year : int
        Last simulation year (e.g. ``SimPeriod.end_date.year``).
    iprint : int, optional
        SWAT print code (0=monthly, 1=daily, 2=annual).  Default 1.
    timescale : str, optional
        Desired aggregation: ``'Daily'``, ``'Monthly'``, or ``'Annual'``.
    sdate : str, optional
        Slice start (pandas label-based).
    edate : str, optional
        Slice end (pandas label-based).

    Returns
    -------
    pd.DataFrame
        Date-indexed water balance columns.
    """
    df = read_output_std(swatmf_folder, end_year, iprint)
    df.index = pd.date_range(start_date, periods=len(df))

    if timescale == "Monthly":
        df = df.resample("ME").mean()
    elif timescale == "Annual":
        df = df.resample("YE").mean()

    if sdate or edate:
        df = df.loc[sdate:edate]

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_WB_NAMES = (
    "Precipitation",
    "Soil Water",
    "Surface Runoff",
    "Lateral Flow",
    "Groundwater Flow to Stream",
    "Seepage: Stream → Aquifer",
    "Deep Percolation to Aquifer",
    "Groundwater Volume",
)
_WB_COLORS = (
    "slateblue",
    "lightgreen",
    "limegreen",
    "forestgreen",
    "darkgreen",
    "b",
    "dodgerblue",
    "skyblue",
)


def plot_water_balance(
    df: pd.DataFrame,
    timescale: str = "Daily",
    show_legend: bool = True,
    dark_theme: bool = False,
    figsize: tuple[float, float] = (14, 7),
) -> tuple[plt.Figure, np.ndarray]:
    """Four-panel water balance plot.

    Panel 1 — Precipitation (inverted y-axis).
    Panel 2 — Soil water (stacked on outflows).
    Panel 3 — Flow components: surface, lateral, groundwater discharge
               above zero; seepage, deep percolation, gw volume below.
    Panel 4 — Subsurface storage components.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_water_balance`.
    timescale : str, optional
        Used in the title.
    show_legend : bool, optional
        Whether to add a legend.  Default ``True``.
    dark_theme : bool, optional
        Dark background.  Default ``False``.
    figsize : tuple, optional
        Figure size.  Default ``(14, 7)``.

    Returns
    -------
    (fig, axes)
        *axes* is a 1-D NumPy array of length 4.
    """
    plt.style.use("dark_background" if dark_theme else "default")

    fig, axes = plt.subplots(
        nrows=4, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [0.2, 0.2, 0.4, 0.2], "hspace": 0.1},
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.05)

    # ── Precipitation ─────────────────────────────────────────────────────
    axes[0].stackplot(df.index, df.prec, colors=["slateblue"])
    axes[0].set_ylim(df.prec.max() * 1.1, 0)
    axes[0].xaxis.tick_top()
    axes[0].spines["bottom"].set_visible(False)
    axes[0].tick_params(axis="both", labelsize=8)
    axes[0].set_title(
        f"Water Balance — {timescale} [mm]", fontsize=12, fontweight="semibold"
    )
    axes[0].title.set_position([0.5, 1.8])

    # ── Soil Water ────────────────────────────────────────────────────────
    for sp in ("top", "bottom"):
        axes[1].spines[sp].set_visible(False)
    axes[1].get_xaxis().set_visible(False)
    axes[1].stackplot(df.index, df.sw, colors=["lightgreen"])
    axes[1].set_ylim(
        (df.gwq + df.latq + df.surq).max(),
        (df.gwq + df.latq + df.surq + df.sw).max(),
    )
    axes[1].tick_params(axis="both", labelsize=8)

    # ── Flow components ───────────────────────────────────────────────────
    for sp in ("top", "bottom"):
        axes[2].spines[sp].set_visible(False)
    axes[2].get_xaxis().set_visible(False)
    axes[2].stackplot(
        df.index, df.gwq, df.latq, df.surq, df.sw,
        colors=["darkgreen", "forestgreen", "limegreen", "lightgreen"],
    )
    axes[2].axhline(y=0, lw=0.3, ls="--", c="grey")
    axes[2].stackplot(df.index, -df.swgw, -df.perco, -df.gw)
    axes[2].set_ylim(
        -(df.swgw + df.perco).max(),
        (df.gwq + df.latq + df.surq).max(),
    )
    axes[2].tick_params(axis="both", labelsize=8)
    axes[2].set_yticklabels([abs(x) for x in axes[2].get_yticks()])

    # ── Subsurface storage ────────────────────────────────────────────────
    axes[3].stackplot(df.index, df.gw + df.perco + df.swgw, colors=["skyblue"])
    axes[3].set_ylim(
        (df.gw + df.perco + df.swgw).max(),
        (df.gw + df.perco + df.swgw).min(),
    )
    axes[3].spines["top"].set_visible(False)
    axes[3].tick_params(axis="both", labelsize=8)

    # ── Broken axis diagonals ─────────────────────────────────────────────
    _add_break_diagonals(axes, dark_theme)

    # ── Legend ───────────────────────────────────────────────────────────
    if show_legend:
        patches = [Rectangle((0, 0), 0.1, 0.1, fc=c, alpha=1) for c in _WB_COLORS]
        legend = axes[0].legend(
            patches, _WB_NAMES,
            loc="upper left",
            title="EXPLANATION",
            edgecolor="none",
            fontsize=8,
            bbox_to_anchor=(-0.02, 1.8),
            ncol=8,
        )
        legend._legend_box.align = "left"

    return fig, axes


def _add_break_diagonals(axes: np.ndarray, dark_theme: bool) -> None:
    d = 0.003
    c = "w" if dark_theme else "k"
    kw = dict(color=c, clip_on=False)
    ax1, ax2, ax3 = axes[1], axes[2], axes[3]
    for ax in (ax1, ax2):
        kw["transform"] = ax.transAxes
        ax.plot((-d, d), (-d, d), **kw)
        ax.plot((1 - d, 1 + d), (-d, d), **kw)
    for ax in (ax2, ax3):
        kw["transform"] = ax.transAxes
        ax.plot((-d, d), (1 - d, 1 + d), **kw)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_water_balance(
    df: pd.DataFrame,
    out_folder: str | os.PathLike,
    timescale: str = "Daily",
) -> str:
    """Export water balance DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_water_balance`.
    out_folder : str or path-like
        Destination directory.
    timescale : str, optional
        Used in the filename.

    Returns
    -------
    str
        Absolute path to the written CSV.
    """
    fname = f"swatmf_water_balance_{timescale.lower()}.csv"
    fpath = os.path.join(str(out_folder), fname)
    df.to_csv(fpath, index_label="Date")
    return fpath
