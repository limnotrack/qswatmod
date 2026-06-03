"""
swatmf.outputs.groundwater
===========================
Read and visualise MODFLOW groundwater-level output (``swatmf_out_MF_obs``).

Mirrors the functions in ``pyfolder/post_ii_gw.py`` without any QGIS/Qt
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

def read_mf_obs(swatmf_folder: str | os.PathLike) -> pd.DataFrame:
    """Read ``modflow.obs`` and return a DataFrame of observation cell metadata.

    Supports both the **new** 3-column format written by this package::

        MODFLOW observation cells (number of cells, I,J,K for each cell)
        <N>
        <row> <col> <layer>
        ...

    and the **legacy** 5-column format::

        # comment
        <N>  # comment
        <row> <col> <layer> <grid_id> <elev>
        ...

    In the new format ``grid_id`` is computed as ``(row-1)*ncol + col``
    using the ``.dis`` file found in *swatmf_folder*, and ``mf_elev`` is
    read from the TOP array in that same ``.dis`` file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    pd.DataFrame
        Index: ``grid_id`` (int).
        Columns: ``row``, ``col``, ``layer``, ``mf_elev``.
    """
    wd = str(swatmf_folder)
    path = os.path.join(wd, "modflow.obs")

    # Parse the file manually to handle both old and new formats
    rows = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                row_val = int(parts[0])
            except ValueError:
                continue                     # text header line
            if len(parts) < 3:
                continue                     # count-only line
            try:
                col_val = int(parts[1])
                lay_val = int(parts[2])
            except (ValueError, IndexError):
                continue
            entry = {"row": row_val, "col": col_val, "layer": lay_val}
            # Legacy format: grid_id and elevation in columns 3-4
            if len(parts) >= 5:
                try:
                    entry["grid_id"] = int(parts[3])
                    entry["mf_elev"] = float(parts[4])
                except (ValueError, IndexError):
                    pass
            rows.append(entry)

    df = pd.DataFrame(rows)

    # If grid_id was not in the file, derive it from the .dis file
    if "grid_id" not in df.columns:
        ncol = _read_ncol(wd)
        df["grid_id"] = (df["row"] - 1) * ncol + df["col"]

    # If elevation was not in the file, read it from the TOP array
    if "mf_elev" not in df.columns:
        top_flat = _read_dis_top(wd)
        if top_flat is not None:
            df["mf_elev"] = df["grid_id"].apply(
                lambda gid: top_flat[int(gid) - 1] if 0 < int(gid) <= len(top_flat) else float("nan")
            )
        else:
            df["mf_elev"] = float("nan")

    df = df.set_index("grid_id")
    return df


def _read_ncol(swatmf_folder: str) -> int:
    """Read NCOL from the first ``.dis`` file found in *swatmf_folder*."""
    import glob as _glob
    dis_files = _glob.glob(os.path.join(swatmf_folder, "*.dis"))
    if not dis_files:
        raise FileNotFoundError(f"No .dis file found in {swatmf_folder!r}")
    with open(dis_files[0]) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                parts = s.split()
                return int(parts[2])   # nlay nrow ncol …
    raise ValueError("Could not parse NCOL from .dis file")


def _read_dis_top(swatmf_folder: str) -> Optional[list]:
    """Return the TOP array from the ``.dis`` file as a flat list, or None.

    Handles Flopy-generated DIS files that use ``CONSTANT`` and ``INTERNAL``
    array control records (with optional format strings and comment tokens).
    """
    import glob as _glob
    dis_files = _glob.glob(os.path.join(swatmf_folder, "*.dis"))
    if not dis_files:
        return None
    try:
        with open(dis_files[0]) as fh:
            lines = [l.rstrip() for l in fh]

        # Strip comment lines
        data_lines = [l for l in lines if not l.lstrip().startswith("#")]

        # ── Parse main header (first non-comment line) ─────────────────────
        hdr = data_lines[0].split()
        nlay, nrow, ncol = int(hdr[0]), int(hdr[1]), int(hdr[2])
        li = 1   # next line index

        # ── Skip LAYCBD (nlay integers on one or more lines) ───────────────
        laycbd_read = 0
        while laycbd_read < nlay:
            vals = data_lines[li].split()
            laycbd_read += len(vals)
            li += 1

        # ── Helper: read one MODFLOW 2D array block ────────────────────────
        def read_array(n_vals: int):
            nonlocal li
            ctrl = data_lines[li].split()
            li += 1
            keyword = ctrl[0].upper()
            if keyword == "CONSTANT":
                return [float(ctrl[1])] * n_vals
            if keyword == "INTERNAL":
                # Read n_vals float tokens from following lines
                vals: list[float] = []
                while len(vals) < n_vals:
                    row_tokens = data_lines[li].split()
                    li += 1
                    vals.extend(float(t) for t in row_tokens)
                return vals[:n_vals]
            # Fallback: treat control line itself as data (old-style)
            vals = [float(t) for t in ctrl]
            while len(vals) < n_vals:
                vals.extend(float(t) for t in data_lines[li].split())
                li += 1
            return vals[:n_vals]

        # ── DELR (ncol) ────────────────────────────────────────────────────
        read_array(ncol)
        # ── DELC (nrow) ────────────────────────────────────────────────────
        read_array(nrow)
        # ── TOP (nrow × ncol) — this is what we want ──────────────────────
        top = read_array(nrow * ncol)
        return top
    except Exception:
        return None


def read_swatmf_out_MF_obs(swatmf_folder: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``modflow.obs`` and ``swatmf_out_MF_obs``.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    (mf_obs, output_wt)
        *mf_obs*: land-surface elevations indexed by grid_id.
        *output_wt*: simulated water-table heads, one column per grid cell.
    """
    mf_obs = read_mf_obs(swatmf_folder)
    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    output_wt = pd.read_csv(
        os.path.join(str(swatmf_folder), "swatmf_out_MF_obs"),
        sep=r"\s+",
        skiprows=1,
        names=grid_id_lst,
    )
    return mf_obs, output_wt


def read_gw_obd(
    swatmf_folder: str | os.PathLike,
    obd_file: str,
) -> pd.DataFrame:
    """Read a groundwater-level observation CSV (``dtw*.obd.csv`` or ``gwl*.obd.csv``).

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    obd_file : str
        Observation filename.

    Returns
    -------
    pd.DataFrame
        Date-indexed; columns are well/site identifiers.
    """
    return pd.read_csv(
        os.path.join(str(swatmf_folder), obd_file),
        index_col=0,
        header=0,
        parse_dates=True,
        na_values=[-999, ""],
    )


# ---------------------------------------------------------------------------
# Time-step resampling
# ---------------------------------------------------------------------------

def _apply_timescale(df: pd.DataFrame, timescale: str) -> pd.DataFrame:
    if timescale == "Daily":
        return df
    codes = {"Monthly": "ME", "Annual": "YE"}
    code = codes.get(timescale)
    if code is None:
        raise ValueError(
            f"timescale must be 'Daily', 'Monthly', or 'Annual'; got {timescale!r}"
        )
    return df.resample(code).mean()


# ---------------------------------------------------------------------------
# Main data-access function
# ---------------------------------------------------------------------------

def get_groundwater(
    swatmf_folder: str | os.PathLike,
    grid_id: int,
    start_date: str,
    timescale: str = "Daily",
    depth_to_water: bool = False,
    obd_file: Optional[str] = None,
    obd_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a DataFrame of simulated (and optionally observed) water levels.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    grid_id : int
        MODFLOW grid cell ID (must appear in ``modflow.obs``).
    start_date : str
        Warmup-adjusted start date as ``"MM/DD/YYYY"``.
    timescale : str, optional
        ``'Daily'``, ``'Monthly'``, or ``'Annual'``.  Default ``'Daily'``.
    depth_to_water : bool, optional
        When ``True``, subtract the land-surface elevation from the
        simulated head to produce depth-to-water.  Default ``False``.
    obd_file : str, optional
        Observation CSV filename.
    obd_col : str, optional
        Column name in the observation file.

    Returns
    -------
    pd.DataFrame
        Column ``str(grid_id)`` holds the simulated head (or depth-to-water);
        if observations were provided, *obd_col* is appended.
    """
    mf_obs, output_wt = read_swatmf_out_MF_obs(swatmf_folder)

    output_wt.index = pd.date_range(start_date, periods=len(output_wt))
    df = _apply_timescale(output_wt, timescale)

    col = str(grid_id)
    if depth_to_water:
        elev = float(mf_obs.loc[int(grid_id), "mf_elev"])
        result = (df[col] - elev).rename(col)
    else:
        result = df[col]

    result = result.to_frame()

    if obd_file is not None:
        if obd_col is None:
            raise ValueError("obd_col must be provided when obd_file is given")
        obd = read_gw_obd(swatmf_folder, obd_file)
        result = pd.concat([result, obd[obd_col]], axis=1).dropna()

    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(
    df: pd.DataFrame,
    grid_id: int,
    obd_col: str,
) -> dict[str, float]:
    """Compute objective-function metrics for groundwater levels.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_groundwater` (must contain both sim and obs).
    grid_id : int
        Grid cell ID — column ``str(grid_id)`` is the simulated series.
    obd_col : str
        Column name for observed values.

    Returns
    -------
    dict with keys ``nse``, ``rmse``, ``pbias``, ``rsq``.
    """
    return all_metrics(df[str(grid_id)].values, df[obd_col].values)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_groundwater(
    df: pd.DataFrame,
    grid_id: int,
    obd_col: Optional[str] = None,
    timescale: str = "Daily",
    depth_to_water: bool = False,
    figsize: tuple[float, float] = (12, 4),
    dark_theme: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot simulated (and optionally observed) groundwater levels.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_groundwater`.
    grid_id : int
        Grid cell ID — used for title and column lookup.
    obd_col : str, optional
        Observed column name.
    timescale : str, optional
        Label for the plot title.
    depth_to_water : bool, optional
        If ``True``, labels the y-axis as depth-to-water.
    figsize : tuple, optional
        Figure size.  Default ``(12, 4)``.
    dark_theme : bool, optional
        Dark background style.  Default ``False``.

    Returns
    -------
    (fig, ax)
    """
    plt.style.use("dark_background" if dark_theme else "default")
    fig, ax = plt.subplots(figsize=figsize)

    col = str(grid_id)
    ylabel = r"Depth to Water $[m]$" if depth_to_water else r"Hydraulic Head $[m]$"
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%Y"))

    ax.plot(df.index, df[col], c="limegreen", lw=1, label="Simulated")

    if obd_col is not None and obd_col in df.columns:
        ax.plot(
            df.index, df[obd_col], c="m", lw=1.5, alpha=0.6, label="Observed"
        )
        if len(df[obd_col].dropna()) > 1:
            stats = compute_statistics(df, grid_id, obd_col)
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
        f"{timescale} Groundwater Level — Grid ID {grid_id}",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_groundwater(
    df: pd.DataFrame,
    out_folder: str | os.PathLike,
    grid_id: int,
    obd_col: str = "",
    timescale: str = "Daily",
    stats: Optional[dict[str, float]] = None,
) -> str:
    """Export groundwater results to a tab-delimited text file.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`get_groundwater`.
    out_folder : str or path-like
        Destination directory.
    grid_id : int
        Grid cell ID — used in the filename.
    obd_col : str, optional
        Observed column name.  Default ``''``.
    timescale : str, optional
        Time-step label.  Default ``'Daily'``.
    stats : dict, optional
        Pre-computed stats from :func:`compute_statistics`.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    from datetime import datetime as _dt
    from swatmf import _EXPORT_VERSION

    version = _EXPORT_VERSION
    ctime = _dt.now().strftime("- %m/%d/%y %H:%M:%S -")
    fname = f"swatmf_gw({grid_id})_obd({obd_col})_{timescale.lower()}.txt"
    fpath = os.path.join(str(out_folder), fname)

    with open(fpath, "w") as fh:
        fh.write(f"# {fname} created by swatmf package {version}{ctime}\n")
        df.to_csv(
            fh, index_label="Date", sep="\t",
            float_format="%10.4f", lineterminator="\n", encoding="utf-8",
        )
        fh.write("\n# Statistics\n")
        if stats is not None:
            fh.write(f"Nash-Sutcliffe: {stats['nse']:.4f}\n")
            fh.write(f"R-squared: {stats['rsq']:.4f}\n")
            fh.write(f"PBIAS: {stats['pbias']:.4f}\n")
            fh.write(f"RMSE: {stats['rmse']:.4f}\n")
        else:
            fh.write("Nash-Sutcliffe: ---\n")
            fh.write("R-squared: ---\n")
            fh.write("PBIAS: ---\n")
            fh.write("RMSE: ---\n")

    return fpath
