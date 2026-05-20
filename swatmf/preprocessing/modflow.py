"""
swatmf.preprocessing.modflow
==============================
Pure-Python helpers for reading and writing MODFLOW-related files that are
part of the SWAT-MODFLOW pre-processing workflow.

The functions here mirror the file-I/O logic in
``pyfolder/modflow_functions.py`` without any dependency on QGIS, PyQt, or
the ``processing`` module.  Geospatial operations (creating/selecting
shapefile features, spatial intersections) must still be performed in QGIS
or via a library such as GeoPandas/Shapely.

Functions exposed
-----------------
parse_dis_file         — Parse a MODFLOW discretisation (.dis) file.
grid_row_col           — Build full (row, col) arrays for every grid cell.
parse_riv_file         — Parse a MODFLOW river-package (.riv) file.
write_riv_file         — Overwrite a MODFLOW river-package (.riv) file.
create_modflow_obs     — Write the ``modflow.obs`` observation file.
create_modflow_mfn     — Generate ``modflow.mfn`` from the MODFLOW name file.
modify_modflow_oc      — Update unit numbers in the MODFLOW output-control file.
"""

from __future__ import annotations

import csv
import datetime
import glob
import os
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from swatmf import _EXPORT_VERSION


# ---------------------------------------------------------------------------
# Named-tuple types
# ---------------------------------------------------------------------------

class DisInfo(NamedTuple):
    """Key parameters extracted from a MODFLOW ``.dis`` file.

    Attributes
    ----------
    nrow : int
        Number of rows in the MODFLOW grid.
    ncol : int
        Number of columns in the MODFLOW grid.
    delr : float
        Cell width along rows (y-spacing, metres).
    delc : float
        Cell width along columns (x-spacing, metres).
    n_cells : int
        Total number of cells (``nrow * ncol``).
    top_elevs : list of float
        Land-surface elevation for every cell (row-major order).
    transient : bool
        ``True`` if the simulation is transient (``TR``); ``False`` if
        steady-state (``SS``).
    """

    nrow: int
    ncol: int
    delr: float
    delc: float
    n_cells: int
    top_elevs: list
    transient: bool


class RivCell(NamedTuple):
    """One river-package cell record.

    Attributes
    ----------
    layer : int
    row : int
    col : int
    stage : float
    cond : float
    rbot : float
    """

    layer: int
    row: int
    col: int
    stage: float
    cond: float
    rbot: float


# ---------------------------------------------------------------------------
# Helpers: find unique MODFLOW package files
# ---------------------------------------------------------------------------

def _find_single_file(folder: str, extension: str) -> str:
    """Return the single file with *extension* in *folder*, or raise."""
    matches = glob.glob(os.path.join(folder, f"*{extension}"))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {extension!r} file found in {folder!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {extension!r} files found in {folder!r}: {matches}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Parse the MODFLOW discretisation file
# ---------------------------------------------------------------------------

def parse_dis_file(swatmf_folder: str | os.PathLike) -> DisInfo:
    """Parse the MODFLOW discretisation (``.dis``) file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (contains the ``.dis`` file).

    Returns
    -------
    DisInfo
        Named-tuple with grid dimensions, cell sizes, and land-surface
        elevations.

    Notes
    -----
    The parser assumes the standard free-format MODFLOW-2005 DIS layout:

    * Line 0 (after comments): ``NLAY NROW NCOL NPER ITMUNI LENUNI``
    * Line 1: ``LAYCBD``
    * Line 2: ``DELR`` (row widths — constant or array keyword)
    * Line 3: ``DELC`` (column widths — constant or array keyword)
    * Lines 4+: ``TOP`` elevations (array keyword + values)
    """
    path = _find_single_file(str(swatmf_folder), ".dis")
    with open(path, "r") as fh:
        data = [
            line.replace("\n", "").split()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]

    nrow = int(data[0][1])
    ncol = int(data[0][2])

    # DELR and DELC — handle both "INTERNAL" keyword and direct value
    if data[2][0].upper() == "INTERNAL":
        delr = float(data[3][1])
        delc = float(data[4][1])
        elev_start = 5
    else:
        delr = float(data[2][1])
        delc = float(data[3][1])
        elev_start = 4

    # TOP elevations: read until the next "INTERNAL" keyword
    top_elevs: list[float] = []
    ii = elev_start
    while ii < len(data) and data[ii][0].upper() != "INTERNAL":
        top_elevs.extend(float(v) for v in data[ii])
        ii += 1

    # Check steady-state vs transient from the DIS header (field index 3 is NPER info)
    transient = any(
        row[3].upper() == "TR"
        for row in data
        if len(row) >= 4
    )

    return DisInfo(
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        n_cells=nrow * ncol,
        top_elevs=top_elevs,
        transient=transient,
    )


# ---------------------------------------------------------------------------
# Build row / column index arrays for the whole grid
# ---------------------------------------------------------------------------

def grid_row_col(dis: DisInfo) -> tuple[list[int], list[int]]:
    """Return (row_list, col_list) for every grid cell in row-major order.

    Parameters
    ----------
    dis : DisInfo
        Output of :func:`parse_dis_file`.

    Returns
    -------
    (rows, cols)
        Each list has length ``dis.n_cells``.  ``rows[i]`` and ``cols[i]``
        are 1-based indices.

    Examples
    --------
    >>> rows, cols = grid_row_col(dis)
    >>> print(rows[0], cols[0])   # first cell: row 1, col 1
    1 1
    """
    rows: list[int] = []
    cols: list[int] = []
    for r in range(1, dis.nrow + 1):
        for c in range(1, dis.ncol + 1):
            rows.append(r)
            cols.append(c)
    return rows, cols


# ---------------------------------------------------------------------------
# Parse the MODFLOW river-package file
# ---------------------------------------------------------------------------

def parse_riv_file(swatmf_folder: str | os.PathLike) -> pd.DataFrame:
    """Parse the MODFLOW river-package (``.riv``) file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``row``, ``col``, ``stage``, ``cond``, ``rbot``.
        One row per river cell.
    """
    path = _find_single_file(str(swatmf_folder), ".riv")
    with open(path, "r") as fh:
        data = [
            line.replace("\n", "").split()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]

    n_riv = int(data[0][0])
    records = []
    # The first two data rows are header rows; river cells start at index 2
    for i in range(2, n_riv + 2):
        row = data[i]
        records.append(
            RivCell(
                layer=int(row[0]),
                row=int(row[1]),
                col=int(row[2]),
                stage=float(row[3]),
                cond=float(row[4]),
                rbot=float(row[5]),
            )
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Overwrite the MODFLOW river-package file
# ---------------------------------------------------------------------------

def write_riv_file(
    swatmf_folder: str | os.PathLike,
    riv_df: pd.DataFrame,
) -> str:
    """Overwrite the MODFLOW ``.riv`` file with updated river-cell data.

    The input DataFrame must contain columns ``layer``, ``row``, ``col``,
    ``stage``, ``cond``, ``rbot`` — i.e. the same columns produced by
    :func:`parse_riv_file` or by the QGIS linking process after
    ``overwriteRivPac``.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    riv_df : pd.DataFrame
        River-cell data sorted by ``grid_id`` (or just sequentially).

    Returns
    -------
    str
        Absolute path to the overwritten ``.riv`` file.
    """
    path = _find_single_file(str(swatmf_folder), ".riv")
    n = len(riv_df)
    ts = datetime.datetime.now().strftime("- %m/%d/%y %H:%M:%S -")
    header = (
        f"# {os.path.basename(path)} overwritten by swatmf package "
        f"{_EXPORT_VERSION}{ts}"
    )
    n_row = f"{n}\t0\t\t\t# Number of river cells"

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([header])
        writer.writerow([n_row])
        writer.writerow([n_row])  # written twice — matches plugin behaviour
        for _, r in riv_df.iterrows():
            writer.writerow(
                [
                    int(r.get("layer", 1)),
                    int(r["row"]),
                    int(r["col"]),
                    f"{r['stage']:f}",
                    f"{r['cond']:f}",
                    f"{r['rbot']:f}",
                    "# Layer, Row, Col, Stage, Cond, Rbot",
                ]
            )
    return path


# ---------------------------------------------------------------------------
# Compute river-package hydraulic parameters from river_grid attributes
# ---------------------------------------------------------------------------

def compute_riv_params(
    river_grid_df: pd.DataFrame,
    riverbed_k: float,
    riverbed_thick: float,
) -> pd.DataFrame:
    """Derive MODFLOW river-package parameters from a ``river_grid`` table.

    This replicates the ``rivInfoTo_mf_riv2`` / ``rivInfoTo_mf_riv2_ii``
    logic from ``modflow_functions.py``, taking a pandas DataFrame instead of
    a QGIS vector layer as input.

    Parameters
    ----------
    river_grid_df : pd.DataFrame
        Must contain columns ``grid_id``, ``Wid2``, ``Dep2``, ``row``,
        ``col``, ``top_elev``, ``rgrid_len``.  Each row is one river-grid
        segment (intersection between the SWAT river network and the MODFLOW
        grid).
    riverbed_k : float
        Riverbed hydraulic conductivity [m/day].
    riverbed_thick : float
        Riverbed thickness [m].

    Returns
    -------
    pd.DataFrame
        One row per unique ``grid_id``; columns ``row``, ``col``,
        ``riv_stage``, ``riv_cond``, ``riv_bot``.
    """
    df = river_grid_df.copy()
    grp = df.groupby("grid_id")

    width_sum  = grp["Wid2"].sum()
    depth_avg  = grp["Dep2"].mean()
    row_avg    = grp["row"].mean().round().astype(int)
    col_avg    = grp["col"].mean().round().astype(int)
    elev_avg   = grp["top_elev"].mean()
    length_sum = grp["rgrid_len"].sum()

    riv_cond  = riverbed_k * length_sum * width_sum / riverbed_thick
    riv_stage = elev_avg + depth_avg + riverbed_thick
    riv_bot   = elev_avg + riverbed_thick

    return pd.DataFrame(
        {
            "grid_id":   width_sum.index,
            "row":       row_avg.values,
            "col":       col_avg.values,
            "riv_stage": riv_stage.values,
            "riv_cond":  riv_cond.values,
            "riv_bot":   riv_bot.values,
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Create modflow.obs
# ---------------------------------------------------------------------------

def create_modflow_obs(
    swatmf_folder: str | os.PathLike,
    obs_df: pd.DataFrame,
) -> str:
    """Write the ``modflow.obs`` observation-cell file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (output destination).
    obs_df : pd.DataFrame
        Observation-cell table with **at least** columns:

        * ``grid_id`` — MODFLOW grid cell ID (integer, 1-based)
        * ``layer``   — MODFLOW layer number (integer, usually 1)
        * ``top_elev``— land-surface elevation at the cell centre (float)

        The ``row`` and ``col`` columns will be **computed automatically**
        from the ``grid_id`` and the ``.dis`` file if they are not already
        present.

    Returns
    -------
    str
        Absolute path to the written ``modflow.obs`` file.

    Notes
    -----
    File format (tab-delimited)::

        # modflow.obs file is created by swatmf package <version> <timestamp>
        <N>    # Number of observation cells
        <row>  <col>  <layer>  <grid_id>  <elev>  # Row, Col, Layer, grid_id, elevation
        ...

    Examples
    --------
    >>> import pandas as pd
    >>> obs = pd.DataFrame({"grid_id": [10, 45], "layer": [1, 1], "top_elev": [123.4, 119.7]})
    >>> path = create_modflow_obs(wd, obs)
    """
    wd = str(swatmf_folder)
    dis = parse_dis_file(wd)
    rows_all, cols_all = grid_row_col(dis)

    df = obs_df.copy()
    df["grid_id"] = df["grid_id"].astype(int)
    df = df.sort_values("grid_id").reset_index(drop=True)

    if "row" not in df.columns or "col" not in df.columns:
        df["row"] = [rows_all[gid - 1] for gid in df["grid_id"]]
        df["col"] = [cols_all[gid - 1] for gid in df["grid_id"]]

    ts = datetime.datetime.now().strftime("- %m/%d/%y %H:%M:%S -")
    out_path = os.path.join(wd, "modflow.obs")

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [f"# modflow.obs file is created by swatmf package {_EXPORT_VERSION}{ts}"]
        )
        writer.writerow([f"{len(df)}", "                # Number of observation cells"])
        for _, r in df.iterrows():
            elev = f"{float(r['top_elev']):.2f}"
            writer.writerow(
                [
                    int(r["row"]),
                    int(r["col"]),
                    int(r["layer"]),
                    int(r["grid_id"]),
                    elev,
                    "# Row, Col, Layer, grid_id, elevation ",
                ]
            )

    return out_path


# ---------------------------------------------------------------------------
# Read modflow.obs
# ---------------------------------------------------------------------------

def read_modflow_obs(swatmf_folder: str | os.PathLike) -> pd.DataFrame:
    """Read an existing ``modflow.obs`` file into a DataFrame.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    pd.DataFrame
        Columns: ``row``, ``col``, ``layer``, ``grid_id``, ``top_elev``.
    """
    path = os.path.join(str(swatmf_folder), "modflow.obs")
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                int(parts[0])  # first token must be a number (row)
            except ValueError:
                continue
            rows.append(
                {
                    "row":      int(parts[0]),
                    "col":      int(parts[1]),
                    "layer":    int(parts[2]),
                    "grid_id":  int(parts[3]),
                    "top_elev": float(parts[4]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Create modflow.mfn
# ---------------------------------------------------------------------------

def create_modflow_mfn(swatmf_folder: str | os.PathLike) -> str:
    """Generate ``modflow.mfn`` from the MODFLOW name (``.nam``) file.

    This mirrors the ``create_modflow_mfn`` function in
    ``modflow_functions.py``.  It reads the ``.nam`` file, adds 5000 to any
    unit-number fields whose current value has fewer than 4 digits, and
    writes the result to ``modflow.mfn`` in the same folder.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    str
        Absolute path to the written ``modflow.mfn`` file.

    Raises
    ------
    FileNotFoundError
        If no ``.nam`` file is found in *swatmf_folder*.
    """
    wd = str(swatmf_folder)
    nam_path = _find_single_file(wd, ".nam")

    ts = datetime.datetime.now().strftime(" - %m/%d/%y %H:%M:%S -")
    info = f"# modflow.mfn file, generated by swatmf package {_EXPORT_VERSION}{ts}\n"

    with open(nam_path, "r") as fh:
        raw_data = [x.strip() for x in fh if x.strip()]

    lines = [info]
    log_entries: list[str] = []

    for line in raw_data:
        if line.startswith("#"):
            lines.append(line + "\n")
            continue
        parts = line.split()
        if len(parts) >= 2 and len(parts[1]) < 4:
            old_unit = parts[1]
            new_unit = int(parts[1]) + 5000
            line = parts[0] + "\t" + str(new_unit) + "\t" + parts[2]
            log_entries.append(
                f"modflow.mfn: Unit number {old_unit} → {new_unit}"
            )
        lines.append(line + "\n")

    mfn_path = os.path.join(wd, "modflow.mfn")
    with open(mfn_path, "w") as fh:
        fh.writelines(lines)

    # Append to the edit log if any unit numbers were changed
    if log_entries:
        log_path = os.path.join(wd, "modflow_EditLog.txt")
        ts2 = datetime.datetime.now().strftime("[%m/%d/%y %H:%M:%S]")
        with open(log_path, "a") as fh:
            for entry in log_entries:
                fh.write(f"{ts2} -> {entry}\n")

    return mfn_path


# ---------------------------------------------------------------------------
# Modify the MODFLOW output-control file
# ---------------------------------------------------------------------------

def modify_modflow_oc(swatmf_folder: str | os.PathLike) -> str:
    """Update ``HEAD SAVE UNIT`` numbers in the MODFLOW output-control file.

    Adds 5000 to any ``HEAD SAVE UNIT`` values with fewer than 4 digits.
    This mirrors ``modify_modflow_oc`` in ``modflow_functions.py``.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    str
        Absolute path to the modified ``.oc`` file.
    """
    wd = str(swatmf_folder)
    oc_path = _find_single_file(wd, ".oc")

    ts = datetime.datetime.now().strftime(" - %m/%d/%y %H:%M:%S -")
    info = f"# Output Control file, modified by swatmf package {_EXPORT_VERSION}{ts}\n"
    log_entries: list[str] = []
    lines = [info]

    with open(oc_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if (
                line.startswith("HEAD SAVE UNIT")
                and len(parts) >= 4
                and len(parts[3]) < 4
            ):
                old_unit = parts[3]
                new_unit = int(parts[3]) + 5000
                parts[3] = str(new_unit)
                line = "\t".join(parts) + "\n"
                log_entries.append(
                    f"{os.path.basename(oc_path)}: Unit number {old_unit} → {new_unit}"
                )
            lines.append(line)

    with open(oc_path, "w") as fh:
        fh.writelines(lines)

    if log_entries:
        log_path = os.path.join(wd, "modflow_EditLog.txt")
        ts2 = datetime.datetime.now().strftime("[%m/%d/%y %H:%M:%S]")
        with open(log_path, "a") as fh:
            for entry in log_entries:
                fh.write(f"{ts2} -> {entry}\n")

    return oc_path
