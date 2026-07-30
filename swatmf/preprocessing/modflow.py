"""
swatmf.preprocessing.modflow
==============================
Pure-Python helpers for reading and writing MODFLOW-related files that are
part of the SWAT-MODFLOW pre-processing workflow.

Where possible the functions delegate to `flopy <https://flopy.readthedocs.io>`_
for robust, format-aware I/O.  Manual text parsing is kept as a lightweight
fallback for the few cases where the full flopy model-load overhead would be
unnecessary.

Functions exposed
-----------------
parse_dis_file         — Parse a MODFLOW discretisation (.dis) file via flopy.
grid_row_col           — Build full (row, col) arrays for every grid cell.
parse_riv_file         — Parse a MODFLOW river-package (.riv) file.
write_riv_file         — Write (or overwrite) a MODFLOW river-package (.riv) file.
compute_riv_params     — Derive RIV parameters from a ``river_grid`` table.
create_modflow_obs     — Write the ``modflow.obs`` observation file.
read_modflow_obs       — Read an existing ``modflow.obs`` into a DataFrame.
create_mf_model        — Build a new MODFLOW model from scratch using flopy.
create_modflow_mfn     — Generate ``modflow.mfn`` from the MODFLOW name file.
modify_modflow_oc      — Update unit numbers in the MODFLOW output-control file.
check_modflow_files    — Validate MODFLOW folder; generate modflow.mfn and fix .oc.
create_mf_grid         — Build the ``mf_grid.gpkg`` polygon grid from a .dis file
                         and NW-corner coordinates (MODFLOW Option 2 / Scenario B).
import_mf_grid         — Import an existing grid shapefile/GeoPackage and annotate it
                         with grid_id, row, col, top_elev (MODFLOW Option 1 / Scenario A).
build_mf_model_from_dem — Build a complete MODFLOW model from a DEM raster (Scenario C):
                          resamples the DEM to a regular grid, derives bot_elev / sy /
                          initial_head from scalars or rasters, calls create_mf_model,
                          and writes mf_grid.gpkg in one step.
"""

from __future__ import annotations

import csv
import datetime
import glob
import os
from typing import NamedTuple, Optional, Union

import flopy
import flopy.modflow as fm
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


class MFModelResult(NamedTuple):
    """Result returned by :func:`build_mf_model_from_dem`.

    Attributes
    ----------
    mf : flopy.modflow.Modflow
        The constructed (and written) flopy model object.
    grid_path : str
        Absolute path to the written ``mf_grid.gpkg`` GeoPackage.
    x_origin : float
        X coordinate (easting) of the north-west corner of the grid used when
        building the polygon grid.
    y_origin : float
        Y coordinate (northing) of the north-west corner of the grid.
    nrow : int
        Number of rows in the MODFLOW grid.
    ncol : int
        Number of columns in the MODFLOW grid.
    """

    mf: object          # flopy.modflow.Modflow — avoid circular import annotation
    grid_path: str
    x_origin: float
    y_origin: float
    nrow: int
    ncol: int


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
    """Parse the MODFLOW discretisation (``.dis``) file using flopy.

    flopy handles the full range of MODFLOW-2005 free-format DIS syntax
    (INTERNAL / CONSTANT / EXTERNAL arrays, comment lines, etc.), making
    this more robust than manual text parsing.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (contains the ``.dis`` file and the
        corresponding ``.nam`` name file).

    Returns
    -------
    DisInfo
        Named-tuple with grid dimensions, cell sizes, and land-surface
        elevations.

    Notes
    -----
    flopy needs the MODFLOW name (``.nam``) file to load the model.  If no
    ``.nam`` file is found the function falls back to lightweight manual
    parsing of the ``.dis`` file.
    """
    wd = str(swatmf_folder)
    dis_path = _find_single_file(wd, ".dis")

    # --- flopy-based load (preferred) -----------------------------------------
    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    if nam_files:
        mf_name = os.path.splitext(os.path.basename(nam_files[0]))[0]
        try:
            mf = fm.Modflow.load(
                mf_name + ".nam",
                model_ws=wd,
                load_only=["dis"],
                check=False,
                verbose=False,
            )
            dis = mf.get_package("DIS")
            nrow = int(dis.nrow)
            ncol = int(dis.ncol)
            # delr / delc may be scalar or array — take first value
            delr = float(np.asarray(dis.delr.array).flat[0])
            delc = float(np.asarray(dis.delc.array).flat[0])
            top_elevs = np.asarray(dis.top.array).flatten().tolist()
            # steady is a boolean array, one entry per stress period
            transient = bool(not np.all(dis.steady.array))
            return DisInfo(
                nrow=nrow,
                ncol=ncol,
                delr=delr,
                delc=delc,
                n_cells=nrow * ncol,
                top_elevs=top_elevs,
                transient=transient,
            )
        except Exception:
            pass  # fall through to manual parse

    # --- Manual fallback (no .nam file, or flopy load failed) -----------------
    with open(dis_path, "r") as fh:
        data = [
            line.replace("\n", "").split()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]

    nrow = int(data[0][1])
    ncol = int(data[0][2])

    if data[2][0].upper() == "INTERNAL":
        delr = float(data[3][1])
        delc = float(data[4][1])
        elev_start = 5
    else:
        delr = float(data[2][1])
        delc = float(data[3][1])
        elev_start = 4

    top_elevs: list[float] = []
    ii = elev_start
    while ii < len(data) and data[ii][0].upper() != "INTERNAL":
        top_elevs.extend(float(v) for v in data[ii])
        ii += 1

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
    """Parse the MODFLOW river-package (``.riv``) file via flopy.

    flopy is used when a ``.nam`` file is available; otherwise the function
    falls back to lightweight manual text parsing.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``row``, ``col``, ``stage``, ``cond``, ``rbot``.
        One row per river cell (all stress-period data from period 0).
    """
    wd = str(swatmf_folder)

    # --- flopy-based load (preferred) -----------------------------------------
    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    if nam_files:
        mf_name = os.path.splitext(os.path.basename(nam_files[0]))[0]
        try:
            mf = fm.Modflow.load(
                mf_name + ".nam",
                model_ws=wd,
                load_only=["riv"],
                check=False,
                verbose=False,
            )
            riv_pkg = mf.get_package("RIV")
            if riv_pkg is not None:
                # stress_period_data[0] is a recarray with fields
                # layer, row, col, stage, cond, rbot (0-based indices)
                spd = riv_pkg.stress_period_data[0]
                df = pd.DataFrame(spd)
                # Convert to 1-based to match the rest of this module
                df["layer"] = df["layer"] + 1
                df["row"]   = df["row"]   + 1
                df["col"]   = df["col"]   + 1
                return df[["layer", "row", "col", "stage", "cond", "rbot"]]
        except Exception:
            pass  # fall through to manual parse

    # --- Manual fallback ------------------------------------------------------
    path = _find_single_file(wd, ".riv")
    with open(path, "r") as fh:
        data = [
            line.replace("\n", "").split()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]

    n_riv = int(data[0][0])
    records = []
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
    """Write (or overwrite) the MODFLOW ``.riv`` file with river-cell data.

    Accepts both the column names produced by :func:`parse_riv_file`
    (``stage``, ``cond``, ``rbot``) and those produced by
    :func:`compute_riv_params` (``riv_stage``, ``riv_cond``, ``riv_bot``).

    If no ``.riv`` file already exists in *swatmf_folder* a new one is
    created, deriving the filename from the ``.nam`` file's model name (or
    falling back to ``"modflow.riv"``).

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    riv_df : pd.DataFrame
        River-cell data.  Required columns: ``row``, ``col``, and one of
        (``stage`` / ``riv_stage``), (``cond`` / ``riv_cond``),
        (``rbot`` / ``riv_bot``).  ``layer`` defaults to 1 if absent.

    Returns
    -------
    str
        Absolute path to the written ``.riv`` file.
    """
    folder = str(swatmf_folder)

    # ── Resolve column name variants ─────────────────────────────────────────
    df = riv_df.rename(columns={
        "riv_stage": "stage",
        "riv_cond":  "cond",
        "riv_bot":   "rbot",
    })

    # ── Resolve file path (create if missing) ────────────────────────────────
    riv_matches = glob.glob(os.path.join(folder, "*.riv"))
    if riv_matches:
        path = riv_matches[0]
        action = "overwritten"
    else:
        # Derive name from the .nam file; fall back to "modflow.riv".
        nam_matches = glob.glob(os.path.join(folder, "*.nam"))
        if nam_matches:
            stem = os.path.splitext(os.path.basename(nam_matches[0]))[0]
        else:
            stem = "modflow"
        path = os.path.join(folder, f"{stem}.riv")
        action = "created"

    n = len(df)
    ts = datetime.datetime.now().strftime("- %m/%d/%y %H:%M:%S -")
    header = (
        f"# {os.path.basename(path)} {action} by swatmf package "
        f"{_EXPORT_VERSION}{ts}"
    )
    # MODFLOW Fortran reads the first two lines as plain integers — must NOT be
    # CSV-quoted.  Write them directly rather than through csv.writer so that
    # fields containing the tab delimiter are never wrapped in double-quotes.
    n_header = f"{n}\t0\t\t\t# Number of river cells"

    with open(path, "w", newline="") as fh:
        fh.write(header + "\r\n")
        fh.write(n_header + "\r\n")
        fh.write(n_header + "\r\n")  # written twice — matches plugin behaviour
        writer = csv.writer(fh, delimiter="\t")
        for _, r in df.iterrows():
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
    *,
    output_dir: str | os.PathLike | None = None,
) -> str:
    """Write the ``modflow.obs`` observation-cell file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory that contains the ``.dis`` file.
        Used to resolve ``row``/``col`` from ``grid_id`` when those columns
        are absent.  Also used as the output directory unless *output_dir*
        is given.
    obs_df : pd.DataFrame
        Observation-cell table.  May be empty (zero rows) — in that case a
        header-only file with ``0`` cells is written and the ``.dis`` file is
        not read.  When non-empty, must contain at least:

        * ``grid_id`` — MODFLOW grid cell ID (integer, 1-based)
        * ``layer``   — MODFLOW layer number (integer, usually 1)
        * ``top_elev``— land-surface elevation at the cell centre (float)

        The ``row`` and ``col`` columns are computed automatically from
        ``grid_id`` and the ``.dis`` file if they are not already present.
    output_dir : str or path-like, optional
        Directory where ``modflow.obs`` is written.  Defaults to
        *swatmf_folder* when omitted.

    Returns
    -------
    str
        Absolute path to the written ``modflow.obs`` file.

    Notes
    -----
    File format matches the reference files shipped with SWAT-MODFLOW::

        MODFLOW observation cells (number of cells, I,J,K for each cell)
        <N>
        <row> <col> <layer>
        ...

    Only row, col, layer are written — the Fortran executable reads exactly
    three integers per data line with list-directed I/O.

    Examples
    --------
    >>> import pandas as pd
    >>> obs = pd.DataFrame({"grid_id": [10, 45], "layer": [1, 1], "top_elev": [123.4, 119.7]})
    >>> path = create_modflow_obs(wd, obs)
    """
    wd      = str(swatmf_folder)
    out_dir = str(output_dir) if output_dir is not None else wd
    df      = obs_df.copy()

    if len(df) == 0:
        # No wells — write a zero-record file without reading the .dis file.
        out_path = os.path.join(out_dir, "modflow.obs")
        with open(out_path, "w", newline="") as fh:
            fh.write("MODFLOW observation cells (number of cells, I,J,K for each cell)\n")
            fh.write("0\n")
        return out_path

    # ── Non-empty: resolve row/col from grid_id via .dis ────────────────────
    dis = parse_dis_file(wd)
    rows_all, cols_all = grid_row_col(dis)

    df["grid_id"] = df["grid_id"].astype(int)
    df = df.sort_values("grid_id").reset_index(drop=True)

    if "row" not in df.columns or "col" not in df.columns:
        n_cells = dis.n_cells
        bad = [gid for gid in df["grid_id"] if gid < 1 or gid > n_cells]
        if bad:
            raise ValueError(
                f"grid_id value(s) {bad} are out of range for this model "
                f"({dis.nrow} rows x {dis.ncol} cols = {n_cells} cells, "
                f"valid range 1-{n_cells}).  "
                "Note: column names in observed-data CSVs (e.g. 'g_5699') "
                "are well identifier numbers, not MODFLOW grid cell IDs.  "
                "Use a spatial join of the observation-well point shapefile "
                "with mf_grid.gpkg to find the correct grid_id for each well."
            )
        df["row"] = [rows_all[gid - 1] for gid in df["grid_id"]]
        df["col"] = [cols_all[gid - 1] for gid in df["grid_id"]]

    out_path = os.path.join(out_dir, "modflow.obs")

    with open(out_path, "w", newline="") as fh:
        fh.write("MODFLOW observation cells (number of cells, I,J,K for each cell)\n")
        fh.write(f"{len(df)}\n")
        for _, r in df.iterrows():
            fh.write(f"{int(r['row'])} {int(r['col'])} {int(r['layer'])}\n")

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
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            # Skip text header lines and the count line (non-numeric first token)
            try:
                row_val = int(parts[0])
            except (ValueError, IndexError):
                continue
            # Count line has exactly 1 token; data lines have 3 (row col layer)
            if len(parts) < 3:
                continue
            try:
                col_val = int(parts[1])
                lay_val = int(parts[2])
            except (ValueError, IndexError):
                continue
            rows.append({"row": row_val, "col": col_val, "layer": lay_val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Create a new MODFLOW model from scratch using flopy
# ---------------------------------------------------------------------------

def create_mf_model(
    mf_folder: str | os.PathLike,
    mf_name: str,
    top_elev: np.ndarray,
    bot_elev: np.ndarray,
    delr: float,
    delc: float,
    hk: Union[float, np.ndarray],
    ss: Union[float, np.ndarray],
    sy: Union[float, np.ndarray],
    initial_head: Union[float, np.ndarray],
    riv_df: Optional[pd.DataFrame],
    sim_duration: int,
    *,
    vka: float = 0.1,
    laytype: int = 1,
    evt: Optional[Union[float, np.ndarray]] = None,
    rch: float = 0.0,
    ibound: Optional[np.ndarray] = None,
    nodata: float = -9999.0,
    headtol: float = 0.1,
    fluxtol: float = 500.0,
    maxiterout: int = 1000,
) -> flopy.modflow.Modflow:
    """Build a new MODFLOW-NWT model using flopy and write its input files.

    This replicates the ``writeMFmodel`` function in ``pyfolder/writeMF.py``
    without any QGIS / PyQt dependency.  All spatial data (elevation arrays,
    river-cell table, etc.) must be supplied as numpy arrays or DataFrames —
    typically loaded from GeoTIFF rasters or GeoPackage files.

    Parameters
    ----------
    mf_folder : str or path-like
        Directory where the MODFLOW input files will be written (the
        SWAT-MODFLOW working directory).
    mf_name : str
        Base name for the MODFLOW model (used for all generated file names).
    top_elev : np.ndarray, shape (nrow, ncol)
        Land-surface elevation array [length units of model].
    bot_elev : np.ndarray, shape (nrow, ncol)
        Aquifer bottom elevation array.
    delr : float
        Cell width along rows (y-spacing) [length units].
    delc : float
        Cell width along columns (x-spacing) [length units].
    hk : float or np.ndarray
        Horizontal hydraulic conductivity [length/time].  A scalar applies
        the same value to every cell.
    ss : float or np.ndarray
        Specific storage [1/length].
    sy : float or np.ndarray
        Specific yield [dimensionless].
    initial_head : float or np.ndarray
        Initial hydraulic head [length units].
    riv_df : pd.DataFrame or None
        River-package cell data.  Must contain columns ``layer``, ``row``,
        ``col``, ``riv_stage``, ``riv_cond``, ``riv_bot``.  Use
        :func:`compute_riv_params` to derive these from ``river_grid``
        attributes.  Pass ``None`` to omit the RIV package entirely (useful
        when building a draft model before river cells are defined).
    sim_duration : int
        Total simulation duration [days].  Adding ~100 days is recommended
        to account for leap years (matches the plugin convention).
    vka : float, optional
        Vertical anisotropy ratio (VKA in UPW package).  Default 0.1.
    laytype : int, optional
        Layer type: 0 = confined, 1 = convertible.  Default 1.
    evt : float or np.ndarray, optional
        Evapotranspiration rate.  If ``None`` the EVT package is omitted.
    rch : float, optional
        Background recharge rate [length/time].  SWAT-MODFLOW overrides this
        at run time; the default of 0 is appropriate.
    ibound : np.ndarray, optional
        IBOUND array.  If ``None`` it is derived from *top_elev* using
        *nodata* as the no-data sentinel value.
    nodata : float, optional
        No-data value in the elevation arrays.  Default ``-9999.0``.
    headtol : float, optional
        NWT solver head tolerance.  Default 0.1.
    fluxtol : float, optional
        NWT solver flux tolerance.  Default 500.
    maxiterout : int, optional
        Maximum outer iterations for NWT solver.  Default 1000.

    Returns
    -------
    flopy.modflow.Modflow
        The constructed (and written) flopy model object.

    Examples
    --------
    >>> import numpy as np, rasterio
    >>> with rasterio.open("top_elev.tif") as src:
    ...     top = src.read(1).astype(float)
    ...     delc = src.res[0]
    ...     delr = src.res[1]
    >>> with rasterio.open("bot_elev.tif") as src:
    ...     bot = src.read(1).astype(float)
    >>> mf = create_mf_model(
    ...     mf_folder=wd, mf_name="mymodel",
    ...     top_elev=top, bot_elev=bot,
    ...     delr=delr, delc=delc,
    ...     hk=5.0, ss=1e-4, sy=0.2,
    ...     initial_head=top - 2.0,
    ...     riv_df=riv_params,
    ...     sim_duration=365,
    ... )
    """
    wd = str(mf_folder)
    nrow, ncol = top_elev.shape

    # IBOUND: 1 where data exists, 0 where nodata
    if ibound is None:
        ibound_arr = np.where(top_elev != nodata, 1, 0).astype(np.int32)
    else:
        ibound_arr = np.asarray(ibound, dtype=np.int32)

    # Build flopy model object
    mf = fm.Modflow(mf_name, model_ws=wd, version="mfnwt")

    # DIS package — single layer, transient, daily time steps
    fm.ModflowDis(
        mf,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top_elev,
        botm=bot_elev,
        itmuni=4,           # 4 = days
        perlen=sim_duration,
        nstp=sim_duration,
        steady=False,
    )

    # BAS package — boundary conditions + initial head
    fm.ModflowBas(mf, ibound=ibound_arr, strt=initial_head)

    # NWT solver
    fm.ModflowNwt(
        mf,
        headtol=headtol,
        fluxtol=fluxtol,
        maxiterout=maxiterout,
        Continue=False,
        iprnwt=1,
        linmeth=2,
    )

    # UPW package — hydraulic properties
    fm.ModflowUpw(mf, hk=hk, ss=ss, sy=sy, vka=vka, laytyp=laytype)

    # EVT package (optional)
    if evt is not None:
        fm.ModflowEvt(mf, nevtop=3, evtr=evt)

    # RIV package — convert 1-based row/col to 0-based for flopy
    if riv_df is not None and len(riv_df) > 0:
        riv_sorted = riv_df.sort_values(
            # Prefer sorting by grid_id for consistent ordering; fall back to row
            # if grid_id is absent (e.g. when riv_df comes directly from parse_riv_file).
            riv_df.columns.intersection(["grid_id", "row"]).tolist() or riv_df.columns[:1].tolist()
        )
        riv_array = np.column_stack([
            riv_sorted.get("layer", pd.Series(np.ones(len(riv_sorted), dtype=int))).values - 1,
            riv_sorted["row"].values - 1,
            riv_sorted["col"].values - 1,
            riv_sorted["riv_stage"].values.astype(float),
            riv_sorted["riv_cond"].values.astype(float),
            riv_sorted["riv_bot"].values.astype(float),
        ])
        fm.ModflowRiv(mf, stress_period_data={0: riv_array})

    # RCH package — background recharge (SWAT overrides at run time)
    fm.ModflowRch(mf, rech=rch)

    # OC package
    fm.ModflowOc(mf, ihedfm=1)

    # Write all input files
    mf.write_input()

    return mf


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

    # If a .riv file is present but not yet listed, inject it before OC/DATA.
    # SWAT-MODFLOW3.exe accesses MODFLOW's internal RIV package arrays when
    # swatmf_river2grid.txt has any river cells; if the RIV package is absent
    # from the name file those arrays are null → error 157 access violation.
    riv_files = glob.glob(os.path.join(wd, "*.riv"))
    already_listed = any(
        ln.strip().upper().startswith("RIV") for ln in lines
        if not ln.strip().startswith("#")
    )
    if riv_files and not already_listed:
        riv_name = os.path.basename(riv_files[0])
        riv_line = f"RIV\t5018\t{riv_name}\n"
        # Insert before the first OC or DATA line
        insert_at = len(lines)
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if stripped and not stripped.startswith("#"):
                tag = stripped.split()[0].upper()
                if tag in ("OC", "DATA", "DATA(BINARY)"):
                    insert_at = i
                    break
        lines.insert(insert_at, riv_line)
        log_entries.append(f"modflow.mfn: injected RIV line for {riv_name}")

    mfn_path = os.path.join(wd, "modflow.mfn")
    with open(mfn_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    # Append to the edit log if any unit numbers were changed
    if log_entries:
        log_path = os.path.join(wd, "modflow_EditLog.txt")
        ts2 = datetime.datetime.now().strftime("[%m/%d/%y %H:%M:%S]")
        with open(log_path, "a", encoding="utf-8") as fh:
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

    with open(oc_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    if log_entries:
        log_path = os.path.join(wd, "modflow_EditLog.txt")
        ts2 = datetime.datetime.now().strftime("[%m/%d/%y %H:%M:%S]")
        with open(log_path, "a", encoding="utf-8") as fh:
            for entry in log_entries:
                fh.write(f"{ts2} -> {entry}\n")

    return oc_path


# ---------------------------------------------------------------------------
# Check MODFLOW folder and prepare modflow.mfn / fix .oc
# ---------------------------------------------------------------------------

def check_modflow_files(swatmf_folder: str | os.PathLike) -> dict:
    """Validate MODFLOW inputs and prepare the ``modflow.mfn`` / ``.oc`` files.

    Replicates the **Check MODFLOW file** button (``pushButton_checkMF →
    checkMF``) from the QSWATMOD2 plugin, which calls
    ``create_modflow_mfn`` and ``modify_modflow_oc``.

    Steps performed
    ~~~~~~~~~~~~~~~
    1. Verify that the required MODFLOW package files (``.dis``, ``.nam``,
       ``.oc``) are present in *swatmf_folder*.
    2. Generate ``modflow.mfn`` from the ``.nam`` file (adjusts unit numbers
       by adding 5000 where required).
    3. Update ``HEAD SAVE UNIT`` numbers in the ``.oc`` output-control file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        The SWAT-MODFLOW working directory (the folder that contains the
        MODFLOW input files — ``.dis``, ``.nam``, ``.oc``, etc.).

    Returns
    -------
    dict
        A summary dictionary with the following keys:

        * ``"status"`` — ``"ok"`` on success, ``"warning"`` if some optional
          files were missing.
        * ``"dis_file"`` — path to the ``.dis`` file found.
        * ``"nam_file"`` — path to the ``.nam`` file found.
        * ``"oc_file"``  — path to the ``.oc`` file found (or ``None``).
        * ``"mfn_file"`` — path to the written ``modflow.mfn`` file.
        * ``"messages"`` — list of informational messages.

    Raises
    ------
    FileNotFoundError
        If the ``.dis`` or ``.nam`` file is missing from *swatmf_folder*.

    Examples
    --------
    >>> from swatmf.preprocessing.modflow import check_modflow_files
    >>> result = check_modflow_files(wd)
    >>> print(result["status"])
    ok
    >>> for msg in result["messages"]:
    ...     print(msg)
    """
    wd = str(swatmf_folder)
    messages: list[str] = []
    status = "ok"

    # 1 — validate required files ------------------------------------------
    dis_path = _find_single_file(wd, ".dis")   # raises if missing
    nam_path = _find_single_file(wd, ".nam")   # raises if missing
    messages.append(f"Found .dis  : {os.path.basename(dis_path)}")
    messages.append(f"Found .nam  : {os.path.basename(nam_path)}")

    # .oc is optional (not all MODFLOW models use it)
    oc_files = glob.glob(os.path.join(wd, "*.oc"))
    oc_path: str | None = oc_files[0] if oc_files else None
    if oc_path:
        messages.append(f"Found .oc   : {os.path.basename(oc_path)}")
    else:
        messages.append("No .oc file found — skipping output-control update.")
        status = "warning"

    # 2 — generate modflow.mfn from .nam -----------------------------------
    mfn_path = create_modflow_mfn(wd)
    messages.append(f"Written     : {os.path.basename(mfn_path)}")

    # 3 — fix .oc unit numbers ---------------------------------------------
    if oc_path:
        oc_out = modify_modflow_oc(wd)
        messages.append(f"Updated     : {os.path.basename(oc_out)}")

    return {
        "status":   status,
        "dis_file": dis_path,
        "nam_file": nam_path,
        "oc_file":  oc_path,
        "mfn_file": mfn_path,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Create mf_grid GeoPackage from a MODFLOW .dis file and NW corner coordinates
# ---------------------------------------------------------------------------

def create_mf_grid(
    swatmf_folder: str | os.PathLike,
    x_origin: float,
    y_origin: float,
    *,
    crs: str | int | None = None,
    output_dir: str | os.PathLike | None = None,
    extra_cols: int = 0,
    extra_rows: int = 0,
    output_filename: str = "mf_grid.gpkg",
) -> str:
    """Build the MODFLOW grid polygon GeoPackage from a ``.dis`` file.

    Replicates the **Create Grid** button (``pushButton_createMF → createMF
    → MF_grid → create_grid_id → create_row → create_col → create_top_elev``)
    from the QSWATMOD2 plugin.  This is the pure-Python / GeoPandas equivalent
    for **MODFLOW Option 2 / Scenario B** — MODFLOW input files exist but no
    ``mf_grid`` shapefile is available yet.

    The function reads the ``.dis`` file to determine the grid dimensions,
    builds one rectangular polygon per cell in row-major order, and annotates
    every cell with ``grid_id``, ``row``, ``col``, and ``top_elev`` attributes.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (must contain a ``.dis`` file).
    x_origin : float
        X coordinate (easting) of the **north-west corner** of the grid
        [same units as the model CRS, typically metres].
    y_origin : float
        Y coordinate (northing) of the north-west corner of the grid.
    crs : str, int, or None, optional
        Coordinate reference system for the output GeoPackage — any value
        accepted by :func:`pyproj.CRS.from_user_input` (e.g. ``"EPSG:27700"``
        or an integer EPSG code).  If ``None``, no CRS is assigned; the layer
        will still be created and usable but QGIS will show a warning.
    output_dir : str or path-like, optional
        Directory where the GeoPackage is written.  Defaults to
        *swatmf_folder*.
    extra_cols : int, optional
        Number of extra columns to add to the right of the grid (replicates
        the ``spinBox_col`` extra-column expansion feature in the plugin).
        Default 0.
    extra_rows : int, optional
        Number of extra rows to add below the grid (replicates
        ``spinBox_row``).  Default 0.
    output_filename : str, optional
        Name of the output GeoPackage file.  Default ``"mf_grid.gpkg"``.

    Returns
    -------
    str
        Absolute path to the written ``mf_grid.gpkg`` GeoPackage.

    Raises
    ------
    FileNotFoundError
        If no ``.dis`` file is found in *swatmf_folder*.
    ImportError
        If :mod:`geopandas` or :mod:`shapely` are not installed.

    Examples
    --------
    >>> from swatmf.preprocessing.modflow import create_mf_grid
    >>> path = create_mf_grid(
    ...     wd,
    ...     x_origin=367000.0,
    ...     y_origin=6234000.0,
    ...     crs="EPSG:27700",
    ... )
    >>> print("mf_grid written to:", path)
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise ImportError(
            "geopandas and shapely are required for create_mf_grid.  "
            "Install with: pip install geopandas"
        ) from exc

    wd = str(swatmf_folder)
    out_dir = str(output_dir) if output_dir is not None else wd
    os.makedirs(out_dir, exist_ok=True)

    dis = parse_dis_file(wd)
    nrow  = dis.nrow  + extra_rows
    ncol  = dis.ncol  + extra_cols
    delr  = dis.delr
    delc  = dis.delc
    # top elevations come from the original (un-extended) grid — pad extras
    # with the mean elevation so they do not break downstream workflows.
    top_elevs_base = list(dis.top_elevs)
    mean_elev = float(np.mean(top_elevs_base)) if top_elevs_base else 0.0
    # For an extended grid repeat the mean for any extra cells
    n_orig = dis.nrow * dis.ncol
    top_elevs_ext = top_elevs_base + [mean_elev] * (nrow * ncol - n_orig)

    rows_list: list[int] = []
    cols_list: list[int] = []
    grid_ids:  list[int] = []
    geoms:     list[Polygon] = []

    gid = 1
    for r in range(1, nrow + 1):
        for c in range(1, ncol + 1):
            x_left  = x_origin + (c - 1) * delc
            x_right = x_origin +  c      * delc
            y_top   = y_origin - (r - 1) * delr
            y_bot   = y_origin -  r      * delr
            geoms.append(Polygon([
                (x_left,  y_top),
                (x_right, y_top),
                (x_right, y_bot),
                (x_left,  y_bot),
                (x_left,  y_top),
            ]))
            grid_ids.append(gid)
            rows_list.append(r)
            cols_list.append(c)
            gid += 1

    gdf = gpd.GeoDataFrame(
        {
            "grid_id":  grid_ids,
            "row":      rows_list,
            "col":      cols_list,
            "top_elev": top_elevs_ext,
        },
        geometry=geoms,
        crs=crs,
    )

    out_path = os.path.normpath(os.path.join(out_dir, output_filename))
    gdf.to_file(out_path, driver="GPKG")
    return out_path


# ---------------------------------------------------------------------------
# Import an existing MODFLOW grid shapefile / GeoPackage and annotate it
# ---------------------------------------------------------------------------

def import_mf_grid(
    src_path: str | os.PathLike,
    swatmf_folder: str | os.PathLike,
    *,
    output_dir: str | os.PathLike | None = None,
    output_filename: str = "mf_grid.gpkg",
    grid_id_col: str | None = None,
    row_col: str | None = None,
    col_col: str | None = None,
) -> str:
    """Import an existing MODFLOW grid shapefile/GeoPackage and annotate it.

    Replicates the **Import MF grid shapefile** button
    (``pushButton_MF_grid_shapefile → import_mf_grid → create_grid_id →
    create_row → create_col → create_top_elev``) from the QSWATMOD2 plugin.
    This is the pure-Python / GeoPandas equivalent for **MODFLOW Option 1 /
    Scenario A** — a pre-existing ``mf_grid`` shapefile is available.

    The function:

    1. Reads *src_path* into a GeoDataFrame and fixes any geometry errors.
    2. Adds ``grid_id`` (1-based sequential integer) if absent.
    3. Derives ``row`` and ``col`` from the ``.dis`` file if absent.
    4. Joins ``top_elev`` from the ``.dis`` land-surface array if absent.
    5. Writes the annotated grid as ``mf_grid.gpkg`` in *output_dir*.

    Parameters
    ----------
    src_path : str or path-like
        Path to the source MODFLOW grid shapefile (``.shp``) or GeoPackage
        (``.gpkg``).
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (must contain a ``.dis`` file used to
        derive ``row``, ``col``, and ``top_elev``).
    output_dir : str or path-like, optional
        Destination directory for the output GeoPackage.  Defaults to
        *swatmf_folder*.
    output_filename : str, optional
        Name of the output file.  Default ``"mf_grid.gpkg"``.
    grid_id_col : str, optional
        If the source file already contains a grid-cell ID column under a
        different name, supply that name here and it will be renamed to
        ``"grid_id"``.  If ``None``, ``grid_id`` will be created (1 … N).
    row_col : str, optional
        Existing column name for row numbers.  If ``None``, ``row`` is
        computed from the ``.dis`` file.
    col_col : str, optional
        Existing column name for column numbers.  If ``None``, ``col`` is
        computed from the ``.dis`` file.

    Returns
    -------
    str
        Absolute path to the written ``mf_grid.gpkg``.

    Raises
    ------
    FileNotFoundError
        If *src_path* does not exist or no ``.dis`` file is in
        *swatmf_folder*.
    ImportError
        If :mod:`geopandas` is not installed.

    Examples
    --------
    >>> from swatmf.preprocessing.modflow import import_mf_grid
    >>> path = import_mf_grid(
    ...     src_path="C:/data/MODFLOW_Grid.shp",
    ...     swatmf_folder=wd,
    ... )
    >>> print("mf_grid written to:", path)
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for import_mf_grid.  "
            "Install with: pip install geopandas"
        ) from exc

    src = str(src_path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Source grid file not found: {src!r}")

    wd = str(swatmf_folder)
    out_dir = str(output_dir) if output_dir is not None else wd
    os.makedirs(out_dir, exist_ok=True)

    # 1 — load and fix geometries
    gdf = gpd.read_file(src)
    gdf["geometry"] = gdf.geometry.buffer(0)  # fix minor topology errors

    # 2 — add / rename grid_id
    if grid_id_col is not None and grid_id_col in gdf.columns:
        if grid_id_col != "grid_id":
            gdf = gdf.rename(columns={grid_id_col: "grid_id"})
    if "grid_id" not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf["grid_id"] = gdf.index + 1

    # 3 — add row / col from .dis if not already present
    dis = None
    if (row_col is None or row_col not in gdf.columns) and "row" not in gdf.columns:
        dis = parse_dis_file(wd)
        rows_all, cols_all = grid_row_col(dis)
        n = len(gdf)
        gdf["row"] = rows_all[:n]
        gdf["col"] = cols_all[:n]
    else:
        if row_col and row_col != "row" and row_col in gdf.columns:
            gdf = gdf.rename(columns={row_col: "row"})
        if col_col and col_col != "col" and col_col in gdf.columns:
            gdf = gdf.rename(columns={col_col: "col"})

    # 4 — add top_elev from .dis if not already present
    if "top_elev" not in gdf.columns:
        if dis is None:
            dis = parse_dis_file(wd)
        top_elevs = dis.top_elevs
        n = len(gdf)
        gdf["top_elev"] = top_elevs[:n] if len(top_elevs) >= n else (
            top_elevs + [float(np.mean(top_elevs))] * (n - len(top_elevs))
        )

    # 5 — write output
    out_path = os.path.normpath(os.path.join(out_dir, output_filename))
    gdf.to_file(out_path, driver="GPKG")
    return out_path


# ---------------------------------------------------------------------------
# swatmf_drain2sub.txt — link drain cells to SWAT subbasins
# ---------------------------------------------------------------------------

def write_drain2sub(
    mf_folder: str | os.PathLike,
    sub_path: "str | os.PathLike | gpd.GeoDataFrame",
    *,
    mfgrid_path: str | os.PathLike | None = None,
    sub_col: str = "Subbasin",
    out_path: str | os.PathLike | None = None,
) -> str:
    """Write ``swatmf_drain2sub.txt`` mapping each DRN cell to a SWAT subbasin.

    When ``drain_cells = True`` is set in ``swatmf_link.txt``, the SWAT-MODFLOW
    executable reads this file at start-up to know which subbasin channel each
    MODFLOW drain cell discharges into.  One subbasin number is written per
    drain cell, in the same order as the entries in the ``.drn`` file.

    Drain cells that straddle multiple subbasins are assigned to the subbasin
    with the largest intersection area.  Drain cells that fall outside all
    subbasins (e.g. in the GW-only domain) are assigned to the nearest subbasin
    centroid.

    File format (matches SWAT-MODFLOW Fortran reader)::

        <N>                   ! number of drain cells
        <subbasin_1>          ! one integer per cell, same order as .drn
        <subbasin_2>
        ...

    Parameters
    ----------
    mf_folder : str or path-like
        MODFLOW model workspace (contains the ``.drn`` and ``.nam`` files).
    sub_path : str, path-like, or GeoDataFrame
        SWAT subbasin polygon layer (must be in the same CRS as the MFgrid).
    mfgrid_path : str or path-like, optional
        Path to the MFgrid shapefile / GeoPackage.  Searched automatically
        inside *mf_folder* and common sibling GIS folders if omitted.
    sub_col : str
        Column name in *sub_path* that holds the integer subbasin number.
        Default ``"Subbasin"``.
    out_path : str or path-like, optional
        Output path for the file.  Defaults to
        ``<mf_folder>/swatmf_drain2sub.txt``.

    Returns
    -------
    str
        Path of the written file.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    wd = str(mf_folder)

    # ── Locate .drn file ─────────────────────────────────────────────────────
    drn_files = glob.glob(os.path.join(wd, "*.drn"))
    if not drn_files:
        raise FileNotFoundError(
            f"No .drn file found in {wd!r}.  "
            "Run add_spring_drain() or burn_drain_to_point() first."
        )
    drn_file = drn_files[0]

    # ── Parse DRN stress-period data (skip comment + 2 header lines) ─────────
    drn_cells: list[tuple[int, int, int]] = []   # (layer, row, col)  1-based
    with open(drn_file) as fh:
        lines = [ln for ln in fh if not ln.strip().startswith("#")]

    # Line 0: MXACTD  IDRNPB
    # Line 1: ITMP  NP   (stress period 1)
    # Lines 2+: layer row col elev conductance
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) >= 3:
            lay, row, col = int(parts[0]), int(parts[1]), int(parts[2])
            drn_cells.append((lay, row, col))

    n = len(drn_cells)
    if n == 0:
        raise ValueError(f"No drain cells found in {drn_file!r}")
    print(f"  Drain cells read from {os.path.basename(drn_file)}: {n}")

    # ── Load MFgrid ───────────────────────────────────────────────────────────
    if mfgrid_path is None:
        candidates = (
            glob.glob(os.path.join(wd, "mf_grid.gpkg"))
            + glob.glob(os.path.join(wd, "MFgrid*.shp"))
            + glob.glob(os.path.join(wd, "*grid*.shp"))
            + glob.glob(os.path.join(wd, "*grid*.gpkg"))
        )
        # Also look in sibling GIS folders
        parent = os.path.dirname(wd)
        for gis_sub in ["GIS/org_shps", "GIS", "org_shps", "shapefiles"]:
            candidates += (
                glob.glob(os.path.join(parent, gis_sub, "mf_grid.gpkg"))
                + glob.glob(os.path.join(parent, gis_sub, "MFgrid*.shp"))
                + glob.glob(os.path.join(parent, gis_sub, "*grid*.gpkg"))
            )
        mfgrid_path = next((p for p in candidates if os.path.isfile(p)), None)
        if mfgrid_path is None:
            raise FileNotFoundError(
                "MFgrid shapefile not found.  Pass mfgrid_path= explicitly."
            )

    grid_gdf = gpd.read_file(str(mfgrid_path))
    col_map  = {c: c.lower() for c in grid_gdf.columns}
    grid_gdf = grid_gdf.rename(columns=col_map)

    # ── Load subbasins ────────────────────────────────────────────────────────
    if isinstance(sub_path, gpd.GeoDataFrame):
        sub_gdf = sub_path.copy()
    else:
        sub_gdf = gpd.read_file(str(sub_path))

    if grid_gdf.crs is not None and sub_gdf.crs is not None:
        if grid_gdf.crs != sub_gdf.crs:
            sub_gdf = sub_gdf.to_crs(grid_gdf.crs)

    # ── Match each drain cell to its primary subbasin ─────────────────────────
    # Build a GeoDataFrame of the drain cells from the grid
    drn_df = gpd.GeoDataFrame(
        {"layer":   [c[0] for c in drn_cells],
         "row":     [c[1] for c in drn_cells],
         "col":     [c[2] for c in drn_cells]},
    )
    drn_df = drn_df.merge(
        grid_gdf[["row", "col", "grid_id", "geometry"]],
        on=["row", "col"], how="left",
    )
    drn_df = gpd.GeoDataFrame(drn_df, geometry="geometry", crs=grid_gdf.crs)
    drn_df["cell_area"] = drn_df.geometry.area

    # Intersection-based join to find subbasins; keep largest overlap
    intersected = gpd.overlay(
        drn_df.reset_index().rename(columns={"index": "drn_idx"}),
        sub_gdf[[sub_col, "geometry"]],
        how="intersection",
    )
    intersected["_area"] = intersected.geometry.area
    best = (
        intersected.sort_values("_area", ascending=False)
        .drop_duplicates(subset=["drn_idx"])
        .set_index("drn_idx")
    )

    subbasin_map: dict[int, int] = {}
    for i in range(n):
        if i in best.index:
            subbasin_map[i] = int(best.loc[i, sub_col])
        else:
            # Drain cell outside all subbasins — assign nearest centroid
            cell_geom = drn_df.iloc[i].geometry
            if cell_geom is None or cell_geom.is_empty:
                subbasin_map[i] = int(sub_gdf[sub_col].iloc[0])
            else:
                cx, cy = cell_geom.centroid.x, cell_geom.centroid.y
                dists = sub_gdf.geometry.centroid.distance(Point(cx, cy))
                subbasin_map[i] = int(sub_gdf.iloc[dists.idxmin()][sub_col])
            print(f"  Warning: drain cell row={drn_cells[i][1]} col={drn_cells[i][2]} "
                  f"is outside all subbasins — assigned to subbasin {subbasin_map[i]}")

    # ── Write the file ────────────────────────────────────────────────────────
    # Format (from smrt_read_drain2sub Fortran source):
    #   Line 1 : ndrn_subs  (count)
    #   Line 2 : blank      (read(6007,*) discards it)
    #   Lines 3+: drn_row  drn_col  sub_basin  (one per drain cell, 1-based)
    if out_path is None:
        out_path = os.path.join(wd, "swatmf_drain2sub.txt")

    with open(str(out_path), "w") as fh:
        fh.write(f"{n:12d}\n")
        fh.write("\n")                          # blank line consumed by read(6007,*)
        for i in range(n):
            lay, row, col = drn_cells[i]
            sub_num = subbasin_map[i]
            fh.write(f"{row:12d}{col:12d}{sub_num:12d}\n")
            print(f"  Drain {i+1}: row={row}  col={col}  → subbasin {sub_num}")

    print(f"  Written: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Internal helper – write a MODFLOW DRN file and splice it into the .nam
# ---------------------------------------------------------------------------

def _write_drn_and_update_nam(
    wd: str,
    nam_file: str,
    drn_records: list,
    layer: int,
) -> str:
    """Write a .drn file and inject the DRN line into *nam_file*.

    Parameters
    ----------
    wd        : model workspace directory
    nam_file  : full path to the .nam file
    drn_records : list of (row1, col1, elev, conductance) tuples (1-based)
    layer     : MODFLOW layer number (1-based)

    Returns the path of the written .drn file.
    """
    mf_name_base = os.path.splitext(os.path.basename(nam_file))[0]
    drn_path = os.path.join(wd, f"{mf_name_base}.drn")

    n = len(drn_records)
    with open(drn_path, "w") as fh:
        fh.write("# DRN package written by swatmf\n")
        fh.write(f"{n:10d}{0:10d}\n")       # MXACTD  IDRNPB
        fh.write(f"{n:10d}{0:10d}\n")       # ITMP    NP  (stress period 1)
        for r, c, elev, cond in drn_records:
            fh.write(f"{layer:10d}{r:10d}{c:10d}{elev:15.4f}{cond:15.4f}\n")

    drn_unit    = 21
    drn_basename = os.path.basename(drn_path)
    with open(nam_file, "r") as fh:
        nam_lines = fh.readlines()

    already_listed = any(
        ln.strip().upper().startswith("DRN")
        for ln in nam_lines
        if not ln.strip().startswith("#")
    )
    if not already_listed:
        drn_nam_line = f"DRN\t{drn_unit}\t{drn_basename}\n"
        insert_at = len(nam_lines)
        for i, ln in enumerate(nam_lines):
            stripped = ln.strip()
            if stripped and not stripped.startswith("#"):
                tag = stripped.split()[0].upper()
                if tag in ("OC", "DATA", "DATA(BINARY)"):
                    insert_at = i
                    break
        nam_lines.insert(insert_at, drn_nam_line)
        with open(nam_file, "w") as fh:
            fh.writelines(nam_lines)
        print(f"  DRN line injected into {os.path.basename(nam_file)}: unit {drn_unit}")
    else:
        # Update the existing line in-place (unit / filename stays the same)
        print(f"  DRN already listed in {os.path.basename(nam_file)} — file overwritten in place")

    return drn_path


# ---------------------------------------------------------------------------
# D8 flow-routing helpers
# ---------------------------------------------------------------------------

def _d8_flow_direction(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fdir_dr, fdir_dc) arrays — the row/col step of each cell's
    steepest-descent neighbour.  NaN cells and flat cells with no downslope
    neighbour get (-99, -99) (no valid direction)."""
    nrow, ncol = dem.shape
    # 8 neighbours: dr, dc, diagonal distance scale
    neighbours = [
        (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
        ( 0, -1, 1.0),                       ( 0, 1, 1.0),
        ( 1, -1, np.sqrt(2)), ( 1, 0, 1.0), ( 1, 1, np.sqrt(2)),
    ]
    best_slope = np.full((nrow, ncol), -np.inf)
    fdir_dr    = np.full((nrow, ncol), -99, dtype=int)
    fdir_dc    = np.full((nrow, ncol), -99, dtype=int)

    for dr, dc, dist in neighbours:
        # Source slice (the cell) and neighbour slice
        r_src = slice(max(0, -dr), nrow + min(0, -dr) or None)
        c_src = slice(max(0, -dc), ncol + min(0, -dc) or None)
        r_nbr = slice(max(0,  dr), nrow + min(0,  dr) or None)
        c_nbr = slice(max(0,  dc), ncol + min(0,  dc) or None)

        src_elev = dem[r_src, c_src]
        nbr_elev = dem[r_nbr, c_nbr]
        slope    = (src_elev - nbr_elev) / dist

        valid = np.isfinite(slope) & np.isfinite(src_elev) & np.isfinite(nbr_elev)
        update = valid & (slope > best_slope[r_src, c_src])

        # Write back into full arrays (need index offsets)
        r0 = max(0, -dr); c0 = max(0, -dc)
        for i, j in zip(*np.where(update)):
            ri, ci = i + r0, j + c0
            if slope[i, j] > best_slope[ri, ci]:
                best_slope[ri, ci] = slope[i, j]
                fdir_dr[ri, ci]    = dr
                fdir_dc[ri, ci]    = dc

    return fdir_dr, fdir_dc


def _upstream_cells(
    fdir_dr: np.ndarray,
    fdir_dc: np.ndarray,
    outlet_r: int,
    outlet_c: int,
) -> set[tuple[int, int]]:
    """BFS upstream from (outlet_r, outlet_c) following D8 flow directions.
    Returns a set of (row, col) indices (0-based) that drain to the outlet."""
    nrow, ncol = fdir_dr.shape

    # Build reverse map: for each cell, which cells point TO it?
    # We do this lazily during BFS using the fdir arrays directly.
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),           ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    upstream: set[tuple[int, int]] = set()
    queue = [(outlet_r, outlet_c)]
    upstream.add((outlet_r, outlet_c))

    while queue:
        r, c = queue.pop()
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrow and 0 <= nc < ncol):
                continue
            if (nr, nc) in upstream:
                continue
            # Does (nr, nc) flow into (r, c)?
            if fdir_dr[nr, nc] == -dr and fdir_dc[nr, nc] == -dc:
                upstream.add((nr, nc))
                queue.append((nr, nc))

    return upstream


# ---------------------------------------------------------------------------
# Shape initial hydraulic head toward a spring / drain outlet
# ---------------------------------------------------------------------------

def run_steady_state_spinup(
    mf_folder: str | os.PathLike,
    exe_path: str | os.PathLike,
    *,
    recharge_scale: float = 1.0,
    silent: bool = True,
) -> np.ndarray:
    """Run a steady-state MODFLOW pass and write the result back as initial heads.

    The workflow is:

    1. Patch the ``.dis`` file: change the final stress-period flag ``TR``
       (transient) to ``SS`` (steady-state).
    2. Run MODFLOW with the patched DIS.
    3. Read the resulting heads from the ``.hds`` file.
    4. Write the steady-state heads as the new ``strt`` array in the BAS6
       file using ``bas.write_file()`` — this avoids calling
       ``mf.write_input()`` so the OC HEAD SAVE UNIT is never reset.
    5. Restore the ``.dis`` file to ``TR`` so the model is ready for the
       transient coupled run.

    The head change you saw (−0.18 m uniform, +1.44 m near rivers) indicates
    the initial heads were slightly above the true steady-state level overall
    and that the river cells were acting as recharge sources.  This function
    eliminates that spin-up artefact.

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace (contains ``.nam``, ``.dis``, ``.bas`` etc.).
    exe_path : path
        Path to the MODFLOW-NWT executable.
    recharge_scale : float
        Multiplier applied to average recharge before the SS run.  Default
        1.0 (use whatever the model already has).  Useful if you want to
        represent long-term mean conditions differently from the transient
        forcing.
    silent : bool
        Suppress MODFLOW console output.  Default True.

    Returns
    -------
    np.ndarray
        Steady-state head array (shape nrow × ncol, layer 1) written to
        BAS6 as the new ``strt``.

    Raises
    ------
    RuntimeError
        If MODFLOW does not converge during the steady-state run.
    """
    import re as _re
    import flopy
    import flopy.utils.binaryfile as bf

    wd = str(mf_folder)

    # ── Locate files ─────────────────────────────────────────────────────────
    dis_files = glob.glob(os.path.join(wd, "*.dis"))
    bas_files = glob.glob(os.path.join(wd, "*.bas"))
    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    hds_files = glob.glob(os.path.join(wd, "*.hds"))

    if not dis_files:
        raise FileNotFoundError(f"No .dis file found in {wd!r}")
    if not bas_files:
        raise FileNotFoundError(f"No .bas file found in {wd!r}")
    if not nam_files:
        raise FileNotFoundError(f"No .nam file found in {wd!r}")

    dis_path = dis_files[0]

    # Resolve the HDS path from the .nam file so we pick the right unit even
    # when multiple .hds files exist.  The OC line "HEAD SAVE UNIT <n>" maps
    # to whatever filename is on unit <n> in the .nam file.
    hds_path = None
    try:
        with open(nam_files[0]) as fh:
            nam_text = fh.read()
        # Find "HEAD SAVE UNIT <n>" in OC file, then look up unit <n> in .nam
        oc_files = glob.glob(os.path.join(wd, "*.oc"))
        if oc_files:
            with open(oc_files[0]) as fh:
                oc_text = fh.read()
            mu = _re.search(r'HEAD\s+SAVE\s+UNIT\s+(\d+)', oc_text, _re.I)
            if mu:
                unit = mu.group(1)
                um = _re.search(
                    rf'^\s*\S+\s+{unit}\s+(\S+)',
                    nam_text, _re.I | _re.MULTILINE
                )
                if um:
                    hds_path = os.path.join(wd, um.group(1))
    except Exception:
        pass
    if hds_path is None:
        hds_path = hds_files[0] if hds_files else os.path.join(wd, "modflow.hds")

    # ── Step 1: patch DIS — TR → SS, NSTP → 1 ───────────────────────────────
    with open(dis_path, encoding="utf-8") as fh:
        dis_original = fh.read()

    # Stress-period data line: PERLEN  NSTP  TSMULT  TR|SS
    # e.g. "  16170.000000         16170  1.000000  TR"
    # We change TR → SS AND force NSTP=1 so that OC's "period 1 step 1"
    # fires at the single converged SS solution.  In MODFLOW-NWT, NSTP and
    # TSMULT are ignored for SS stress periods, but the OC step counter
    # still needs to see step 1 = last step for the save to occur.

    def _patch_sp_line(m):
        perlen, _, tsmult = m.group(1), m.group(2), m.group(3)
        return f"{perlen}         1{tsmult}SS"

    dis_patched = _re.sub(
        r'([ \t]+[\d.E+\-]+[ \t]+)(\d+)([ \t]+[\d.E+\-]+[ \t]+)TR\b',
        _patch_sp_line,
        dis_original,
    )
    if dis_patched == dis_original:
        print("  WARNING: stress-period 'TR' not found in .dis — may already "
              "be SS or uses a numeric flag.  Proceeding anyway.")

    with open(dis_path, "w", encoding="utf-8") as fh:
        fh.write(dis_patched)
    print(f"  DIS patched: TR → SS, NSTP → 1  ({os.path.basename(dis_path)})")

    # ── Step 2: run MODFLOW (steady-state) ───────────────────────────────────
    try:
        success, buff = flopy.run_model(
            str(exe_path),
            os.path.basename(nam_files[0]),
            model_ws=wd,
            silent=silent,
            report=True,
        )
    finally:
        # ── Step 5: always restore DIS to TR, even if MODFLOW crashes ────────
        with open(dis_path, "w", encoding="utf-8") as fh:
            fh.write(dis_original)
        print(f"  DIS restored: SS → TR  ({os.path.basename(dis_path)})")

    if not success:
        raise RuntimeError(
            "MODFLOW did not converge during steady-state spin-up. "
            "Check the .list file for details."
        )
    print("  Steady-state MODFLOW converged successfully.")

    # ── Step 3: read steady-state heads ──────────────────────────────────────
    hf = bf.HeadFile(hds_path)
    ss_head = hf.get_data(totim=hf.get_times()[-1])[0]   # (nrow, ncol), layer 0

    # ── Step 4: write new strt via BAS6.write_file() ─────────────────────────
    # Load only DIS + BAS6 so we don't disturb OC unit numbers
    with open(nam_files[0]) as fh:
        nam_text = fh.read().upper()
    load_only = [p for p in ["DIS", "BAS6"] if p in nam_text]

    mf = flopy.modflow.Modflow.load(
        os.path.basename(nam_files[0]),
        model_ws=wd,
        load_only=load_only,
        check=False,
    )
    bas = mf.get_package("BAS6")
    ibound = bas.ibound.array

    # Inactive cells → 0
    ss_strt = np.where(ibound[0] > 0, ss_head, 0.0)

    bas_new = flopy.modflow.ModflowBas(
        mf,
        ibound=ibound,
        strt=ss_strt[np.newaxis, :, :],
    )
    bas_new.write_file()
    print(f"  BAS6 strt updated with steady-state heads  ({bas_files[0]})")

    active = np.where(ibound[0] > 0, ss_strt, np.nan)
    print(f"  Head range (active): {np.nanmin(active):.1f} – {np.nanmax(active):.1f} m")

    return ss_strt


def set_head_gradient_to_spring(
    mf_folder: str | os.PathLike,
    outlet_point,
    *,
    head_at_outlet: float | None = None,
    head_offset_at_outlet: float = 0.5,
    gradient: float | None = None,
    fraction_of_relief: float = 0.15,
    mfgrid_path: str | os.PathLike | None = None,
    plot: bool = True,
    plot_path: str | os.PathLike | None = None,
) -> np.ndarray:
    """Set the MODFLOW initial head (strt) as a smooth gradient toward the spring.

    Rather than adding artificial drain cells to steer groundwater, this
    function directly shapes the potentiometric surface so that the hydraulic
    gradient already points toward the outlet.  MODFLOW's own physics then
    routes water there without needing prescribed flow paths.

    The head surface is constructed as::

        h(r, c) = h_outlet + gradient × dist(r, c)

    where ``dist`` is the Euclidean distance (m) from each active cell to the
    outlet cell and ``gradient`` [m/m] is either supplied directly or estimated
    from the active-domain extent and ``fraction_of_relief``.

    Parameters
    ----------
    mf_folder : str or path-like
        MODFLOW model workspace.
    outlet_point : GeoDataFrame, (x, y) tuple, or (row, col) tuple
        Spring / drain outlet location.  Same formats accepted as
        ``burn_drain_to_point``.
    head_at_outlet : float, optional
        Hydraulic head prescribed at the outlet cell (m NZTM).  If omitted,
        the cell's top elevation minus *head_offset_at_outlet* is used.
    head_offset_at_outlet : float
        Depth below land surface at the outlet cell used when
        *head_at_outlet* is not supplied.  Default 0.5 m.
    gradient : float, optional
        Hydraulic gradient [m/m] applied uniformly away from the outlet.
        If omitted, inferred from *fraction_of_relief* and the domain extent.
    fraction_of_relief : float
        Fraction of the active-domain topographic relief used to set the
        head range when *gradient* is not supplied.  Default 0.15 (15%).
        Increase toward 0.3 for steep catchments; decrease toward 0.05 for
        nearly flat aquifers.
    mfgrid_path : str or path-like, optional
        Path to MFgrid shapefile for spatial snapping.  Searched
        automatically if omitted.
    plot : bool
        If True (default), generate a two-panel verification figure showing
        the new head surface and the implied flow direction vectors.
    plot_path : str or path-like, optional
        Save the figure to this path.  If omitted the figure is shown
        interactively (or written to ``<mf_folder>/head_gradient_check.png``
        when running non-interactively).

    Returns
    -------
    np.ndarray
        The new strt array (shape nrow × ncol), with nodata cells set to 0.
    """
    wd = str(mf_folder)

    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    if not nam_files:
        raise FileNotFoundError(f"No .nam file found in {wd}")

    mf = flopy.modflow.Modflow.load(
        os.path.basename(nam_files[0]),
        model_ws=wd,
        load_only=["DIS", "BAS6"],
        check=False,
    )
    dis    = mf.get_package("DIS")
    bas    = mf.get_package("BAS6")
    top    = dis.top.array.astype(float)
    ibound = bas.ibound.array[0]
    nrow, ncol = dis.nrow, dis.ncol
    cell_size = float(dis.delr.array[0])

    # ── Resolve outlet cell ───────────────────────────────────────────────────
    try:
        import geopandas as gpd
        _has_gpd = True
    except ImportError:
        _has_gpd = False

    if _has_gpd and hasattr(outlet_point, "geometry"):
        pt = outlet_point.geometry.iloc[0].centroid
        ox, oy = pt.x, pt.y

        if mfgrid_path is None:
            candidates = glob.glob(os.path.join(wd, "MFgrid*.shp")) or \
                         glob.glob(os.path.join(wd, "*grid*.shp"))
            mfgrid_path = candidates[0] if candidates else None

        if mfgrid_path and os.path.isfile(str(mfgrid_path)):
            grid_gdf = gpd.read_file(str(mfgrid_path))
            col_map  = {c: c.lower() for c in grid_gdf.columns}
            grid_gdf = grid_gdf.rename(columns=col_map)
            pt_gdf   = gpd.GeoDataFrame(
                geometry=[outlet_point.geometry.iloc[0]],
                crs=outlet_point.crs,
            ).to_crs(grid_gdf.crs)
            joined = gpd.sjoin(
                grid_gdf[["row", "col", "geometry"]],
                pt_gdf, how="inner", predicate="intersects",
            )
            if joined.empty:
                raise ValueError("Outlet point does not fall within any active grid cell.")
            outlet_r0 = int(joined["row"].iloc[0]) - 1
            outlet_c0 = int(joined["col"].iloc[0]) - 1
        else:
            xmin = float(getattr(dis, "xul", 0.0))
            ymax = float(getattr(dis, "yul", nrow * cell_size))
            outlet_c0 = int((ox - xmin) / cell_size)
            outlet_r0 = int((ymax - oy) / cell_size)
    elif isinstance(outlet_point, (tuple, list)) and len(outlet_point) == 2:
        a, b = outlet_point
        if isinstance(a, float) and a > 1000:
            xmin = float(getattr(dis, "xul", 0.0))
            ymax = float(getattr(dis, "yul", nrow * cell_size))
            outlet_c0 = int((a - xmin) / cell_size)
            outlet_r0 = int((ymax - b) / cell_size)
        else:
            outlet_r0, outlet_c0 = int(a) - 1, int(b) - 1
    else:
        raise TypeError(
            "outlet_point must be a GeoDataFrame, (x, y) coordinate tuple, "
            "or (row, col) index tuple."
        )

    print(f"  Outlet cell: row={outlet_r0+1}  col={outlet_c0+1}  "
          f"top={top[outlet_r0, outlet_c0]:.2f} m")

    # ── Build distance-from-outlet grid ──────────────────────────────────────
    rows_idx = np.arange(nrow)
    cols_idx = np.arange(ncol)
    C, R     = np.meshgrid(cols_idx, rows_idx)
    dist     = np.sqrt((R - outlet_r0)**2 + (C - outlet_c0)**2) * cell_size  # metres

    # ── Determine head at outlet and gradient ─────────────────────────────────
    outlet_top = float(top[outlet_r0, outlet_c0])
    if head_at_outlet is None:
        head_at_outlet = outlet_top - head_offset_at_outlet
        print(f"  DEM elevation at outlet : {outlet_top:.2f} m")
        print(f"  Head at outlet (DEM − {head_offset_at_outlet} m) : {head_at_outlet:.2f} m")

    if gradient is None:
        # Estimate from fraction of topographic relief over the domain extent
        active_top = np.where(ibound > 0, top, np.nan)
        relief     = float(np.nanmax(active_top) - np.nanmin(active_top))
        head_range = fraction_of_relief * relief
        max_dist   = float(np.nanmax(dist[ibound > 0]))
        gradient   = head_range / max_dist if max_dist > 0 else 0.001
        print(f"  Topographic relief : {relief:.1f} m")
        print(f"  Head range applied : {head_range:.1f} m  "
              f"({fraction_of_relief*100:.0f}% of relief)")
        print(f"  Implied gradient   : {gradient*1000:.2f} m/km")

    # ── Construct new head surface ────────────────────────────────────────────
    strt_new = head_at_outlet + gradient * dist

    # Cap heads at cell top (head cannot exceed land surface)
    strt_new = np.minimum(strt_new, top)

    # Inactive cells → 0 (MODFLOW ignores them)
    strt_new = np.where(ibound > 0, strt_new, 0.0)

    print(f"  Head at outlet     : {head_at_outlet:.2f} m")
    print(f"  Head range (active): {float(np.nanmin(np.where(ibound>0, strt_new, np.nan))):.1f}"
          f" – {float(np.nanmax(np.where(ibound>0, strt_new, np.nan))):.1f} m")

    # ── Write updated BAS6 file ───────────────────────────────────────────────
    bas_new = flopy.modflow.ModflowBas(
        mf,
        ibound=bas.ibound.array,
        strt=strt_new[np.newaxis, :, :],   # shape (nlay, nrow, ncol)
    )
    bas_new.write_file()
    bas_files = glob.glob(os.path.join(wd, "*.bas"))
    if bas_files:
        print(f"  BAS6 file updated  : {bas_files[0]}")

    # ── Verification plot ─────────────────────────────────────────────────────
    if plot:
        from swatmf.outputs.modflow_diagnostics import plot_head_gradient_check
        import matplotlib.pyplot as plt

        fig = plot_head_gradient_check(
            strt_new, wd,
            outlet_rc=(outlet_r0, outlet_c0),
        )
        fig.suptitle(
            f"Head gradient verification  |  gradient ≈ {gradient*1000:.2f} m/km  "
            f"|  outlet head = {head_at_outlet:.1f} m",
            fontsize=11, fontweight="bold",
        )

        if plot_path is None:
            import matplotlib
            if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "cairo"):
                plot_path = os.path.join(wd, "head_gradient_check.png")

        if plot_path is not None:
            fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            print(f"  Gradient plot saved : {plot_path}")
        else:
            plt.show()

        plt.close(fig)

    return strt_new


# ---------------------------------------------------------------------------
# Drain burn-in to a single outlet point (DEM-based, retained for reference)
# ---------------------------------------------------------------------------

def burn_drain_to_point(
    mf_folder: str | os.PathLike,
    outlet_point,
    *,
    conductance: float = 500.0,
    elev_offset: float = 0.5,
    layer: int = 1,
    min_accumulation: int = 3,
    smooth_dem: bool = True,
    mfgrid_path: str | os.PathLike | None = None,
) -> str:
    """Compute a D8 drainage network to *outlet_point* and burn it in as DRN.

    No stream shapefile is required.  The function:

    1. Loads the MODFLOW top (DEM) array.
    2. Optionally smooths it to remove pits that would trap flow before the
       outlet.
    3. Runs a D8 steepest-descent flow-direction analysis.
    4. Delineates all active cells that drain to the outlet cell via BFS.
    5. Filters to cells with upstream-area ≥ *min_accumulation* cells (the
       main channel, not every hillslope cell).
    6. Writes a DRN file for those cells and regenerates ``modflow.mfn``.

    Parameters
    ----------
    mf_folder : str or path-like
        MODFLOW model workspace.
    outlet_point : GeoDataFrame, (x, y) tuple, or (row, col) tuple
        The spring / drain outlet location.  If a GeoDataFrame the first
        geometry's centroid is used.  Coordinates must be in the same CRS as
        the MODFLOW grid.  Alternatively supply a ``(row, col)`` tuple of
        1-based grid indices directly.
    conductance : float
        Drain conductance in m²/day applied to every channel cell.
        Default 500 m²/day.
    elev_offset : float
        Drain elevation = cell top − *elev_offset* (m).  Default 0.5 m.
    layer : int
        MODFLOW layer (1-based).  Default 1.
    min_accumulation : int
        Minimum number of upstream cells for a cell to be included as a drain.
        Raise this to restrict DRN to main channels only; set to 1 to drain
        every cell upstream of the outlet.  Default 3.
    smooth_dem : bool
        If True, apply a 3×3 uniform filter to the top array before computing
        flow directions.  This fills local pits that would otherwise capture
        flow before it reaches the outlet.  Default True.
    mfgrid_path : str or path-like, optional
        Path to the MFgrid shapefile.  Searched automatically if omitted.

    Returns
    -------
    str
        Path to the written ``.drn`` file.
    """
    from scipy.ndimage import uniform_filter

    wd = str(mf_folder)

    # ── Locate model files ────────────────────────────────────────────────────
    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    if not nam_files:
        raise FileNotFoundError(f"No .nam file found in {wd}")

    mf = flopy.modflow.Modflow.load(
        os.path.basename(nam_files[0]),
        model_ws=wd,
        load_only=["DIS", "BAS6"],
        check=False,
    )
    dis    = mf.get_package("DIS")
    bas    = mf.get_package("BAS6")
    top    = dis.top.array          # (nrow, ncol)
    ibound = bas.ibound.array[0]    # (nrow, ncol)
    nrow, ncol = dis.nrow, dis.ncol

    # Infer cell size from DIS (assume uniform square cells)
    cell_size = float(dis.delr.array[0])

    # ── Resolve outlet cell ───────────────────────────────────────────────────
    try:
        import geopandas as gpd
        _has_gpd = True
    except ImportError:
        _has_gpd = False

    if _has_gpd and hasattr(outlet_point, "geometry"):
        # GeoDataFrame — use first point centroid
        pt = outlet_point.geometry.iloc[0].centroid
        ox, oy = pt.x, pt.y

        # Need grid origin to convert coords → row/col
        if mfgrid_path is None:
            candidates = glob.glob(os.path.join(wd, "MFgrid*.shp")) or \
                         glob.glob(os.path.join(wd, "*grid*.shp"))
            mfgrid_path = candidates[0] if candidates else None

        if mfgrid_path and os.path.isfile(str(mfgrid_path)):
            grid_gdf = gpd.read_file(str(mfgrid_path))
            col_map  = {c: c.lower() for c in grid_gdf.columns}
            grid_gdf = grid_gdf.rename(columns=col_map)
            pt_gdf   = gpd.GeoDataFrame(
                geometry=[outlet_point.geometry.iloc[0]],
                crs=outlet_point.crs,
            ).to_crs(grid_gdf.crs)
            joined = gpd.sjoin(
                grid_gdf[["row", "col", "geometry"]],
                pt_gdf, how="inner", predicate="intersects",
            )
            if joined.empty:
                raise ValueError("Outlet point does not fall within any active grid cell.")
            outlet_r0 = int(joined["row"].iloc[0]) - 1   # 0-based
            outlet_c0 = int(joined["col"].iloc[0]) - 1
        else:
            # Fall back: infer origin from DIS
            xmin = float(dis.xul) if hasattr(dis, "xul") else 0.0
            ymax = float(dis.yul) if hasattr(dis, "yul") else nrow * cell_size
            outlet_c0 = int((ox - xmin) / cell_size)
            outlet_r0 = int((ymax - oy) / cell_size)

        print(f"  Outlet point ({ox:.1f}, {oy:.1f}) → row={outlet_r0+1}  col={outlet_c0+1}")

    elif isinstance(outlet_point, (tuple, list)) and len(outlet_point) == 2:
        first, second = outlet_point
        if isinstance(first, float) and first > 1000:
            # Treat as (x, y) coordinates
            ox, oy = first, second
            xmin = float(dis.xul) if hasattr(dis, "xul") else 0.0
            ymax = float(dis.yul) if hasattr(dis, "yul") else nrow * cell_size
            outlet_c0 = int((ox - xmin) / cell_size)
            outlet_r0 = int((ymax - oy) / cell_size)
        else:
            # Treat as (row, col) — 1-based
            outlet_r0 = int(first)  - 1
            outlet_c0 = int(second) - 1
        print(f"  Outlet cell: row={outlet_r0+1}  col={outlet_c0+1}")
    else:
        raise TypeError(
            "outlet_point must be a GeoDataFrame, (x, y) coordinate tuple, "
            "or (row, col) index tuple."
        )

    if not (0 <= outlet_r0 < nrow and 0 <= outlet_c0 < ncol):
        raise ValueError(
            f"Outlet cell ({outlet_r0+1}, {outlet_c0+1}) is outside the grid "
            f"({nrow} rows × {ncol} cols)."
        )

    # ── Build DEM for flow routing (active cells only, NaN elsewhere) ─────────
    dem = np.where(ibound > 0, top.astype(float), np.nan)

    if smooth_dem:
        # Fill NaN with local mean before smoothing so edges don't bleed
        filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
        smoothed = uniform_filter(filled, size=3)
        dem = np.where(np.isnan(dem), np.nan, smoothed)
        print("  DEM smoothed (3×3 uniform filter) to reduce pit trapping")

    # ── D8 flow directions ────────────────────────────────────────────────────
    print("  Computing D8 flow directions ...")
    fdir_dr, fdir_dc = _d8_flow_direction(dem)

    # ── Delineate all upstream cells via BFS ──────────────────────────────────
    print("  Delineating upstream drainage area ...")
    upstream = _upstream_cells(fdir_dr, fdir_dc, outlet_r0, outlet_c0)
    print(f"  Total cells draining to outlet: {len(upstream)}")

    # ── Compute flow accumulation (upstream cell count) for each cell ─────────
    # Simple O(n) pass: sort cells from high to low elevation, accumulate
    acc = {rc: 1 for rc in upstream}
    sorted_cells = sorted(upstream, key=lambda rc: -dem[rc[0], rc[1]]
                          if np.isfinite(dem[rc[0], rc[1]]) else -1e9)
    for r, c in sorted_cells:
        dr, dc = int(fdir_dr[r, c]), int(fdir_dc[r, c])
        if dr == -99:
            continue
        nr, nc = r + dr, c + dc
        if (nr, nc) in acc:
            acc[(nr, nc)] += acc[(r, c)]

    # ── Filter to channel cells (accumulation ≥ threshold) ───────────────────
    channel_cells = [(r, c) for (r, c), a in acc.items() if a >= min_accumulation]
    print(f"  Channel cells (accumulation ≥ {min_accumulation}): {len(channel_cells)}")

    if not channel_cells:
        raise ValueError(
            f"No channel cells found with min_accumulation={min_accumulation}. "
            "Lower the threshold or check that the outlet is inside the active domain."
        )

    # Always include the outlet cell itself
    if (outlet_r0, outlet_c0) not in {(r, c) for r, c in channel_cells}:
        channel_cells.append((outlet_r0, outlet_c0))

    # ── Build DRN records (1-based row/col) ───────────────────────────────────
    drn_records = []
    for r0, c0 in channel_cells:
        elev = float(top[r0, c0]) - elev_offset
        drn_records.append((r0 + 1, c0 + 1, elev, conductance))

    print(f"  DRN records to write: {len(drn_records)}")

    # ── Write DRN file, update .nam, regenerate modflow.mfn + OC ─────────────
    drn_path = _write_drn_and_update_nam(wd, nam_files[0], drn_records, layer)
    create_modflow_mfn(wd)
    modify_modflow_oc(wd)

    print(f"  DRN file written    : {drn_path}")
    print(f"  modflow.mfn updated : {os.path.join(wd, 'modflow.mfn')}")
    return drn_path


# ---------------------------------------------------------------------------
# Stream drain burn-in (from shapefile)
# ---------------------------------------------------------------------------

def burn_stream_drains(
    mf_folder: str | os.PathLike,
    stream_path: "str | os.PathLike | gpd.GeoDataFrame",
    *,
    conductance: float = 500.0,
    elev_offset: float = 0.5,
    layer: int = 1,
    mfgrid_path: str | os.PathLike | None = None,
    spring_points: "gpd.GeoDataFrame | None" = None,
    spring_conductance: float = 100.0,
    spring_elev_offset: float = 0.5,
) -> str:
    """Burn stream-network drains into the MODFLOW model.

    Analogous to DEM stream-burning in SWAT: every MODFLOW cell that the
    stream polylines cross receives a Drain (DRN) entry.  When the simulated
    head exceeds the drain elevation the cell discharges to the surface,
    drawing groundwater toward the valley floor regardless of the initial-head
    configuration.

    An optional ``spring_points`` argument lets you combine the stream-network
    burn-in with one or more point spring outlets in a single DRN file.

    Parameters
    ----------
    mf_folder : str or path-like
        MODFLOW model workspace (contains ``.dis``, ``.nam``, ``modflow.mfn``).
    stream_path : str, path-like, or GeoDataFrame
        Stream network polylines.  Must be in the same CRS as the MODFLOW
        grid (EPSG:2193 for NZTM).  Accepted formats: any file readable by
        ``geopandas.read_file`` or an already-loaded GeoDataFrame.
    conductance : float, optional
        Streambed drain conductance in m²/d applied to every stream cell.
        Higher values → stronger hydraulic connection.  Default 500 m²/d.
    elev_offset : float, optional
        Subtract this from the cell top elevation to set the drain elevation
        (i.e. the drain sits *elev_offset* metres below the land surface).
        Default 0.5 m.
    layer : int, optional
        MODFLOW layer that receives the drain cells.  Default 1 (top layer).
    mfgrid_path : str or path-like, optional
        Path to ``MFgrid.shp`` (MODFLOW cell polygon grid).  If omitted the
        function looks for ``MFgrid.shp`` in *mf_folder*.
    spring_points : GeoDataFrame, optional
        Additional point spring locations merged into the same DRN file.
    spring_conductance : float, optional
        Conductance for the spring point drains.  Default 100 m²/d.
    spring_elev_offset : float, optional
        Drain elevation offset (below land surface) for spring points.
        Default 0.5 m.

    Returns
    -------
    str
        Path to the written ``.drn`` file.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    wd = str(mf_folder)

    # ── Locate model files ────────────────────────────────────────────────────
    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    if not nam_files:
        raise FileNotFoundError(f"No .nam file found in {wd}")

    dis_files = glob.glob(os.path.join(wd, "*.dis"))
    if not dis_files:
        raise FileNotFoundError(f"No .dis file found in {wd}")

    # ── Load MODFLOW grid shapefile ───────────────────────────────────────────
    if mfgrid_path is None:
        candidates = glob.glob(os.path.join(wd, "MFgrid*.shp")) or \
                     glob.glob(os.path.join(wd, "*grid*.shp"))
        if not candidates:
            raise FileNotFoundError(
                "MFgrid.shp not found — pass mfgrid_path explicitly."
            )
        mfgrid_path = candidates[0]

    grid_gdf = gpd.read_file(mfgrid_path)
    print(f"  Grid: {len(grid_gdf)} cells from {os.path.basename(str(mfgrid_path))}")

    # Standardise row/col column names (case-insensitive)
    col_map = {c: c.lower() for c in grid_gdf.columns}
    grid_gdf = grid_gdf.rename(columns=col_map)

    # ── Load MODFLOW DIS for cell-top elevations ──────────────────────────────
    mf = flopy.modflow.Modflow.load(
        os.path.basename(nam_files[0]),
        model_ws=wd,
        load_only=["DIS", "BAS6"],
        check=False,
    )
    dis  = mf.get_package("DIS")
    top_arr = dis.top.array          # shape (nrow, ncol)

    # ── Load stream network ───────────────────────────────────────────────────
    if isinstance(stream_path, gpd.GeoDataFrame):
        streams = stream_path.copy()
    else:
        streams = gpd.read_file(str(stream_path))
    print(f"  Streams: {len(streams)} features")

    # Ensure CRS matches
    if grid_gdf.crs is not None and streams.crs is not None:
        if grid_gdf.crs != streams.crs:
            streams = streams.to_crs(grid_gdf.crs)

    # ── Intersect streams with grid cells ─────────────────────────────────────
    stream_cells = gpd.sjoin(
        grid_gdf[["row", "col", "geometry"]],
        streams[["geometry"]],
        how="inner",
        predicate="intersects",
    ).drop_duplicates(subset=["row", "col"])

    print(f"  Stream cells identified: {len(stream_cells)}")
    if len(stream_cells) == 0:
        raise ValueError(
            "No grid cells intersect the stream network — check CRS alignment."
        )

    # ── Build DRN records from stream cells ───────────────────────────────────
    drn_records: list = []
    for _, cell in stream_cells.iterrows():
        r = int(cell["row"])
        c = int(cell["col"])
        elev = float(top_arr[r - 1, c - 1]) - elev_offset
        drn_records.append((r, c, elev, conductance))

    # ── Optionally add spring point drains ────────────────────────────────────
    if spring_points is not None:
        if isinstance(spring_points, gpd.GeoDataFrame):
            pts = spring_points.copy()
        else:
            pts = gpd.GeoDataFrame(
                spring_points,
                geometry=gpd.points_from_xy(
                    spring_points["x"], spring_points["y"]
                ),
                crs=grid_gdf.crs,
            )

        if grid_gdf.crs is not None and pts.crs is not None:
            if pts.crs != grid_gdf.crs:
                pts = pts.to_crs(grid_gdf.crs)

        pt_cells = gpd.sjoin(
            grid_gdf[["row", "col", "geometry"]],
            pts[["geometry"]],
            how="inner",
            predicate="intersects",
        ).drop_duplicates(subset=["row", "col"])

        existing_rc = {(r["row"], r["col"]) for _, r in stream_cells.iterrows()}
        for _, cell in pt_cells.iterrows():
            r, c = int(cell["row"]), int(cell["col"])
            if (r, c) not in existing_rc:
                elev = float(top_arr[r - 1, c - 1]) - spring_elev_offset
                drn_records.append((r, c, elev, spring_conductance))
                print(f"  Spring point added: row={r}  col={c}  "
                      f"drain_elev={elev:.2f} m  conductance={spring_conductance:.1f} m²/d")

    print(f"  Total DRN cells: {len(drn_records)}")

    # ── Write DRN file and update .nam ────────────────────────────────────────
    drn_path = _write_drn_and_update_nam(wd, nam_files[0], drn_records, layer)

    # ── Regenerate modflow.mfn and re-apply OC unit bump ─────────────────────
    create_modflow_mfn(wd)
    modify_modflow_oc(wd)

    print(f"  DRN file written    : {drn_path}")
    print(f"  modflow.mfn updated : {os.path.join(wd, 'modflow.mfn')}")
    return drn_path


# ---------------------------------------------------------------------------
# Spring / drain outlet
# ---------------------------------------------------------------------------

def add_spring_drain(
    mf_folder: str | os.PathLike,
    spring_points: "gpd.GeoDataFrame | pd.DataFrame",
    *,
    conductance: float = 100.0,
    elev_col: str | None = None,
    elev_offset: float = 0.0,
    layer: int = 1,
    mfgrid_path: str | os.PathLike | None = None,
) -> str:
    """Add a MODFLOW Drain (DRN) package representing one or more springs.

    Springs discharge groundwater to the surface only when the simulated head
    exceeds the drain (spring orifice) elevation.  This is more physically
    correct than a Constant Head cell, which can inject water, and more
    appropriate than a River cell, which implies a surface-water body.

    When ``drain_cells = True`` is set in ``swatmf_link.txt`` the SWAT-MODFLOW
    executable routes all DRN discharge to SWAT subbasin channels, so each
    spring automatically becomes a GW→SW flux in the coupled model.

    Parameters
    ----------
    mf_folder : str or path-like
        MODFLOW model folder (the folder containing ``.dis``, ``.nam`` etc.).
        The ``.drn`` file is written here and ``modflow.mfn`` is regenerated
        so the new package is included in the next model run.
    spring_points : GeoDataFrame or DataFrame
        Point locations of the spring(s).  Accepted forms:

        * **GeoDataFrame** with point geometry (any CRS — reprojected to match
          the grid if needed).  The grid cell containing each point is found
          by a spatial join with ``mf_grid.gpkg``.
        * **DataFrame** with columns ``row`` and ``col`` (1-based, matching
          the MODFLOW convention) if you already know the cell indices.

    conductance : float, optional
        Drain conductance [L²/T] applied to all spring cells.  Controls how
        rapidly discharge responds to head above the drain elevation.  Higher
        values produce a more responsive spring; lower values damp the
        response.  A value of 100 m²/day is a reasonable starting point for
        a high-flow spring.  Default ``100.0``.
    elev_col : str or None, optional
        Column in *spring_points* containing the drain (spring orifice)
        elevation [m].  If ``None`` (default) the land-surface elevation
        (``top_elev`` from ``mf_grid.gpkg``, or the DIS ``top`` array) is
        used, minus *elev_offset*.
    elev_offset : float, optional
        Subtracted from the drain elevation when *elev_col* is ``None``.
        A small positive value (e.g. ``0.5``) lowers the effective drain
        threshold slightly so discharge begins before the water table reaches
        the exact surface.  Default ``0.0``.
    layer : int, optional
        MODFLOW layer number for the drain cells (1-based).  Default ``1``.
    mfgrid_path : str, path-like, or None, optional
        Path to ``mf_grid.gpkg``.  Required when *spring_points* is a
        GeoDataFrame.  If ``None``, the function searches *mf_folder* and its
        parent ``GIS/org_shps/`` for ``mf_grid.gpkg``.

    Returns
    -------
    str
        Absolute path to the written ``.drn`` file.

    Examples
    --------
    From a point shapefile of spring locations:

    >>> import geopandas as gpd
    >>> from swatmf.preprocessing.modflow import add_spring_drain
    >>>
    >>> springs = gpd.read_file("GIS/org_shps/springs.shp")
    >>> drn_path = add_spring_drain(
    ...     mf_folder    = run_dir,
    ...     spring_points = springs,
    ...     conductance  = 200.0,   # high-flow spring
    ...     elev_offset  = 0.5,     # drain 0.5 m below surface
    ...     mfgrid_path  = "GIS/org_shps/mf_grid.gpkg",
    ... )
    >>> print("DRN written to:", drn_path)

    From known row/col indices (no shapefile needed):

    >>> import pandas as pd
    >>> springs_rc = pd.DataFrame({"row": [34], "col": [187]})
    >>> drn_path = add_spring_drain(run_dir, springs_rc, conductance=100.0)
    """
    try:
        import geopandas as gpd_mod
    except ImportError:
        gpd_mod = None

    wd = str(mf_folder)

    # ── Load MODFLOW model to get top elevations and grid dims ────────────────
    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    if not nam_files:
        raise FileNotFoundError(f"No .nam file found in {wd!r}")

    import flopy
    import flopy.modflow as fm
    mf = flopy.modflow.Modflow.load(
        os.path.basename(nam_files[0]),
        model_ws=wd,
        check=False,
        verbose=False,
    )
    dis_pkg = mf.get_package("DIS")
    top_arr = dis_pkg.top.array     # (nrow, ncol) land-surface elevation
    nrow, ncol = dis_pkg.nrow, dis_pkg.ncol

    # ── Resolve spring cell locations ─────────────────────────────────────────
    has_geo = (
        gpd_mod is not None
        and hasattr(spring_points, "geometry")
        and spring_points.geometry is not None
    )

    if has_geo:
        # Spatial join: find which mf_grid cell each spring falls in
        grid_path = mfgrid_path
        if grid_path is None:
            candidates = [
                os.path.join(wd, "mf_grid.gpkg"),
                os.path.join(os.path.dirname(wd), "GIS", "org_shps", "mf_grid.gpkg"),
                os.path.join(wd, "GIS", "org_shps", "mf_grid.gpkg"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    grid_path = c
                    break
        if grid_path is None or not os.path.isfile(str(grid_path)):
            raise FileNotFoundError(
                "mf_grid.gpkg not found.  Supply mfgrid_path= explicitly."
            )

        grid_gdf = gpd_mod.read_file(str(grid_path))
        pts = spring_points.copy()
        if pts.crs is not None and pts.crs != grid_gdf.crs:
            pts = pts.to_crs(grid_gdf.crs)

        joined = gpd_mod.sjoin(
            pts,
            grid_gdf[["grid_id", "row", "col", "top_elev", "geometry"]],
            how="left",
            predicate="within",
        )
        missing = joined["row"].isna().sum()
        if missing:
            import warnings
            warnings.warn(
                f"{missing} spring point(s) did not fall within any active grid cell "
                "and will be skipped.",
                stacklevel=2,
            )
        joined = joined.dropna(subset=["row", "col"]).copy()
        joined["row"] = joined["row"].astype(int)
        joined["col"] = joined["col"].astype(int)

        rows = joined["row"].tolist()
        cols = joined["col"].tolist()
        top_elevs = joined["top_elev"].tolist()

        if elev_col and elev_col in joined.columns:
            drain_elevs = joined[elev_col].tolist()
        else:
            drain_elevs = [e - elev_offset for e in top_elevs]

    else:
        # DataFrame with row/col columns (1-based)
        import pandas as _pd
        df = spring_points if isinstance(spring_points, _pd.DataFrame) else _pd.DataFrame(spring_points)
        rows = df["row"].astype(int).tolist()
        cols = df["col"].astype(int).tolist()

        if elev_col and elev_col in df.columns:
            drain_elevs = df[elev_col].tolist()
        else:
            drain_elevs = [
                float(top_arr[r - 1, c - 1]) - elev_offset
                for r, c in zip(rows, cols)
            ]

    if not rows:
        raise ValueError("No valid spring cells found — check spring_points geometry and grid overlap.")

    print(f"  Adding {len(rows)} spring drain cell(s):")
    for r, c, elev in zip(rows, cols, drain_elevs):
        print(f"    row={r}  col={c}  drain_elev={elev:.2f} m  conductance={conductance:.1f} m2/d")

    drn_records = [(r, c, elev, conductance) for r, c, elev in zip(rows, cols, drain_elevs)]
    drn_path = _write_drn_and_update_nam(wd, nam_files[0], drn_records, layer)

    create_modflow_mfn(wd)
    modify_modflow_oc(wd)

    print(f"  DRN file written    : {drn_path}")
    print(f"  modflow.mfn updated : {os.path.join(wd, 'modflow.mfn')}")
    print()
    print("  Next steps:")
    print("  1. Re-run the standalone MODFLOW QA (Section 2.8) to verify spring discharge")
    print("  2. Set cfg.drain_cells = True in swatmf_link.txt (Section 3.1)")
    print("     so SWAT-MODFLOW routes spring discharge to SWAT subbasin channels")

    return drn_path


# ---------------------------------------------------------------------------
# Build a complete MODFLOW model from a DEM raster (Scenario C)
# ---------------------------------------------------------------------------

def build_mf_model_from_dem(
    dem_path: str | os.PathLike,
    mf_folder: str | os.PathLike,
    mf_name: str,
    *,
    boundary_path: str | os.PathLike | None = None,
    cell_size: float = 200.0,
    aquifer_thickness: Union[float, str, os.PathLike] = 30.0,
    sy: Union[float, str, os.PathLike] = 0.2,
    initial_head: Union[float, str, os.PathLike, None] = None,
    water_table_depth: Union[float, str, os.PathLike, None] = None,
    head_depth: float = 2.0,
    hk: float = 5.0,
    ss: float = 1e-4,
    vka: float = 0.1,
    crs: str | int | None = None,
    output_dir: str | os.PathLike | None = None,
    nodata: float = -9999.0,
    sim_duration: int = 365,
    riv_df: Optional[pd.DataFrame] = None,
) -> MFModelResult:
    """Build a complete MODFLOW model from a DEM raster.

    This is the high-level entry point for **Scenario C — Build a new MODFLOW
    model from scratch**.  It combines DEM resampling, parameter array
    construction, :func:`create_mf_model`, and :func:`create_mf_grid` into a
    single call.

    The DEM is resampled to *cell_size* × *cell_size* cells.  Aquifer
    thickness, specific yield, and initial hydraulic head can each be supplied
    as either a constant scalar or the path to a GeoTIFF raster.  All rasters
    are reprojected and resampled to match the output grid automatically.

    Parameters
    ----------
    dem_path : str or path-like
        Path to the land-surface DEM raster (any format supported by
        :mod:`rasterio`, e.g. GeoTIFF).
    mf_folder : str or path-like
        Directory where MODFLOW input files will be written (the SWAT-MODFLOW
        working directory).
    mf_name : str
        Base name for the MODFLOW model files.
    boundary_path : str, path-like, or None, optional
        Path to a polygon shapefile or GeoPackage used to clip the DEM before
        building the grid.  If ``None`` the full DEM extent is used.  A
        subbasin polygon file (e.g. ``sub_link.gpkg``) is a natural choice.
    cell_size : float, optional
        Target cell size in the units of the output CRS (metres for a
        projected CRS).  Default ``200.0``.
    aquifer_thickness : float or path-like, optional
        Vertical thickness of the aquifer layer [same units as DEM].  Supply
        a scalar to use a uniform value, or a raster path for spatially
        variable thickness.  Default ``30.0``.
    sy : float or path-like, optional
        Specific yield [dimensionless].  Scalar or raster path.  Default
        ``0.2``.
    initial_head : float, path-like, or None, optional
        Initial hydraulic head [same units as DEM].  Scalar or raster path
        (e.g. a GeoTIFF of observed/modelled piezometric surface).  When
        provided this takes priority over *water_table_depth* and *head_depth*.
        If ``None`` (default), head is derived from *water_table_depth* or
        *head_depth*.
    water_table_depth : float, path-like, or None, optional
        Depth to the water table below the land surface [same units as DEM].
        Scalar or raster path (e.g. a GeoTIFF of water-table depth).  When
        provided, ``initial_head = top_elev - water_table_depth`` is computed
        cell-by-cell after resampling to the model grid.  Ignored when
        *initial_head* is not ``None``.  If both *water_table_depth* and
        *initial_head* are ``None``, the scalar *head_depth* is used instead.
    head_depth : float, optional
        Fallback scalar depth below the land surface used to compute the
        default initial head when both *initial_head* and *water_table_depth*
        are ``None``.  Default ``2.0`` [same units as DEM].
    hk : float, optional
        Horizontal hydraulic conductivity [length/time].  Scalar only.
        Default ``5.0``.
    ss : float, optional
        Specific storage [1/length].  Scalar only.  Default ``1e-4``.
    vka : float, optional
        Vertical anisotropy ratio (VKA in the UPW package).  Default ``0.1``.
    crs : str, int, or None, optional
        Coordinate reference system for the output grid — any value accepted
        by :func:`rasterio.crs.CRS.from_user_input` (e.g. ``"EPSG:27700"`` or
        an integer EPSG code).  If ``None``, the DEM's own CRS is used.
    output_dir : str or path-like, optional
        Directory where ``mf_grid.gpkg`` is written.  Defaults to
        *mf_folder*.
    nodata : float, optional
        Sentinel value for cells outside the model domain.  Default
        ``-9999.0``.
    sim_duration : int, optional
        Simulation duration [days].  A buffer of ~100 days beyond the SWAT
        run length is recommended.  Default ``365``.
    riv_df : pd.DataFrame or None, optional
        River-package cell table (columns: ``row``, ``col``, ``riv_stage``,
        ``riv_cond``, ``riv_bot``).  Pass ``None`` (default) to omit the RIV
        package; add it later via :func:`write_riv_file`.

    Returns
    -------
    MFModelResult
        A named-tuple with fields:

        * ``mf``         — the flopy model object (input files already written)
        * ``grid_path``  — absolute path to the written ``mf_grid.gpkg``
        * ``x_origin``   — X (easting) of the NW grid corner
        * ``y_origin``   — Y (northing) of the NW grid corner
        * ``nrow``       — number of grid rows
        * ``ncol``       — number of grid columns

    Raises
    ------
    ImportError
        If :mod:`rasterio` or :mod:`geopandas` are not installed.
    FileNotFoundError
        If *dem_path* (or *boundary_path*) does not exist.

    Examples
    --------
    >>> from swatmf.preprocessing.modflow import build_mf_model_from_dem
    >>> result = build_mf_model_from_dem(
    ...     dem_path    = "GIS/dem.tif",
    ...     mf_folder   = wd,
    ...     mf_name     = "mymodel",
    ...     boundary_path = paths.sm_shps + "/sub_link.gpkg",
    ...     cell_size   = 200.0,
    ...     aquifer_thickness = 30.0,
    ...     sy          = 0.2,
    ...     crs         = "EPSG:32632",
    ...     output_dir  = paths.sm_shps,
    ...     sim_duration = sim_period + 100,
    ... )

    Supplying a hydraulic head raster directly (e.g. from a regional model):

    >>> result = build_mf_model_from_dem(
    ...     dem_path     = "GIS/dem.tif",
    ...     mf_folder    = wd,
    ...     mf_name      = "mymodel",
    ...     initial_head = "GIS/gw_head.tif",   # piezometric surface raster
    ... )

    Supplying a water-table depth raster (head = DEM − depth):

    >>> result = build_mf_model_from_dem(
    ...     dem_path           = "GIS/dem.tif",
    ...     mf_folder          = wd,
    ...     mf_name            = "mymodel",
    ...     water_table_depth  = "GIS/wt_depth.tif",  # depth-to-WT raster
    ... )
    >>> print("mf_grid written to:", result.grid_path)
    >>> print(f"Grid: {result.nrow} rows × {result.ncol} cols")
    """
    # ── Lazy imports ────────────────────────────────────────────────────────
    try:
        import rasterio
        import rasterio.crs as rasterio_crs
        from rasterio.transform import from_origin, array_bounds
        from rasterio.warp import reproject, Resampling, transform_bounds
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for build_mf_model_from_dem.  "
            "Install with: pip install rasterio"
        ) from exc

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for build_mf_model_from_dem.  "
            "Install with: pip install geopandas"
        ) from exc

    dem_str = str(dem_path)
    if not os.path.isfile(dem_str):
        raise FileNotFoundError(f"DEM raster not found: {dem_str!r}")

    wd = str(mf_folder)
    os.makedirs(wd, exist_ok=True)

    # ── Open DEM, optionally clip to boundary ────────────────────────────────
    with rasterio.open(dem_str) as src:
        src_crs = src.crs
        src_nodata = src.nodata if src.nodata is not None else nodata

        if boundary_path is not None:
            from rasterio.mask import mask as rasterio_mask
            bnd_path = str(boundary_path)
            if not os.path.isfile(bnd_path):
                raise FileNotFoundError(
                    f"Boundary file not found: {bnd_path!r}"
                )
            bnd_gdf = gpd.read_file(bnd_path)
            if bnd_gdf.crs is not None and bnd_gdf.crs != src_crs:
                bnd_gdf = bnd_gdf.to_crs(src_crs)
            geoms = bnd_gdf.geometry.tolist()
            raw_arr, raw_transform = rasterio_mask(
                src, geoms, crop=True, nodata=src_nodata, filled=True
            )
            raw_dem = raw_arr[0].astype(np.float64)
            raw_h, raw_w = raw_dem.shape
            raw_bounds = array_bounds(raw_h, raw_w, raw_transform)
        else:
            raw_dem = src.read(1).astype(np.float64)
            raw_transform = src.transform
            raw_h, raw_w = src.height, src.width
            raw_bounds = array_bounds(raw_h, raw_w, raw_transform)

    # ── Determine output CRS and bounds ─────────────────────────────────────
    if crs is not None:
        dst_crs = rasterio_crs.CRS.from_user_input(crs)
    else:
        dst_crs = src_crs

    left, bottom, right, top_b = (
        transform_bounds(src_crs, dst_crs, *raw_bounds)
        if dst_crs != src_crs
        else raw_bounds
    )

    # ── Compute output grid dimensions and transform ─────────────────────────
    ncol = max(1, int(np.ceil((right - left) / cell_size)))
    nrow = max(1, int(np.ceil((top_b - bottom) / cell_size)))
    x_origin = left
    y_origin = top_b
    dst_transform = from_origin(x_origin, y_origin, cell_size, cell_size)

    # ── Resample DEM to target grid ──────────────────────────────────────────
    top_elev = np.full((nrow, ncol), nodata, dtype=np.float64)
    reproject(
        source=raw_dem,
        destination=top_elev,
        src_transform=raw_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=src_nodata,
        dst_nodata=nodata,
    )

    shape = (nrow, ncol)

    # ── Helper: resolve scalar or raster-path to a (nrow, ncol) array ────────
    def _resolve(val: Union[float, str, os.PathLike]) -> np.ndarray:
        if isinstance(val, (int, float)):
            return np.full(shape, float(val), dtype=np.float64)
        arr = np.full(shape, nodata, dtype=np.float64)
        with rasterio.open(str(val)) as rsrc:
            reproject(
                source=rasterio.band(rsrc, 1),
                destination=arr,
                src_transform=rsrc.transform,
                src_crs=rsrc.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=rsrc.nodata,
                dst_nodata=nodata,
            )
        return arr

    # ── Derive parameter arrays ──────────────────────────────────────────────
    valid = top_elev != nodata

    thickness_arr = _resolve(aquifer_thickness)
    bot_elev = np.where(valid, top_elev - thickness_arr, nodata)

    sy_arr = _resolve(sy)

    if initial_head is not None:
        # Explicit head raster or scalar — highest priority
        head_arr: np.ndarray = _resolve(initial_head)
    elif water_table_depth is not None:
        # Depth-to-WT raster or scalar: head = top - depth, cell by cell
        depth_arr = _resolve(water_table_depth)
        head_arr = np.where(valid, top_elev - depth_arr, nodata)
    else:
        # Fallback: uniform scalar depth below land surface
        head_arr = np.where(valid, top_elev - head_depth, nodata)

    ibound = np.where(valid, 1, 0).astype(np.int32)

    # ── Build MODFLOW model ──────────────────────────────────────────────────
    mf = create_mf_model(
        mf_folder=wd,
        mf_name=mf_name,
        top_elev=top_elev,
        bot_elev=bot_elev,
        delr=cell_size,
        delc=cell_size,
        hk=hk,
        ss=ss,
        sy=sy_arr,
        initial_head=head_arr,
        riv_df=riv_df,
        sim_duration=sim_duration,
        vka=vka,
        ibound=ibound,
        nodata=nodata,
    )

    # ── Build mf_grid.gpkg ───────────────────────────────────────────────────
    # Resolve crs to a value accepted by create_mf_grid (str or int)
    if crs is not None:
        grid_crs: str | int | None = crs
    else:
        epsg = dst_crs.to_epsg() if dst_crs is not None else None
        grid_crs = epsg if epsg is not None else (
            dst_crs.to_wkt() if dst_crs is not None else None
        )

    out_dir = str(output_dir) if output_dir is not None else wd
    grid_path = create_mf_grid(
        swatmf_folder=wd,
        x_origin=x_origin,
        y_origin=y_origin,
        crs=grid_crs,
        output_dir=out_dir,
    )

    return MFModelResult(
        mf=mf,
        grid_path=grid_path,
        x_origin=x_origin,
        y_origin=y_origin,
        nrow=nrow,
        ncol=ncol,
    )

