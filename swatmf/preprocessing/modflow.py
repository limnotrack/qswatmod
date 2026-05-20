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
write_riv_file         — Overwrite a MODFLOW river-package (.riv) file.
compute_riv_params     — Derive RIV parameters from a ``river_grid`` table.
create_modflow_obs     — Write the ``modflow.obs`` observation file.
read_modflow_obs       — Read an existing ``modflow.obs`` into a DataFrame.
create_mf_model        — Build a new MODFLOW model from scratch using flopy.
create_modflow_mfn     — Generate ``modflow.mfn`` from the MODFLOW name file.
modify_modflow_oc      — Update unit numbers in the MODFLOW output-control file.
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
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            # Skip the count header — it has fewer than 5 numeric-looking tokens
            if len(parts) < 5:
                continue
            # The data rows have exactly 5 numeric fields (the rest is a comment)
            try:
                row_val  = int(parts[0])
                col_val  = int(parts[1])
                lay_val  = int(parts[2])
                gid_val  = int(parts[3])
                elev_val = float(parts[4])
            except (ValueError, IndexError):
                continue
            rows.append(
                {
                    "row":      row_val,
                    "col":      col_val,
                    "layer":    lay_val,
                    "grid_id":  gid_val,
                    "top_elev": elev_val,
                }
            )
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
    riv_df: pd.DataFrame,
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
    riv_df : pd.DataFrame
        River-package cell data.  Must contain columns ``layer``, ``row``,
        ``col``, ``riv_stage``, ``riv_cond``, ``riv_bot``.  Use
        :func:`compute_riv_params` to derive these from ``river_grid``
        attributes.
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
