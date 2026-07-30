"""
swatmf.simulation
==================
Read and write the ``swatmf_link.txt`` configuration file that controls
all SWAT-MODFLOW run-time options; copy link tables to the working
directory; and launch the SWAT-MODFLOW executable.

This mirrors the ``create_swatmf_link``, ``copylinkagefiles``, and
``run_SM`` functions from the QSWATMOD2 plugin without any dependency
on QGIS, PyQt, or the plugin GUI.

Typical usage
-------------
1. Read the template that ships with QSWATMOD2::

    cfg = SwatmfLinkConfig.defaults()

2. Adjust the flags you care about::

    cfg = cfg._replace(
        pumping_mf=True,
        drain_cells=True,
        gw_delay=7.0,
    )

3. Write to the SWAT-MODFLOW working directory::

    write_swatmf_link(wd, cfg)

4. Or read back an existing file::

    cfg = read_swatmf_link(wd)

5. Copy link tables from ``GIS/Table/`` to the working directory::

    copy_link_files(table_dir, wd)

6. Launch the SWAT-MODFLOW executable::

    proc = run_simulation(wd)
    proc.wait()
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Output-frequency helper
# ---------------------------------------------------------------------------

def _output_days(sim_period: int, step: int) -> list[int]:
    """Return the list of Julian days on which SWAT-MODFLOW writes output.

    Mirrors the ``spinBox_freq_sm_output`` logic in the plugin.

    Parameters
    ----------
    sim_period : int
        Total simulation duration in days.
    step : int
        Output frequency (every *step* days).  Use 1 for daily output.

    Returns
    -------
    list of int
    """
    if step == 1:
        return list(range(1, sim_period + 1, step))
    else:
        days = [1]  # force the first day
        days.extend(range(step, sim_period + 1, step))
        return days


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SwatmfLinkConfig:
    """All SWAT-MODFLOW run-time flags and settings.

    Attributes
    ----------
    swatmf_active : bool
        ``True`` → run SWAT + MODFLOW fully linked.
        ``False`` → run SWAT only.
    pumping_mf : bool
        Activate irrigation supplied by MODFLOW pumping.
        (Pumping rate dictated by MODFLOW.)
    pumping_swat : bool
        Activate SWAT auto-irrigation → MODFLOW pumping.
        (Pumping rate dictated by SWAT.)
    drain_cells : bool
        Route MODFLOW DRAIN-cell fluxes to SWAT subbasin channels.
    rt3d_active : bool
        Activate RT3D groundwater reactive transport.
    read_mf_obs : bool
        Read observation cell list from ``modflow.obs``.
    output_swat_dp : bool
        Write SWAT deep-percolation (mm, per HRU).
    output_mf_recharge : bool
        Write MODFLOW recharge (m³/day, per cell).
    output_channel_depth : bool
        Write SWAT channel depth (m, per subbasin).
    output_river_stage : bool
        Write MODFLOW river stage (m, per river cell).
    output_gwsw_grid : bool
        Write GW/SW exchange (m³/day, per MODFLOW river cell).
    output_gwsw_sub : bool
        Write GW/SW exchange (m³/day, per SWAT subbasin).
    output_print_avg : bool
        Write monthly/annual average SWAT-MODFLOW outputs.
    output_step : int
        Write results every *output_step* days (1 = every day).
    gw_delay : float
        Single-value groundwater delay [days] applied to all HRUs.
        Used only when ``gw_delay_per_hru`` is ``False``.
    gw_delay_per_hru : bool
        ``True`` → read one GW-delay value per HRU from the ``gw_delay``
        table; ``False`` → use the single value in ``gw_delay``.
    gw_delay_values : list of float
        Per-HRU GW delay values (only used when ``gw_delay_per_hru=True``).
    """

    swatmf_active: bool = True

    # --- Pre-Processing / Simulation tab flags --------------------------------
    pumping_mf: bool = False       # MODFLOW pumping → SWAT irrigation
    pumping_swat: bool = False     # SWAT auto-irrigation → MODFLOW pumping
    drain_cells: bool = False      # MODFLOW DRN cells → SWAT channels
    rt3d_active: bool = False

    # --- Observation output ---------------------------------------------------
    read_mf_obs: bool = False

    # --- Optional SWAT-MODFLOW outputs ----------------------------------------
    output_swat_dp: bool = True
    output_mf_recharge: bool = True
    output_channel_depth: bool = True
    output_river_stage: bool = True
    output_gwsw_grid: bool = True
    output_gwsw_sub: bool = True
    output_print_avg: bool = True

    # --- Output frequency -----------------------------------------------------
    output_step: int = 1           # write every N days

    # --- Groundwater delay ----------------------------------------------------
    gw_delay: float = 31.0
    gw_delay_per_hru: bool = False
    gw_delay_values: list = field(default_factory=list)

    @classmethod
    def defaults(cls) -> "SwatmfLinkConfig":
        """Return the factory-default configuration (matches the template file)."""
        return cls()


# ---------------------------------------------------------------------------
# _flag: convert bool → "1" or "0"
# ---------------------------------------------------------------------------

def _flag(value: bool) -> str:
    return "1" if value else "0"


# ---------------------------------------------------------------------------
# Write swatmf_link.txt
# ---------------------------------------------------------------------------

def write_swatmf_link(
    swatmf_folder: str | os.PathLike,
    cfg: SwatmfLinkConfig,
    sim_period: Optional[int] = None,
) -> str:
    """Write ``swatmf_link.txt`` from a :class:`SwatmfLinkConfig`.

    If *sim_period* is ``None`` the output-frequency section (the list of
    Julian days) is omitted — useful when you only need to update the flags.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    cfg : SwatmfLinkConfig
        Run-time configuration.
    sim_period : int, optional
        Total number of simulation days.  If provided, the output-day
        schedule and groundwater-delay block are appended.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(swatmf_folder)
    out_path = os.path.join(wd, "swatmf_link.txt")

    lines: list[str] = [
        f"{_flag(cfg.swatmf_active)}    SWAT-MODFLOW is activated\n",
        f"{_flag(cfg.pumping_mf)}    MODFLOW Pumping --> SWAT Irrigation\n",
        f"{_flag(cfg.pumping_swat)}    SWAT Auto-Irrigation --> MODFLOW Pumping\n",
        f"{_flag(cfg.drain_cells)}    MODFLOW Drains --> SWAT subbasin channels\n",
        f"{_flag(cfg.rt3d_active)}    RT3D is active (N and P groundwater reactive transport)\n",
        f"{_flag(cfg.read_mf_obs)}    Read in observation cells from \"modflow.obs\"\n",
        "Optional output for SWAT-MODFLOW (0=no; 1=yes)\n",
        f"{_flag(cfg.output_swat_dp)}    SWAT Deep Percolation (mm) (for each HRU)\n",
        f"{_flag(cfg.output_mf_recharge)}    MODFLOW Recharge (m3/day) (for each MODFLOW Cell)\n",
        f"{_flag(cfg.output_channel_depth)}    SWAT Channel Depth (m) (for each SWAT Subbasin)\n",
        f"{_flag(cfg.output_river_stage)}    MODFLOW River Stage (m) (for each MODFLOW River Cell)\n",
        f"{_flag(cfg.output_gwsw_grid)}    Groundwater/Surface Water Exchange (m3/day) (for each MODFLOW River Cell)\n",
        f"{_flag(cfg.output_gwsw_sub)}    Groundwater/Surface Water Exchange (m3/day) (for each SWAT Subbasin)\n",
        f"{_flag(cfg.output_print_avg)}    Print out average values for SWAT-MODFLOW and RT3D output variables\n",
    ]

    if sim_period is not None:
        lines.append("Write SWAT-MODFLOW output only on specified days\n")
        days = _output_days(sim_period, cfg.output_step)
        lines.append(f"{len(days)}\n")
        lines.extend(f"{d}\n" for d in days)

        lines.append("Groundwater delay\n")
        if cfg.gw_delay_per_hru and cfg.gw_delay_values:
            lines.append(
                "1    0 = read in a single value for all HRUs; "
                "1 = read in one value for each HRU\n"
            )
            lines.extend(f"{v}\n" for v in cfg.gw_delay_values)
        else:
            lines.append(
                "0    0 = read in a single value for all HRUs; "
                "1 = read in one value for each HRU\n"
            )
            lines.append(
                f"{cfg.gw_delay}    GW_DELAY : Groundwater delay [days]\n"
            )

    with open(out_path, "w") as fh:
        fh.writelines(lines)

    return out_path


# ---------------------------------------------------------------------------
# Read swatmf_link.txt back into a SwatmfLinkConfig
# ---------------------------------------------------------------------------

def read_swatmf_link(swatmf_folder: str | os.PathLike) -> SwatmfLinkConfig:
    """Parse an existing ``swatmf_link.txt`` file.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.

    Returns
    -------
    SwatmfLinkConfig
        Configuration recovered from the file.  The ``gw_delay_values``
        list is populated when per-HRU delays are present.
    """
    path = os.path.join(str(swatmf_folder), "swatmf_link.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"swatmf_link.txt not found in {swatmf_folder!r}")

    with open(path, "r") as fh:
        lines = fh.readlines()

    def _b(flag: str) -> bool:
        return flag.strip() == "1"

    cfg = SwatmfLinkConfig()
    gw_delay_mode: Optional[int] = None
    gw_delay_values: list[float] = []
    in_gw_delay = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if "Groundwater delay" in stripped:
                in_gw_delay = True
            continue

        parts = stripped.split(None, 1)
        if not parts:
            continue
        val = parts[0]

        if "SWAT-MODFLOW is activated" in stripped:
            cfg.swatmf_active = _b(val)
        elif "MODFLOW Pumping --> SWAT" in stripped:
            cfg.pumping_mf = _b(val)
        elif "SWAT Auto-Irrigation" in stripped:
            cfg.pumping_swat = _b(val)
        elif "MODFLOW Drains" in stripped:
            cfg.drain_cells = _b(val)
        elif "RT3D is active" in stripped:
            cfg.rt3d_active = _b(val)
        elif "Read in observation" in stripped:
            cfg.read_mf_obs = _b(val)
        elif "SWAT Deep Percolation" in stripped:
            cfg.output_swat_dp = _b(val)
        elif "MODFLOW Recharge" in stripped:
            cfg.output_mf_recharge = _b(val)
        elif "SWAT Channel Depth" in stripped:
            cfg.output_channel_depth = _b(val)
        elif "MODFLOW River Stage" in stripped:
            cfg.output_river_stage = _b(val)
        elif "Groundwater/Surface Water Exchange (m3/day) (for each MODFLOW River Cell)" in stripped:
            cfg.output_gwsw_grid = _b(val)
        elif "Groundwater/Surface Water Exchange (m3/day) (for each SWAT Subbasin)" in stripped:
            cfg.output_gwsw_sub = _b(val)
        elif "Print out average" in stripped:
            cfg.output_print_avg = _b(val)
        elif "GW_DELAY" in stripped:
            cfg.gw_delay = float(val)
        elif "read in a single value" in stripped:
            gw_delay_mode = int(val)
            cfg.gw_delay_per_hru = gw_delay_mode == 1
        elif in_gw_delay and gw_delay_mode == 1:
            try:
                gw_delay_values.append(float(val))
            except ValueError:
                pass

    if gw_delay_values:
        cfg.gw_delay_values = gw_delay_values

    return cfg


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarise_link_config(cfg: SwatmfLinkConfig) -> str:
    """Return a human-readable summary of the configuration.

    Parameters
    ----------
    cfg : SwatmfLinkConfig

    Returns
    -------
    str
        Multi-line string.
    """
    yn = {True: "YES", False: "no"}
    lines = [
        "swatmf_link.txt configuration",
        "=" * 40,
        f"  SWAT-MODFLOW active          : {yn[cfg.swatmf_active]}",
        "",
        "  Simulation options",
        f"    Pumping (MF → SWAT irrig.) : {yn[cfg.pumping_mf]}",
        f"    Pumping (SWAT → MF pump.)  : {yn[cfg.pumping_swat]}",
        f"    DRAIN cells → SWAT ch.     : {yn[cfg.drain_cells]}",
        f"    RT3D reactive transport    : {yn[cfg.rt3d_active]}",
        "",
        "  Output flags",
        f"    SWAT deep percolation      : {yn[cfg.output_swat_dp]}",
        f"    MODFLOW recharge           : {yn[cfg.output_mf_recharge]}",
        f"    SWAT channel depth         : {yn[cfg.output_channel_depth]}",
        f"    MODFLOW river stage        : {yn[cfg.output_river_stage]}",
        f"    GW-SW exchange (grid)      : {yn[cfg.output_gwsw_grid]}",
        f"    GW-SW exchange (subbasin)  : {yn[cfg.output_gwsw_sub]}",
        f"    Print monthly/annual avg   : {yn[cfg.output_print_avg]}",
        f"    Output every N days        : {cfg.output_step}",
        "",
        "  Groundwater delay",
        f"    Per-HRU mode               : {yn[cfg.gw_delay_per_hru]}",
        f"    Single-value delay [days]  : {cfg.gw_delay}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Copy link tables from GIS/Table → SWAT-MODFLOW folder
# ---------------------------------------------------------------------------

#: Link table file names written by :func:`generate_link_tables
#: <swatmf.preprocessing.linking.generate_link_tables>`.
_LINK_TABLE_NAMES = ("hru_dhru", "dhru_grid", "grid_dhru", "river_grid")

#: Executable-format files written by write_swatmf_dhru2hru / dhru2grid / grid2dhru.
#: These are the files the SWAT-MODFLOW Fortran executable actually reads.
_SWATMF_EXE_LINK_FILES = (
    "swatmf_dhru2hru.txt",
    "swatmf_dhru2grid.txt",
    "swatmf_grid2dhru.txt",
)

#: Legacy rename map kept for back-compat but no longer used for primary copy.
_LINK_TABLE_RENAME: dict[str, str] = {
    "hru_dhru":  "swatmf_hru2dhru.txt",   # human-readable → (not read by exe)
    "dhru_grid": "swatmf_dhru2grid.txt",   # same name, different format
    "grid_dhru": "swatmf_grid2dhru.txt",   # same name, different format
}


def copy_link_files(
    table_dir: str | os.PathLike,
    swatmf_folder: str | os.PathLike,
    *,
    extra_extensions: tuple[str, ...] = (".txt",),
) -> list[str]:
    """Copy SWAT-MODFLOW link tables from ``GIS/Table`` to the working directory.

    Replicates the **Create linkage files** button (``pushButton_create_linkfiles
    → createLinkFiles → copylinkagefiles``) from the QSWATMOD2 plugin.

    The following files are copied when present:

    * ``hru_dhru``, ``dhru_grid``, ``grid_dhru``, ``river_grid``
      (no extension — the primary link tables).
    * Any ``*.txt`` files in *table_dir* (mirrors the plugin behaviour of
      calling ``copylinkagefiles``, which copies ``*.txt`` files).

    Parameters
    ----------
    table_dir : str or path-like
        Source directory containing the link tables
        (``GIS/Table`` in the QSWATMOD2 project layout).
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (files are copied here).
    extra_extensions : tuple of str, optional
        Additional file extensions to copy from *table_dir*.
        Default ``(".txt",)``.

    Returns
    -------
    list of str
        Absolute paths of the files that were actually copied.

    Examples
    --------
    >>> from swatmf import Paths
    >>> from swatmf.simulation import copy_link_files
    >>>
    >>> paths = Paths("/data/my_project", "my_project")
    >>> copied = copy_link_files(paths.table_folder, paths.swatmf_folder)
    >>> for f in copied:
    ...     print("Copied:", f)
    """
    src = str(table_dir)
    dst = str(swatmf_folder)
    os.makedirs(dst, exist_ok=True)

    copied: list[str] = []

    # 1 — copy the executable-format link files (produced by write_swatmf_dhru2hru etc.)
    #     These are the files the SWAT-MODFLOW Fortran executable actually reads.
    for name in _SWATMF_EXE_LINK_FILES:
        src_path = os.path.join(src, name)
        if os.path.isfile(src_path):
            dst_path = shutil.copy2(src_path, os.path.join(dst, name))
            copied.append(dst_path)

    # 2 — copy river_grid (kept with its bare name; river package support)
    rg_src = os.path.join(src, "river_grid")
    if os.path.isfile(rg_src):
        copied.append(shutil.copy2(rg_src, os.path.join(dst, "river_grid")))

    # 3 — copy any extra *.txt files from table_dir (e.g. swatmf_link.txt backup)
    for ext in extra_extensions:
        for src_path in glob.glob(os.path.join(src, f"*{ext}")):
            fname = os.path.basename(src_path)
            # Skip the exe-format files already copied above
            if fname in _SWATMF_EXE_LINK_FILES:
                continue
            dst_path = shutil.copy2(src_path, os.path.join(dst, fname))
            copied.append(dst_path)

    return copied


# ---------------------------------------------------------------------------
# Copy MODFLOW model files to the SWAT-MODFLOW working directory
# ---------------------------------------------------------------------------

#: MODFLOW input file extensions that must be present in the working directory.
_MF_INPUT_EXTS = (
    ".dis", ".bas", ".nam", ".nwt", ".upw",
    ".riv", ".rch", ".oc", ".evt", ".drn",
)

#: Extra helper files created by swatmf that must accompany the MODFLOW files.
_MF_HELPER_FILES = ("modflow.mfn", "modflow.obs")


def write_swatmf_river2grid(
    river_grid_df: "pd.DataFrame",
    swatmf_folder: str | os.PathLike,
    *,
    layer: int = 1,
) -> str:
    """Write ``swatmf_river2grid.txt`` in the format read by SWAT-MODFLOW.

    Parameters
    ----------
    river_grid_df : DataFrame
        Output of :func:`swatmf.preprocessing.linking.generate_river_grid`.
        Must have columns ``grid_id`` (int), ``subbasin`` (int),
        ``rgrid_len`` (float).
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (TxtInOut).
    layer : int, optional
        MODFLOW layer number for all river cells.  Default 1.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    df = river_grid_df.copy()
    df["grid_id"]  = df["grid_id"].astype(int)
    df["subbasin"] = df["subbasin"].astype(int)
    df = df.sort_values(["grid_id", "subbasin"]).reset_index(drop=True)

    # Group by grid_id so each unique river cell becomes one record.
    # The third field on line 1 is n_subbasins (the count of subbasins
    # crossed by that cell), NOT the subbasin ID.  Subsequent lines list
    # all subbasin IDs, then all lengths — matching CreateSWATMF.exe output.
    groups = df.groupby("grid_id", sort=True)
    n_riv = len(groups)

    out_path = os.path.join(str(swatmf_folder), "swatmf_river2grid.txt")
    with open(out_path, "wb") as fh:
        # Header: %12d (CreateSWATMF.exe uses 12-wide, not 13)
        fh.write(f"{n_riv:12d}\r\n".encode())
        for river_id, (grid_id, grp) in enumerate(groups, 1):
            subs = grp["subbasin"].tolist()
            lens = grp["rgrid_len"].tolist()
            n_sub = len(subs)
            # Line 1: river_id  grid_id  n_subbasins
            fh.write(
                ("".join(f"{v:13d}" for v in [river_id, int(grid_id), n_sub])
                 + "\r\n").encode()
            )
            # Line 2: subbasin IDs (all on one line)
            fh.write(("".join(f"{s:13d}" for s in subs) + "\r\n").encode())
            # Line 3: river lengths (all on one line)
            fh.write(("".join(f"{l:13.5f}" for l in lens) + "\r\n").encode())
    return out_path


def copy_modflow_files(
    mf_folder: str | os.PathLike,
    swatmf_folder: str | os.PathLike,
) -> list[str]:
    """Copy MODFLOW input files from *mf_folder* to the SWAT-MODFLOW working
    directory, then regenerate ``modflow.mfn`` so it uses bare filenames
    (relative to the working directory).

    The SWAT-MODFLOW executable expects **all** inputs — SWAT files, MODFLOW
    files, link tables, and ``modflow.mfn`` — to live in a single working
    directory.  This function replicates the file-copying behaviour of the
    QSWATMOD2 plugin's *Create linkage files* step for the MODFLOW side.

    Files copied:

    * All MODFLOW input packages (``*.dis``, ``*.bas``, ``*.nam``, ``*.nwt``,
      ``*.upw``, ``*.riv``, ``*.rch``, ``*.oc``, ``*.evt``, ``*.drn`` …)
    * ``modflow.obs`` (if present)
    * ``modflow.mfn`` is **regenerated** in *swatmf_folder* rather than copied,
      so that the unit-number assignments and file paths are correct for the
      new location.

    Parameters
    ----------
    mf_folder : str or path-like
        Folder containing the MODFLOW model files (the ``.dis``, ``.nam`` …
        files written by flopy or the pre-built example model).
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (``TxtInOut``).  Files are copied here.

    Returns
    -------
    list[str]
        Absolute paths of every file written to *swatmf_folder*.
    """
    from swatmf.preprocessing.modflow import create_modflow_mfn

    src = str(mf_folder)
    dst = str(swatmf_folder)
    os.makedirs(dst, exist_ok=True)
    copied: list[str] = []

    # Copy all MODFLOW input package files.
    for fname in os.listdir(src):
        ext = os.path.splitext(fname)[1].lower()
        if ext in _MF_INPUT_EXTS:
            dst_path = shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
            copied.append(dst_path)

    # Copy optional helper files (modflow.obs etc.) but NOT modflow.mfn —
    # that is regenerated below with correct paths.
    for helper in _MF_HELPER_FILES:
        if helper == "modflow.mfn":
            continue
        src_path = os.path.join(src, helper)
        if os.path.isfile(src_path):
            dst_path = shutil.copy2(src_path, os.path.join(dst, helper))
            copied.append(dst_path)

    # Regenerate modflow.mfn in the destination so unit numbers and file
    # references are consistent with the files now in swatmf_folder.
    mfn_path = create_modflow_mfn(dst)
    copied.append(mfn_path)

    return copied


# ---------------------------------------------------------------------------
# Run SWAT-MODFLOW simulation
# ---------------------------------------------------------------------------

#: Recognised executable names, in order of preference.
_EXE_CANDIDATES = (
    "SWAT-MODFLOW3.exe",
    "swatmf_rel230818.exe",
    "swatmf.exe",
)


def run_simulation(
    swatmf_folder: str | os.PathLike,
    exe_name: str | None = None,
    *,
    wait: bool = True,
) -> subprocess.Popen:
    """Launch the SWAT-MODFLOW executable.

    Replicates the **Run simulation** button (``pushButton_run_SM → run_SM``)
    from the QSWATMOD2 plugin.

    The function searches *swatmf_folder* for a known executable in this
    order: ``SWAT-MODFLOW3.exe`` → ``swatmf_rel230818.exe`` → ``swatmf.exe``.
    You can override the search by passing *exe_name* explicitly.

    The executable is launched with *swatmf_folder* as its working directory
    so that all relative file paths inside the executable resolve correctly.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory (contains the executable and all input
        files such as ``swatmf_link.txt``, ``hru_dhru``, etc.).
    exe_name : str, optional
        Base name (or absolute path) of the executable to run.  If omitted,
        the function auto-detects from :data:`_EXE_CANDIDATES`.
    wait : bool, optional
        If ``True`` (default), block until the process finishes.
        If ``False``, return the :class:`subprocess.Popen` object immediately
        so the caller can monitor or terminate the process.

    Returns
    -------
    subprocess.Popen
        The running (or completed) process object.

    Raises
    ------
    FileNotFoundError
        If no recognised SWAT-MODFLOW executable can be found in
        *swatmf_folder* and *exe_name* is not provided.

    Examples
    --------
    Run synchronously (block until complete):

    >>> from swatmf.simulation import run_simulation
    >>> proc = run_simulation(wd)
    >>> print("Return code:", proc.returncode)

    Run in the background and poll for completion:

    >>> proc = run_simulation(wd, wait=False)
    >>> while proc.poll() is None:
    ...     print("Still running …")
    >>> print("Finished with return code", proc.returncode)
    """
    wd = str(swatmf_folder)

    if exe_name is not None:
        # Allow either a basename or an absolute path
        exe_path = exe_name if os.path.isabs(exe_name) else os.path.join(wd, exe_name)
    else:
        exe_path = None
        for candidate in _EXE_CANDIDATES:
            candidate_path = os.path.join(wd, candidate)
            if os.path.isfile(candidate_path):
                exe_path = candidate_path
                break

    if exe_path is None or not os.path.isfile(exe_path):
        searched = ", ".join(_EXE_CANDIDATES)
        raise FileNotFoundError(
            f"No SWAT-MODFLOW executable found in {wd!r}.  "
            f"Searched for: {searched}.  "
            "Supply the exe_name parameter to specify a custom executable."
        )

    # Explicitly redirect stdin to DEVNULL to prevent the process from inheriting
    # invalid console handles (fixes OSError/WinError 6 in R's reticulate).
    #
    # stdout is redirected to DEVNULL (discarded): the executable produces no
    # meaningful stdout and the pipe buffer would deadlock if we used PIPE with
    # proc.wait() on a long run.
    #
    # stderr is captured via communicate() which drains the pipe continuously
    # to prevent the same buffer-full deadlock.
    proc = subprocess.Popen(
        os.path.normpath(exe_path),
        cwd=wd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,   # discard — avoids pipe-buffer deadlock
        stderr=subprocess.PIPE,      # capture for error reporting
    )

    if wait:
        # communicate() drains stderr while waiting — safe for long runs.
        # For progress output use run_simulation_monitored() instead.
        _, stderr_bytes = proc.communicate()
        if proc.returncode != 0:
            log_tail    = _read_log_tail(wd)
            stderr_text = (stderr_bytes or b"").decode(errors="replace").strip()
            msg_parts   = [
                f"SWAT-MODFLOW exited with return code {proc.returncode}.",
            ]
            if log_tail:
                msg_parts.append(f"swatmf_log (last lines):\n{log_tail}")
            if stderr_text:
                msg_parts.append(f"stderr:\n{stderr_text}")
            raise RuntimeError("\n".join(msg_parts))
        # Store captured stderr on the object so callers can inspect it.
        proc._stderr_output = (stderr_bytes or b"").decode(errors="replace")

    return proc


# ---------------------------------------------------------------------------
# Monitored run — prints swatmf_log lines as they are written
# ---------------------------------------------------------------------------

def run_simulation_monitored(
    swatmf_folder: str | os.PathLike,
    exe_name: str | None = None,
    *,
    poll_interval: float = 2.0,
) -> subprocess.Popen:
    """Launch SWAT-MODFLOW and stream ``swatmf_log`` to stdout in real time.

    Unlike :func:`run_simulation` (which blocks silently), this function
    prints each new line written to ``swatmf_log`` as the executable runs,
    then prints a dot every *poll_interval* seconds while the time-stepping
    loop is active (the executable only writes to the log during init).

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    exe_name : str, optional
        Executable name or path.  Auto-detected when omitted.
    poll_interval : float, optional
        Seconds between log-file checks.  Default 2.

    Returns
    -------
    subprocess.Popen
        Completed process object (``returncode`` is set).

    Raises
    ------
    RuntimeError
        If the executable exits with a non-zero return code.
    """
    import time
    import threading

    wd = str(swatmf_folder)

    # Launch (non-blocking)
    proc = run_simulation(wd, exe_name=exe_name, wait=False)

    log_path    = os.path.join(wd, "swatmf_log")
    seen_bytes  = 0
    init_done   = False
    dot_count   = 0
    _COLS       = 60   # dots per line before wrapping

    print("Running SWAT-MODFLOW  (log output below — dots = time-stepping)")
    print("-" * 60)

    # Drain stderr in a background thread so it never blocks
    stderr_buf: list[bytes] = []
    def _drain() -> None:
        if proc.stderr:
            stderr_buf.append(proc.stderr.read())
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    try:
        while proc.poll() is None:
            # Read any new log lines
            if os.path.isfile(log_path):
                with open(log_path, "rb") as fh:
                    fh.seek(seen_bytes)
                    chunk = fh.read()
                if chunk:
                    seen_bytes += len(chunk)
                    for line in chunk.decode(errors="replace").splitlines():
                        line = line.strip()
                        if line:
                            if dot_count:
                                print()   # end the dots line
                                dot_count = 0
                            print(f"  {line}")
                    # After init the log goes quiet — switch to dot mode
                    if "initialization finished" in chunk.decode(errors="replace"):
                        init_done = True

            if init_done:
                print(".", end="", flush=True)
                dot_count += 1
                if dot_count >= _COLS:
                    print()
                    dot_count = 0

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        proc.terminate()
        print("\n[interrupted]")
        return proc

    t.join(timeout=5)

    # Print any remaining log content
    if os.path.isfile(log_path):
        with open(log_path, "rb") as fh:
            fh.seek(seen_bytes)
            remainder = fh.read()
        if remainder:
            if dot_count:
                print()
            for line in remainder.decode(errors="replace").splitlines():
                if line.strip():
                    print(f"  {line.strip()}")

    if dot_count:
        print()

    print("-" * 60)

    if proc.returncode != 0:
        log_tail    = _read_log_tail(wd)
        stderr_text = (b"".join(stderr_buf)).decode(errors="replace").strip()
        msg_parts   = [f"SWAT-MODFLOW exited with return code {proc.returncode}."]
        if log_tail:
            msg_parts.append(f"swatmf_log (last lines):\n{log_tail}")
        if stderr_text:
            msg_parts.append(f"stderr:\n{stderr_text}")
        raise RuntimeError("\n".join(msg_parts))

    print(f"Simulation complete.  Return code: {proc.returncode}")
    return proc


# ---------------------------------------------------------------------------
# Simulation validation
# ---------------------------------------------------------------------------

#: Mapping from SwatmfLinkConfig flag name → expected output files.
#: Files marked with '*' use glob patterns (prefix match).
_CFG_TO_OUTPUTS: dict[str, list[str]] = {
    # SWAT standard outputs
    "_always": [
        "output.rch",
        "output.std",
    ],
    # SWAT-MODFLOW coupled outputs
    "read_mf_obs":         ["swatmf_out_MF_obs"],
    "output_swat_dp":      ["swatmf_out_SWAT_recharge"],
    "output_mf_recharge":  ["swatmf_out_MF_recharge"],
    "output_channel_depth":["swatmf_out_SWAT_channel"],
    "output_river_stage":  ["swatmf_out_MF_riverstage"],
    "output_gwsw_grid":    ["swatmf_out_MF_gwsw"],
    "output_gwsw_sub":     ["swatmf_out_SWAT_gwsw"],
}


def _read_log_tail(swatmf_folder: str, n: int = 15) -> str:
    """Return the last *n* lines of ``swatmf_log``, or empty string."""
    log_path = os.path.join(swatmf_folder, "swatmf_log")
    if not os.path.isfile(log_path):
        return ""
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return "".join(lines[-n:]).rstrip()


def validate_simulation(
    swatmf_folder: str | os.PathLike,
    cfg: "SwatmfLinkConfig | None" = None,
    *,
    print_report: bool = True,
) -> dict:
    """Check simulation outputs against the configuration and the run log.

    Reads ``swatmf_log`` to determine how far the executable progressed,
    then verifies that every output file expected from the active
    ``swatmf_link.txt`` flags actually exists and is non-empty.

    Parameters
    ----------
    swatmf_folder : str or path-like
        SWAT-MODFLOW working directory.
    cfg : SwatmfLinkConfig, optional
        Configuration object used for the run.  If omitted, the function
        re-reads ``swatmf_link.txt`` from *swatmf_folder*.
    print_report : bool, optional
        If ``True`` (default) print a human-readable report to stdout.

    Returns
    -------
    dict with keys:

    ``"log_lines"``
        All lines from ``swatmf_log`` as a list of strings.
    ``"log_complete"``
        ``True`` if the log contains the expected completion marker
        (``"swatmf: simulation is complete"``).
    ``"log_last"``
        Last non-blank line of the log (useful for diagnosing crashes).
    ``"outputs"``
        Dict mapping each expected filename to ``"ok"``, ``"empty"``,
        or ``"missing"``.
    ``"all_ok"``
        ``True`` only if the log is complete and all outputs are ``"ok"``.
    """
    wd = str(swatmf_folder)

    # ── Read log ─────────────────────────────────────────────────────────────
    log_path = os.path.join(wd, "swatmf_log")
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log_lines = fh.readlines()
    else:
        log_lines = []

    log_text = "".join(log_lines)
    # The executable writes to swatmf_log only during initialisation; it does
    # not append a "simulation complete" footer.  We treat the run as complete
    # if the log contains the last init step ("initialization finished") AND
    # all expected output files are non-trivially sized.
    log_complete = (
        "swatmf: simulation is complete" in log_text.lower()
        or "initialization finished" in log_text.lower()
    )
    non_blank = [l.rstrip() for l in log_lines if l.strip()]
    log_last = non_blank[-1] if non_blank else "(log is empty)"

    # ── Determine expected outputs from config ────────────────────────────────
    if cfg is None:
        try:
            cfg = read_swatmf_link(wd)
        except Exception:
            cfg = None

    expected_files: list[str] = list(_CFG_TO_OUTPUTS["_always"])
    if cfg is not None:
        for flag, files in _CFG_TO_OUTPUTS.items():
            if flag == "_always":
                continue
            if getattr(cfg, flag, False):
                expected_files.extend(files)

    # ── Check each file ───────────────────────────────────────────────────────
    # Files with only a header line (< 200 bytes) are treated as effectively
    # empty — the executable writes the header before running any timesteps,
    # so a header-only file indicates a crash during initialisation.
    _MIN_DATA_BYTES = 200

    outputs: dict[str, str] = {}
    for fname in expected_files:
        fpath = os.path.join(wd, fname)
        if not os.path.isfile(fpath):
            outputs[fname] = "missing"
        elif os.path.getsize(fpath) < _MIN_DATA_BYTES:
            outputs[fname] = "empty"
        else:
            outputs[fname] = "ok"

    all_ok = log_complete and all(v == "ok" for v in outputs.values())

    # ── Pre-flight: required input files ────────────────────────────────────
    required_inputs = [
        "swatmf_link.txt", "modflow.mfn",
        "swatmf_dhru2hru.txt", "swatmf_dhru2grid.txt", "swatmf_grid2dhru.txt",
        "file.cio",
    ]
    if cfg is not None and cfg.read_mf_obs:
        required_inputs.append("modflow.obs")

    missing_inputs: list[str] = [
        f for f in required_inputs
        if not os.path.isfile(os.path.join(wd, f))
    ]

    # ── Print report ─────────────────────────────────────────────────────────
    if print_report:
        sep = "-" * 56
        print(sep)
        print("SWAT-MODFLOW Simulation Validation Report")
        print(sep)

        # Log status
        print("\n[1] Run log  (swatmf_log)")
        if not log_lines:
            print("    [!] swatmf_log not found -- was the model run?")
        else:
            complete_str = "yes" if log_complete else "NO  <- model did not finish"
            print(f"    Complete : {complete_str}")
            print(f"    Last line: {log_last.strip()}")
            if not log_complete:
                print("\n    Last 15 log lines:")
                for line in log_lines[-15:]:
                    print(f"      {line}", end="")

        # Missing inputs
        print("\n[2] Required input files")
        if missing_inputs:
            for f in missing_inputs:
                print(f"    [!] MISSING  {f}")
        else:
            print("    [ok] all present")

        # Output files
        print("\n[3] Expected output files")
        for fname, status in outputs.items():
            icon = "[ok]" if status == "ok" else "[!] "
            size = ""
            fpath = os.path.join(wd, fname)
            if os.path.isfile(fpath):
                kb = os.path.getsize(fpath) / 1024
                size = f"  ({kb:.1f} KB)"
            print(f"    {icon}  {status.upper():<8}  {fname}{size}")

        print(f"\n{'[ok] All checks passed.' if all_ok else '[!]  Issues found -- see above.'}")
        print(sep)

    return {
        "log_lines":    log_lines,
        "log_complete": log_complete,
        "log_last":     log_last,
        "outputs":      outputs,
        "missing_inputs": missing_inputs,
        "all_ok":       all_ok,
    }
