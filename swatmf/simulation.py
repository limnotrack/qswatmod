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
        "# Optional output for SWAT-MODFLOW (0=no; 1=yes)\n",
        f"{_flag(cfg.output_swat_dp)}    SWAT Deep Percolation (mm) (for each HRU)\n",
        f"{_flag(cfg.output_mf_recharge)}    MODFLOW Recharge (m3/day) (for each MODFLOW Cell)\n",
        f"{_flag(cfg.output_channel_depth)}    SWAT Channel Depth (m) (for each SWAT Subbasin)\n",
        f"{_flag(cfg.output_river_stage)}    MODFLOW River Stage (m) (for each MODFLOW River Cell)\n",
        f"{_flag(cfg.output_gwsw_grid)}    Groundwater/Surface Water Exchange (m3/day) (for each MODFLOW River Cell)\n",
        f"{_flag(cfg.output_gwsw_sub)}    Groundwater/Surface Water Exchange (m3/day) (for each SWAT Subbasin)\n",
        f"{_flag(cfg.output_print_avg)}    Print out average values for SWAT-MODFLOW and RT3D output variables\n",
    ]

    if sim_period is not None:
        lines.append("# == Write SWAT-MODFLOW output only on specified days ==\n")
        days = _output_days(sim_period, cfg.output_step)
        lines.append(f"{len(days)}\n")
        lines.extend(f"{d}\n" for d in days)

        lines.append("# == Groundwater delay ==\n")
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

    # 1 — copy the primary link tables (no extension)
    for name in _LINK_TABLE_NAMES:
        src_path = os.path.join(src, name)
        if os.path.isfile(src_path):
            dst_path = shutil.copy2(src_path, os.path.join(dst, name))
            copied.append(dst_path)

    # 2 — copy extra-extension files (default: *.txt)
    for ext in extra_extensions:
        for src_path in glob.glob(os.path.join(src, f"*{ext}")):
            fname = os.path.basename(src_path)
            dst_path = shutil.copy2(src_path, os.path.join(dst, fname))
            copied.append(dst_path)

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

    proc = subprocess.Popen(
        os.path.normpath(exe_path),
        cwd=wd,
    )
    if wait:
        proc.wait()

    return proc
