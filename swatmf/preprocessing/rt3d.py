"""
swatmf.preprocessing.rt3d
==========================
Pure-Python helpers for creating RT3D (Reactive Transport in 3 Dimensions)
input files required by the SWAT-MODFLOW-RT3D coupled model.

RT3D is a reactive-transport code that simulates the fate and transport
of nitrogen (NO₃) and phosphorus (P) in the groundwater system.  When
activated, SWAT passes recharge concentrations to RT3D daily via the
SMRT linkage.

The functions here replicate the ``write_rt3d_inputs`` pipeline from the
QSWATMOD2 plugin's ``pyfolder/write_rt3d.py`` without any QGIS/PyQt
dependency.

Functions exposed
-----------------
write_rt3d_btn       — Write the Basic Transport (.btn) package file.
write_rt3d_dsp       — Write the Dispersion (.dsp) package file.
write_rt3d_rct       — Write the Reaction (.rct) package file.
write_rt3d_gcg       — Write the implicit solver (.gcg) package file.
write_rt3d_adv       — Write the Advection (.adv) package file.
write_rt3d_ssm       — Write the Source/Sink Mixing (.ssm) package file.
write_rt3d_filenames — Write the ``rt3d_filenames`` master file list.
write_rt3d_model     — Convenience wrapper that writes all RT3D files.
RT3DConfig           — Dataclass holding all RT3D parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RT3D configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RT3DConfig:
    """All parameters needed to write a complete set of RT3D input files.

    Attributes
    ----------
    model_name : str
        Base name for RT3D files (e.g. ``"rt3d_model"``).
    nrow : int
        Number of grid rows (from MODFLOW .dis).
    ncol : int
        Number of grid columns (from MODFLOW .dis).
    nlay : int
        Number of grid layers. Default 1.
    ncomp : int
        Number of transported species. Default 2 (NO₃ and P).
    mcomp : int
        Number of mobile species. Default 2.
    species : list of str
        Species names. Default ``["NO3", "P"]``.
    porosity : float or np.ndarray
        Aquifer porosity — a scalar or 2-D array (nrow × ncol).
        Default 0.30.
    icbund : np.ndarray or None
        ICBUND array (same shape as MODFLOW IBOUND).  Cells with 0 are
        inactive for transport.  If ``None``, derived from the MODFLOW BAS
        IBOUND array (all active cells = 1).
    initial_conc_no3 : float or np.ndarray
        Initial NO₃ concentration [mg/L].  Scalar or array.  Default 0.0.
    initial_conc_p : float or np.ndarray
        Initial P concentration [mg/L].  Scalar or array.  Default 0.0.
    longitudinal_dispersivity : float
        Longitudinal dispersivity [length units].  Default 10.0.
    ratio_h_to_l : float
        Ratio of horizontal transverse to longitudinal dispersivity.
        Default 0.1.
    ratio_v_to_l : float
        Ratio of vertical transverse to longitudinal dispersivity.
        Default 0.01.
    effective_molecular_diffusion : float
        Effective molecular diffusion coefficient.  Default 0.0.
    bulk_density : float
        Aquifer bulk density [g/cm³ or consistent mass/volume].
        Default 1.5.
    sorption_no3 : float
        Linear partition coefficient for NO₃.  Default 0.0 (no sorption).
    sorption_p : float
        Linear partition coefficient for P.  Default 0.0.
    k_denitrification : float
        First-order denitrification rate constant [1/day].  Default 0.01.
    k_half_saturation : float
        Monod half-saturation term for denitrification [mg/L].
        Default 10.0.
    obs_cells : pd.DataFrame or None
        Observation cells for concentration output.  Must contain columns
        ``row``, ``col``, ``layer``.  If ``None``, no observation cells
        are written.
    output_times : list of int or None
        Days on which RT3D writes output.  If ``None``, daily output is
        assumed and must be supplied at write time.
    """

    model_name: str = "rt3d_model"
    nrow: int = 1
    ncol: int = 1
    nlay: int = 1
    ncomp: int = 2
    mcomp: int = 2
    species: list = field(default_factory=lambda: ["NO3", "P"])

    # Transport properties
    porosity: Union[float, np.ndarray] = 0.30
    icbund: Optional[np.ndarray] = None

    # Initial concentrations
    initial_conc_no3: Union[float, np.ndarray] = 0.0
    initial_conc_p: Union[float, np.ndarray] = 0.0

    # Dispersion
    longitudinal_dispersivity: float = 10.0
    ratio_h_to_l: float = 0.1
    ratio_v_to_l: float = 0.01
    effective_molecular_diffusion: float = 0.0

    # Reaction
    bulk_density: float = 1.5
    sorption_no3: float = 0.0
    sorption_p: float = 0.0
    k_denitrification: float = 0.01
    k_half_saturation: float = 10.0

    # Observation / output
    obs_cells: Optional[pd.DataFrame] = None
    output_times: Optional[list] = None


# ---------------------------------------------------------------------------
# Individual file writers
# ---------------------------------------------------------------------------

def _write_array(f, arr: Union[float, np.ndarray], nrow: int, ncol: int,
                 fmt: str = "%.5e", flag_line: bool = True) -> None:
    """Write a 2-D array or constant to an open file handle."""
    if np.isscalar(arr):
        if flag_line:
            f.write(f"0 {arr}\n")
        else:
            a = np.full((nrow, ncol), float(arr))
            for row in a:
                f.write("\t".join(fmt % v for v in row) + "\n")
    else:
        a = np.asarray(arr)
        if flag_line:
            f.write("1 0.00\n")
        for row in a:
            f.write("\t".join(fmt % v for v in row) + "\n")


def write_rt3d_btn(
    output_dir: str | os.PathLike,
    cfg: RT3DConfig,
    output_times: Optional[list] = None,
) -> str:
    """Write the RT3D Basic Transport Package (``.btn``) file.

    Parameters
    ----------
    output_dir : path-like
        Directory where the file will be written.
    cfg : RT3DConfig
        RT3D configuration.
    output_times : list of int, optional
        Override for output times.  Falls back to ``cfg.output_times``.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(output_dir)
    os.makedirs(wd, exist_ok=True)
    fname = cfg.model_name + ".btn"
    path = os.path.join(wd, fname)

    times = output_times or cfg.output_times or list(range(1, 366))

    with open(path, "w", newline="") as f:
        # Packages header
        f.write("'PACKAGES USED: ADV,DSP,SSM,RCT,GCG,SALT"
                "---------------------------------------'\n")
        f.write("T T T F T F F F F F\n")

        # Species
        f.write("'NCOMP, MCOMP ---------------------------------"
                "------------------------------------'\n")
        f.write(f"{cfg.ncomp} {cfg.mcomp}\n")

        # Output format
        f.write("'TYPE OF WRITING FOR OUTPUT FILES "
                "(0=ASCII/1=BINARY/2=BOTH) -------------------'\n")
        f.write("0\n")

        # Species names and flags
        f.write("'SPECIES (MOBILE/IMMOBILE) ------"
                "----------------------------------------------'\n")
        for sp in cfg.species:
            f.write(f"'{sp}' 1 1\n")

        # Units
        f.write("'d'  'm'  'g'\n")

        # Porosity
        f.write("'POROSITY FOR EACH LAYER --------"
                "----------------------------------------------'\n")
        if cfg.icbund is not None:
            icb = np.asarray(cfg.icbund)
        else:
            icb = np.ones((cfg.nrow, cfg.ncol), dtype=int)

        # Porosity array: multiply icbund by porosity value
        if np.isscalar(cfg.porosity):
            por_arr = icb.astype(float) * float(cfg.porosity)
        else:
            por_arr = np.asarray(cfg.porosity)

        for row in por_arr:
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")

        # ICBUND
        f.write("'ICBUND ARRAY ---------------------------------"
                "------------------------------------'\n")
        for row in icb:
            f.write("\t".join(str(int(v)) for v in row) + "\n")

        # Initial concentrations
        f.write("'INITIAL CONCENTRATIONS: EACH SPECIES ---------"
                "------------------------------------'\n")
        # NO3
        if np.isscalar(cfg.initial_conc_no3):
            f.write(f"0 {cfg.initial_conc_no3} CNO3\n")
        else:
            f.write("1 0.00 CNO3\n")
            arr = np.asarray(cfg.initial_conc_no3)
            for row in arr:
                f.write(" ".join(f"{v:.5e}" for v in row) + "\n")

        # P
        if np.isscalar(cfg.initial_conc_p):
            f.write(f"0 {cfg.initial_conc_p} P\n")
        else:
            f.write("1 0.00 P\n")
            arr = np.asarray(cfg.initial_conc_p)
            for row in arr:
                f.write(" ".join(f"{v:.5e}" for v in row) + "\n")

        # Inactive cell marker
        f.write("'VALUE INDICATING INACTIVE CELL CONCENTRATION "
                "---------------------------------'\n")
        f.write(" -999.0000\n")

        # Output format flags
        f.write("'IFMTCN(print), IFMTNP(particle), IFMTRF(R), "
                "IFMTDP(D), SAVUCN(binary) --------'\n")
        f.write("6\t0\t0\t0\tF\n")

        # Output times
        f.write("'NUMBER OF OUTPUT TIMES --------"
                "-----------------------------------------------'\n")
        f.write(f"{len(times)}\n")
        f.write("'OUTPUT TIMES\t---------------------------------"
                "-----------------------------------'\n")
        f.write(" ".join(str(t) for t in times) + "\n")

        # Observation cells
        f.write("'OBSERVATION CELLS: I,J,K\t-----"
                "-----------------------------------------------'\n")
        if cfg.obs_cells is not None and len(cfg.obs_cells) > 0:
            obs = cfg.obs_cells
            f.write(f"{len(obs)} 1\n")
            for _, r in obs.iterrows():
                f.write(f"{int(r['row'])} {int(r['col'])} "
                        f"{int(r.get('layer', 1))}\n")
        else:
            f.write("0 1\n")

        # Mass budget
        f.write("'OUTPUT MASS BUDGET FILES\t------"
                "----------------------------------------------'\n")
        f.write("F\n")

    return os.path.abspath(path)


def write_rt3d_dsp(output_dir: str | os.PathLike, cfg: RT3DConfig) -> str:
    """Write the RT3D Dispersion Package (``.dsp``) file.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(output_dir)
    os.makedirs(wd, exist_ok=True)
    fname = cfg.model_name + ".dsp"
    path = os.path.join(wd, fname)

    with open(path, "w", newline="") as f:
        f.write(" 'LONGITUDINAL DISPERSIVITY --------"
                "-------------------------------------------'\n")
        f.write(f"       0 {cfg.longitudinal_dispersivity}\n")
        f.write(" 'RATIO OF HORIZ. TRANSVERSE TO LONG. DISP. "
                "-----------------------------------'\n")
        f.write(f"       0 {cfg.ratio_h_to_l}\n")
        f.write(" 'RATIO OF VERTIC. TRANSVERSE TO LONG. DISP. "
                "----------------------------------'\n")
        f.write(f"       0 {cfg.ratio_v_to_l}\n")
        f.write(" 'EFFECTIVE MOLECULAR DIFFUSION COEFFICIENT "
                "-----------------------------------'\n")
        f.write(f"       0 {cfg.effective_molecular_diffusion}\n")

    return os.path.abspath(path)


def write_rt3d_rct(output_dir: str | os.PathLike, cfg: RT3DConfig) -> str:
    """Write the RT3D Reaction Package (``.rct``) file.

    The reaction model uses IREACT=10 (custom denitrification kinetics
    from the SWAT-MODFLOW-RT3D source code) with linear sorption
    (ISOTHM=1).

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(output_dir)
    os.makedirs(wd, exist_ok=True)
    fname = cfg.model_name + ".rct"
    path = os.path.join(wd, fname)

    with open(path, "w", newline="") as f:
        f.write("'ISOTHM,IREACT,NCRXNDATA,NVRXNDATA,ISOLVER,"
                "IRCTOP -----------------------------'\n")
        f.write("1 10 2 0 1 0\n")
        f.write("'Bulk density --------"
                "--------------------------------------'\n")
        f.write(f"0  {cfg.bulk_density}\n")
        f.write("'Sorption parameters --------"
                "--------------------------------'\n")
        f.write(f"0  {cfg.sorption_no3}  partition coefficient "
                "for NO3  (linear sorption)\n")
        f.write(f"0  {cfg.sorption_p}  partition coefficient "
                "for PO4  (linear sorption)\n")
        f.write("0  0.0  second parameter for NO3  "
                "(not used for linear sorption)\n")
        f.write("0  0.0  second parameter for PO4  "
                "(not used for linear sorption)\n")
        f.write("'Spatially Constant Values for reaction rates' "
                "--------------------------------'\n")
        f.write(f"{cfg.k_denitrification}  First-Order Rate "
                "Constant of Denitrification\n")
        f.write(f"{cfg.k_half_saturation}  Monod Half-Saturation "
                "Term for Denitrification\n")

    return os.path.abspath(path)


def write_rt3d_gcg(output_dir: str | os.PathLike, cfg: RT3DConfig) -> str:
    """Write the RT3D GCG implicit solver (``.gcg``) file.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(output_dir)
    os.makedirs(wd, exist_ok=True)
    fname = cfg.model_name + ".gcg"
    path = os.path.join(wd, fname)

    with open(path, "w", newline="") as f:
        f.write("50 1 3 0\t\t\t\t       ITER1,MXITER,ISOLVE,NCRS\n")
        f.write("1.000 0.00001 0\t\t\t       ACCL,CCLOSE,IPRGCG\n")

    return os.path.abspath(path)


def write_rt3d_adv(output_dir: str | os.PathLike, cfg: RT3DConfig) -> str:
    """Write the RT3D Advection (``.adv``) package file.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(output_dir)
    os.makedirs(wd, exist_ok=True)
    fname = cfg.model_name + ".adv"
    path = os.path.join(wd, fname)

    with open(path, "w", newline="") as f:
        f.write("         0 1.0000000         1\n")

    return os.path.abspath(path)


def write_rt3d_ssm(output_dir: str | os.PathLike, cfg: RT3DConfig) -> str:
    """Write the RT3D Source/Sink Mixing (``.ssm``) package file.

    The SSM file is minimal because SWAT-MODFLOW passes recharge
    concentrations at run time via the SMRT linkage — the file only
    needs a flag for constant-concentration cells.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(output_dir)
    os.makedirs(wd, exist_ok=True)
    fname = cfg.model_name + ".ssm"
    path = os.path.join(wd, fname)

    with open(path, "w", newline="") as f:
        f.write(" Flag for constant concentration cells\n F\n")

    return os.path.abspath(path)


def write_rt3d_filenames(
    output_dir: str | os.PathLike,
    cfg: RT3DConfig,
) -> str:
    """Write the ``rt3d_filenames`` master file that lists all RT3D packages.

    This file is read by the SWAT-MODFLOW-RT3D executable to locate the
    individual package files.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    wd = str(output_dir)
    os.makedirs(wd, exist_ok=True)
    path = os.path.join(wd, "rt3d_filenames")

    name = cfg.model_name
    with open(path, "w", newline="") as f:
        f.write(f"'{name}.btn'           INBTN=1\t\t\t"
                "Basic Transport Package\n")
        f.write(f"'{name}.adv'\t\t\tINADV=2\t\t\t"
                "Advection Package\n")
        f.write(f"'{name}.dsp'\t\t\tINDSP=3\t\t\t"
                "Dispersion Package\n")
        f.write(f"'{name}.ssm'\t\t\tINSSM=4\t\t\t"
                "Source/Sink Mixing Package\n")
        f.write(f"'{name}.rct'\t\t\tINRCT=5\t\t\t"
                "Reaction Package\n")
        f.write(f"'{name}.gcg'\t\t\tINGCG=6\t\t\t"
                "Implicit Solver Package\n")
        f.write(f"'{name}.restart'\t\t    OUTRES=10\t\t"
                "Restart File\n")

    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def write_rt3d_model(
    output_dir: str | os.PathLike,
    cfg: RT3DConfig,
    output_times: Optional[list] = None,
) -> dict[str, str]:
    """Write all RT3D input files for a SWAT-MODFLOW-RT3D simulation.

    This is the top-level convenience function that writes every package
    file required to activate RT3D.

    Parameters
    ----------
    output_dir : path-like
        Directory where all RT3D files will be written (typically the
        SWAT-MODFLOW working directory / TxtInOut).
    cfg : RT3DConfig
        Complete RT3D configuration.
    output_times : list of int, optional
        Output times override (passed to :func:`write_rt3d_btn`).

    Returns
    -------
    dict
        Mapping of package name to absolute file path for each written
        file.

    Examples
    --------
    >>> from swatmf.preprocessing.rt3d import RT3DConfig, write_rt3d_model
    >>> cfg = RT3DConfig(
    ...     model_name="rt3d_mbrw",
    ...     nrow=46, ncol=81,
    ...     porosity=0.30,
    ...     initial_conc_no3=0.5,
    ...     initial_conc_p=0.01,
    ...     longitudinal_dispersivity=10.0,
    ...     k_denitrification=0.01,
    ...     k_half_saturation=10.0,
    ... )
    >>> files = write_rt3d_model(wd, cfg, output_times=list(range(1, 366)))
    >>> for pkg, path in files.items():
    ...     print(f"{pkg}: {path}")
    """
    result = {}
    result["btn"] = write_rt3d_btn(output_dir, cfg, output_times)
    result["dsp"] = write_rt3d_dsp(output_dir, cfg)
    result["rct"] = write_rt3d_rct(output_dir, cfg)
    result["gcg"] = write_rt3d_gcg(output_dir, cfg)
    result["adv"] = write_rt3d_adv(output_dir, cfg)
    result["ssm"] = write_rt3d_ssm(output_dir, cfg)
    result["filenames"] = write_rt3d_filenames(output_dir, cfg)
    return result
