"""
swatmf.outputs.modflow_diagnostics
====================================
Diagnostic plots for the standalone MODFLOW model used within a
SWAT-MODFLOW setup.

All functions accept a MODFLOW model workspace directory and optional
shapefiles/arrays, and return a ``matplotlib.figure.Figure`` so callers
can either display or save as they choose.

Public API
----------
plot_flow_direction        — potentiometric surface + GW flow vectors
plot_head_gradient_check   — verify set_head_gradient_to_spring output
plot_head_before_after     — initial vs simulated head + head change
plot_flow_and_head_change  — potentiometric surface + Δhead (matches section 2.8c workflow)
plot_water_table_depth     — depth to water table (top − head)
plot_budget_summary        — volumetric budget time series from .list file
plot_boundary_flux         — identify cells where water enters / leaves
plot_drain_head            — hydraulic head at drain cell(s) over time
plot_drain_flux            — drain discharge at drain cell(s) over time
"""

from __future__ import annotations

import os
import re
import glob
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(mf_folder: str, load_only: list[str] | None = None):
    """Load a flopy Modflow object from *mf_folder*.

    Packages listed in *load_only* that are absent from the model are silently
    dropped so callers don't need to know which optional packages exist.
    """
    import flopy
    wd = str(mf_folder)
    nam_files = glob.glob(os.path.join(wd, "*.nam"))
    if not nam_files:
        raise FileNotFoundError(f"No .nam file found in {wd!r}")

    packages = load_only or ["DIS", "BAS6"]

    # Read the .nam file to find which packages are actually present
    with open(nam_files[0]) as fh:
        nam_text = fh.read().upper()
    available = [p for p in packages if p.upper() in nam_text]

    return flopy.modflow.Modflow.load(
        os.path.basename(nam_files[0]),
        model_ws=wd,
        load_only=available,
        check=False,
    )


def _grid_coords(
    dis,
    mf_folder: str | None = None,
    mfgrid_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (X, Y, cell_size) meshgrids of cell centres.

    Origin is read from (in order of preference):
    1. Explicit *mfgrid_path*   — caller-supplied shapefile path
    2. MFgrid shapefile found inside *mf_folder*
    3. dis.xul / dis.yul        — present in newer flopy models
    4. (0, 0)                   — last resort (pixel-space coordinates)
    """
    nrow, ncol = dis.nrow, dis.ncol
    cs = float(dis.delr.array[0])

    xmin, ymax = None, None

    # Build candidate shapefile list
    candidates: list[str] = []
    if mfgrid_path is not None:
        candidates.append(str(mfgrid_path))
    if mf_folder is not None:
        candidates += (
            glob.glob(os.path.join(mf_folder, "MFgrid*.shp"))
            + glob.glob(os.path.join(mf_folder, "*grid*.shp"))
            + glob.glob(os.path.join(mf_folder, "MFgrid*.gpkg"))
            + glob.glob(os.path.join(mf_folder, "*grid*.gpkg"))
        )

    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            import geopandas as gpd
            _gdf = gpd.read_file(cand)
            xmin = float(_gdf.total_bounds[0])
            ymax = float(_gdf.total_bounds[3])
            break
        except Exception:
            continue

    # dis attributes
    if xmin is None:
        _xul = getattr(dis, "xul", None)
        _yul = getattr(dis, "yul", None)
        if _xul is not None and float(_xul) != 0.0:
            xmin = float(_xul)
            ymax = float(_yul)

    # Fallback
    if xmin is None:
        xmin, ymax = 0.0, float(nrow * cs)

    col_cx = xmin + (np.arange(ncol) + 0.5) * cs
    row_cy = ymax - (np.arange(nrow) + 0.5) * cs
    X, Y = np.meshgrid(col_cx, row_cy)
    return X, Y, cs


def _unit_flow_vectors(
    head: np.ndarray,
    ibound: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cell_size: float,
    stride: int | None = None,
):
    """Return (Xq, Yq, Uq, Vq) unit-length quiver arrays from −∇h."""
    nrow, ncol = head.shape
    st = stride or max(1, min(nrow, ncol) // 20)
    filled = np.where(np.isnan(head), np.nanmean(head), head)
    smooth = uniform_filter(filled, size=5)
    dy, dx = np.gradient(smooth, cell_size)
    u, v   = -dx, dy
    rs = np.arange(0, nrow, st)
    cs = np.arange(0, ncol, st)
    Xq = X[np.ix_(rs, cs)]
    Yq = Y[np.ix_(rs, cs)]
    mq = ibound[np.ix_(rs, cs)] > 0
    Uq = np.where(mq, u[np.ix_(rs, cs)], np.nan)
    Vq = np.where(mq, v[np.ix_(rs, cs)], np.nan)
    mag = np.hypot(Uq, Vq)
    mag = np.where(mag == 0, np.nan, mag)
    return Xq, Yq, Uq / mag, Vq / mag


def _add_overlays(ax, watershed=None, rivers=None, springs=None):
    if watershed is not None:
        watershed.boundary.plot(ax=ax, color="red", linewidth=1.5, zorder=4)
    if rivers is not None:
        rivers.plot(ax=ax, color="cyan", linewidth=1.2, zorder=5)
    if springs is not None:
        springs.plot(ax=ax, color="yellow", markersize=80, zorder=6,
                     edgecolor="black", linewidth=1.5)


def _load_hds(mf_folder: str, hds_path: str | None = None):
    """Return (head_file, times) from the binary HDS file."""
    import flopy.utils.binaryfile as bf
    wd = str(mf_folder)
    if hds_path is None:
        candidates = glob.glob(os.path.join(wd, "*.hds"))
        if not candidates:
            raise FileNotFoundError(f"No .hds file found in {wd!r}")
        hds_path = candidates[0]
    hf = bf.HeadFile(hds_path)
    return hf, hf.get_times()


# ---------------------------------------------------------------------------
# 1. Flow direction
# ---------------------------------------------------------------------------

def plot_flow_direction(
    mf_folder: str | os.PathLike,
    *,
    hds_path: str | os.PathLike | None = None,
    totim: float | None = None,
    watershed=None,
    rivers=None,
    springs=None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 7),
    cmap: str = "terrain",
    stride: int | None = None,
    mfgrid_path: str | os.PathLike | None = None,
    ax: "plt.Axes | None" = None,
) -> plt.Figure:
    """Potentiometric surface contours with GW flow-direction arrows.

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace.
    hds_path : path, optional
        Binary head file.  Searched in *mf_folder* if omitted.
    totim : float, optional
        Simulation time (days) to read.  Defaults to the last saved time.
    watershed, rivers, springs : GeoDataFrame, optional
        Overlay shapefiles plotted on top of the head surface.
    title : str, optional
        Figure title.
    figsize : tuple
        Figure size in inches.
    cmap : str
        Colormap for the head surface.  Default ``"terrain"``.
    stride : int, optional
        Quiver subsampling stride.  Auto-computed if omitted.
    ax : Axes, optional
        Plot into an existing Axes instead of creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mf = _load_model(mf_folder)
    dis, bas = mf.get_package("DIS"), mf.get_package("BAS6")
    X, Y, cs = _grid_coords(dis, str(mf_folder), str(mfgrid_path) if mfgrid_path else None)
    ibound = bas.ibound.array[0]

    hf, times = _load_hds(mf_folder, hds_path)
    t = totim if totim is not None else times[-1]
    head = np.where(ibound > 0, hf.get_data(totim=t)[0], np.nan)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cf = ax.contourf(X, Y, head, levels=25, cmap=cmap, alpha=0.85)
    ax.contour(X, Y, head, levels=15, colors="white", linewidths=0.35, alpha=0.5)

    Xq, Yq, Uq, Vq = _unit_flow_vectors(head, ibound, X, Y, cs, stride)
    ax.quiver(Xq, Yq, Uq, Vq, scale=50, scale_units="inches",
              width=0.003, headwidth=4, color="navy", alpha=0.7)

    _add_overlays(ax, watershed, rivers, springs)
    plt.colorbar(cf, ax=ax, label="Head (m)", shrink=0.8)
    ax.set_title(title or f"GW flow direction  (t = {t:.0f} d)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Head gradient verification
# ---------------------------------------------------------------------------

def plot_head_gradient_check(
    strt: np.ndarray,
    mf_folder: str | os.PathLike,
    *,
    outlet_rc: tuple[int, int] | None = None,
    springs=None,
    mfgrid_path: str | os.PathLike | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Two-panel verification of a prescribed initial head gradient.

    Left  — potentiometric surface + unit flow vectors.
    Right — gradient magnitude |∇h| (m/m); should be smooth with no
            discontinuities.

    Parameters
    ----------
    strt : ndarray
        2-D initial head array (nrow × ncol).
    mf_folder : path
        MODFLOW model workspace (needed for DIS/BAS6).
    outlet_rc : (row, col) tuple, optional
        0-based outlet cell indices to mark with a cross.
    springs : GeoDataFrame, optional
        Spring point(s) to overlay.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    wd = str(mf_folder)
    mf = _load_model(wd)
    dis, bas = mf.get_package("DIS"), mf.get_package("BAS6")
    X, Y, cs = _grid_coords(dis, wd, str(mfgrid_path) if mfgrid_path else None)
    ibound = bas.ibound.array[0]

    active = np.where(ibound > 0, strt, np.nan)

    filled = np.where(np.isnan(active), np.nanmean(active), active)
    smooth = uniform_filter(filled, size=5)
    dy, dx = np.gradient(smooth, cs)
    grad_mag = np.where(ibound > 0, np.hypot(dx, dy), np.nan)
    u, v = -dx, dy

    Xq, Yq, Uq, Vq = _unit_flow_vectors(active, ibound, X, Y, cs)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: potentiometric surface
    ax = axes[0]
    cf = ax.contourf(X, Y, active, levels=25, cmap="Blues_r", alpha=0.85)
    ax.contour(X, Y, active, levels=15, colors="navy", linewidths=0.35, alpha=0.55)
    ax.quiver(Xq, Yq, Uq, Vq, scale=50, scale_units="inches",
              width=0.003, headwidth=4, color="white", alpha=0.85)
    if outlet_rc is not None:
        col_cx = X[0, :]
        row_cy = Y[:, 0]
        ax.scatter([col_cx[outlet_rc[1]]], [row_cy[outlet_rc[0]]],
                   s=150, c="yellow", edgecolors="black",
                   linewidths=1.5, zorder=7, label="Outlet")
        ax.legend(fontsize=8)
    _add_overlays(ax, springs=springs)
    plt.colorbar(cf, ax=ax, label="Initial head (m)", shrink=0.8)
    ax.set_title("Prescribed initial head\nArrows = −∇h (should all → outlet)",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Right: gradient magnitude
    ax2 = axes[1]
    vmax_g = np.nanpercentile(grad_mag, 98)
    cf2 = ax2.contourf(X, Y, grad_mag, levels=25, cmap="YlOrRd",
                       vmin=0, vmax=vmax_g)
    ax2.quiver(Xq, Yq, Uq, Vq, scale=50, scale_units="inches",
               width=0.003, headwidth=4, color="black", alpha=0.45)
    if outlet_rc is not None:
        ax2.scatter([col_cx[outlet_rc[1]]], [row_cy[outlet_rc[0]]],
                    s=150, c="yellow", edgecolors="black",
                    linewidths=1.5, zorder=7)
    _add_overlays(ax2, springs=springs)
    plt.colorbar(cf2, ax=ax2, label="|∇h|  (m/m)", shrink=0.8)
    ax2.set_title("Gradient magnitude\nShould be smooth — spikes indicate pits",
                  fontsize=10)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Easting (m)")

    fig.suptitle("Initial head gradient verification", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Before / after head comparison
# ---------------------------------------------------------------------------

def plot_head_before_after(
    mf_folder: str | os.PathLike,
    strt: np.ndarray,
    *,
    hds_path: str | os.PathLike | None = None,
    totim: float | None = None,
    watershed=None,
    rivers=None,
    springs=None,
    mfgrid_path: str | os.PathLike | None = None,
    figsize: tuple[float, float] = (21, 6),
) -> plt.Figure:
    """Three-panel comparison: initial head, final head, head change.

    Panels share a common colour scale on the head panels so any shift in
    the potentiometric surface is immediately visible.  The third panel
    (Δhead = final − initial, RdBu_r) shows how far MODFLOW moved the heads;
    a near-zero map means the initial conditions were close to equilibrium.

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace.
    strt : ndarray
        2-D initial head array (nrow × ncol) — e.g. from
        ``set_head_gradient_to_spring``.
    hds_path : path, optional
        Binary head output file.  Searched in *mf_folder* if omitted.
    totim : float, optional
        Time step to read from HDS.  Defaults to the last saved time.
    watershed, rivers, springs : GeoDataFrame, optional
        Overlay shapefiles.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    mf = _load_model(mf_folder)
    dis, bas = mf.get_package("DIS"), mf.get_package("BAS6")
    X, Y, cs = _grid_coords(dis, str(mf_folder), str(mfgrid_path) if mfgrid_path else None)
    ibound = bas.ibound.array[0]

    hf, times = _load_hds(mf_folder, hds_path)
    t = totim if totim is not None else times[-1]
    hfinal_raw = hf.get_data(totim=t)[0]

    hinit  = np.where(ibound > 0, strt,       np.nan)
    hfinal = np.where(ibound > 0, hfinal_raw, np.nan)
    hdelta = hfinal - hinit

    vmin = np.nanmin([np.nanmin(hinit), np.nanmin(hfinal)])
    vmax = np.nanmax([np.nanmax(hinit), np.nanmax(hfinal)])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, head, label in [
        (axes[0], hinit,  "Initial head (strt)"),
        (axes[1], hfinal, f"Final simulated head  (t = {t:.0f} d)"),
    ]:
        cf = ax.contourf(X, Y, head, levels=25, cmap="Blues_r",
                         vmin=vmin, vmax=vmax, alpha=0.85)
        ax.contour(X, Y, head, levels=15, colors="navy",
                   linewidths=0.35, alpha=0.5)
        Xq, Yq, Uq, Vq = _unit_flow_vectors(head, ibound, X, Y, cs)
        ax.quiver(Xq, Yq, Uq, Vq, scale=50, scale_units="inches",
                  width=0.003, headwidth=4, color="white", alpha=0.85)
        _add_overlays(ax, watershed, rivers, springs)
        plt.colorbar(cf, ax=ax, label="Head (m)", shrink=0.8)
        ax.set_title(label + "\nArrows = GW flow direction", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")

    lim = np.nanpercentile(np.abs(hdelta[np.isfinite(hdelta)]), 98)
    cf2 = axes[2].contourf(X, Y, hdelta, levels=25, cmap="RdBu_r",
                           vmin=-lim, vmax=lim)
    _add_overlays(axes[2], watershed, rivers, springs)
    plt.colorbar(cf2, ax=axes[2], label="Δhead (m)", shrink=0.8)
    axes[2].set_title("Head change  (final − initial)\n"
                      "Near-zero = close to equilibrium", fontsize=10)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("Easting (m)")

    fig.suptitle("MODFLOW head: initial vs simulated", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Flow direction + head change (workflow section 2.8c equivalent)
# ---------------------------------------------------------------------------

def plot_flow_and_head_change(
    mf_folder: str | os.PathLike,
    *,
    hds_path: str | os.PathLike | None = None,
    totim: float | None = None,
    watershed=None,
    rivers=None,
    springs=None,
    suptitle: str | None = None,
    mfgrid_path: str | os.PathLike | None = None,
    figsize: tuple[float, float] = (18, 7),
    save_path: str | os.PathLike | None = None,
    print_stats: bool = True,
) -> "plt.Figure":
    """Two-panel QA plot: potentiometric surface + head change from initial.

    Left panel — filled contour head map with labelled contour lines and
    unit-length flow-direction arrows (``−∇h``).

    Right panel — head change (final − initial) using a diverging
    ``RdBu_r`` colourmap centred on zero, so rises and falls are
    immediately distinguishable.

    Optionally prints a brief head-change summary to help decide whether
    the initial heads are close to steady state.

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace.
    hds_path : path, optional
        Binary head file.  Searched in *mf_folder* if omitted.
    totim : float, optional
        Simulation time (days) to read.  Defaults to the last saved time.
    watershed, rivers, springs : GeoDataFrame, optional
        Overlay shapefiles.
    suptitle : str, optional
        Overall figure title.  Defaults to the folder name.
    mfgrid_path : path, optional
        MFgrid shapefile used to derive real-world coordinates.
    figsize : tuple
    save_path : path, optional
        If supplied, the figure is saved here (dpi 150).
    print_stats : bool
        Print head-change summary statistics.  Default True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.colors as mcolors

    mf = _load_model(mf_folder, load_only=["DIS", "BAS6"])
    dis, bas = mf.get_package("DIS"), mf.get_package("BAS6")
    X, Y, cs = _grid_coords(dis, str(mf_folder), str(mfgrid_path) if mfgrid_path else None)
    ibound = bas.ibound.array[0]
    strt   = bas.strt.array[0]

    hf, times = _load_hds(mf_folder, hds_path)
    t = totim if totim is not None else times[-1]
    head_raw = hf.get_data(totim=t)[0]

    head  = np.where(ibound > 0, head_raw, np.nan)
    strt_ma = np.where(ibound > 0, strt,    np.nan)
    dh    = head - strt_ma

    Xq, Yq, Uq, Vq = _unit_flow_vectors(head, ibound, X, Y, cs)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: potentiometric surface + flow vectors ───────────────────────
    ax = axes[0]
    cf = ax.contourf(X, Y, head, levels=30, cmap="terrain", alpha=0.85)
    cl = ax.contour( X, Y, head, levels=20, colors="white",
                     linewidths=0.4, alpha=0.5)
    ax.clabel(cl, inline=True, fontsize=6, fmt="%.0f m", colors="white")
    plt.colorbar(cf, ax=ax, label="Groundwater head (m)", shrink=0.75)

    ax.quiver(Xq, Yq, Uq, Vq,
              scale=30, scale_units="width",
              width=0.003, headwidth=3, headlength=4,
              color="navy", alpha=0.7, label="GW flow direction")

    _add_overlays(ax, watershed, rivers, springs)
    ax.set_title(f"Potentiometric Surface & GW Flow Direction\n(t = {t:.0f} d)",
                 fontsize=11)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    # ── Right: head change (final − initial) ─────────────────────────────
    ax2 = axes[1]
    dh_abs = np.nanpercentile(np.abs(dh[np.isfinite(dh)]), 95)
    norm   = mcolors.TwoSlopeNorm(vmin=-dh_abs, vcenter=0, vmax=dh_abs)
    cf2 = ax2.contourf(X, Y, dh, levels=30, cmap="RdBu_r", norm=norm, alpha=0.85)
    plt.colorbar(cf2, ax=ax2,
                 label="Head change from initial (m)\n+ = rise,  − = fall",
                 shrink=0.75)

    _add_overlays(ax2, watershed, rivers, springs)
    ax2.set_title("Head Change: Final − Initial\n"
                  "(near-zero = initial heads close to steady state)",
                  fontsize=11)
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    ax2.set_aspect("equal")

    name = suptitle or os.path.basename(os.path.abspath(str(mf_folder)))
    fig.suptitle(f"{name} — MODFLOW standalone QA", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if print_stats:
        dh_vals = dh[np.isfinite(dh)]
        print(f"\nHead change summary (final − initial):")
        print(f"  Mean : {dh_vals.mean():+.2f} m")
        print(f"  Std  : {dh_vals.std():.2f} m")
        print(f"  Range: {dh_vals.min():+.2f} to {dh_vals.max():+.2f} m")
        if abs(dh_vals.mean()) < 1.0 and dh_vals.std() < 5.0:
            print("  ✔ Initial heads appear close to steady state.")
        else:
            print("  ⚠ Large head changes detected — consider:")
            print("    1. Supplying a hydraulic head raster via initial_head= in Section 2.2")
            print("    2. Using water_table_depth= with a depth-to-WT raster")
            print("    3. Running a steady-state stress period first to equilibrate")

    return fig


# ---------------------------------------------------------------------------
# 5. Water table depth
# ---------------------------------------------------------------------------

def plot_water_table_depth(
    mf_folder: str | os.PathLike,
    *,
    hds_path: str | os.PathLike | None = None,
    totim: float | None = None,
    watershed=None,
    rivers=None,
    springs=None,
    max_depth: float = 10.0,
    mfgrid_path: str | os.PathLike | None = None,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Figure:
    """Map of depth-to-water-table (land surface − simulated head).

    Shallow cells (near-zero depth) indicate the water table is close to the
    surface — likely gaining reaches or spring zones.  Cells where head > top
    are artesian and shown in a distinct colour.

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace.
    hds_path : path, optional
        Binary head file.
    totim : float, optional
        Time to read; defaults to last saved time.
    watershed, rivers, springs : GeoDataFrame, optional
        Overlay shapefiles.
    max_depth : float
        Colour-scale maximum depth (m).  Cells deeper than this are all the
        same colour.  Default 10 m.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    mf = _load_model(mf_folder)
    dis, bas = mf.get_package("DIS"), mf.get_package("BAS6")
    X, Y, cs = _grid_coords(dis, str(mf_folder), str(mfgrid_path) if mfgrid_path else None)
    ibound = bas.ibound.array[0]
    top = dis.top.array

    hf, times = _load_hds(mf_folder, hds_path)
    t = totim if totim is not None else times[-1]
    head = hf.get_data(totim=t)[0]

    depth = np.where(ibound > 0, top - head, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(X, Y, depth, levels=25, cmap="RdYlGn_r",
                     vmin=-1.0, vmax=max_depth)
    # Mark artesian cells (head > top → depth < 0) with a hatch
    artesian = np.where((ibound > 0) & (depth < 0), 1.0, np.nan)
    if np.any(np.isfinite(artesian)):
        ax.contourf(X, Y, artesian, levels=[0.5, 1.5],
                    hatches=["///"], alpha=0, colors="none")
        ax.contour(X, Y, artesian, levels=[0.5], colors="blue",
                   linewidths=0.8, linestyles="--", alpha=0.7)

    _add_overlays(ax, watershed, rivers, springs)
    cbar = plt.colorbar(cf, ax=ax, label="Depth to water table (m)", shrink=0.8)
    cbar.ax.axhline(0, color="blue", linewidth=1.5, linestyle="--",
                    label="Artesian (head > surface)")
    ax.set_title(f"Depth to water table  (t = {t:.0f} d)\n"
                 "Dashed blue = artesian / above-surface head", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Budget summary
# ---------------------------------------------------------------------------


def plot_budget_summary(
    mf_folder: str | os.PathLike,
    *,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Parse the MODFLOW list file and plot the volumetric budget components.

    Reads every ``VOLUMETRIC BUDGET`` block from the ``.list`` file and
    extracts IN / OUT totals for each package (STORAGE, RECHARGE, DRAINS,
    WELLS, etc.).  Produces a stacked bar chart comparing cumulative IN vs
    OUT terms at each saved time step.

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace (contains the ``.list`` file).
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    wd = str(mf_folder)
    list_files = glob.glob(os.path.join(wd, "*.list"))
    if not list_files:
        raise FileNotFoundError(f"No .list file found in {wd!r}")

    with open(list_files[0]) as fh:
        text = fh.read()

    # Find every budget block
    blocks = re.split(r"VOLUMETRIC BUDGET FOR ENTIRE MODEL", text)[1:]
    if not blocks:
        raise ValueError("No VOLUMETRIC BUDGET blocks found in list file.")

    times_out: list[float] = []
    in_records:  list[dict] = []
    out_records: list[dict] = []

    time_re  = re.compile(r"TIME STEP\s+\d+,\s+STRESS PERIOD\s+\d+.*?SECONDS\s+=\s+([\d.E+\-]+)", re.S)
    entry_re = re.compile(r"([A-Z][A-Z0-9 _\-]+?)\s*=\s*([\d.E+\-]+)", re.I)

    for block in blocks:
        m = time_re.search(block)
        times_out.append(float(m.group(1)) / 86400 if m else np.nan)  # seconds → days

        in_part  = re.split(r"TOTAL IN\b|OUT:", block, flags=re.I)
        out_part = re.split(r"TOTAL OUT\b|DISCREPANCY", block, flags=re.I)

        in_dict: dict[str, float] = {}
        if len(in_part) > 1:
            for name, val in entry_re.findall(in_part[0].split("IN:")[-1]):
                name = name.strip()
                if name not in ("TOTAL", "PERCENT"):
                    in_dict[name] = float(val)

        out_dict: dict[str, float] = {}
        if len(out_part) > 1:
            raw = out_part[0].split("OUT:")[-1] if "OUT:" in out_part[0] else out_part[0]
            for name, val in entry_re.findall(raw):
                name = name.strip()
                if name not in ("TOTAL", "PERCENT"):
                    out_dict[name] = float(val)

        in_records.append(in_dict)
        out_records.append(out_dict)

    if not any(in_records):
        raise ValueError("Could not parse any budget entries from list file.")

    import pandas as pd
    df_in  = pd.DataFrame(in_records,  index=times_out).fillna(0)
    df_out = pd.DataFrame(out_records, index=times_out).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    colors = plt.cm.tab10.colors

    for ax, df, direction in [(axes[0], df_in, "IN"), (axes[1], df_out, "OUT")]:
        bottom = np.zeros(len(df))
        for i, col in enumerate(df.columns):
            ax.bar(range(len(df)), df[col].values, bottom=bottom,
                   label=col, color=colors[i % len(colors)], alpha=0.85)
            bottom += df[col].values
        ax.set_title(f"Budget {direction} terms", fontsize=10)
        ax.set_xlabel("Budget step")
        ax.set_ylabel("Volume (m³)")
        ax.legend(fontsize=7, loc="upper left")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle("MODFLOW volumetric budget", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Boundary flux map
# ---------------------------------------------------------------------------

def plot_boundary_flux(
    mf_folder: str | os.PathLike,
    *,
    hds_path: str | os.PathLike | None = None,
    totim: float | None = None,
    watershed=None,
    rivers=None,
    springs=None,
    mfgrid_path: str | os.PathLike | None = None,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Figure:
    """Map showing likely groundwater inflow and outflow zones.

    At the model boundary, the sign of the head gradient perpendicular to
    each edge indicates whether water is flowing in (inflow) or out (outflow)
    of the domain.  Interior no-flow boundaries and active cells near the
    perimeter are identified from the ibound array.

    The plot shows:
    - Background head surface (contours)
    - Boundary cells coloured by inward (blue) vs outward (red) gradient
    - DRN / CHD cells highlighted if present

    Parameters
    ----------
    mf_folder : path
    hds_path : path, optional
    totim : float, optional
    watershed, rivers, springs : GeoDataFrame, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    mf = _load_model(mf_folder, load_only=["DIS", "BAS6", "DRN", "CHD"])
    dis, bas = mf.get_package("DIS"), mf.get_package("BAS6")
    X, Y, cs = _grid_coords(dis, str(mf_folder), str(mfgrid_path) if mfgrid_path else None)
    ibound = bas.ibound.array[0]
    nrow, ncol = dis.nrow, dis.ncol

    hf, times = _load_hds(mf_folder, hds_path)
    t = totim if totim is not None else times[-1]
    head = np.where(ibound > 0, hf.get_data(totim=t)[0], np.nan)

    # Smooth head for gradient calculation
    filled = np.where(np.isnan(head), np.nanmean(head), head)
    smooth = uniform_filter(filled, size=3)
    dy, dx = np.gradient(smooth, cs)

    # Identify perimeter active cells and their outward-normal gradient
    # Outward normal: top row → −dy, bottom row → +dy, left → −dx, right → +dx
    outward_grad = np.full((nrow, ncol), np.nan)

    def _is_boundary(r, c):
        return (
            r == 0 or r == nrow - 1 or c == 0 or c == ncol - 1
            or ibound[r - 1, c] == 0 or ibound[r + 1, c] == 0
            or ibound[r, c - 1] == 0 or ibound[r, c + 1] == 0
        )

    for r in range(nrow):
        for c in range(ncol):
            if ibound[r, c] <= 0:
                continue
            if not _is_boundary(r, c):
                continue
            normals = []
            if r == 0 or (r > 0 and ibound[r - 1, c] == 0):
                normals.append(-dy[r, c])    # outward = -y (northward)
            if r == nrow - 1 or (r < nrow - 1 and ibound[r + 1, c] == 0):
                normals.append(dy[r, c])     # outward = +y (southward)
            if c == 0 or (c > 0 and ibound[r, c - 1] == 0):
                normals.append(-dx[r, c])    # outward = -x (westward)
            if c == ncol - 1 or (c < ncol - 1 and ibound[r, c + 1] == 0):
                normals.append(dx[r, c])     # outward = +x (eastward)
            if normals:
                outward_grad[r, c] = np.mean(normals)

    fig, ax = plt.subplots(figsize=figsize)

    # Background head
    cf = ax.contourf(X, Y, head, levels=20, cmap="Greys", alpha=0.4)
    ax.contour(X, Y, head, levels=12, colors="grey", linewidths=0.3, alpha=0.5)

    # Boundary cells: red = outward gradient (losing / outflow)
    #                 blue = inward gradient (gaining / inflow)
    out_mask = outward_grad > 0
    in_mask  = outward_grad < 0

    ax.scatter(X[out_mask], Y[out_mask], c="red",  s=18, zorder=5,
               label="Outflow boundary (∇h outward)")
    ax.scatter(X[in_mask],  Y[in_mask],  c="blue", s=18, zorder=5,
               label="Inflow boundary (∇h inward)")

    # Highlight DRN cells if present
    drn = mf.get_package("DRN")
    if drn is not None:
        drn_spd = drn.stress_period_data.data.get(0, [])
        if len(drn_spd):
            drn_r = [rec[0] for rec in drn_spd]
            drn_c = [rec[1] for rec in drn_spd]
            ax.scatter(X[drn_r, drn_c], Y[drn_r, drn_c],
                       c="green", s=40, marker="^", zorder=6,
                       label="DRN cells")

    _add_overlays(ax, watershed, rivers, springs)
    plt.colorbar(cf, ax=ax, label="Head (m)", shrink=0.8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Boundary flux direction  (t = {t:.0f} d)\n"
                 "Red = GW leaving domain  |  Blue = GW entering domain",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Drain cell head time series
# ---------------------------------------------------------------------------

def plot_drain_head(
    mf_folder: str | os.PathLike,
    drain_cells: "Sequence[tuple[int, int]] | tuple[int, int]",
    *,
    hds_path: str | os.PathLike | None = None,
    layer: int = 1,
    drain_elev: "float | Sequence[float] | None" = None,
    observed=None,
    observed_label: str = "Observed",
    labels: "Sequence[str] | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> "plt.Figure":
    """Hydraulic head at one or more drain cells over the full simulation.

    The drain elevation is drawn as a dashed reference line so you can see
    when the head drops below it and the drain switches off.

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace.
    drain_cells : (row, col) or list of (row, col)
        1-based row/column indices of the drain cell(s) to inspect.
    hds_path : path, optional
        Binary head file; searched in *mf_folder* if omitted.
    layer : int
        Model layer to read (1-based).  Default 1.
    drain_elev : float or list of float, optional
        Drain elevation(s) (m) for the reference line.  Read from the
        DRN package automatically when *None*.
    observed : pd.Series, optional
        Observed head / spring stage for comparison.  Index should be
        numeric (days) or datetime.
    observed_label : str
        Legend label for the observed series.
    labels : list of str, optional
        One label per drain cell.  Defaults to ``"Cell (r, c)"``.
    title : str, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    wd = str(mf_folder)
    # Normalise to list of (row, col)
    if isinstance(drain_cells, tuple) and len(drain_cells) == 2 and isinstance(drain_cells[0], int):
        drain_cells = [drain_cells]
    drain_cells = list(drain_cells)

    hf, times = _load_hds(mf_folder, hds_path)
    lyr = layer - 1  # 0-based

    # Fetch drain elevations from DRN package if not supplied
    elev_by_cell: dict[tuple, float] = {}
    if drain_elev is None:
        mf = _load_model(wd, load_only=["DIS", "DRN"])
        drn = mf.get_package("DRN")
        if drn is not None:
            spd = drn.stress_period_data.data.get(0, [])
            for rec in spd:
                # rec fields: layer(0-based), row(0-based), col(0-based), elev, cond
                key = (int(rec[1]) + 1, int(rec[2]) + 1)
                elev_by_cell[key] = float(rec[3])
    elif isinstance(drain_elev, (int, float)):
        for cell in drain_cells:
            elev_by_cell[tuple(cell)] = float(drain_elev)
    else:
        for cell, ev in zip(drain_cells, drain_elev):
            elev_by_cell[tuple(cell)] = float(ev)

    # Extract head at each cell for every time step
    series: dict[tuple, list[float]] = {tuple(c): [] for c in drain_cells}
    for t in times:
        data = hf.get_data(totim=t)[lyr]
        for cell in drain_cells:
            r, c = cell[0] - 1, cell[1] - 1
            series[tuple(cell)].append(float(data[r, c]))

    days = np.asarray(times)
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors

    for i, cell in enumerate(drain_cells):
        key = tuple(cell)
        lbl = labels[i] if labels and i < len(labels) else f"Cell ({cell[0]}, {cell[1]})"
        ax.plot(days, np.asarray(series[key]), color=colors[i % len(colors)],
                linewidth=1.8, label=lbl)
        if key in elev_by_cell:
            ax.axhline(elev_by_cell[key], color=colors[i % len(colors)],
                       linestyle="--", linewidth=1.2, alpha=0.65,
                       label=f"Drain elev ({cell[0]},{cell[1]}): {elev_by_cell[key]:.2f} m")

    if observed is not None:
        import pandas as pd
        obs_idx = observed.index
        if hasattr(obs_idx, "astype") and str(obs_idx.dtype).startswith("datetime"):
            obs_x = obs_idx.astype(np.int64) / 1e9 / 86400
        else:
            obs_x = np.asarray(obs_idx, dtype=float)
        ax.scatter(obs_x, observed.values, color="black", s=25, zorder=5,
                   label=observed_label)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Hydraulic head (m)")
    ax.set_title(title or "Hydraulic head at drain cell(s)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Drain cell flux time series
# ---------------------------------------------------------------------------

def plot_drain_flux(
    mf_folder: str | os.PathLike,
    drain_cells: "Sequence[tuple[int, int]] | tuple[int, int]",
    *,
    cbc_path: str | os.PathLike | None = None,
    layer: int = 1,
    observed=None,
    observed_label: str = "Observed (m³/d)",
    labels: "Sequence[str] | None" = None,
    title: str | None = None,
    sign: float = -1.0,
    figsize: tuple[float, float] = (10, 5),
) -> "plt.Figure":
    """Drain discharge at one or more drain cells over the full simulation.

    Drain flux in the CBC file is negative (water leaving the aquifer).
    By default ``sign=-1`` flips the values to positive discharge (m³/d).

    Parameters
    ----------
    mf_folder : path
        MODFLOW model workspace.
    drain_cells : (row, col) or list of (row, col)
        1-based row/column indices of the drain cell(s).
    cbc_path : path, optional
        Cell-by-cell budget file (``.cbc`` / ``.bud`` / ``.cbb``).
        Searched in *mf_folder* if omitted.  The OC package must have
        ``COMPACT BUDGET`` enabled to produce this file.
    layer : int
        Model layer (1-based).  Default 1.
    observed : pd.Series, optional
        Observed spring flow / discharge for comparison.
    observed_label : str
        Legend label for the observed series.
    labels : list of str, optional
        One label per drain cell.
    title : str, optional
    sign : float
        Multiplier applied to raw CBC values.  ``-1`` (default) converts
        MODFLOW's negative drain flux to positive discharge.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    import flopy.utils.binaryfile as bf

    wd = str(mf_folder)
    if isinstance(drain_cells, tuple) and len(drain_cells) == 2 and isinstance(drain_cells[0], int):
        drain_cells = [drain_cells]
    drain_cells = list(drain_cells)

    # Locate CBC file
    if cbc_path is None:
        candidates = (
            glob.glob(os.path.join(wd, "*.cbc"))
            + glob.glob(os.path.join(wd, "*.bud"))
            + glob.glob(os.path.join(wd, "*.cbb"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No cell-by-cell budget file (.cbc/.bud/.cbb) found in {wd!r}. "
                "Ensure COMPACT BUDGET is set in the OC package and re-run MODFLOW."
            )
        cbc_path = candidates[0]

    cbf = bf.CellBudgetFile(str(cbc_path))
    times = cbf.get_times()
    lyr   = layer - 1

    record_names = [
        t.decode().strip() if isinstance(t, bytes) else t.strip()
        for t in cbf.get_unique_record_names()
    ]
    drn_label = next((n for n in record_names if "DRAIN" in n.upper()), None)
    if drn_label is None:
        raise ValueError(
            f"No DRAIN record found in {cbc_path!r}. "
            f"Available records: {record_names}"
        )

    # Load DIS once for potential node-index conversion
    _mf  = _load_model(wd)
    _dis = _mf.get_package("DIS")
    nrow, ncol = _dis.nrow, _dis.ncol

    series: dict[tuple, list[float]] = {tuple(c): [] for c in drain_cells}
    for t in times:
        try:
            data_list = cbf.get_data(text=drn_label, totim=t)
        except Exception:
            for cell in drain_cells:
                series[tuple(cell)].append(np.nan)
            continue

        if not data_list:
            for cell in drain_cells:
                series[tuple(cell)].append(np.nan)
            continue

        data = data_list[0]
        if data.ndim == 3:
            # Full 3-D array (nlay × nrow × ncol)
            for cell in drain_cells:
                r, c = cell[0] - 1, cell[1] - 1
                series[tuple(cell)].append(float(data[lyr, r, c]))
        else:
            # Record array — COMPACT BUDGET format
            flux_by_node: dict[int, float] = {}
            for rec in data:
                node = int(rec[0]) - 1   # 0-based node number
                flux_by_node[node] = float(rec[-1])
            for cell in drain_cells:
                r, c  = cell[0] - 1, cell[1] - 1
                node  = lyr * nrow * ncol + r * ncol + c
                series[tuple(cell)].append(flux_by_node.get(node, 0.0))

    days = np.asarray(times)
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors

    for i, cell in enumerate(drain_cells):
        key = tuple(cell)
        lbl = labels[i] if labels and i < len(labels) else f"Cell ({cell[0]}, {cell[1]})"
        flux_ts = sign * np.asarray(series[key])
        ax.plot(days, flux_ts, color=colors[i % len(colors)], linewidth=1.8, label=lbl)

    if observed is not None:
        import pandas as pd
        obs_idx = observed.index
        if hasattr(obs_idx, "astype") and str(obs_idx.dtype).startswith("datetime"):
            obs_x = obs_idx.astype(np.int64) / 1e9 / 86400
        else:
            obs_x = np.asarray(obs_idx, dtype=float)
        ax.scatter(obs_x, observed.values, color="black", s=25, zorder=5,
                   label=observed_label)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Drain discharge (m³/d)")
    ax.set_title(title or "Drain cell discharge", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
