"""
swatmf.preprocessing.linking
==============================
Pure-Python / GeoPandas functions that replicate the **Linking Process**
panel of the QSWATMOD2 QGIS plugin.

The linking process spatially joins SWAT HRU polygons with the MODFLOW grid
and writes two tab-delimited ASCII tables that the SWAT-MODFLOW executable
reads at run-time:

``hru_dhru``
    Maps each disaggregated HRU (dHRU) to its parent HRU and subbasin.
``dhru_grid``
    Maps each dHRU fragment to the MODFLOW grid cells it overlaps.

These files were previously only generated inside QGIS.  This module
reproduces the same pipeline using :mod:`geopandas` so that the full
pre-processing workflow can run in a standard Python environment.

Functions exposed
-----------------
create_dhru              — Explode multipart HRU polygons → singlepart dHRUs.
build_hru_dhru           — Intersect dHRUs × subbasins → hru_dhru GeoDataFrame.
export_hru_dhru          — Write ``hru_dhru`` table file from a GeoDataFrame.
build_dhru_grid          — Intersect dHRUs × MODFLOW grid → dhru_grid GeoDataFrame.
export_dhru_grid         — Write ``dhru_grid`` table file from a GeoDataFrame.
generate_link_tables     — Full pipeline: HRU + sub + mf_grid → both table files.

Notes
-----
*All area values are in square metres* (the CRS of the input layers).  The
SWAT-MODFLOW executable expects integer values, so areas are rounded before
they are written.

The ``area_filter_m2`` thresholds (default 9 m² for hru_dhru and 30 m² for
dhru_grid) match the sliver-removal thresholds used in the QGIS plugin.
"""

from __future__ import annotations

import csv
import os
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_crs_match(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject *gdf2* to match *gdf1* if their CRS differ."""
    if gdf1.crs is None or gdf2.crs is None:
        return gdf2
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    return gdf2


def _fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with any invalid geometries repaired."""
    return gdf.copy().assign(geometry=gdf.geometry.buffer(0))


# ---------------------------------------------------------------------------
# Step 1 — Disaggregate HRUs
# ---------------------------------------------------------------------------

def create_dhru(
    hru_gdf: gpd.GeoDataFrame,
    hru_id_col: str = "HRU_ID",
    hrugis_col: str = "HRUGIS",
) -> gpd.GeoDataFrame:
    """Explode multipart HRU polygons to singlepart disaggregated HRUs (dHRUs).

    Mirrors the ``multipart_to_singlepart → create_temp_id → create_dhru_id
    → calculate_area(dhru_area)`` sequence from the QGIS plugin.

    Parameters
    ----------
    hru_gdf : GeoDataFrame
        SWAT HRU polygons.  Must contain *hru_id_col* (typically ``HRU_ID``).
        If the column is absent it is created by sorting on *hrugis_col*
        (``HRUGIS``).  The field ``hru_area`` is added if missing.
    hru_id_col : str, optional
        Column name for the integer HRU identifier.  Default ``"HRU_ID"``.
    hrugis_col : str, optional
        Column used to derive a stable sort order when *hru_id_col* must be
        created.  Default ``"HRUGIS"``.

    Returns
    -------
    GeoDataFrame
        Singlepart polygons with columns:

        * ``HRU_ID`` — parent HRU identifier (1-based integer)
        * ``hru_area`` — area of the *original* (multipart) HRU polygon [m²]
        * ``dhru_id`` — sequential dHRU identifier (1-based, sorted by HRU_ID)
        * ``dhru_area`` — area of this singlepart polygon [m²]

    Examples
    --------
    >>> import geopandas as gpd
    >>> hru = gpd.read_file("GIS/SMshps/hru_link.gpkg")
    >>> dhru = create_dhru(hru)
    >>> dhru[["HRU_ID", "hru_area", "dhru_id", "dhru_area"]].head()
    """
    gdf = _fix_geometries(hru_gdf.copy())

    # --- ensure HRU_ID -------------------------------------------------------
    if hru_id_col not in gdf.columns:
        if hrugis_col in gdf.columns:
            gdf = gdf.sort_values(hrugis_col).reset_index(drop=True)
            gdf["HRU_ID"] = range(1, len(gdf) + 1)
        else:
            gdf = gdf.reset_index(drop=True)
            gdf["HRU_ID"] = range(1, len(gdf) + 1)
    else:
        gdf = gdf.rename(columns={hru_id_col: "HRU_ID"})

    # --- hru_area (original polygon area) ------------------------------------
    if "hru_area" not in gdf.columns:
        gdf["hru_area"] = gdf.geometry.area

    # --- multipart → singlepart (dhru) ---------------------------------------
    dhru = gdf.explode(index_parts=False).reset_index(drop=True)

    # --- dhru_id: sequential, sorted by HRU_ID ------------------------------
    dhru = dhru.sort_values("HRU_ID").reset_index(drop=True)
    dhru["dhru_id"] = range(1, len(dhru) + 1)

    # --- dhru_area (area of each singlepart polygon) -------------------------
    dhru["dhru_area"] = dhru.geometry.area

    return dhru[["HRU_ID", "hru_area", "dhru_id", "dhru_area", "geometry"]]


# ---------------------------------------------------------------------------
# Step 2 — hru_dhru: intersect dHRUs × subbasins
# ---------------------------------------------------------------------------

def build_hru_dhru(
    dhru_gdf: gpd.GeoDataFrame,
    sub_gdf: gpd.GeoDataFrame,
    subbasin_col: str = "Subbasin",
    area_filter_m2: float = 9.0,
) -> gpd.GeoDataFrame:
    """Intersect dHRU polygons with SWAT subbasin polygons.

    Replicates ``hru_dhru → create_hru_dhru_filter`` from the QGIS plugin.

    Parameters
    ----------
    dhru_gdf : GeoDataFrame
        Disaggregated HRU polygons from :func:`create_dhru`.
    sub_gdf : GeoDataFrame
        SWAT subbasin polygons.  Must contain *subbasin_col* (``Subbasin``).
    subbasin_col : str, optional
        Subbasin identifier column in *sub_gdf*.  Default ``"Subbasin"``.
    area_filter_m2 : float, optional
        Sliver threshold [m²].  Intersection fragments smaller than this
        value are dropped (plugin default: 9 m²).

    Returns
    -------
    GeoDataFrame
        Intersection polygons with columns:
        ``dhru_id``, ``area_f``, ``HRU_ID``, ``Subbasin``, ``hru_area``.

    Examples
    --------
    >>> sub = gpd.read_file("GIS/SMshps/sub_link.gpkg")
    >>> hd  = build_hru_dhru(dhru, sub)
    """
    sub = _ensure_crs_match(dhru_gdf, sub_gdf.copy())
    sub = _fix_geometries(sub)

    # Rename subbasin column to canonical name
    if subbasin_col != "Subbasin" and subbasin_col in sub.columns:
        sub = sub.rename(columns={subbasin_col: "Subbasin"})

    # Spatial intersection (dhru × sub)
    intersected = gpd.overlay(
        dhru_gdf[["HRU_ID", "hru_area", "dhru_id", "dhru_area", "geometry"]],
        sub[["Subbasin", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    if intersected.empty:
        raise ValueError(
            "hru_dhru intersection returned no features. Check that the HRU "
            "and subbasin shapefiles share the same CRS and spatial extent."
        )

    # Dissolve at dhru_id × Subbasin level so each row is unique, then
    # compute area_f from the dissolved geometry in a single pass.
    hru_dhru = (
        intersected
        .dissolve(by=["dhru_id", "Subbasin"], aggfunc="first")
        .reset_index()
    )
    hru_dhru["area_f"] = hru_dhru.geometry.area

    # Drop slivers
    hru_dhru = hru_dhru[hru_dhru["area_f"] >= area_filter_m2].copy()

    return hru_dhru[["HRU_ID", "hru_area", "dhru_id", "dhru_area", "Subbasin", "area_f", "geometry"]]


# ---------------------------------------------------------------------------
# Export hru_dhru table
# ---------------------------------------------------------------------------

def export_hru_dhru(
    hru_dhru_gdf: gpd.GeoDataFrame,
    table_dir: str | os.PathLike,
) -> str:
    """Write the ``hru_dhru`` link table to *table_dir*.

    The file format exactly matches the tab-delimited ASCII layout expected
    by the SWAT-MODFLOW executable:

    ::

        <n_records>
        <max_hru_id>
        dhru_id dhru_area hru_id subbasin hru_area
        <data rows …>

    Parameters
    ----------
    hru_dhru_gdf : GeoDataFrame
        Output of :func:`build_hru_dhru`.
    table_dir : str or path-like
        Destination folder (``GIS/Table`` in the QSWATMOD2 project).

    Returns
    -------
    str
        Absolute path to the written file.

    Examples
    --------
    >>> path = export_hru_dhru(hd, paths.table_folder)
    >>> print("Written to:", path)
    """
    df = hru_dhru_gdf.sort_values(["HRU_ID", "dhru_id"]).reset_index(drop=True)

    n_records  = len(df)
    max_hru_id = int(df["HRU_ID"].max())

    os.makedirs(str(table_dir), exist_ok=True)
    output_file = os.path.normpath(os.path.join(str(table_dir), "hru_dhru"))

    with open(output_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([str(n_records)])
        writer.writerow([str(max_hru_id)])
        writer.writerow(["dhru_id dhru_area hru_id subbasin hru_area"])
        for _, row in df.iterrows():
            writer.writerow([
                f"{int(row['dhru_id']):>10d}",
                f"{int(round(row['area_f'])):>14d}",
                f"{int(row['HRU_ID']):>7d}",
                f"{int(row['Subbasin']):>7d}",
                f"{int(round(row['hru_area'])):>14d}",
            ])

    return output_file


# ---------------------------------------------------------------------------
# Step 3 — dhru_grid: intersect dHRUs × MODFLOW grid
# ---------------------------------------------------------------------------

def build_dhru_grid(
    dhru_gdf: gpd.GeoDataFrame,
    mfgrid_gdf: gpd.GeoDataFrame,
    grid_id_col: str = "grid_id",
    grid_area_col: str = "grid_area",
    area_filter_m2: float = 30.0,
) -> gpd.GeoDataFrame:
    """Intersect dHRU polygons with the MODFLOW grid.

    Replicates ``dhru_grid → create_dhru_grid_filter`` from the QGIS plugin.

    Parameters
    ----------
    dhru_gdf : GeoDataFrame
        Disaggregated HRU polygons from :func:`create_dhru`.
    mfgrid_gdf : GeoDataFrame
        MODFLOW grid polygons.  Must contain *grid_id_col* (``grid_id``).
        If *grid_area_col* (``grid_area``) is absent it is derived from the
        polygon geometry.
    grid_id_col : str, optional
        Grid cell ID column in *mfgrid_gdf*.  Default ``"grid_id"``.
    grid_area_col : str, optional
        Grid cell area column in *mfgrid_gdf*.  Default ``"grid_area"``.
    area_filter_m2 : float, optional
        Sliver threshold [m²].  Intersection fragments smaller than this
        value are dropped (plugin default: 30 m²).

    Returns
    -------
    GeoDataFrame
        Intersection polygons with columns:
        ``dhru_id``, ``dhru_area``, ``HRU_ID``, ``hru_area``,
        ``grid_id``, ``grid_area``, ``ol_area``.

    Examples
    --------
    >>> mfgrid = gpd.read_file("GIS/SMshps/mf_grid.gpkg")
    >>> dg = build_dhru_grid(dhru, mfgrid)
    """
    grid = _ensure_crs_match(dhru_gdf, mfgrid_gdf.copy())
    grid = _fix_geometries(grid)

    if grid_id_col != "grid_id" and grid_id_col in grid.columns:
        grid = grid.rename(columns={grid_id_col: "grid_id"})

    # Ensure grid_area is present
    if grid_area_col not in grid.columns:
        grid["grid_area"] = grid.geometry.area
    elif grid_area_col != "grid_area":
        grid = grid.rename(columns={grid_area_col: "grid_area"})

    # Spatial intersection (dhru × mf_grid)
    intersected = gpd.overlay(
        dhru_gdf[["HRU_ID", "hru_area", "dhru_id", "dhru_area", "geometry"]],
        grid[["grid_id", "grid_area", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    if intersected.empty:
        raise ValueError(
            "dhru_grid intersection returned no features.  Check that the "
            "dHRU and MODFLOW grid shapefiles share the same CRS and extent."
        )

    # Compute overlap area
    intersected["ol_area"] = intersected.geometry.area

    # Drop slivers
    intersected = intersected[intersected["ol_area"] >= area_filter_m2].copy()

    return intersected[
        ["HRU_ID", "hru_area", "dhru_id", "dhru_area", "grid_id", "grid_area", "ol_area", "geometry"]
    ]


# ---------------------------------------------------------------------------
# Export dhru_grid table
# ---------------------------------------------------------------------------

def export_dhru_grid(
    dhru_grid_gdf: gpd.GeoDataFrame,
    mfgrid_gdf: gpd.GeoDataFrame,
    table_dir: str | os.PathLike,
    grid_id_col: str = "grid_id",
) -> str:
    """Write the ``dhru_grid`` link table to *table_dir*.

    The file format exactly matches the tab-delimited ASCII layout expected
    by the SWAT-MODFLOW executable:

    ::

        <n_records>
        <total_grid_cells>
        grid_id grid_area dhru_id overlap_area dhru_area
        <data rows …>

    Parameters
    ----------
    dhru_grid_gdf : GeoDataFrame
        Output of :func:`build_dhru_grid`.
    mfgrid_gdf : GeoDataFrame
        Full MODFLOW grid GeoDataFrame (used to get total cell count).
    table_dir : str or path-like
        Destination folder (``GIS/Table`` in the QSWATMOD2 project).
    grid_id_col : str, optional
        Grid cell ID column in *mfgrid_gdf*.  Default ``"grid_id"``.

    Returns
    -------
    str
        Absolute path to the written file.

    Examples
    --------
    >>> mfgrid = gpd.read_file("GIS/SMshps/mf_grid.gpkg")
    >>> path = export_dhru_grid(dg, mfgrid, paths.table_folder)
    >>> print("Written to:", path)
    """
    df = dhru_grid_gdf.sort_values(["grid_id", "dhru_id"]).reset_index(drop=True)

    n_records       = len(df)
    total_grid_cells = len(mfgrid_gdf)

    os.makedirs(str(table_dir), exist_ok=True)
    output_file = os.path.normpath(os.path.join(str(table_dir), "dhru_grid"))

    with open(output_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([str(n_records)])
        writer.writerow([str(total_grid_cells)])
        writer.writerow(["grid_id grid_area dhru_id overlap_area dhru_area"])
        for _, row in df.iterrows():
            writer.writerow([
                f"{int(row['grid_id']):>10d}",
                f"{int(round(row['grid_area'])):>14d}",
                f"{int(row['dhru_id']):>10d}",
                f"{int(round(row['ol_area'])):>14d}",
                f"{int(round(row['dhru_area'])):>14d}",
            ])

    return output_file


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def generate_link_tables(
    hru_path: str | os.PathLike,
    sub_path: str | os.PathLike,
    mfgrid_path: str | os.PathLike,
    table_dir: str | os.PathLike,
    *,
    hru_id_col: str = "HRU_ID",
    hrugis_col: str = "HRUGIS",
    subbasin_col: str = "Subbasin",
    grid_id_col: str = "grid_id",
    grid_area_col: str = "grid_area",
    hru_area_filter_m2: float = 9.0,
    grid_area_filter_m2: float = 30.0,
    save_intermediate: bool = False,
    intermediate_dir: str | os.PathLike | None = None,
) -> dict[str, str]:
    """Full SWAT-MODFLOW linking pipeline — generate ``hru_dhru`` and ``dhru_grid``.

    This function replicates the complete **Linking Process** sequence from
    the QSWATMOD2 plugin:

    1. Load HRU, subbasin, and MODFLOW grid shapefiles.
    2. Disaggregate multipart HRU polygons → singlepart dHRUs.
    3. Intersect dHRUs × subbasins → ``hru_dhru`` table.
    4. Intersect dHRUs × MODFLOW grid → ``dhru_grid`` table.
    5. Write both tab-delimited files to *table_dir*.

    Parameters
    ----------
    hru_path : str or path-like
        Path to the SWAT HRU shapefile or GeoPackage
        (e.g. ``GIS/SMshps/hru_link.gpkg``).  Must contain integer HRU IDs
        (column *hru_id_col*) and an ``hru_area`` field (or it will be
        computed from polygon geometry).
    sub_path : str or path-like
        Path to the SWAT subbasin shapefile or GeoPackage
        (e.g. ``GIS/SMshps/sub_link.gpkg``).  Must contain *subbasin_col*.
    mfgrid_path : str or path-like
        Path to the MODFLOW grid shapefile or GeoPackage
        (e.g. ``GIS/SMshps/mf_grid.gpkg``).  Must contain *grid_id_col*.
    table_dir : str or path-like
        Output directory for the link table files (``GIS/Table``).

    hru_id_col : str, optional
        Column name for the integer HRU ID in the HRU shapefile.
        Default ``"HRU_ID"``.
    hrugis_col : str, optional
        Column used to create ``HRU_ID`` when it is absent.
        Default ``"HRUGIS"``.
    subbasin_col : str, optional
        Column name for the subbasin ID in the subbasin shapefile.
        Default ``"Subbasin"``.
    grid_id_col : str, optional
        Column name for the grid cell ID in the MODFLOW grid shapefile.
        Default ``"grid_id"``.
    grid_area_col : str, optional
        Column name for grid cell area.  Derived from geometry if absent.
        Default ``"grid_area"``.
    hru_area_filter_m2 : float, optional
        Sliver-removal threshold for ``hru_dhru`` [m²].  Default 9.
    grid_area_filter_m2 : float, optional
        Sliver-removal threshold for ``dhru_grid`` [m²].  Default 30.
    save_intermediate : bool, optional
        If ``True``, write the intermediate ``dhru``, ``hru_dhru``, and
        ``dhru_grid`` GeoPackages to *intermediate_dir* for inspection.
        Default ``False``.
    intermediate_dir : str or path-like, optional
        Folder for intermediate GeoPackages.  Defaults to *table_dir*.

    Returns
    -------
    dict
        ``{"hru_dhru": <path>, "dhru_grid": <path>}``

    Examples
    --------
    >>> from swatmf import Paths
    >>> from swatmf.preprocessing.linking import generate_link_tables
    >>>
    >>> paths = Paths("/data/my_project", "my_project")
    >>>
    >>> result = generate_link_tables(
    ...     hru_path    = paths.sm_shps + "/hru_link.gpkg",
    ...     sub_path    = paths.sm_shps + "/sub_link.gpkg",
    ...     mfgrid_path = paths.sm_shps + "/mf_grid.gpkg",
    ...     table_dir   = paths.table_folder,
    ... )
    >>> print(result)
    {'hru_dhru': '.../GIS/Table/hru_dhru', 'dhru_grid': '.../GIS/Table/dhru_grid'}
    """
    # 1 — load inputs
    hru_gdf    = _fix_geometries(gpd.read_file(str(hru_path)))
    sub_gdf    = _fix_geometries(gpd.read_file(str(sub_path)))
    mfgrid_gdf = _fix_geometries(gpd.read_file(str(mfgrid_path)))

    # 2 — disaggregate HRUs → dHRUs
    dhru_gdf = create_dhru(
        hru_gdf,
        hru_id_col=hru_id_col,
        hrugis_col=hrugis_col,
    )

    # 3 — hru_dhru
    hru_dhru_gdf = build_hru_dhru(
        dhru_gdf,
        sub_gdf,
        subbasin_col=subbasin_col,
        area_filter_m2=hru_area_filter_m2,
    )

    # 4 — dhru_grid
    dhru_grid_gdf = build_dhru_grid(
        dhru_gdf,
        mfgrid_gdf,
        grid_id_col=grid_id_col,
        grid_area_col=grid_area_col,
        area_filter_m2=grid_area_filter_m2,
    )

    # 5 — write table files
    hd_path = export_hru_dhru(hru_dhru_gdf, table_dir)
    dg_path = export_dhru_grid(dhru_grid_gdf, mfgrid_gdf, table_dir, grid_id_col=grid_id_col)

    # optional: save intermediate GeoPackages for QA/QC
    if save_intermediate:
        idir = str(intermediate_dir or table_dir)
        os.makedirs(idir, exist_ok=True)
        dhru_gdf.to_file(os.path.join(idir, "dhru_link.gpkg"),      driver="GPKG")
        hru_dhru_gdf.to_file(os.path.join(idir, "hru_dhru_link.gpkg"), driver="GPKG")
        dhru_grid_gdf.to_file(os.path.join(idir, "dhru_grid_link.gpkg"), driver="GPKG")

    return {"hru_dhru": hd_path, "dhru_grid": dg_path}
