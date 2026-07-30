"""swatmf.outputs — sub-package for reading SWAT-MODFLOW output files.

Modules
-------
streamflow          — Read and plot SWAT streamflow output.
groundwater         — Read and plot MODFLOW observation well output.
recharge            — Read and plot MODFLOW recharge output.
gwsw                — Read and plot GW-SW exchange output.
water_balance       — Read and plot water-balance components.
rt3d                — Read and plot RT3D concentration and nutrient flux output.
modflow_diagnostics — Diagnostic plots for standalone MODFLOW QA.
"""

from .modflow_diagnostics import (
    plot_flow_direction,
    plot_head_gradient_check,
    plot_head_before_after,
    plot_flow_and_head_change,
    plot_water_table_depth,
    plot_budget_summary,
    plot_boundary_flux,
    plot_drain_head,
    plot_drain_flux,
)
