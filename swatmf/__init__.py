"""
swatmf
======
A pure-Python package for scripted SWAT-MODFLOW pre-processing and
post-processing workflows.

This package replicates the analysis and visualisation logic of the QSWATMOD2
QGIS plugin without requiring QGIS, PyQt, or any other GUI dependency.  All
functions operate directly on the input and output files produced by the
SWAT-MODFLOW executable.

Modules
-------
paths                        : project directory/path management
sim_period                   : simulation period parsing (file.cio)
metrics                      : objective functions (NSE, RMSE, PBias, R²)
preprocessing.modflow        : parse/write MODFLOW files; build new MF model
preprocessing.linking        : generate hru_dhru and dhru_grid link tables
simulation                   : read/write swatmf_link.txt configuration
outputs/streamflow           : read & visualise output.rch
outputs/groundwater          : read & visualise swatmf_out_MF_obs
outputs/recharge             : read swatmf_out_MF_recharge* files
outputs/gwsw                 : read swatmf_out_MF_gwsw* files
outputs/water_balance        : read output.std
"""

__version__ = "0.1.0"
# Version string used in exported file headers to match the QSWATMOD2 plugin convention
_EXPORT_VERSION = "version 2.10.1."

from .paths import Paths
from .sim_period import parse_file_cio

__all__ = ["Paths", "parse_file_cio"]
