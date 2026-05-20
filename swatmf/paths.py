"""
swatmf.paths
============
Project directory and path management.

Mirrors the ``dirs_and_paths`` method from the QSWATMOD2 plugin, but
implemented as a plain Python class that does not require a QGIS project
to be open.
"""

from __future__ import annotations

import os


class Paths:
    """Resolve all QSWATMOD2 project sub-directories from the project root.

    Parameters
    ----------
    project_dir : str or path-like
        The folder that contains the QGIS project file (``*.qgz`` / ``*.qgs``).
        This is the folder whose contents are copied from *FOLDER_FOR_COPY*
        when a new project is created.
    project_name : str
        The base name of the QGIS project file (without extension).  This is
        used to locate the project sub-folder that holds all SWAT-MODFLOW
        inputs and outputs.

    Examples
    --------
    >>> from swatmf import Paths
    >>> p = Paths("/data/my_project", "my_project")
    >>> p.swatmf_folder
    '/data/my_project/my_project/SWAT-MODFLOW'
    """

    def __init__(self, project_dir: str | os.PathLike, project_name: str) -> None:
        self._root = os.path.normpath(str(project_dir))
        self._name = project_name

    def _sub(self, *parts: str) -> str:
        return os.path.normpath(os.path.join(self._root, self._name, *parts))

    # ------------------------------------------------------------------
    # Public path properties
    # ------------------------------------------------------------------

    @property
    def org_shps(self) -> str:
        """Original shapefiles folder (``GIS/org_shps``)."""
        return self._sub("GIS", "org_shps")

    @property
    def sm_shps(self) -> str:
        """SWAT-MODFLOW shapefiles folder (``GIS/SMshps``)."""
        return self._sub("GIS", "SMshps")

    @property
    def swatmf_folder(self) -> str:
        """SWAT-MODFLOW model folder (``SWAT-MODFLOW``).

        This is the working directory (``wd``) referenced throughout the
        plugin — it contains ``file.cio``, ``output.rch``,
        ``swatmf_out_MF_obs``, etc.
        """
        return self._sub("SWAT-MODFLOW")

    @property
    def table_folder(self) -> str:
        """Table / link-file template folder (``GIS/Table``)."""
        return self._sub("GIS", "Table")

    @property
    def sm_exes(self) -> str:
        """Executables folder (``SM_exes``)."""
        return self._sub("SM_exes")

    @property
    def exported_files(self) -> str:
        """Exported results folder (``exported_files``)."""
        return self._sub("exported_files")

    @property
    def scenarios(self) -> str:
        """Scenarios folder (``Scenarios``)."""
        return self._sub("Scenarios")

    @property
    def db_files(self) -> str:
        """SQLite database folder (``DB``)."""
        return self._sub("DB")

    # ------------------------------------------------------------------
    # Convenience: dict-style access (backwards-compatible with plugin)
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, str]:
        """Return all paths as a dictionary matching the plugin convention.

        Returns
        -------
        dict
            Keys: ``org_shps``, ``SMshps``, ``SMfolder``, ``Table``,
            ``SM_exes``, ``exported_files``, ``Scenarios``, ``db_files``.
        """
        return {
            "org_shps": self.org_shps,
            "SMshps": self.sm_shps,
            "SMfolder": self.swatmf_folder,
            "Table": self.table_folder,
            "SM_exes": self.sm_exes,
            "exported_files": self.exported_files,
            "Scenarios": self.scenarios,
            "db_files": self.db_files,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Paths(project_dir={self._root!r}, project_name={self._name!r})"
        )
