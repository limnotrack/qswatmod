"""
swatmf.sim_period
=================
Parse the SWAT ``file.cio`` control file to determine the simulation period.

This mirrors the ``define_sim_period`` method in ``QSWATMOD2.py`` but does
not interact with any GUI widget.
"""

from __future__ import annotations

import datetime
import os
from typing import NamedTuple


class SimPeriod(NamedTuple):
    """Container for the four key simulation dates.

    Attributes
    ----------
    start_date : datetime.datetime
        First calendar date of the simulation (no warmup skipped).
    end_date : datetime.datetime
        Last calendar date of the simulation.
    start_date_warmup : datetime.datetime
        First calendar date after the warmup period has been skipped.
    end_date_warmup : datetime.datetime
        Last calendar date of the effective simulation (i.e. ``end_date``
        minus the number of warm-up years that were skipped from the output).
    skipyear : int
        Number of warm-up years skipped (``NYSKIP`` in file.cio).
    iprint : int
        Output print code (0 = monthly, 1 = daily, 2 = annual).
    """

    start_date: datetime.datetime
    end_date: datetime.datetime
    start_date_warmup: datetime.datetime
    end_date_warmup: datetime.datetime
    skipyear: int
    iprint: int


def parse_file_cio(swatmf_folder: str | os.PathLike) -> SimPeriod:
    """Parse ``file.cio`` and return a :class:`SimPeriod` named-tuple.

    Parameters
    ----------
    swatmf_folder : str or path-like
        Directory that contains the ``file.cio`` file (the SWAT-MODFLOW
        working directory, i.e. ``Paths.swatmf_folder``).

    Returns
    -------
    SimPeriod

    Raises
    ------
    FileNotFoundError
        If ``file.cio`` is not present in *swatmf_folder*.
    ValueError
        If the file cannot be parsed correctly.

    Notes
    -----
    Line indices used (0-based):

    * line  7 (``NBYR``)   — number of years of simulation
    * line  8 (``IYR``)    — beginning year of simulation
    * line  9 (``IDAF``)   — beginning Julian day of simulation
    * line 10 (``IDAD``)   — ending Julian day of simulation
    * line 58 (``IPRINT``) — print code (0=monthly, 1=daily, 2=annual)
    * line 59 (``NYSKIP``) — number of years to skip output
    """
    cio_path = os.path.join(str(swatmf_folder), "file.cio")
    if not os.path.isfile(cio_path):
        raise FileNotFoundError(f"file.cio not found in {swatmf_folder!r}")

    with open(cio_path, "r") as fh:
        lines = fh.readlines()

    def _read_int(line: str) -> int:
        return int(line[12:16])

    skipyear = _read_int(lines[59])
    iprint = _read_int(lines[58])
    styear = _read_int(lines[8])
    nbyr = _read_int(lines[7])
    idaf = _read_int(lines[9])   # beginning Julian day
    idad = _read_int(lines[10])  # ending Julian day

    edyear = styear + nbyr - 1
    styear_warmup = styear + skipyear
    edyear_warmup = styear_warmup + nbyr - 1 - skipyear

    # When there is no warmup the beginning Julian day is respected; otherwise
    # the simulation restarts on January 1st of the warmup year.
    begin_day = idaf if skipyear == 0 else 1

    stdate = datetime.datetime(styear, 1, 1) + datetime.timedelta(idaf - 1)
    eddate = datetime.datetime(edyear, 1, 1) + datetime.timedelta(idad - 1)
    stdate_warmup = datetime.datetime(styear_warmup, 1, 1) + datetime.timedelta(begin_day - 1)
    eddate_warmup = datetime.datetime(edyear_warmup, 1, 1) + datetime.timedelta(idad - 1)

    return SimPeriod(
        start_date=stdate,
        end_date=eddate,
        start_date_warmup=stdate_warmup,
        end_date_warmup=eddate_warmup,
        skipyear=skipyear,
        iprint=iprint,
    )
