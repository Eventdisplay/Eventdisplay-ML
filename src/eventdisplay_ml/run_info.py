"""
Read run information from Eventdisplay ROOT files without a compiled Eventdisplay.

Replaces ``printRunParameter -runinfo`` / ``-updated-runinfo`` for the XGB wrapper
scripts: instrument epoch, atmosphere ID, DB run type, telescopes, run duration,
target, zenith angle and wobble offset.

The run parameters are stored as a serialized ``VEvndispRunParameter`` object
(``runparameterV2``). ``uproot`` cannot deserialize this object completely (it fails in
the nested image-cleaning parameters), but all members needed here are stored before the
failing one. A tolerant reader keeps the members read until the failure.
"""

import datetime as dt
import logging
import math
import os
from pathlib import Path

import numpy as np
import uproot
import uproot.deserialization

_logger = logging.getLogger(__name__)

RUN_PARAMETER_CLASS = "VEvndispRunParameter"
RUN_PARAMETER_KEY = "runparameterV2"
EPOCH_FILE_NAME = "VERITAS.Epochs.runparameter"
# Members required (all are stored early in the serialized object).
REQUIRED_MEMBERS = (
    "fInstrumentEpoch",
    "fAtmosphereID",
    "fDBRunType",
    "fTelToAnalyze",
    "fRunDuration",
    "fTargetName",
    "fWobbleNorth",
    "fWobbleEast",
    "fDBDataStartTimeMJD",
    "fDBDataStoppTimeMJD",
)
# Telescope elevations below this value (deg) are not pointing (unused telescopes).
MIN_TELESCOPE_ELEVATION = 5.0
# MJD of the proleptic Gregorian ordinal 1 (0001-01-01 is day 1; MJD 0 is 1858-11-17).
_MJD_ORDINAL_OFFSET = 678576


def _tolerant_model(versioned_cls):
    """Return a subclass of a generated uproot model that keeps partially read members."""

    class TolerantModel(versioned_cls):  # pylint: disable=too-few-public-methods
        """Stop reading members at the first deserialization error."""

        def read_members(self, chunk, cursor, context, file):
            try:
                super().read_members(chunk, cursor, context, file)
            except uproot.deserialization.DeserializationError:
                self._members["_truncated"] = True

        def check_numbytes(self, chunk, cursor, context):
            """Skip the byte count check (object is read only partially)."""

    TolerantModel.__name__ = versioned_cls.__name__
    return TolerantModel


def read_run_parameters(root_file):
    """
    Read the leading members of ``runparameterV2`` from an open ROOT file.

    Parameters
    ----------
    root_file : uproot.ReadOnlyDirectory
        Open Eventdisplay mscw, evndisp or anasum-run file.

    Returns
    -------
    dict
        Values of ``REQUIRED_MEMBERS``.

    Raises
    ------
    KeyError
        If the run parameters or one of the required members are missing.
    """
    if RUN_PARAMETER_KEY not in root_file:
        raise KeyError(f"{RUN_PARAMETER_KEY} not found in {root_file.file_path}")
    model_class = root_file.file.class_named(RUN_PARAMETER_CLASS)
    try:
        run_parameter = root_file[RUN_PARAMETER_KEY]
    except uproot.deserialization.DeserializationError:
        # Failed read has defined the class versions; replace them by tolerant readers.
        for version, versioned in list(model_class.known_versions.items()):
            model_class.known_versions[version] = _tolerant_model(versioned)
        run_parameter = root_file[RUN_PARAMETER_KEY]
    members = run_parameter.all_members
    missing = [name for name in REQUIRED_MEMBERS if name not in members]
    if missing:
        raise KeyError(
            f"Run parameter members {missing} not found in {root_file.file_path}"
            " (unexpected VEvndispRunParameter layout)"
        )
    return {name: members[name] for name in REQUIRED_MEMBERS}


def sql_date_to_mjd(sql_string):
    """
    Convert an SQL date ("YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS") to the MJD of its day.

    Follows ``VSkyCoordinatesUtilities::getMJD_from_SQLstring`` (time of day is ignored).
    """
    day = dt.date.fromisoformat(sql_string[:10])
    return float(day.toordinal() - _MJD_ORDINAL_OFFSET)


def read_epoch_file(epoch_file, keyword, mjd_start, mjd_stop):
    """
    Find the EPOCH or ATMOSPHERE entry covering a run.

    Lines have the format ``* <KEYWORD> <value> <sql min> <sql max> [<mjd min> <mjd max>]``.
    The entry must cover the complete run (start and stop MJD).

    Parameters
    ----------
    epoch_file : str or Path
        Epoch file (``VERITAS.Epochs.runparameter``).
    keyword : str
        "EPOCH" or "ATMOSPHERE".
    mjd_start, mjd_stop : float
        Run start and stop MJD.

    Returns
    -------
    str
        Epoch name or atmosphere ID.

    Raises
    ------
    ValueError
        If no entry covers the run, or the run MJDs are not available.
    """
    if keyword not in ("EPOCH", "ATMOSPHERE"):
        raise ValueError(f"Invalid keyword {keyword}")
    if not mjd_start > 0.0 or not mjd_stop > 0.0:
        raise ValueError(f"Run MJDs not available (start {mjd_start}, stop {mjd_stop})")
    with Path(epoch_file).open(encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) < 5 or tokens[0] != "*" or tokens[1] != keyword:
                continue
            value = tokens[2]
            if len(tokens) >= 7:
                mjd_min, mjd_max = float(tokens[5]), float(tokens[6])
            else:
                mjd_min, mjd_max = sql_date_to_mjd(tokens[3]), sql_date_to_mjd(tokens[4])
            if mjd_start >= mjd_min and mjd_stop <= mjd_max:
                return value
    raise ValueError(
        f"No {keyword} interval in {epoch_file} covers run MJDs {mjd_start} to {mjd_stop}"
    )


def read_mean_elevation(root_file):
    """
    Return mean telescope elevation (deg) from the first and last event.

    Only telescopes with an elevation above ``MIN_TELESCOPE_ELEVATION`` are used
    (as in ``printRunParameter``). Returns -999 if no telescope qualifies.
    """
    n_tel = root_file["telconfig"].num_entries
    tree = root_file["data"]
    n_events = tree.num_entries
    if n_events == 0:
        return -999.0
    first = np.asarray(tree["TelElevation"].array(entry_start=0, entry_stop=1, library="np")[0])
    elevations = [first[:n_tel]]
    if n_events > 1:
        last = np.asarray(
            tree["TelElevation"].array(entry_start=n_events - 1, entry_stop=n_events, library="np")[
                0
            ]
        )
        elevations.append(last[:n_tel])
    values = np.concatenate(elevations)
    values = values[values > MIN_TELESCOPE_ELEVATION]
    if values.size == 0:
        return -999.0
    return float(values.mean())


def default_epoch_file():
    """Return the epoch file in ``$VERITAS_EVNDISP_AUX_DIR/ParameterFiles``."""
    aux_dir = os.environ.get("VERITAS_EVNDISP_AUX_DIR")
    if not aux_dir:
        raise ValueError("Epoch file not given and VERITAS_EVNDISP_AUX_DIR not set")
    return Path(aux_dir) / "ParameterFiles" / EPOCH_FILE_NAME


def get_run_info(input_file, update_epoch=False, epoch_file=None):
    """
    Collect run information for XGB model selection.

    Parameters
    ----------
    input_file : str or Path
        Eventdisplay mscw (or evndisp) ROOT file.
    update_epoch : bool
        Take epoch and atmosphere from the epoch file instead of the stored values
        (equivalent to ``printRunParameter updated-runinfo``).
    epoch_file : str or Path, optional
        Epoch file; default ``$VERITAS_EVNDISP_AUX_DIR/ParameterFiles/VERITAS.Epochs.runparameter``.

    Returns
    -------
    dict
        Keys: epoch, major_epoch, atmosphere, run_type, telescopes, duration, target,
        zenith, wobble, wobble_int.
    """
    with uproot.open(input_file) as root_file:
        parameters = read_run_parameters(root_file)
        elevation = read_mean_elevation(root_file)

    epoch = parameters["fInstrumentEpoch"]
    atmosphere = str(parameters["fAtmosphereID"])
    if update_epoch:
        epoch_file = epoch_file or default_epoch_file()
        mjd_start, mjd_stop = parameters["fDBDataStartTimeMJD"], parameters["fDBDataStoppTimeMJD"]
        epoch = read_epoch_file(epoch_file, "EPOCH", mjd_start, mjd_stop)
        atmosphere = read_epoch_file(epoch_file, "ATMOSPHERE", mjd_start, mjd_stop)
    wobble = math.hypot(parameters["fWobbleNorth"], parameters["fWobbleEast"])
    return {
        "epoch": epoch,
        "major_epoch": epoch.split("_", maxsplit=1)[0],
        "atmosphere": atmosphere,
        "run_type": parameters["fDBRunType"] or "no_db_runtype",
        "telescopes": "".join(str(int(tel) + 1) for tel in parameters["fTelToAnalyze"]),
        "duration": parameters["fRunDuration"],
        "target": parameters["fTargetName"],
        "zenith": 90.0 - elevation,
        "wobble": wobble,
        "wobble_int": math.floor(wobble * 100.0 + 0.5),
    }


def format_run_info(info):
    """
    Format run info as one tab-separated line (layout of ``printRunParameter -runinfo``).

    Columns: epoch, major epoch, atmosphere, run type, telescopes, duration, target,
    zenith, wobble, wobble x100 (integer). Floating point values use six significant
    digits as the C++ stream output.
    """
    return "\t".join(
        [
            info["epoch"],
            info["major_epoch"],
            info["atmosphere"],
            info["run_type"],
            info["telescopes"],
            f"{info['duration']:g}",
            info["target"],
            f"{info['zenith']:g}",
            f"{info['wobble']:g}",
            str(info["wobble_int"]),
        ]
    )
