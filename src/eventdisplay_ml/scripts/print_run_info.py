"""
Print run information of an Eventdisplay ROOT file in one line.

Python replacement for ``printRunParameter -runinfo`` (and ``updated-runinfo`` with
``--updated``), so that XGB model selection does not require a compiled Eventdisplay.
Columns (tab-separated): epoch, major epoch, atmosphere, run type, telescopes,
run duration, target, zenith angle (deg), wobble offset, wobble offset x100.
"""

import argparse
import logging
import sys

from eventdisplay_ml.run_info import format_run_info, get_run_info

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
_logger = logging.getLogger(__name__)


def main():
    """Print run information."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n", maxsplit=1)[0])
    parser.add_argument("input_file", help="Eventdisplay mscw (or evndisp) ROOT file")
    parser.add_argument(
        "--updated",
        action="store_true",
        help="take epoch and atmosphere from the epoch file (like 'updated-runinfo')",
    )
    parser.add_argument(
        "--epoch_file",
        default=None,
        help="epoch file (default: $VERITAS_EVNDISP_AUX_DIR/ParameterFiles/VERITAS.Epochs.runparameter)",
    )
    args = parser.parse_args()
    try:
        info = get_run_info(args.input_file, update_epoch=args.updated, epoch_file=args.epoch_file)
    except (KeyError, ValueError, OSError) as err:
        _logger.error("Cannot read run information from %s: %s", args.input_file, err)
        sys.exit(1)
    print(format_run_info(info))


if __name__ == "__main__":
    main()
