"""Reproducible, optional perturbations of stereo image parameters."""

from pathlib import Path

import numpy as np
import yaml

_STRENGTHS = ("size", "width", "length", "centroid", "asym", "tgrad")


def load_smearing(path):
    """Read and validate image-smearing settings from YAML."""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) - {*_STRENGTHS, "seed", "fraction"}:
        raise ValueError("Smearing YAML must contain only seed, fraction and supported strengths")
    result = {"seed": 0, "fraction": 1.0, **config}
    if (
        isinstance(result["seed"], bool)
        or not isinstance(result["seed"], int)
        or not 0 <= result["seed"] < 2**32
    ):
        raise ValueError("Smearing seed must be an integer in [0, 2**32)")
    for name in ("fraction", *_STRENGTHS):
        value = result.get(name, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
            or (name == "fraction" and value > 1)
        ):
            raise ValueError(f"Invalid smearing value for {name}: {value}")
        result[name] = float(value)
    return result


def smear_images(values, variable, events, active, config):
    """Smear physical telescope-ID-indexed values without altering invalid images.

    Draws are generated one event at a time, so temporary random-number storage
    is independent of the number of events in a chunk.
    """
    key = {"cen_x": "centroid", "cen_y": "centroid", "tgrad_x": "tgrad"}.get(variable, variable)
    sigma = config.get(key, 0.0)
    if not sigma or not config.get("fraction", 1.0):
        return values
    run = np.asarray(events["runNumber"], dtype=np.uint64)
    event = np.asarray(events["eventNumber"], dtype=np.uint64)
    result = np.array(values, dtype=float, copy=True)
    stream = {"size": 0, "width": 1, "length": 2, "cen_x": 3, "cen_y": 4, "asym": 5, "tgrad_x": 6}[
        variable
    ]
    for i, (run_id, event_id) in enumerate(zip(run, event, strict=True)):
        mask = active[i] & np.isfinite(result[i])
        if variable in ("size", "width", "length"):
            mask &= result[i] > 0
        if not mask.any():
            continue
        run_id, event_id = int(run_id), int(event_id)
        rng = np.random.default_rng(
            [
                config["seed"],
                run_id & 0xFFFFFFFF,
                run_id >> 32,
                event_id & 0xFFFFFFFF,
                event_id >> 32,
            ]
        )
        if rng.random() >= config.get("fraction", 1.0):
            continue
        z = rng.standard_normal((7, values.shape[1]))[stream, mask]
        if variable == "size":
            result[i, mask] *= 10 ** (sigma * z)
        else:
            result[i, mask] += sigma * z
            if variable in ("width", "length"):
                result[i, mask] = np.maximum(result[i, mask], 1e-6)
    return result
