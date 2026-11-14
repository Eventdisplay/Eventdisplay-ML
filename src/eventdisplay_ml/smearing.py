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


def image_draws(events, n_tel, seed):
    """Draw per-event standard normals with NumPy, independently of chunking."""
    run = np.asarray(events["runNumber"], dtype=np.uint64)
    event = np.asarray(events["eventNumber"], dtype=np.uint64)
    selection = np.empty(len(run))
    normal = np.empty((len(run), 7, n_tel))
    for i, (run_id, event_id) in enumerate(zip(run, event, strict=True)):
        run_id, event_id = int(run_id), int(event_id)
        rng = np.random.default_rng(
            [seed, run_id & 0xFFFFFFFF, run_id >> 32, event_id & 0xFFFFFFFF, event_id >> 32]
        )
        selection[i] = rng.random()
        normal[i] = rng.standard_normal((7, n_tel))
    return selection, normal


def smear_images(values, variable, events, active, config):
    """Smear physical telescope-ID-indexed values without altering invalid images."""
    key = {"cen_x": "centroid", "cen_y": "centroid", "tgrad_x": "tgrad"}.get(variable, variable)
    sigma = config.get(key, 0.0)
    if not sigma or not config.get("fraction", 1.0):
        return values
    selection, normal = config.get("_draws", (None, None))
    if selection is None:
        selection, normal = image_draws(events, values.shape[1], config["seed"])
    result = np.array(values, dtype=float, copy=True)
    stream = {"size": 0, "width": 1, "length": 2, "cen_x": 3, "cen_y": 4, "asym": 5, "tgrad_x": 6}[
        variable
    ]
    selected = selection < config.get("fraction", 1.0)
    for tel in range(result.shape[1]):
        z = normal[:, stream, tel]
        mask = active[:, tel] & selected & np.isfinite(result[:, tel])
        if variable in ("size", "width", "length"):
            mask &= result[:, tel] > 0
        if variable == "size":
            result[mask, tel] *= 10 ** (sigma * z[mask])
        else:
            result[mask, tel] += sigma * z[mask]
            if variable in ("width", "length"):
                result[mask, tel] = np.maximum(result[mask, tel], 1e-6)
    return result
