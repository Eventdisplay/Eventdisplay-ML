"""Unit tests for geomag.py."""

import numpy as np
import pytest

from eventdisplay_ml.geomag import FIELD_COMPONENTS, calculate_geomagnetic_angles

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _expected_angle(azimuth_deg, elevation_deg, observatory):
    """Recompute the expected angle from scratch for verification."""
    obs = observatory.upper()
    bx = FIELD_COMPONENTS[obs]["BX"]
    by = FIELD_COMPONENTS[obs]["BY"]
    bz = FIELD_COMPONENTS[obs]["BZ"]

    sx = np.cos(np.radians(elevation_deg)) * np.cos(np.radians(azimuth_deg))
    sy = np.cos(np.radians(elevation_deg)) * np.sin(np.radians(azimuth_deg))
    sz = np.sin(np.radians(elevation_deg))

    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    bx_n = bx / b_mag
    by_n = by / b_mag
    bz_n = -bz / b_mag  # sign flip as in source

    cos_theta = sx * bx_n + sy * by_n + sz * bz_n
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))


# ---------------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observatory", ["VERITAS", "CTAO-NORTH", "CTAO-SOUTH"])
def test_angle_at_zenith_is_in_valid_range(observatory):
    """Pointing straight up should give a valid angle for any observatory."""
    angle = calculate_geomagnetic_angles(0.0, 90.0, observatory=observatory)
    assert 0.0 <= float(angle) <= 180.0


@pytest.mark.parametrize(("az", "el"), [(0, 90), (0, 45), (180, 45), (90, 0), (270, 30)])
def test_angle_always_between_0_and_180(az, el):
    angle = calculate_geomagnetic_angles(az, el)
    assert 0.0 <= float(angle) <= 180.0


def test_angle_matches_manual_computation_veritas():
    az, el = 30.0, 60.0
    expected = _expected_angle(az, el, "VERITAS")
    result = calculate_geomagnetic_angles(az, el, observatory="VERITAS")
    assert result == pytest.approx(expected, rel=1e-5)


def test_angle_matches_manual_computation_ctao_north():
    az, el = 120.0, 45.0
    expected = _expected_angle(az, el, "CTAO-NORTH")
    result = calculate_geomagnetic_angles(az, el, observatory="CTAO-NORTH")
    assert result == pytest.approx(expected, rel=1e-5)


def test_angle_matches_manual_computation_ctao_south():
    az, el = 200.0, 30.0
    expected = _expected_angle(az, el, "CTAO-SOUTH")
    result = calculate_geomagnetic_angles(az, el, observatory="CTAO-SOUTH")
    assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Array-valued inputs
# ---------------------------------------------------------------------------


def test_accepts_numpy_arrays():
    az = np.array([0.0, 90.0, 180.0, 270.0])
    el = np.array([45.0, 45.0, 45.0, 45.0])
    result = calculate_geomagnetic_angles(az, el)
    assert result.shape == (4,)
    assert np.all((result >= 0) & (result <= 180))


def test_array_matches_scalar_results():
    azimuths = [0.0, 30.0, 90.0]
    elevations = [45.0, 60.0, 30.0]
    scalar_results = [
        float(calculate_geomagnetic_angles(az, el)) for az, el in zip(azimuths, elevations)
    ]
    array_result = calculate_geomagnetic_angles(np.array(azimuths), np.array(elevations))
    np.testing.assert_allclose(array_result, scalar_results, rtol=1e-5)


# ---------------------------------------------------------------------------
# Case-insensitivity and different observatories
# ---------------------------------------------------------------------------


def test_observatory_name_is_case_insensitive():
    angle_upper = calculate_geomagnetic_angles(0.0, 45.0, observatory="VERITAS")
    angle_lower = calculate_geomagnetic_angles(0.0, 45.0, observatory="veritas")
    assert float(angle_upper) == pytest.approx(float(angle_lower))


def test_different_observatories_give_different_angles():
    """VERITAS and CTAO-SOUTH have different field components → different angles."""
    az, el = 45.0, 45.0
    angle_veritas = float(calculate_geomagnetic_angles(az, el, observatory="VERITAS"))
    angle_ctao_south = float(calculate_geomagnetic_angles(az, el, observatory="CTAO-SOUTH"))
    assert angle_veritas != pytest.approx(angle_ctao_south)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_observatory_raises_key_error():
    with pytest.raises(KeyError, match="not defined"):
        calculate_geomagnetic_angles(0.0, 45.0, observatory="UNKNOWN_SITE")


# ---------------------------------------------------------------------------
# Physical consistency
# ---------------------------------------------------------------------------


def test_field_components_dict_has_required_keys():
    for obs in ("VERITAS", "CTAO-NORTH", "CTAO-SOUTH"):
        assert obs in FIELD_COMPONENTS
        for comp in ("BX", "BY", "BZ"):
            assert comp in FIELD_COMPONENTS[obs]


def test_veritas_by_is_zero():
    """BY is assumed 0 for all defined observatories (CORSIKA convention)."""
    for obs in FIELD_COMPONENTS.values():
        assert obs["BY"] == pytest.approx(0.0)
