"""Calculate shower angles with respect to geomagnetic field."""

import numpy as np

# Follow CORSIKA definitions
# BX horizontal component toward North
# BZ: vertical component downward
FIELD_COMPONENTS = {
    "VERITAS": {
        "BX": 25.2e-6,  # Tesla
        "BZ": 40.88e-6,  # Tesla
    }
}


def calculate_geomagnetic_angles(azimuth, elevation, site="VERITAS"):
    """
    Calculate the angle between the shower direction and the geomagnetic field.

    Parameters
    ----------
    azimuth : array-like
        Azimuth angles of the showers in degrees.
    elevation : array-like
        Elevation angles of the showers in degrees.
    site : str
        Site identifier to get geomagnetic field components.

    Returns
    -------
    theta_B : array-like
        Angle between shower direction and geomagnetic field in degrees.
    """
    try:
        bx = FIELD_COMPONENTS[site]["BX"]
        bz = FIELD_COMPONENTS[site]["BZ"]
    except KeyError as exc:
        raise KeyError(f"Geomagnetic field components for site '{site}' are not defined.") from exc

    # Shower direction unit vector
    sx = np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))  # North
    sy = np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))  # East
    sz = np.sin(np.radians(elevation))  # Up
    sx = np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))  # North

    # Geomagnetic field unit vector
    b_magnitude = np.sqrt(bx**2 + bz**2)
    bx = bx / b_magnitude
    by = 0.0
    bz = -bz / b_magnitude  # magnetic field points downward

    # Dot product to find cos(theta_B)
    cos_theta_b = sx * bx + sy * by + sz * bz
    return np.degrees(np.arccos(cos_theta_b))
