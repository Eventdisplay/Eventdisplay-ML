"""Unit tests for telescope sorting logic (mirror area first, then size)."""

import numpy as np


def test_sort_logic_equal_area_by_size():
    """Test sorting with equal mirror areas: secondary sort by size descending."""
    max_tel_id = 3
    mirror_lookup = np.array([np.nan, 100.0, np.nan, 100.0], dtype=np.float32)
    sizes_row = np.array([np.nan, 3725.51, np.nan, 2640.01], dtype=np.float32)

    # Apply the new sort logic: area priority, then size within equal area
    tel_entries = []
    for tel_idx in range(max_tel_id + 1):
        area = mirror_lookup[tel_idx]
        size_val = sizes_row[tel_idx]
        area_valid = 0 if not np.isnan(area) else 1
        size_valid = 0 if not np.isnan(size_val) else 1
        area_key = -area if area_valid == 0 else 0.0
        size_key = -size_val if size_valid == 0 else 0.0
        tel_entries.append((tel_idx, area_valid, area_key, size_valid, size_key))

    tel_entries.sort(key=lambda x: (x[1], x[2], x[3], x[4]))
    sort_indices = np.array([t[0] for t in tel_entries])

    # Expect: tel 1 (area 100, size 3725.51), tel 3 (area 100, size 2640.01), then NaN areas
    assert np.array_equal(sort_indices, np.array([1, 3, 0, 2]))


def test_sort_logic_mixed_areas_and_nan_sizes():
    """Test sorting with mixed mirror areas and some NaN sizes: area priority always."""
    max_tel_id = 4
    # Areas: tel 0=200, tel 1=100 (NaN size), tel 2=200, tel 3=100, tel 4=NaN
    mirror_lookup = np.array([200.0, 100.0, 200.0, 100.0, np.nan], dtype=np.float32)
    # Sizes: tel 0=1000, tel 1=NaN, tel 2=500, tel 3=2000, tel 4=3000
    sizes_row = np.array([1000.0, np.nan, 500.0, 2000.0, 3000.0], dtype=np.float32)

    # Apply sorting
    tel_entries = []
    for tel_idx in range(max_tel_id + 1):
        area = mirror_lookup[tel_idx]
        size_val = sizes_row[tel_idx]
        area_valid = 0 if not np.isnan(area) else 1
        size_valid = 0 if not np.isnan(size_val) else 1
        area_key = -area if area_valid == 0 else 0.0
        size_key = -size_val if size_valid == 0 else 0.0
        tel_entries.append((tel_idx, area_valid, area_key, size_valid, size_key))

    tel_entries.sort(key=lambda x: (x[1], x[2], x[3], x[4]))
    sort_indices = np.array([t[0] for t in tel_entries])

    # Expect: area 200 desc (tel 0 size 1000, tel 2 size 500), then area 100 desc (tel 3 size 2000, tel 1 size NaN), then NaN area (tel 4)
    assert np.array_equal(sort_indices, np.array([0, 2, 3, 1, 4]))
