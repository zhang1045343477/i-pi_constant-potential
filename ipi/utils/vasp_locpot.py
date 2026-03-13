#!/usr/bin/env python3
"""Utilities for reading and processing VASP LOCPOT files.

This module provides a minimal, dependency-free reader for VASP LOCPOT
files and helpers to compute planar-averaged electrostatic potentials
along the Cartesian Z direction.

The implementation is intentionally conservative and focused on the
use case of workfunction calculations in slab geometries:

* Only VASP-style LOCPOT/CHGCAR grid files are supported.
* Grid data are reshaped using Fortran (column-major) order, consistent
  with how VASP writes the real-space grid.
* For LOCPOT files, the stored values are already in eV; the planar
  average is returned in the same units without further scaling.

Note
----
This reader mirrors the logic used in the standalone test utility
`1-vtotav.py` in the test7-cp-vasp directory, but is packaged as a
reusable library function for use inside i-PI (e.g. for VASP-based
workfunction calculations).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def _read_vasp_grid(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read cell and 3D grid data from a VASP LOCPOT/CHGCAR-like file.

    Parameters
    ----------
    filename
        Path to the VASP LOCPOT (or CHGCAR-style) file.

    Returns
    -------
    cell : (3, 3) ndarray
        Lattice vectors in Angstrom.
    grid : (nx, ny, nz) ndarray
        Real-space grid data reshaped in Fortran (column-major) order so
        that ``grid[i, j, k]`` corresponds to the value at that grid
        point in the VASP real-space mesh.
    """

    with open(filename, "r") as f:
        # Header: comment line + global scaling
        _ = f.readline()  # comment
        scale = float(f.readline().split()[0])

        # Lattice vectors (3 lines)
        cell = np.zeros((3, 3), float)
        for i in range(3):
            parts = f.readline().split()
            cell[i, :] = [float(x) for x in parts[:3]]
        cell *= scale

        # Element symbols or atom counts (VASP5 vs VASP4 style)
        line = f.readline().split()
        try:
            # If the first token is an int, this was actually the counts line
            int(line[0])
            natoms_per_type = [int(x) for x in line]
        except ValueError:
            # VASP 5: line contains element symbols, next line has counts
            natoms_per_type = [int(x) for x in f.readline().split()]

        natoms = sum(natoms_per_type)

        # Coordinate mode, with optional "Selective dynamics" line
        line = f.readline().strip()
        if line and line[0].upper() == "S":
            # Selective dynamics present; next line is coordinate mode
            coord_mode = f.readline().strip()
        else:
            coord_mode = line
        # coord_mode is not used here; we just need to skip atomic positions

        # Skip atomic positions
        for _ in range(natoms):
            f.readline()

        # Optional blank line before grid dimensions
        line = f.readline()
        if not line.strip():
            line = f.readline()
        if not line:
            raise RuntimeError(
                "Unexpected end of file while looking for grid dimensions in %s" % filename
            )

        parts = line.split()
        if len(parts) != 3:
            raise RuntimeError(
                "Could not parse grid dimensions from line: %s" % line.strip()
            )
        nx, ny, nz = [int(x) for x in parts]

        # Read the remaining floating-point values (may contain multiple blocks)
        data_vals = []
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            data_vals.extend(tokens)

        data = np.array([float(x) for x in data_vals], float)
        npoints = nx * ny * nz
        if data.size % npoints != 0:
            raise RuntimeError(
                "Grid data size (%d) is not a multiple of nx*ny*nz (%d)" % (data.size, npoints)
            )

        # Use the last grid block (e.g. total potential or total charge density)
        nblocks = data.size // npoints
        block = data[(nblocks - 1) * npoints :]

        # VASP writes grid data in Fortran (column-major) order; reshape accordingly.
        grid = block.reshape((nx, ny, nz), order="F")

        return cell, grid


def planar_average_z_vasp(grid: np.ndarray) -> np.ndarray:
    """Compute planar-averaged profile along Z from a 3D VASP grid.

    The input ``grid`` is assumed to have shape ``(nx, ny, nz)`` and to
    correspond to a LOCPOT-style potential grid in eV. The function
    averages over the X and Y indices to obtain a 1D profile vs the
    third (Z) index.

    Parameters
    ----------
    grid
        3D numpy array with shape (nx, ny, nz).

    Returns
    -------
    profile : (nz,) ndarray
        Planar-averaged potential along Z, in the same units as ``grid``
        (for LOCPOT this is eV).
    """

    if grid.ndim != 3:
        raise ValueError(
            "planar_average_z_vasp expects a 3D array of shape (nx, ny, nz), got ndim=%d" % grid.ndim
        )

    # Average over x and y, keep z
    return grid.mean(axis=(0, 1))


def z_coords_from_cell(cell: np.ndarray, nz: int) -> np.ndarray:
    """Construct Z coordinates (in Angstrom) for a VASP grid.

    This helper mirrors the convention used by typical LOCPOT post-
    processing scripts (including the original vtotav.py): the Z axis is
    assumed to follow the third lattice vector, and the grid is treated
    as spanning [0, Lz] with ``nz`` points, giving a spacing of

        dz = Lz / (nz - 1)

    and coordinates ``z[i] = i * dz``.

    Parameters
    ----------
    cell
        (3, 3) lattice matrix in Angstrom.
    nz
        Number of grid points along Z.

    Returns
    -------
    z : (nz,) ndarray
        Z coordinates in Angstrom.
    """

    if cell.shape != (3, 3):
        raise ValueError("cell must be a (3,3) array of lattice vectors in Angstrom")
    if nz < 2:
        # Degenerate case; just return zeros
        return np.zeros(int(max(nz, 0)), float)

    # Lattice vector lengths
    latticelength = np.dot(cell, cell.T).diagonal() ** 0.5
    Lz = float(latticelength[2])

    dz = Lz / float(nz - 1)
    indices = np.arange(nz, dtype=float)
    return indices * dz


__all__ = [
    "_read_vasp_grid",
    "planar_average_z_vasp",
    "z_coords_from_cell",
]
