"""Utilities for processing CP2K cube files (e.g. V_HARTREE_CUBE).

Provides helpers to read cube files, compute planar-averaged electrostatic
potentials along Z, and construct Z coordinates in Angstrom.

All energy values are converted from Hartree (CP2K default) to eV using
ipi.utils.units.Constants.EV_PER_HARTREE.
"""

import os
from typing import Tuple

import numpy as np

from ipi.utils.units import Constants

# Conversion factor from Bohr to Angstrom, consistent with CP2K and common
# quantum chemistry conventions.
BOHR_TO_ANGSTROM: float = 0.529177210903


def read_cube(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Read a Gaussian/CP2K cube file.

    Parameters
    ----------
    filename : str
        Path to the cube file.

    Returns
    -------
    cube_data : np.ndarray
        3D array of shape (nx, ny, nz) with the scalar field values (Hartree).
    origin_bohr : np.ndarray
        Origin of the grid in Bohr, shape (3,).
    dx, dy, dz : np.ndarray
        Grid vectors along x, y, z in Bohr, each of shape (3,).
    shape : tuple of int
        Grid dimensions (nx, ny, nz).
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cube file not found: {filename}")

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"{filename}: content too short to be a cube file")

    # Third line: natoms x0 y0 z0
    parts = lines[2].split()
    if len(parts) < 4:
        raise ValueError(f"{filename}: malformed header line 3")

    natoms = int(float(parts[0]))
    origin_bohr = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)

    def _parse_axis(line: str):
        tokens = line.split()
        if len(tokens) != 4:
            raise ValueError("Malformed axis line (expect: n vx vy vz)")
        n = int(float(tokens[0]))
        vec = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=float)
        return n, vec

    nx, dx = _parse_axis(lines[3])
    ny, dy = _parse_axis(lines[4])
    nz, dz = _parse_axis(lines[5])

    # Atom block: natoms lines starting from line 6
    data_start = 6 + abs(natoms)

    # Data: nx*ny*nz floats in Fortran order (x fastest, z slowest)
    data_flat = np.fromstring(" ".join(lines[data_start:]), sep=" ")
    expected = nx * ny * nz
    if data_flat.size != expected:
        raise ValueError(f"{filename}: expected {expected} values, got {data_flat.size}")

    cube_data = data_flat.reshape((nx, ny, nz))
    return cube_data, origin_bohr, dx, dy, dz, (nx, ny, nz)


def planar_average_z(cube_data: np.ndarray, to_eV: bool = True) -> np.ndarray:
    """Compute planar-averaged potential along Z.

    Parameters
    ----------
    cube_data : np.ndarray
        3D array of shape (nx, ny, nz) in Hartree units.
    to_eV : bool, optional
        If True, convert from Hartree to eV using Constants.EV_PER_HARTREE.

    Returns
    -------
    np.ndarray
        1D array of length nz with planar-averaged potential in eV (if
        ``to_eV`` is True) or Hartree otherwise.
    """

    z_profile = cube_data.mean(axis=(0, 1))
    if to_eV:
        z_profile = z_profile * Constants.EV_PER_HARTREE
    return z_profile


def z_coordinates_A(origin_bohr: np.ndarray, dz_vec_bohr: np.ndarray, nz: int) -> np.ndarray:
    """Construct Z coordinates (in Angstrom) for the third grid dimension.

    This uses the norm of the DZ vector as the inter-layer spacing and
    returns relative coordinates z_i = i * |dz|, i = 0..nz-1.

    Parameters
    ----------
    origin_bohr : np.ndarray
        Origin of the grid in Bohr (unused for now, kept for completeness).
    dz_vec_bohr : np.ndarray
        Grid vector along Z in Bohr, shape (3,).
    nz : int
        Number of grid points along Z.

    Returns
    -------
    np.ndarray
        1D array of length nz with Z coordinates in Angstrom.
    """

    dz_len_A = float(np.linalg.norm(dz_vec_bohr)) * BOHR_TO_ANGSTROM
    z_vals = np.arange(nz, dtype=float) * dz_len_A
    return z_vals
