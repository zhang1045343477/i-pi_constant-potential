"""Asymmetric double-well potential with constant potential / workfunction capability for i-PI.

In this toy model we define the *work function* as the derivative of the
total energy with respect to electronic charge, ``Phi = dE/dq``. This
quantity is exposed to i-PI via ``extras['workfunction_eV']`` in units of eV
and is used for interpreting constant-potential runs in terms of an
effective work function.

To remain compatible with the existing constant-potential (Fermi-level)
infrastructure, we also define an effective Fermi level as
``Ef = -Phi = -dE/dq`` (taking the vacuum electrostatic potential as the
zero of energy). The corresponding scalar is exported via
``extras['fermi_level']`` and ``extras['fermi_level_eV']``.
"""

import json
import numpy as np
try:
    from .dummy import Dummy_driver
    from ipi.utils import units
    from ipi.engine.potentiostat import ElectronicChargeError
except ImportError:
    from dummy import Dummy_driver
    from ipi.utils import units
    from ipi.engine.potentiostat import ElectronicChargeError

__DRIVER_NAME__  = "asym_dw_constant_potential"
__DRIVER_CLASS__ = "AsymDwConstantPotential_driver"

A2au = units.unit_to_internal("length", "angstrom", 1.0)      # 1 Å in a.u.
eV2au = units.unit_to_internal("energy", "electronvolt", 1.0)  # 1 eV in a.u.

# Default harmonic confinement strength (removed as requested - no k parameter)
DEFAULT_K_EV_PER_A2 = 0.0  # No confinement in y,z directions

# Minimum charge threshold - simulation stops if q falls below this value
Q_MIN = 1e-6

class AsymDwConstantPotential_driver(Dummy_driver):
    """Asymmetric double-well potential with constant potential capability.

    Implements the coupled potential: E(R,q) = V(R) + q*PZC + q^2/(2C)
    where V(R) is the asymmetric double-well in x direction:
    V(R) = a*(x-x0)^4 - b*(x-x0)^2 + d*(x-x0) + c

    Physical relations:
    - Total energy: E(R,q) = V(R) + q*PZC + q^2/(2C)
    - Atomic forces: F_r = -∂E/∂r = -∂V/∂r (PZC and C are constants here)
    - Work function (toy definition): Phi = ∂E/∂q = PZC + q/C
    - Effective Fermi level: Ef = -Phi = -(PZC + q/C)

    Thus, running *constant Fermi level* dynamics with a target Ef is
    equivalent to running a constant-workfunction simulation with target
    Phi = -Ef in this toy model (vacuum electrostatic potential set to 0).

    Usage for constant potential simulations:
        driver.set_electronic_state(q)  # Set current electronic charge
        pot, forces, virial, extras = driver(cell, pos)
        fermi_level = extras["fermi_level"]

    Parameters:
        a  (eV/Å^4) : Quartic coefficient (must be positive for stability)
        b  (eV/Å^2) : Quadratic coefficient (positive creates double-well)
        d  (eV/Å)   : Linear coefficient (creates asymmetry)
        x0 (Å)      : Reference position in x direction
        c  (eV)     : Constant energy offset
        q_default (e) : Default electron number if not set via set_electronic_state
    """

    def __init__(
        self,
        a=None,
        b=None,
        d=None,
        x0=None,
        c=None,
        q_default=None,
        pzc=None,
        cap=None,
        *args,
        **kwargs,
    ):
    
        if (a is None) or (b is None) or (d is None) or (x0 is None) or (c is None):
            # print("using default values: a=1.0 (eV/Å^4), b=3.0 (eV/Å^2), d=1.0 (eV/Å), x0=0.0 (Å), c=-10.0 (eV)")
            a, b, d, x0, c = 1.0, 3.0, 1.0, 0.0, -10.0
        
        if q_default is None:
            q_default = 0.0  # Default net charge for non-constant-potential simulations
            # print(f"using default electronic charge: q_default={q_default:.6f} (e)")

        if pzc is None:
            pzc = 5.0
            # print(f"using default PZC: pzc={pzc:.6f} (eV)")

        if cap is None:
            cap = 10.0
            # print(f"using default capacitance: cap={cap:.6f} (e^2/eV)")
        
        # Parameter validation
        if float(a) <= 0:
            raise ValueError("Parameter 'a' must be positive for potential stability")
        if float(b) <= 0:
            raise ValueError("Parameter 'b' must be positive for double-well formation")

        # Convert parameters to atomic units
        self.a  = float(a)  * (eV2au / (A2au**4))
        self.b  = float(b)  * (eV2au / (A2au**2))
        self.d  = float(d)  * (eV2au / A2au)
        self.c  = float(c)  * eV2au
        self.x0 = float(x0) * A2au
        # Store default electron number; internal net charge is defined as -electron_number
        self.q_default = float(q_default)

        # Constant electrostatic parameters (PZC and capacitance)
        # pzc: energy per unit charge (eV), cap: e^2/eV
        # Convert to atomic units so that Q*pzc and Q^2/(2*cap) yield energies in Hartree,
        # where Q is the net charge in units of e.
        self.pzc = float(pzc) * eV2au
        self.cap = float(cap) / eV2au

        # Electronic state - can be updated via set_electronic_state.
        # Interpret q_default as electron number; internal net charge is Q = -q_default.
        self.q_current = -self.q_default
        
        super().__init__(*args, **kwargs)

    def set_electronic_state(self, q):
        """Set the current electronic state for constant potential simulations.

        Args:
            q (float): Electron number from i-PI (ElectronicState.q).
                      Internally this driver uses the net charge
                      ``Q = -q`` (in units of e).
        """
        # q is the electron number (>0); convert to net charge Q = -q
        self.q_current = -float(q)

    def get_electronic_state(self):
        """Get the current electronic charge.
        
        Returns:
            float: Current electronic charge
        """
        return self.q_current

    def _compute_V_and_derivatives(self, positions_3d):
        """Compute V(R) and its derivatives in x direction.
        
        Args:
            positions_3d (ndarray): Atomic positions shaped (natoms, 3)
            
        Returns:
            tuple: (V_total, dV_dx_array) where V_total is scalar, 
                   dV_dx_array is (natoms,) array of x-force contributions
        """
        x_displacement = positions_3d[:, 0] - self.x0
        
        # V(R) per atom: a*(x-x0)^4 - b*(x-x0)^2 + d*(x-x0) + c
        V_per_atom = (self.a * x_displacement**4 - 
                      self.b * x_displacement**2 + 
                      self.d * x_displacement + 
                      self.c)
        
        # Total V(R)
        V_total = np.sum(V_per_atom, dtype=np.float64)
        
        # dV/dx per atom: 4*a*(x-x0)^3 - 2*b*(x-x0) + d
        dV_dx = (4.0 * self.a * x_displacement**3 - 
                 2.0 * self.b * x_displacement + 
                 self.d)
        
        return V_total, dV_dx

    def __call__(self, cell, pos):
        """Compute potential energy and forces for coupled E(R,q).
        
        Returns:
            tuple: (potential_energy, forces, virial, extras)
                   where extras contains {"fermi_level": Ef}
        """
        positions_3d = pos.reshape(-1, 3)

        # Store position data for get_fermi_level calculations
        self._last_position = positions_3d.copy()

        # Compute V(R) and its derivatives
        V_total, dV_dx = self._compute_V_and_derivatives(positions_3d)

        # Internal convention:
        #   q_electrons = number of electrons (from i-PI ElectronicState.q)
        #   Q_net       = -q_electrons  (net charge in units of e)
        Q_net = float(self.q_current)
        q_electrons = -Q_net

        # Total energy in terms of net charge Q:
        #   E(R,Q) = V(R) + Q*PZC + Q^2/(2C)
        potential_energy = V_total + Q_net * self.pzc + (Q_net * Q_net) / (2.0 * self.cap)

        # Work function / electronic driving force: Phi = dE/dQ = PZC + Q/C
        workfunction = self.pzc + Q_net / self.cap
        # Effective Fermi level: Ef = -Phi (vacuum potential taken as zero)
        fermi_level = -workfunction

        self._last_workfunction = workfunction
        self._last_fermi_level = fermi_level

        # Forces: F_r = -∂E/∂r = -∂V/∂r (PZC and C are constants here)
        force_3d = np.zeros(positions_3d.shape, dtype=np.float64)
        force_3d[:, 0] = -dV_dx  # x direction only
        # Explicitly ensure y and z forces are zero (no confinement in these directions)
        force_3d[:, 1] = 0.0  # y direction: explicitly zero
        force_3d[:, 2] = 0.0  # z direction: explicitly zero
        
        # Zero virial for this potential
        virial = cell * 0.0

        # Pass work function / dE/dq and effective Fermi level via extras.
        # - workfunction, workfunction_eV: Phi = dE/dq
        # - fermi_level, fermi_level_eV:  Ef = -Phi (vacuum potential = 0 reference)
        workfunction_eV = workfunction / eV2au
        fermi_level_eV = fermi_level / eV2au
        extras = {
            # Hartree and eV versions of the driving force dE/dQ and Ef = -Phi
            "fermi_level": fermi_level,
            "fermi_level_eV": fermi_level_eV,
            "workfunction": workfunction,
            "workfunction_eV": workfunction_eV,
            "V_total": V_total,
            # Net charge and electron number for diagnostics
            "q_current": Q_net,
            "q_electrons": q_electrons,
            "pzc": self.pzc,
            "cap": self.cap,
        }
        
        # Reshape to match input format
        forces = force_3d.reshape(pos.shape)
        
        return potential_energy, forces, virial, json.dumps(extras)
    
    def get_fermi_level(self):
        """Get the current effective Fermi level.

        For historical reasons the constant-potential interface in i-PI
        expects a method called ``get_fermi_level``. In this toy driver we
        return the *effective* Fermi level

            Ef = -Phi = -dE/dq = -(PZC + q/C),

        where Phi = dE/dq is the toy definition of the work function.
        """
        # Priority 1: Use cached value from last __call__ if available
        if hasattr(self, '_last_fermi_level') and self._last_fermi_level is not None:
            return self._last_fermi_level

        # Priority 2: Recalculate based on current position and charge
        if hasattr(self, '_last_position') and self._last_position is not None:
            Q_net = float(self.q_current)
            return -(self.pzc + Q_net / self.cap)

        # Priority 3: No calculation has been done yet
        else:
            # This should not happen in normal MD - forces should be calculated first
            raise RuntimeError("get_fermi_level() called before any PES calculation. "
                             "Forces must be calculated first to establish current state.")