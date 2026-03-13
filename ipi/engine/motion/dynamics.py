"""Contains the classes that deal with the different dynamics required in
different types of ensembles.

Holds the algorithms required for normal mode propagators, and the objects to
do the constant temperature and pressure algorithms. Also calculates the
appropriate conserved energy quantity for the ensemble of choice.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import math

import numpy as np

from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat
from ipi.engine.potentiostat import ElectronicState, Potentiostat
from ipi.utils.softexit import softexit
from ipi.utils.messages import warning, verbosity
from ipi.utils.units import Constants


class Dynamics(Motion):
    """self (path integral) molecular dynamics class.

    Gives the standard methods and attributes needed in all the
    dynamics classes.

    Attributes:
        beads: A beads object giving the atoms positions.
        cell: A cell object giving the system box.
        forces: A forces object giving the virial and the forces acting on
            each bead.
        prng: A random number generator object.
        nm: An object which does the normal modes transformation.

    Depend objects:
        econs: The conserved energy quantity appropriate to the given
            ensemble. Depends on the various energy terms which make it up,
            which are different depending on the ensemble.he
        temp: The system temperature.
        dt: The timestep for the algorithms.
        ntemp: The simulation temperature. Will be nbeads times higher than
            the system temperature as PIMD calculations are done at this
            effective classical temperature.
    """

    def __init__(
        self,
        timestep,
        mode="nve",
        splitting="obabo",
        thermostat=None,
        barostat=None,
        fixcom=False,
        fixatoms_dof=None,
        nmts=None,
        efield=None,
        bec=None,
        electrons=None,
    ):
        """Initialises a "dynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """

        super(Dynamics, self).__init__(fixcom=fixcom, fixatoms_dof=fixatoms_dof)

        # initialize time step. this is the main time step that covers a full time step
        self._dt = depend_value(name="dt", value=timestep)
        
        # Initialize simulation step counter for linear ramping
        self._simulation_step = 0

        if thermostat is None:
            self.thermostat = Thermostat()
        else:
            # if (
            #     thermostat.__class__.__name__ is ("ThermoPILE_G" or "ThermoNMGLEG ")
            # ) and (len(fixatoms_dof) > 0):
            if (
                thermostat.__class__.__name__ in ("ThermoPILE_G", "ThermoNMGLEG")
            ) and (len(fixatoms_dof) > 0):
                softexit.trigger(
                    status="bad",
                    message="!! Sorry, fixed atoms and global thermostat on the centroid not supported. Use a local thermostat. !!",
                )
            self.thermostat = thermostat

        if nmts is None or len(nmts) == 0:
            self._nmts = depend_array(name="nmts", value=np.asarray([1], int))
        else:
            self._nmts = depend_array(name="nmts", value=np.asarray(nmts, int))

        if barostat is None:
            self.barostat = Barostat()
        else:
            self.barostat = barostat
        self.enstype = mode
        if self.enstype == "nve":
            self.integrator = NVEIntegrator()
        elif self.enstype == "nvt":
            self.integrator = NVTIntegrator()
        elif self.enstype == "nvt-cc":
            self.integrator = NVTCCIntegrator()
        elif self.enstype == "npt":
            self.integrator = NPTIntegrator()
        elif self.enstype == "nst":
            self.integrator = NSTIntegrator()
        elif self.enstype == "sc":
            self.integrator = SCIntegrator()
        elif self.enstype == "scnpt":
            self.integrator = SCNPTIntegrator()
        else:
            self.integrator = DummyIntegrator()

        # splitting mode for the integrators
        self._splitting = depend_value(name="splitting", value=splitting)

        # constraints
        self.fixcom = fixcom
        if fixatoms_dof is None:
            self.fixatoms_dof = np.zeros(0, int)
        else:
            self.fixatoms_dof = fixatoms_dof
        
        # electronic degrees of freedom for constant potential simulation
        # Only initialize if electrons config is provided
        if electrons is not None:
            self.electrons_config = electrons
            self.electronic_state = None
            self.potentiostat = None
        else:
            # No electrons config - maintain original behavior
            self.electrons_config = None
            self.electronic_state = None
            self.potentiostat = None

    def get_fixdof(self):
        """Calculate the number of fixed degrees of freedom, required for
        temperature and pressure calculations.
        """

        fixdof = len(self.fixatoms_dof) * self.beads.nbeads
        if self.fixcom:
            fixdof += 3
        return fixdof

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        """Binds ensemble beads, cell, bforce, and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
            beads: The beads object from which the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.
        """

        super(Dynamics, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        # Checks if the number of mts levels is equal to the dimensionality of the mts weights.
        if len(self.nmts) != self.forces.nmtslevels:
            raise ValueError(
                "The number of mts levels for the integrator does not agree with the mts_weights of the force components."
            )

        # n times the temperature (for path integral partition function)
        self._ntemp = depend_value(
            name="ntemp", func=self.get_ntemp, dependencies=[self.ensemble._temp]
        )

        # fixed degrees of freedom count
        fixdof = self.get_fixdof()

        # first makes sure that the thermostat has the correct temperature and timestep, then proceeds with binding it.
        dpipe(self._ntemp, self.thermostat._temp)

        # depending on the kind, the thermostat might work in the normal mode or the bead representation.
        self.thermostat.bind(beads=self.beads, nm=self.nm, prng=prng, fixdof=fixdof)

        # Initialize electronic degrees of freedom if config is provided and enabled
        if self.electrons_config is not None and self.electrons_config.get("enabled", False):
            import ipi.engine.potentiostat as epotentiostat
            from ipi.utils.messages import info, verbosity
            
            q_init = self.electrons_config.get("q_init", 1.0)
            from_restart = self.electrons_config.get("_from_restart", False)
            
            # Log source of electronic charge
            if from_restart:
                # info(f" @ELECTRONS: Using electronic charge q = {q_init:.6f} from RESTART file", verbosity.medium)
            else:
                # info(f" @ELECTRONS: Using electronic charge q = {q_init:.6f} from input.xml", verbosity.medium)
            
            # Create electronic state using the InputElectrons method to handle linear ramping
            from ipi.inputs.electrons import InputElectrons
            input_electrons = InputElectrons()
            input_electrons.store(self.electrons_config)
            self.electronic_state = input_electrons.create_electronic_state()
            
            # Get potentiostat object from configuration (it's already created by InputElectrons.fetch())
            potentiostat_obj = self.electrons_config.get("potentiostat", None)
            if potentiostat_obj is None:
                raise ValueError("No potentiostat found in electrons configuration")
                
            # Assign the pre-created potentiostat object
            self.potentiostat = potentiostat_obj

            # Configure potentiostat temperature: use user value if provided, else fall back to ntemp
            pot_temp = self.electrons_config.get("potentiostat_temp", float("nan"))
            if math.isnan(pot_temp):
                pot_temp = getattr(self.potentiostat, "_input_temp", float("nan"))
            if math.isnan(pot_temp):
                # Fall back to the path-integral temperature (n * T)
                pot_temp = float(self.ntemp)

            # Use the public property so the underlying depend_value is updated correctly
            self.potentiostat.temp = float(pot_temp)

            dpipe(self._dt, self.potentiostat._dt)  # Use same timestep
            self.potentiostat.bind(self.electronic_state, prng)
            
            # Propagate neutral_electrons from the electronic_state to all
            # forcefield instances that support this attribute (e.g.
            # FFMixTwoSockets). This must happen before the first call to
            # set_electronic_state, otherwise charge initialization in the
            # forcefields will fail due to missing neutral_electrons.
            if hasattr(self.electronic_state, "neutral_electrons"):
                ne = int(self.electronic_state.neutral_electrons)
                for fcomp in self.forces.mforces:
                    for fb in fcomp._forces:
                        ff = getattr(fb, "ff", None)
                        if ff is not None and hasattr(ff, "neutral_electrons"):
                            ff.neutral_electrons = ne

            # Initialize step 0 Fermi level by synchronizing charge and getting initial Fermi level
            self._initialize_step0_fermi_level()

        # first makes sure that the barostat has the correct stress and timestep, then proceeds with binding it.
        dpipe(self._ntemp, self.barostat._temp)
        dpipe(self.ensemble._pext, self.barostat._pext)
        dpipe(self.ensemble._stressext, self.barostat._stressext)
        self.barostat.bind(
            beads,
            nm,
            cell,
            bforce,
            bias=self.ensemble.bias,
            prng=prng,
            fixdof=fixdof,
            nmts=len(self.nmts),
        )

        self.integrator.bind(self)

        self.ensemble.add_econs(self.thermostat._ethermo)
        self.ensemble.add_econs(self.barostat._ebaro)

        # adds the potential, kinetic energy and the cell Jacobian to the ensemble
        self.ensemble.add_xlpot(self.barostat._pot)
        self.ensemble.add_xlpot(self.barostat._cell_jacobian)
        self.ensemble.add_xlkin(self.barostat._kin)

        # applies constraints immediately after initialization.
        self.integrator.pconstraints()

        # TODO THOROUGH CLEAN-UP AND CHECK
        if (
            self.enstype == "nvt"
            or self.enstype == "nvt-cc"
            or self.enstype == "npt"
            or self.enstype == "nst"
        ):
            if self.ensemble.temp < 0:
                raise ValueError(
                    "Negative or unspecified temperature for a constant-T integrator"
                )
            if self.enstype == "npt":
                if type(self.barostat) is Barostat:
                    raise ValueError(
                        "The barostat and its mode have to be specified for constant-p integrators"
                    )
                if np.allclose(self.ensemble.pext, -12345):
                    raise ValueError("Unspecified pressure for a constant-p integrator")
            elif self.enstype == "nst":
                if np.allclose(self.ensemble.stressext.diagonal(), -12345):
                    raise ValueError("Unspecified stress for a constant-s integrator")
        if self.enstype == "nve" and self.beads.nbeads > 1:
            if self.ensemble.temp < 0:
                raise ValueError(
                    "You need to provide a positive value for temperature inside ensemble to run a PIMD simulation, even when choosing NVE propagation."
                )

        # 避免依赖depend机制的直接费米能级缓存
        self._fermi_cache = {}  # 费米能级缓存字典
        self._fermi_cache_valid = False  # 缓存有效性标志

    def get_ntemp(self):
        """Returns the PI simulation temperature (P times the physical T)."""

        return self.ensemble.temp * self.beads.nbeads

    def _clear_fermi_cache(self):
        """清理费米能级缓存"""
        self._fermi_cache.clear()
        self._fermi_cache_valid = False

    def _read_ef_or_die(self):
        """安全地从forces.extras读取费米能级，读取失败就报错"""
        try:
            # 尝试读取extras
            if hasattr(self.forces, 'extras') and self.forces.extras is not None:
                extras = self.forces.extras
                if isinstance(extras, dict) and 'fermi_level_eV' in extras:
                    return float(extras['fermi_level_eV'])

            # 如果读取失败，抛出错误
            raise RuntimeError(
                "Could not read Fermi level from forces.extras. "
                "Force providers must provide 'fermi_level_eV' (eV) or 'fermi_level' (Hartree) in extras."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read Fermi level: {e}")

    def step(self, step=None):
        """Advances the dynamics by one time step"""

        self.integrator.step(step)
        self.ensemble.time += self.dt  # increments internal time

    def _initialize_step0_fermi_level(self):
        """Initialize step 0 Fermi level by syncing charge and getting initial Fermi level.
        
        This ensures that the initial (step 0) Fermi level is correctly retrieved from
        the potential energy surface after setting the initial electronic charge.
        """
        if (self.electrons_config is None or
            not hasattr(self, 'electronic_state') or
            self.electronic_state is None):
            return

        # Lightweight initialization: just sync charge without forcing calculation
        # The first B step will naturally trigger force calculation and get Fermi level
        try:
            # Sync electronic charge to force providers (all beads)
            self._sync_electronic_charge()

            # Set initial Fermi level to 0.0 for output - will be updated in first B step
            self.electronic_state.current_ef = 0.0

        except Exception as e:
            # If we can't sync charge, warn but don't crash
            from ipi.utils.messages import warning, verbosity
            warning(f"Could not sync electronic charge: {e}.", verbosity.low)
            self.electronic_state.current_ef = 0.0

    def _sync_electronic_charge(self):
        """Sync electronic charge to force providers without forcing calculation."""
        if (self.electrons_config is None or
            not hasattr(self, 'electronic_state') or
            self.electronic_state is None):
            return

        q_current = self.electronic_state.q
        for fcomp in self.forces.mforces:
            # Each force component has a list of _forces (one per bead)
            for fb in fcomp._forces:
                # Each ForceBead has a ff (forcefield) attribute
                if hasattr(fb.ff, 'set_electronic_state'):
                    fb.ff.set_electronic_state(q_current)
    

    def _get_fermi_level_from_forces(self):
        """Get Fermi level from forces.extras using depend mechanism.

        This method is called by the depend object and has the same update timing as forces.
        """
        # Get extras from forces object
        extras = getattr(self.forces, 'extras', None)

        # Handle dictionary format (preferred and required)
        # Priority: fermi_level_eV (in eV units) > fermi_level (in Hartree units)
        if isinstance(extras, dict):
            ef = None
            if 'fermi_level_eV' in extras:
                # Prefer fermi_level_eV for consistent units
                ef = extras['fermi_level_eV']
            elif 'fermi_level_au' in extras:
                # Explicit atomic units -> convert to eV
                ef = extras['fermi_level_au'] * Constants.EV_PER_HARTREE
            elif 'fermi_level' in extras:
                # Fallback to fermi_level; treat as Hartree unless explicit flag says otherwise
                ef = extras['fermi_level'] * Constants.EV_PER_HARTREE  # Hartree to eV conversion

            if ef is not None:
                # Ensure ef is scalar
                if hasattr(ef, '__iter__') and not isinstance(ef, str):
                    ef = float(ef[0]) if len(ef) > 0 else 0.0
                else:
                    ef = float(ef)
                if np.isfinite(ef):
                    return ef
                else:
                    raise RuntimeError(f"Invalid Fermi level in forces.extras: {ef} (not finite)")

        # Handle JSON string format (for backward compatibility only)
        if isinstance(extras, str):
            try:
                import json
                extras_dict = json.loads(extras)
                ef = None
                if 'fermi_level_eV' in extras_dict:
                    # Prefer fermi_level_eV for consistent units
                    ef = extras_dict['fermi_level_eV']
                elif 'fermi_level_au' in extras_dict:
                    ef = extras_dict['fermi_level_au'] * Constants.EV_PER_HARTREE
                elif 'fermi_level' in extras_dict:
                    # Fallback to fermi_level (convert from Hartree to eV)
                    ef = extras_dict['fermi_level'] * Constants.EV_PER_HARTREE  # Hartree to eV conversion

                if ef is not None:
                    # Ensure ef is scalar
                    if hasattr(ef, '__iter__') and not isinstance(ef, str):
                        ef = float(ef[0]) if len(ef) > 0 else 0.0
                    else:
                        ef = float(ef)
                    if np.isfinite(ef):
                        return ef
                    else:
                        raise RuntimeError(f"Invalid Fermi level in JSON extras: {ef} (not finite)")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                raise RuntimeError(f"Invalid JSON format in forces.extras: {e}")

        # Failure: no valid Fermi level found
        raise RuntimeError(
            "Missing Fermi level in forces.extras. "
            "Force providers must set: forces.extras = {'fermi_level': value}"
        )

    def _update_fermi_cache(self):
        """Update Fermi level cache after force calculation.

        This method should be called after each atomic force calculation
        to cache the Fermi level for use by electronic B steps.
        """
        if not hasattr(self, 'electrons_config') or self.electrons_config is None:
            return

        try:
            # Read Fermi level from forces.extras - this should NOT trigger calculation
            # because forces were just calculated
            fermi_level = self._get_fermi_level_from_forces()

            # Cache the Fermi level for electronic B steps
            for integrator in [getattr(self, 'integrator', None)]:
                if integrator is not None:
                    integrator._cached_fermi_level = fermi_level
                    integrator._cached_fermi_valid = True

        except Exception as e:
            # If we can't read Fermi level, invalidate cache
            from ipi.utils.messages import warning
            warning(f"Could not update Fermi cache: {e}")
            for integrator in [getattr(self, 'integrator', None)]:
                if integrator is not None:
                    integrator._cached_fermi_valid = False


dproperties(Dynamics, ["dt", "nmts", "splitting", "ntemp"])


class DummyIntegrator:
    """No-op integrator for (PI)MD"""

    def __init__(self):
        pass

    def get_qdt(self):
        return self.dt * 0.5 / self.inmts

    def get_pdt(self):
        dtl = 1.0 / self.nmts
        for i in range(1, len(dtl)):
            dtl[i] *= dtl[i - 1]
        dtl *= self.dt * 0.5
        return dtl

    def get_tdt(self):
        if self.splitting == "obabo":
            return self.dt * 0.5
        elif self.splitting == "baoab":
            return self.dt
        else:
            raise ValueError(
                "Invalid splitting requested. Only OBABO and BAOAB are supported."
            )

    def bind(self, motion):
        """Reference all the variables for simpler access."""

        self.dynamics = motion  # Keep reference to dynamics object for electronic methods
        self.beads = motion.beads
        self.bias = motion.ensemble.bias
        self.ensemble = motion.ensemble
        self.forces = motion.forces
        self.prng = motion.prng
        self.nm = motion.nm
        self.thermostat = motion.thermostat
        self.barostat = motion.barostat
        self.fixcom = motion.fixcom
        self.fixatoms_dof = motion.fixatoms_dof
        self.enstype = motion.enstype
        
        # Bind electronic degrees of freedom if present
        if hasattr(motion, 'electrons_config'):
            self.electrons_config = motion.electrons_config
        else:
            self.electrons_config = None
        if hasattr(motion, 'electronic_state'):
            self.electronic_state = motion.electronic_state
        else:
            self.electronic_state = None
        if hasattr(motion, 'potentiostat'):
            self.potentiostat = motion.potentiostat
        else:
            self.potentiostat = None

        # no need to dpipe these are really just references
        self._splitting = motion._splitting
        self._dt = motion._dt
        self._nmts = motion._nmts

        # Initialize Fermi level cache - no depend mechanism
        if self.electrons_config is not None:
            self._cached_fermi_level = None
            self._cached_fermi_valid = False

        # check whether fixed indexes make sense
        if np.any(self.fixatoms_dof >= (3 * self.beads.natoms)):
            raise ValueError(
                "Constrained indexes are out of bounds wrt. number of atoms."
            )

        # calculate active dofs
        if len(self.fixatoms_dof) > 0:
            full_indices = np.arange(3 * self.beads.natoms)
            # in the next line, we check whether the fixed indexes are in the full indexes and invert the boolean result with the tilde
            self.activeatoms_mask = ~np.isin(full_indices, self.fixatoms_dof)
        else:
            self.activeatoms_mask = False

        # total number of iteration in the inner-most MTS loop
        self._inmts = depend_value(name="inmts", func=lambda: np.prod(self.nmts))
        self._nmtslevels = depend_value(name="nmtslevels", func=lambda: len(self.nmts))
        # these are the time steps to be used for the different parts of the integrator
        self._qdt = depend_value(
            name="qdt",
            func=self.get_qdt,
            dependencies=[self._splitting, self._dt, self._inmts],
        )  # positions
        self._qdt_on_m = depend_array(
            name="qdt_on_m",
            value=np.zeros(3 * self.beads.natoms),
            func=lambda: self.qdt / dstrip(self.beads.m3)[0],
        )
        self._pdt = depend_array(
            name="pdt",
            func=self.get_pdt,
            value=np.zeros(len(self.nmts)),
            dependencies=[self._splitting, self._dt, self._nmts],
        )  # momenta
        self._tdt = depend_value(
            name="tdt",
            func=self.get_tdt,
            dependencies=[self._splitting, self._dt, self._nmts],
        )  # thermostat

        dpipe(self._qdt, self.nm._dt)
        dpipe(self._dt, self.barostat._dt)
        dpipe(self._qdt, self.barostat._qdt)
        dpipe(self._pdt, self.barostat._pdt)
        dpipe(self._tdt, self.barostat._tdt)
        dpipe(self._tdt, self.thermostat._dt)

        if motion.enstype == "sc" or motion.enstype == "scnpt":
            # coefficients to get the (baseline) trotter to sc conversion
            self.coeffsc = np.ones((self.beads.nbeads, 3 * self.beads.natoms), float)
            self.coeffsc[::2] /= -3.0
            self.coeffsc[1::2] /= 3.0

        # check stress tensor
        self._stresscheck = True

    def pstep(self):
        """Dummy momenta propagator which does nothing."""
        pass

    def qcstep(self):
        """Dummy centroid position propagator which does nothing."""
        pass

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def electronic_B_step(self, dt, force_recalc=False):
        """Electronic momentum kick step using cached Fermi level.

        This step NEVER triggers force calculations. It only reads the Fermi level
        that was cached after the last atomic force calculation.

        Args:
            dt: Time step
            force_recalc: Ignored - this step never triggers calculations
        """
        # Skip if no electrons config or electronic state not initialized
        if (not hasattr(self.dynamics, 'electrons_config') or
            self.dynamics.electrons_config is None or
            not hasattr(self.dynamics, 'electronic_state') or
            self.dynamics.electronic_state is None):
            return

        # Sync electronic charge to force providers (without forcing calculation)
        self.dynamics._sync_electronic_charge()

        # Ensure Fermi cache is populated: if empty, try to read from forces.extras
        if not self.dynamics._fermi_cache_valid or len(self.dynamics._fermi_cache) == 0:
            try:
                fermi_level = self.dynamics._get_fermi_level_from_forces()
                self.dynamics._fermi_cache = {"vasp_fermi": fermi_level}
                self.dynamics._fermi_cache_valid = True
            except Exception:
                raise RuntimeError(
                    "Electronic B step requires valid Fermi cache, but no cached Fermi levels found. "
                    "This indicates that force evaluation did not properly cache Fermi levels. "
                    "Check that force calculation completed successfully and that Fermi level extras are present."
                )
        else:
            fermi_level = next(iter(self.dynamics._fermi_cache.values()))

        from ipi.utils.messages import info, verbosity
        # info(f" @ELECTRONIC_B_STEP: Using cached Fermi level: {fermi_level:.6f} eV", verbosity.medium)

        # Update electronic state's current Fermi level for output
        # CRITICAL: current_ef must be in atomic units (Hartree), but cached value is in eV
        # Convert eV to Hartree for i-PI's property output system
        fermi_level_au = fermi_level / Constants.EV_PER_HARTREE  # eV to Hartree conversion
        self.dynamics.electronic_state.current_ef = fermi_level_au

        # info(f" @ELECTRONIC_B_STEP: Set current_ef = {fermi_level_au:.6f} Hartree for property output", verbosity.medium)

        # Apply electronic momentum kick using cached Fermi level
        if hasattr(self.dynamics, 'potentiostat') and self.dynamics.potentiostat is not None:
            self.dynamics.potentiostat.half_B(dt, fermi_level)
    
    def electronic_A_step(self, dt):
        """Electronic position drift step."""
        # Skip if no electrons config or electronic state not initialized
        if (not hasattr(self.dynamics, 'electrons_config') or 
            self.dynamics.electrons_config is None or
            not hasattr(self.dynamics, 'electronic_state') or
            self.dynamics.electronic_state is None):
            return

        if hasattr(self.dynamics, 'potentiostat') and self.dynamics.potentiostat is not None:
            # Store previous charge to detect changes
            old_q = self.dynamics.electronic_state.q if hasattr(self.dynamics.electronic_state, 'q') else None
            
            # Single step - no sub-stepping to avoid multiple force calculations
            self.dynamics.potentiostat.A(dt)
            
            # Depend mechanism automatically handles Fermi level updates - no manual cache clearing needed

    def pconstraints(self):
        """This removes the centre of mass contribution to the kinetic energy.

        Calculates the centre of mass momenta, then removes the mass weighted
        contribution from each atom. If the ensemble defines a thermostat, then
        the contribution to the conserved quantity due to this subtraction is
        added to the thermostat heat energy, as it is assumed that the centre of
        mass motion is due to the thermostat.

        If there is a choice of thermostats, the thermostat
        connected to the centroid is chosen.
        """

        beads = self.beads
        if self.fixcom:
            nb = beads.nbeads
            p = dstrip(beads.p)
            m3 = dstrip(beads.m3).reshape((-1, 3))
            M = beads[0].M
            Mnb = M * nb

            vcom = np.sum(p.reshape(-1, 3), axis=0) / Mnb
            beads.p -= (m3 * vcom).reshape(nb, -1)

            self.ensemble.eens += np.sum(vcom**2) * 0.5 * Mnb  # COM kinetic energy.

        if len(self.fixatoms_dof) > 0:
            m3 = dstrip(beads.m3)
            p = dstrip(beads.p)
            self.ensemble.eens += 0.5 * np.sum(
                p[:, self.fixatoms_dof] ** 2 / m3[:, self.fixatoms_dof]
            )
            beads.p[:, self.fixatoms_dof] = 0.0


dproperties(
    DummyIntegrator,
    ["splitting", "nmts", "dt", "inmts", "nmtslevels", "qdt", "pdt", "tdt", "qdt_on_m"],
)


class NVEIntegrator(DummyIntegrator):
    """Integrator object for constant energy simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant energy ensemble. Note that a temperature of some kind must be
    defined so that the spring potential can be calculated.

    Attributes:
        ptime: The time taken in updating the velocities.
        qtime: The time taken in updating the positions.
        ttime: The time taken in applying the thermostat steps.

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, and the spring potential energy.
    """

    def pstep(self, level=0):
        """Velocity Verlet momentum propagator."""

        # halfdt/alpha
        if len(self.fixatoms_dof) > 0:
            self.beads.p[:, self.activeatoms_mask] += (
                dstrip(self.forces.mts_forces[level].f)[:, self.activeatoms_mask]
                * self.pdt[level]
            )
            if level == 0 and self.ensemble.has_bias:  # adds bias in the outer loop
                self.beads.p[:, self.activeatoms_mask] += (
                    dstrip(self.bias.f)[:, self.activeatoms_mask] * self.pdt[level]
                )
        else:
            self.beads.p[:] += dstrip(self.forces.mts_forces[level].f) * self.pdt[level]
            if level == 0 and self.ensemble.has_bias:  # adds bias in the outer loop
                self.beads.p[:] += dstrip(self.bias.f) * self.pdt[level]

    def qcstep(self):
        """Velocity Verlet centroid position propagator."""
        # dt/inmts
        self.nm.qnm[0, :] += dstrip(self.nm.pnm)[0, :] * dstrip(self.qdt_on_m)

    # now the idea is that for BAOAB the MTS should work as follows:
    # take the BAB MTS, and insert the O in the very middle. This might imply breaking a A step in two, e.g. one could have
    # Bbabb(a/2) O (a/2)bbabB
    def mtsprop_ba(self, index):
        """Recursive MTS step"""

        mk = int(self.nmts[index] / 2)

        for i in range(mk):  # do nmts/2 full sub-steps
            self.pstep(index)
            self.pconstraints()
            # Skip electronic B step here if NVTIntegrator is handling electronic splitting properly
            # This avoids redundant force calculations that violate lazy update design
            # Electronic B step removed from MTS inner layer to avoid conflicts with outer layer electronic BAOAB
                
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.qcstep()
                # Electronic A step removed from MTS inner layer
                self.nm.free_qstep()
                # Depend mechanism automatically handles Fermi level updates
                self.qcstep()
                # Electronic A step removed from MTS inner layer
                self.nm.free_qstep()
                # Depend mechanism automatically handles Fermi level updates

            else:
                self.mtsprop(index + 1)

            self.pstep(index)
            self.pconstraints()
            # Skip electronic B step here if NVTIntegrator is handling electronic splitting properly
            # This avoids redundant force calculations that violate lazy update design
            # Electronic B step removed from MTS inner layer to avoid conflicts with outer layer electronic BAOAB

        if self.nmts[index] % 2 == 1:
            # propagate p for dt/2alpha with force at level index
            self.pstep(index)
            self.pconstraints()
            # Skip electronic B step here if NVTIntegrator is handling electronic splitting properly
            # This avoids redundant force calculations that violate lazy update design
            # Electronic B step removed from MTS inner layer to avoid conflicts with outer layer electronic BAOAB
                
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.qcstep()
                # Electronic A step removed from MTS inner layer
                self.nm.free_qstep()
                # Depend mechanism automatically handles Fermi level updates
            else:
                self.mtsprop_ba(index + 1)

    def mtsprop_ab(self, index):
        """Recursive MTS step"""

        if self.nmts[index] % 2 == 1:
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.qcstep()
                # Electronic A step removed from MTS inner layer
                self.nm.free_qstep()
            else:
                self.mtsprop_ab(index + 1)

            # propagate p for dt/2alpha with force at level index
            self.pstep(index)
            self.pconstraints()
            # Skip electronic B step here if NVTIntegrator is handling electronic splitting properly
            # This avoids redundant force calculations that violate lazy update design
            # Electronic B step removed from MTS inner layer to avoid conflicts with outer layer electronic BAOAB

        for i in range(int(self.nmts[index] / 2)):  # do nmts/2 full sub-steps
            self.pstep(index)
            self.pconstraints()
            # Skip electronic B step here if NVTIntegrator is handling electronic splitting properly
            # This avoids redundant force calculations that violate lazy update design
            # Electronic B step removed from MTS inner layer to avoid conflicts with outer layer electronic BAOAB
                
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.qcstep()
                # Electronic A step removed from MTS inner layer
                self.nm.free_qstep()
                # Depend mechanism automatically handles Fermi level updates
                self.qcstep()
                # Electronic A step removed from MTS inner layer
                self.nm.free_qstep()
                # Depend mechanism automatically handles Fermi level updates
            else:
                self.mtsprop(index + 1)

            self.pstep(index)
            self.pconstraints()
            # Skip electronic B step here if NVTIntegrator is handling electronic splitting properly
            # This avoids redundant force calculations that violate lazy update design
            # Electronic B step removed from MTS inner layer to avoid conflicts with outer layer electronic BAOAB

    def mtsprop(self, index):
        # just calls the two pieces together
        self.mtsprop_ba(index)
        self.mtsprop_ab(index)

    def step(self, step=None):
        """Does one simulation time step."""

        self.mtsprop(0)


class NVTIntegrator(NVEIntegrator):
    """Integrator object for constant temperature simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.
    """

    def tstep(self):
        """Velocity Verlet thermostat step"""

        self.thermostat.step()
    

    def step(self, step=None):
        """Does one simulation time step."""
        from ipi.utils.messages import info, verbosity
        # info(f"[DEBUG-STEP] Starting step method, splitting={self.splitting}", verbosity.medium)

        dt = self.dt
        dt2 = 0.5 * dt
        
        # NOTE: Do NOT clear Fermi cache at step beginning - B1 needs the cached value
        # Cache will be cleared only after atomic A steps (position updates)
        
        # Check if we have electronic degrees of freedom
        has_electrons = (self.electrons_config is not None and
                        hasattr(self, 'electronic_state') and
                        self.electronic_state is not None and
                        hasattr(self, 'potentiostat') and
                        self.potentiostat is not None)

        # Debug: Print detailed has_electrons check
        from ipi.utils.messages import info, verbosity
        # info(f" @HAS_ELECTRONS_CHECK: electrons_config={self.electrons_config is not None}, "
        #      f"electronic_state={hasattr(self, 'electronic_state') and self.electronic_state is not None}, "
        #      f"potentiostat_attr={hasattr(self, 'potentiostat')}, "
        #      f"potentiostat_obj={hasattr(self, 'potentiostat') and self.potentiostat is not None}, "
        #      f"final_result={has_electrons}", verbosity.medium)
        
        # Update target Fermi level for linear ramping mode
        if has_electrons and hasattr(self.electronic_state, 'update_target_fermi_level'):
            current_step = getattr(self, '_simulation_step', 0)
            self.electronic_state.update_target_fermi_level(current_step)
            # Increment step counter
            self._simulation_step = current_step + 1

        # Determine electronic integration scheme based on potentiostat type
        # Electronic thermostat uses its own splitting, independent of atomic splitting
        # All electronic thermostats use BAOAB splitting for consistency
        electronic_splitting = "none"
        if has_electrons:
            potentiostat_type = type(self.potentiostat).__name__
            from ipi.utils.messages import info, verbosity
            # info(f" @ELECTRONIC_SPLITTING: has_electrons={has_electrons}, potentiostat_type='{potentiostat_type}'", verbosity.medium)
            if potentiostat_type in ["PotentiostatLangevin", "PotentiostatSVR"]:
                electronic_splitting = "baoab"  # Always use BAOAB for all electronic thermostats
                # info(f" @ELECTRONIC_SPLITTING: Set electronic_splitting='{electronic_splitting}'", verbosity.medium)
            else:
                # info(f" @ELECTRONIC_SPLITTING: Potentiostat type '{potentiostat_type}' not recognized", verbosity.medium)


        
        if self.splitting == "obabo":
            # OBABO+baoab: OBAbaoaABbO
            # Force calculation triggered by dependency network in final B/b steps

            # === O(dt/2): Atomic thermostat half-step ===
            self.tstep()
            self.pconstraints()

            # === BA: Atomic momentum and position first half ===
            self.mtsprop_ba(0)  # B: atomic momentum, A: atomic position

            # === bao: Electronic momentum, position, and thermostat ===
            if electronic_splitting == "baoab":
                from ipi.utils.messages import info, verbosity
                # info(f" @ELECTRONIC_B_STEP: Calling electronic_B_step with dt={dt2}", verbosity.medium)
                self.electronic_B_step(dt2, force_recalc=False)  # b: electronic momentum [uses cached Fermi level if valid]
                self.electronic_A_step(dt2)  # a: electronic position [charge changes]
                self.potentiostat.O_step(dt)  # o: electronic thermostat

            # === aA: Position steps (second half) ===
            if electronic_splitting == "baoab":
                self.electronic_A_step(dt2)  # a: electronic position [charge changes again]

            # A: atomic position second half is embedded in mtsprop_ab
            self.mtsprop_ab(0)  # Contains final atomic B step

            # === b: Electronic momentum final step ===
            # Uses cached Fermi level from atomic force calculation
            if electronic_splitting == "baoab":
                self.electronic_B_step(dt2, force_recalc=False)  # b: electronic momentum [uses cached Fermi level]

            # === O(dt/2): Atomic thermostat half-step ===
            self.tstep()
            self.pconstraints()

        elif self.splitting == "baoab":
            # BAOAB+baoab: BAbaOoaABb
            # Force calculation triggered by dependency network in B/b steps

            # === BA: Atomic momentum and position half-steps ===
            self.mtsprop_ba(0)  # B: atomic momentum, A: atomic position

            # === ba: Electronic momentum and position half-steps ===
            if electronic_splitting == "baoab":
                self.electronic_B_step(dt2, force_recalc=False)  # b: electronic momentum [uses cached Fermi level if valid]
                self.electronic_A_step(dt2)  # a: electronic position [charge changes]

            # === Oo: All thermostat steps ===
            # Atomic thermostat
            self.tstep()
            self.pconstraints()

            if electronic_splitting == "baoab":
                # Electronic thermostat
                self.potentiostat.O_step(dt)  # o: electronic thermostat

            # === aA: Position half-steps (second half) ===
            if electronic_splitting == "baoab":
                self.electronic_A_step(dt2)  # a: electronic position [charge changes again]

            # Atomic position update (second part)
            self.mtsprop_ab(0)  # A: atomic position

            # === Bb: Momentum half-steps (final) ===
            # Uses cached Fermi level from atomic force calculation
            if electronic_splitting == "baoab":
                self.electronic_B_step(dt2, force_recalc=False)  # b: electronic momentum [uses cached Fermi level]


class NVTCCIntegrator(NVTIntegrator):
    """Integrator object for constant temperature simulations with constrained centroid.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.
    """

    def pstep(self):
        """Velocity Verlet momenta propagator."""

        # propagates in NM coordinates
        self.nm.pnm += dstrip(self.nm.fnm) * (self.dt * 0.5)
        # self.beads.p += dstrip(self.forces.f)*(self.dt*0.5)
        # also adds the bias force
        # self.beads.p += dstrip(self.bias.f)*(self.dt*0.5)

    def step(self, step=None):
        """Does one simulation time step."""

        self.thermostat.step()
        self.pconstraints()
        # NB we only have to take into account the energy balance of zeroing centroid velocity when we had added energy through the thermostat
        self.ensemble.eens += 0.5 * np.dot(
            self.nm.pnm[0], self.nm.pnm[0] / self.nm.dynm3[0]
        )
        self.nm.pnm[0, :] = 0.0

        self.pstep()
        self.nm.pnm[0, :] = 0.0
        self.pconstraints()

        # self.qcstep() # for the moment I just avoid doing the centroid step.
        self.nm.free_qstep()

        self.pstep()
        self.nm.pnm[0, :] = 0.0
        self.pconstraints()

        self.thermostat.step()
        self.ensemble.eens += 0.5 * np.dot(
            self.nm.pnm[0], self.nm.pnm[0] / self.nm.dynm3[0]
        )
        self.nm.pnm[0, :] = 0.0
        self.pconstraints()


class NPTIntegrator(NVTIntegrator):
    """Integrator object for constant pressure simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.
    """

    # should be enough to redefine these functions, and the step() from NVTIntegrator should do the trick

    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        if self._stresscheck and np.array_equiv(
            dstrip(self.forces.vir), np.zeros(len(self.forces.vir))
        ):
            warning(
                "Forcefield returned a zero stress tensor. NPT simulation will likely make no sense",
                verbosity.low,
            )
            # if verbosity.medium: will uncomment one day
            #    raise ValueError(
            #        "Zero stress terminates simulation for medium verbosity and above."
            #    )

        self._stresscheck = False

        self.barostat.pstep(level)
        super(NPTIntegrator, self).pstep(level)
        # self.pconstraints()

    def qcstep(self):
        """Velocity Verlet centroid position propagator."""

        self.barostat.qcstep()

    def tstep(self):
        """Velocity Verlet thermostat step"""

        self.thermostat.step()
        self.barostat.thermostat.step()
        # self.pconstraints()


class NSTIntegrator(NPTIntegrator):
    """Ensemble object for constant pressure simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.

    Attributes:
    barostat: A barostat object to keep the pressure constant.

    Depend objects:
    econs: Conserved energy quantity. Depends on the bead and cell kinetic
    and potential energy, the spring potential energy, the heat
    transferred to the beads and cell thermostat, the temperature and
    the cell volume.
    pext: External pressure.
    """


class SCIntegrator(NVTIntegrator):
    """Integrator object for constant temperature simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, the spring potential energy and the heat
            transferred to the thermostat.
    """

    def bind(self, mover):
        """Binds ensemble beads, cell, bforce, bbias and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
        beads: The beads object from whcih the bead positions are taken.
        nm: A normal modes object used to do the normal modes transformation.
        cell: The cell object from which the system box is taken.
        bforce: The forcefield object from which the force and virial are
            taken.
        prng: The random number generator object which controls random number
            generation.
        """

        super(SCIntegrator, self).bind(mover)
        self.ensemble.add_econs(self.forces._potsc)
        self.ensemble.add_xlpot(self.forces._potsc)

    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        if level == 0:
            # bias goes in the outer loop
            self.beads.p += dstrip(self.bias.f) * self.pdt[level]
        # just integrate the Trotter force scaled with the SC coefficients, which is a cheap approx to the SC force
        self.beads.p += (
            self.forces.mts_forces[level].f
            * (1.0 + self.forces.coeffsc_part_1)
            * self.pdt[level]
        )

    def step(self, step=None):
        # the |f|^2 term is considered to be slowest (for large enough P) and is integrated outside everything.
        # if nmts is not specified, this is just the same as doing the full SC integration

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

            # forces are integerated for dt with MTS.
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop(0)
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5

            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

        elif self.splitting == "baoab":
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop_ba(0)
            # thermostat is applied for dt
            self.tstep()
            self.pconstraints()
            self.mtsprop_ab(0)
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5


class SCNPTIntegrator(SCIntegrator):
    """Integrator object for constant pressure Suzuki-Chin simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.
    """

    # should be enough to redefine these functions, and the step() from NVTIntegrator should do the trick
    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        if self._stresscheck and np.array_equiv(
            dstrip(self.forces.vir), np.zeros(len(self.forces.vir))
        ):
            warning(
                "Forcefield returned a zero stress tensor. NPT simulation will likely make no sense",
                verbosity.low,
            )
            if verbosity.medium:
                raise ValueError(
                    "Zero stress terminates simulation for medium verbosity and above."
                )

        self._stresscheck = False

        self.barostat.pstep(level)
        super(SCNPTIntegrator, self).pstep(level)

    def qcstep(self):
        """Velocity Verlet centroid position propagator."""

        self.barostat.qcstep()

    def tstep(self):
        """Velocity Verlet thermostat step"""

        self.thermostat.step()
        self.barostat.thermostat.step()

    def step(self, step=None):
        # the |f|^2 term is considered to be slowest (for large enough P) and is integrated outside everything.
        # if nmts is not specified, this is just the same as doing the full SC integration

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

            # forces are integerated for dt with MTS.
            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop(0)
            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5

            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

        elif self.splitting == "baoab":
            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop_ba(0)
            # thermostat is applied for dt
            self.tstep()
            self.pconstraints()
            self.mtsprop_ab(0)
            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
