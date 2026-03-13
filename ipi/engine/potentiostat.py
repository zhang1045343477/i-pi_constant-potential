"""Potentiostat (electronic thermostat) for constant Fermi level MD simulations.

This module implements electronic thermostats that couple the electronic degree of
freedom (charge q) to a target Fermi level, enabling constant potential MD simulations.
Supports Langevin and SVR thermostats using a unified 'tau' parameter interface.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.utils.depend import *
from ipi.utils.messages import verbosity, warning, info
from ipi.utils.prng import Random
from ipi.utils.units import Constants

__all__ = ["ElectronicState", "Potentiostat", "PotentiostatLangevin", "PotentiostatSVR", "ElectronicChargeError"]

# Minimum charge threshold - simulation stops if q falls below this value
Q_MIN = 1e-6  # Minimum allowed electronic charge - simulation will terminate if q < Q_MIN


class ElectronicChargeError(Exception):
    """Exception raised when electronic charge falls below minimum threshold."""
    pass

class ElectronicState:
    """Container for electronic degrees of freedom (q, p, mass) in constant potential MD.

    Manages the electronic charge q, momentum p, and mass parameters needed for
    constant Fermi level simulations. Supports both constant and linearly ramped
    target Fermi levels.

    Attributes:
        q: Electronic charge (> 0)
        p: Electronic momentum
        mass: Electronic mass parameter
        target_ef: Current target Fermi level (may change over time for linear ramping)
        current_ef: Current Fermi level from PES
        current_step: Current simulation step (for linear ramping)
    """
    
    def __init__(
        self,
        q_init=1.0,
        mass=1.0,
        mode="fermi",
        target_ef=None,
        initial_target_ef=None,
        final_target_ef=None,
        target_workfunction=None,
        initial_target_workfunction=None,
        final_target_workfunction=None,
        transition_steps=None,
    ):
        """Initialize electronic state.

        Args:
            q_init (float): Initial electronic charge
            mass (float): Electronic mass parameter
            mode (str): Control mode, either "fermi" or "workfunction".
            target_ef (float, optional): Constant target Fermi level
            initial_target_ef (float, optional): Initial target Fermi level for linear ramping
            final_target_ef (float, optional): Final target Fermi level for linear ramping  
            target_workfunction (float, optional): Constant target work function
            initial_target_workfunction (float, optional): Initial target work function for linear ramping
            final_target_workfunction (float, optional): Final target work function for linear ramping
            transition_steps (int, optional): Number of steps for linear transition
        """
        if q_init <= 0:
            raise ValueError(f"Initial charge must be positive, got {q_init}")

        self.mass = float(mass)
        self.current_ef = 0.0
        self.current_workfunction = 0.0
        self.current_step = 0

        self._q = max(q_init, Q_MIN)
        self._p = 0.0  # momentum conjugate to q

        # Control mode: "fermi" (default) or "workfunction"
        self.mode = mode or "fermi"
        if self.mode not in ("fermi", "workfunction"):
            raise ValueError(f"Unsupported electronic control mode: {self.mode}")

        # Initialize ramping state
        self._linear_mode = False
        self._initial_target_ef = None
        self._final_target_ef = None
        self._initial_target_workfunction = None
        self._final_target_workfunction = None
        self._transition_steps = None

        # Set up target mode depending on control family
        if self.mode == "workfunction":
            # Workfunction control
            if (
                initial_target_workfunction is not None
                or final_target_workfunction is not None
                or transition_steps is not None
            ):
                # Linear ramping mode
                self._linear_mode = True
                self._initial_target_workfunction = float(initial_target_workfunction or 0.0)
                self._final_target_workfunction = float(final_target_workfunction or 0.0)
                self._transition_steps = int(transition_steps or 0)

                if self._transition_steps <= 0:
                    raise ValueError(
                        f"transition_steps must be positive, got {self._transition_steps}"
                    )

                # Set initial target
                self.target_workfunction = self._initial_target_workfunction

            elif target_workfunction is not None:
                # Constant target mode
                self._linear_mode = False
                self.target_workfunction = float(target_workfunction)
            else:
                raise ValueError(
                    "Must specify either target_workfunction or workfunction linear ramping parameters"
                )

        else:
            # Fermi level control (backward compatible)
            if (
                initial_target_ef is not None
                or final_target_ef is not None
                or transition_steps is not None
            ):
                # Linear ramping mode
                self._linear_mode = True
                self._initial_target_ef = float(initial_target_ef or 0.0)
                self._final_target_ef = float(final_target_ef or 0.0)
                self._transition_steps = int(transition_steps or 0)

                if self._transition_steps <= 0:
                    raise ValueError(
                        f"transition_steps must be positive, got {self._transition_steps}"
                    )

                # Set initial target
                self.target_ef = self._initial_target_ef

            elif target_ef is not None:
                # Constant target mode
                self._linear_mode = False
                self.target_ef = float(target_ef)
            else:
                raise ValueError("Must specify either target_ef or linear ramping parameters")
    
    def update_target_fermi_level(self, step):
        """Update target Fermi level for current simulation step.
        
        For linear ramping mode, this updates the target Fermi level according to:
        target_ef = initial + (final - initial) * min(step / transition_steps, 1.0)
        
        Args:
            step (int): Current simulation step
        """
        self.current_step = step
        
        if self._linear_mode:
            if step >= self._transition_steps:
                # Transition complete, use final target
                if self.mode == "workfunction":
                    self.target_workfunction = self._final_target_workfunction
                else:
                    self.target_ef = self._final_target_ef
            else:
                # Linear interpolation
                progress = float(step) / float(self._transition_steps)
                if self.mode == "workfunction":
                    self.target_workfunction = (
                        self._initial_target_workfunction
                        + progress
                        * (self._final_target_workfunction - self._initial_target_workfunction)
                    )
                else:
                    self.target_ef = (
                        self._initial_target_ef
                        + progress
                        * (self._final_target_ef - self._initial_target_ef)
                    )
    
    @property
    def is_linear_mode(self):
        """Check if using linear ramping mode."""
        return self._linear_mode
        
    def get_ramping_info(self):
        """Get information about linear ramping setup.
        
        Returns:
            dict: Dictionary containing ramping parameters, or None if constant mode
        """
        if not self._linear_mode:
            return None
            
        info = {
            "transition_steps": self._transition_steps,
            "current_step": self.current_step,
            "progress": min(1.0, float(self.current_step) / float(self._transition_steps)),
        }

        if self.mode == "workfunction":
            info.update(
                {
                    "initial_target_workfunction": self._initial_target_workfunction,
                    "final_target_workfunction": self._final_target_workfunction,
                    "current_target_workfunction": self.target_workfunction,
                }
            )
        else:
            info.update(
                {
                    "initial_target_ef": self._initial_target_ef,
                    "final_target_ef": self._final_target_ef,
                    "current_target_ef": self.target_ef,
                }
            )

        return info
    
    @property
    def q(self):
        """Electronic charge (always positive)."""
        current_q = self._q
        if current_q < Q_MIN:
            raise ElectronicChargeError(
                f"Electronic charge {current_q:.2e} fell below minimum threshold {Q_MIN:.2e}. "
                f"Simulation terminated to prevent numerical instability."
            )
        return current_q

    @q.setter
    def q(self, value):
        """Set electronic charge."""
        if value <= 0:
            raise ValueError(f"Charge must be positive, got {value}")
        if value < Q_MIN:
            raise ElectronicChargeError(
                f"Attempting to set electronic charge to {value:.2e}, which is below "
                f"minimum threshold {Q_MIN:.2e}. Simulation terminated."
            )
        self._q = value
    
    @property
    def p(self):
        """Electronic momentum."""
        return self._p

    @p.setter
    def p(self, value):
        """Set electronic momentum."""
        self._p = float(value)
    
    @property
    def kinetic_energy(self):
        """Electronic kinetic energy."""
        return 0.5 * self.p**2 / self.mass
    
    @property
    def force(self):
        """Electronic force F_q.

        For Fermi-level control: F_q = Ef_target - Ef_current.
        For workfunction control: F_q = workfunction_current - workfunction_target.
        """
        if getattr(self, "mode", "fermi") == "workfunction":
            return self.current_workfunction - self.target_workfunction
        return self.target_ef - self.current_ef
    
    def drift(self, dt):
        """Position drift step: q += p/mass * dt."""
        drift_term = self._p * dt / self.mass
        new_q = self._q + drift_term
        
        if new_q < Q_MIN:
            raise ElectronicChargeError(
                f"Electronic charge would become {new_q:.2e} after drift step, "
                f"below minimum threshold {Q_MIN:.2e}. Current q={self._q:.2e}, "
                f"p={self._p:.2e}, dt={dt:.2e}. Simulation terminated."
            )
        self._q = new_q
    
    def kick(self, dt, force=None):
        """Momentum kick step: p += force * dt."""
        if force is None:
            force = self.force
        self.p += force * dt


class Potentiostat:
    """Base class for electronic thermostats (potentiostats).
    
    Provides common interface for thermostatting the electronic degree of freedom
    in constant Fermi level MD simulations.
    """
    
    def __init__(self, temp=1.0, dt=1.0, tau=100.0):
        """Initialize potentiostat.
        
        Args:
            temp (float): Temperature (same as atomic system)
            dt (float): Time step
            tau (float): Thermostat time constant
        """
        self._temp = depend_value(name="temp", value=temp)
        self._dt = depend_value(name="dt", value=dt)
        self._tau = depend_value(name="tau", value=tau)
        
        self.electronic_state = None
        self.prng = None
        
    def bind(self, electronic_state, prng=None):
        """Bind electronic state and random number generator.
        
        Args:
            electronic_state (ElectronicState): Electronic degrees of freedom
            prng (Random): Random number generator
        """
        self.electronic_state = electronic_state
        if prng is None:
            self.prng = Random()
        else:
            self.prng = prng
    
    def step(self, dt=None):
        """Perform one thermostat step (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement step method")
    
    def half_B(self, dt, fermi_level):
        """Half momentum kick using current Fermi level.

        Args:
            dt: Time step
            fermi_level: Fermi level in eV units (from cache)
        """
        # CRITICAL: current_ef must be in atomic units (Hartree), but fermi_level is in eV
        # Convert eV to Hartree for i-PI's property output system
        fermi_level_au = fermi_level / Constants.EV_PER_HARTREE  # eV to Hartree conversion
        self.electronic_state.current_ef = fermi_level_au
        self.electronic_state.kick(dt)
    
    def A(self, dt):
        """Position drift step."""
        self.electronic_state.drift(dt)
    
    @property
    def temp(self):
        """Temperature."""
        return self._temp.value
    
    @property
    def dt(self):
        """Time step."""
        return self._dt.value
    
    @property
    def tau(self):
        """Thermostat time constant."""
        return self._tau.value


class PotentiostatLangevin(Potentiostat):
    """Langevin thermostat for electronic degree of freedom.
    
    Implements BAOAB integration with Ornstein-Uhlenbeck process for the O step.
    Maps tau to gamma = 1/tau internally.
    """
    
    def __init__(self, temp=1.0, dt=1.0, tau=100.0):
        super().__init__(temp=temp, dt=dt, tau=tau)

        # Set up OU coefficients as depend objects
        def get_c():
            gamma = 1.0 / float(self.tau)
            return np.exp(-gamma * float(self.dt))

        self._c = depend_value(name="c", func=get_c, dependencies=[self._dt, self._tau])
        # sigma will be set up in bind() method after electronic_state is available
        self._sigma = None
    
    def bind(self, electronic_state, prng=None):
        """Bind and set up OU coefficients."""
        super().bind(electronic_state, prng)

        # Set up sigma dependency after electronic_state is bound
        def get_sigma():
            gamma = 1.0 / float(self.tau)
            c = np.exp(-gamma * float(self.dt))
            return np.sqrt(self.electronic_state.mass * Constants.kb * float(self.temp) * (1.0 - c**2))

        self._sigma = depend_value(name="sigma", func=get_sigma,
                                  dependencies=[self._dt, self._tau, self._temp])
    
    def O_step(self, dt):
        """Ornstein-Uhlenbeck thermostat step."""
        c = self.c
        sigma = self.sigma
        
        # p <- c * p + sigma * Normal(0,1)
        self.electronic_state.p = (c * self.electronic_state.p + 
                                   sigma * self.prng.gvec(1)[0])
    
    def step(self, dt=None):
        """BAOAB Langevin step (O step only, B and A handled separately)."""
        self.O_step(dt or self.dt)


class PotentiostatSVR(Potentiostat):
    """Stochastic Velocity Rescaling (SVR) thermostat for electronic degree of freedom.
    
    Rescales electronic momentum so kinetic energy follows canonical distribution.
    For single electronic DOF, Nf = 1.
    """
    
    def __init__(self, temp=1.0, dt=1.0, tau=100.0):
        super().__init__(temp=temp, dt=dt, tau=tau)
        
        # Set up SVR coefficient
        def get_c():
            return np.exp(-float(self.dt) / float(self.tau))
        
        self._c = depend_value(name="c", func=get_c, dependencies=[self._dt, self._tau])
    
    def O_step(self, dt):
        """SVR rescaling step for electronic momentum."""
        if self.electronic_state is None:
            return
        
        c = self.c
        K = self.electronic_state.kinetic_energy
        K_target = 0.5 * Constants.kb * self.temp  # For single DOF (Nf=1)
        
        if K <= 0:
            # If kinetic energy is zero, just sample from Maxwell distribution
            sigma = np.sqrt(self.electronic_state.mass * Constants.kb * self.temp)
            self.electronic_state.p = sigma * self.prng.gvec(1)[0]
            return
        
        # For Nf=1 case (single electronic DOF)
        R = self.prng.gvec(1)[0]  # Normal(0,1)

        # More robust calculation to avoid numerical issues
        ratio = K_target / K
        sqrt_term = np.sqrt(c * (1.0 - c) * ratio)

        alpha_squared = (c +
                        (1.0 - c) * ratio * R**2 +
                        2.0 * R * sqrt_term)

        # Ensure alpha_squared is positive with better bounds
        alpha_squared = max(alpha_squared, 1e-12)
        alpha = np.sqrt(alpha_squared)

        sign_argument = np.sqrt(c) + R * np.sqrt((1.0 - c) * ratio)
        if sign_argument < 0.0:
            alpha *= -1.0

        self.electronic_state.p *= alpha
    
    def step(self, dt=None):
        """SVR step (O step only, B and A handled separately)."""
        self.O_step(dt or self.dt)


# Add property decorators for depend objects
dproperties(Potentiostat, ["temp", "dt", "tau"])
dproperties(PotentiostatLangevin, ["c", "sigma"])
dproperties(PotentiostatSVR, ["c"])