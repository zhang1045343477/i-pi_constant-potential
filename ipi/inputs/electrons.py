"""Input classes for electronic degrees of freedom and potentiostat configuration.

Handles parsing of <electrons> or <constant_potential> XML blocks for constant 
Fermi level MD simulations.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

import ipi.engine.potentiostat as epotentiostat
from ipi.utils.depend import *
from ipi.utils.inputvalue import *

__all__ = ["InputElectrons", "InputPotentiostat"]


class InputPotentiostat(Input):
    """Input class for potentiostat (electronic thermostat) configuration.

    Handles different potentiostat types (langevin, svr) with unified
    'tau' parameter interface.
    """
    
    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "langevin",
                "options": ["langevin", "svr"],
                "help": "Potentiostat type: 'langevin' for Langevin thermostat, 'svr' for stochastic velocity rescaling."
            },
        )
    }
    
    fields = {
        "tau": (
            InputValue,
            {
                "dtype": float,
                "default": 100.0,
                "help": "Potentiostat time constant. Maps to different internal parameters based on mode.",
                "dimension": "time",
            },
        ),
        "temp": (
            InputValue,
            {
                "dtype": float,
                "default": float("nan"),
                "help": "Electronic heat bath temperature. Defaults to the atomic temperature if not specified.",
                "dimension": "temperature",
            },
        ),

    }
    
    dynamic = {}
    default_help = "Electronic thermostat (potentiostat) for constant Fermi level simulations."
    default_label = "POTENTIOSTAT"
    
    def _store_dict_config(self, config_dict):
        """Store potentiostat configuration from dictionary.

        Args:
            config_dict: Dictionary containing potentiostat configuration
        """
        self.mode.store(config_dict.get("mode", "langevin"))
        self.tau.store(config_dict.get("tau", 100.0))
        self.temp.store(config_dict.get("temp", float("nan")))

    def store(self, potentiostat):
        """Store potentiostat configuration."""
        if potentiostat is None:
            return

        # Handle dictionary input
        if isinstance(potentiostat, dict):
            self._store_dict_config(potentiostat)
            return

        # Handle actual potentiostat objects
        super(InputPotentiostat, self).store(potentiostat)
        
        if isinstance(potentiostat, epotentiostat.PotentiostatLangevin):
            self.mode.store("langevin")
            self.tau.store(potentiostat.tau)
            self.temp.store(potentiostat.temp)
        elif isinstance(potentiostat, epotentiostat.PotentiostatSVR):
            self.mode.store("svr")
            self.tau.store(potentiostat.tau)
            self.temp.store(potentiostat.temp)

        elif hasattr(potentiostat, '__class__') and potentiostat.__class__.__name__ == "InputPotentiostat":
            # Handle the case where an InputPotentiostat is passed
            return
        else:
            raise TypeError("Unknown potentiostat type: " + type(potentiostat).__name__)
    
    def fetch(self):
        """Create potentiostat object from configuration."""
        super(InputPotentiostat, self).fetch()
        
        mode = self.mode.fetch()
        tau = self.tau.fetch()
        temp = self.temp.fetch()
        
        if mode == "langevin":
            potentiostat = epotentiostat.PotentiostatLangevin(tau=tau)
        elif mode == "svr":
            potentiostat = epotentiostat.PotentiostatSVR(tau=tau)

        else:
            raise ValueError(f"Invalid potentiostat mode: {mode}")
        
        # Record requested temperature (NaN if not provided) for later configuration
        potentiostat._input_temp = temp

        return potentiostat


class InputElectrons(Input):
    """Input class for electronic degrees of freedom configuration.
    
    Handles configuration of electronic state and potentiostat for constant
    Fermi level MD simulations.
    """
    
    attribs = {
        "enabled": (
            InputAttribute,
            {
                "dtype": bool,
                "default": False,
                "help": "Enable constant potential (constant Fermi level) MD simulation."
            },
        )
    }
    
    fields = {
        "target_fermi_level": (
            InputValue,
            {
                "dtype": float,
                "default": float('nan'),
                "help": "Target Fermi level for constant potential simulation. Leave unset to disable constant target mode.",
                "dimension": "energy",
            },
        ),
        "initial_target_fermi_level": (
            InputValue,
            {
                "dtype": float,
                "default": float('nan'),
                "help": "Initial target Fermi level for linear ramping. Must be used with final_target_fermi_level and transition_steps.",
                "dimension": "energy",
            },
        ),
        "final_target_fermi_level": (
            InputValue,
            {
                "dtype": float,
                "default": float('nan'),
                "help": "Final target Fermi level for linear ramping. Must be used with initial_target_fermi_level and transition_steps.",
                "dimension": "energy",
            },
        ),
        "target_workfunction": (
            InputValue,
            {
                "dtype": float,
                "default": float('nan'),
                "help": "Target work function for constant workfunction simulation. Mutually exclusive with Fermi level targets.",
                "dimension": "energy",
            },
        ),
        "initial_target_workfunction": (
            InputValue,
            {
                "dtype": float,
                "default": float('nan'),
                "help": "Initial target work function for linear ramping. Must be used with final_target_workfunction and transition_steps.",
                "dimension": "energy",
            },
        ),
        "final_target_workfunction": (
            InputValue,
            {
                "dtype": float,
                "default": float('nan'),
                "help": "Final target work function for linear ramping. Must be used with initial_target_workfunction and transition_steps.",
                "dimension": "energy",
            },
        ),
        "transition_steps": (
            InputValue,
            {
                "dtype": int,
                "default": -1,
                "help": "Number of steps for linear transition from initial to final target Fermi level. Use -1 to disable linear ramping.",
            },
        ),
        "q_init": (
            InputValue,
            {
                "dtype": float,
                "default": 1.0,
                "help": "Initial electronic charge.",
            },
        ),
        "neutral_electrons": (
            InputValue,
            {
                "dtype": int,
                "default": -1,
                "help": "Total electron number for the neutral reference system. Must be a positive integer when electrons are enabled.",
            },
        ),
        "mass": (
            InputValue,
            {
                "dtype": float,
                "default": 1.0,
                "help": "Electronic mass parameter.",
                "dimension": "mass",
            },
        ),

        "z_average_region": (
            InputArray,
            {
                "dtype": float,
                "default": np.array([], float),
                "help": "Z coordinate range [z_min, z_max] used to define the averaging region for workfunction calculations.",
                "dimension": "length",
            },
        ),

        "potentiostat": (
            InputPotentiostat,
            {
                "default": input_default(factory=InputPotentiostat),
                "help": "Potentiostat (electronic thermostat) configuration.",
            },
        ),
        "log_stride": (
            InputValue,
            {
                "dtype": int,
                "default": 1,
                "help": "Stride for logging electronic properties.",
            },
        ),
        "solvation_stride": (
            InputValue,
            {
                "dtype": int,
                "default": 1,
                "help": "Stride (in MD steps) for enabling implicit solvent/workfunction coupling. 0 disables solvent, 1 enables every step.",
            },
        ),
    }
    
    dynamic = {}
    default_help = "Electronic degrees of freedom for constant Fermi level MD."
    default_label = "ELECTRONS"
    
    def store(self, electrons_config):
        """Store electrons configuration."""
        if electrons_config is None:
            return
            
        # Handle InputElectrons object (for default initialization)
        if hasattr(electrons_config, '__class__') and electrons_config.__class__.__name__ == "InputElectrons":
            return
        
        # Handle dictionary input
        if isinstance(electrons_config, dict):
            # Store basic electronic state parameters
            self.enabled.store(electrons_config.get("enabled", False))
            self.target_fermi_level.store(electrons_config.get("target_fermi_level", float('nan')))
            self.initial_target_fermi_level.store(electrons_config.get("initial_target_fermi_level", float('nan')))
            self.final_target_fermi_level.store(electrons_config.get("final_target_fermi_level", float('nan')))
            self.target_workfunction.store(electrons_config.get("target_workfunction", float('nan')))
            self.initial_target_workfunction.store(electrons_config.get("initial_target_workfunction", float('nan')))
            self.final_target_workfunction.store(electrons_config.get("final_target_workfunction", float('nan')))
            self.transition_steps.store(electrons_config.get("transition_steps", -1))
            self.q_init.store(electrons_config.get("q_init", 1.0))
            self.neutral_electrons.store(electrons_config.get("neutral_electrons", -1))
            self.mass.store(electrons_config.get("mass", 1.0))
            self.log_stride.store(electrons_config.get("log_stride", 1))
            self.solvation_stride.store(electrons_config.get("solvation_stride", 1))

            # Workfunction-related configuration
            if "z_average_region" in electrons_config:
                self.z_average_region.store(electrons_config["z_average_region"])
            
            # Store potentiostat configuration
            if "potentiostat" in electrons_config:
                self.potentiostat.store(electrons_config["potentiostat"])
            if "potentiostat_temp" in electrons_config:
                self.potentiostat.temp.store(electrons_config["potentiostat_temp"])

    def store_runtime_state(self, dynamics_obj):
        """Store runtime electronic state for restart files.
        
        Args:
            dynamics_obj: The dynamics object containing current electronic state
        """
        if dynamics_obj is None:
            return
            
        # If electronic system is active, store current charge as q_init for restart
        if (hasattr(dynamics_obj, 'electronic_state') and 
            dynamics_obj.electronic_state is not None and
            hasattr(dynamics_obj, 'electrons_config') and
            dynamics_obj.electrons_config is not None):
            
            # Store configuration
            self.store(dynamics_obj.electrons_config)
            
            # Override q_init with current charge for restart
            current_q = dynamics_obj.electronic_state.q
            self.q_init.store(current_q)
            
            # Mark as coming from restart for informational purposes
            self._from_restart = True
        else:
            # Handle other types - just pass through to superclass
            super(InputElectrons, self).store(dynamics_obj)
    
    def fetch(self):
        """Create electrons configuration from input."""
        try:
            # Try to fetch all field values - if any fail, return None
            config = {
                "enabled": self.enabled.fetch(),
                "target_fermi_level": self.target_fermi_level.fetch(),
                "initial_target_fermi_level": self.initial_target_fermi_level.fetch(),
                "final_target_fermi_level": self.final_target_fermi_level.fetch(),
                "target_workfunction": self.target_workfunction.fetch(),
                "initial_target_workfunction": self.initial_target_workfunction.fetch(),
                "final_target_workfunction": self.final_target_workfunction.fetch(),
                "transition_steps": self.transition_steps.fetch(),
                "q_init": self.q_init.fetch(),
                "neutral_electrons": self.neutral_electrons.fetch(),
                "mass": self.mass.fetch(),
                "solvation_stride": self.solvation_stride.fetch(),
                "log_stride": self.log_stride.fetch(),
                "potentiostat": self.potentiostat.fetch(),
                "potentiostat_temp": self.potentiostat.temp.fetch(),
                "z_average_region": self.z_average_region.fetch(),
            }
            
            # Add restart detection info
            config["_from_restart"] = getattr(self, '_from_restart', False)
            
        except (ValueError, AttributeError) as e:
            # Return None if not initialized (not specified in XML) or any other fetch error
            return None

        # Parameter validation
        if config["enabled"]:
            if config["q_init"] <= 0:
                raise ValueError(f"q_init must be positive, got {config['q_init']}")
            if config.get("neutral_electrons", -1) <= 0:
                raise ValueError(
                    f"neutral_electrons must be a positive integer when electrons are enabled, got {config.get('neutral_electrons', -1)}"
                )
            if config["mass"] <= 0:
                raise ValueError(f"mass must be positive, got {config['mass']}")
            if config["log_stride"] <= 0:
                raise ValueError(f"log_stride must be positive, got {config['log_stride']}")
            
            # Validate mutually exclusive parameter groups for Fermi level vs workfunction
            import math
            import numpy as np

            tF = config["target_fermi_level"]
            tW = config["target_workfunction"]
            iF = config["initial_target_fermi_level"]
            fF = config["final_target_fermi_level"]
            iW = config["initial_target_workfunction"]
            fW = config["final_target_workfunction"]
            steps = config["transition_steps"]

            stride = config.get("solvation_stride", 1)
            if stride < 0:
                raise ValueError(f"solvation_stride must be non-negative, got {stride}")

            has_fermi_constant = not math.isnan(tF)
            has_wf_constant = not math.isnan(tW)

            fermi_endpoints_set = (not math.isnan(iF) or not math.isnan(fF))
            wf_endpoints_set = (not math.isnan(iW) or not math.isnan(fW))
            steps_positive = steps > 0

            # Disallow simultaneous constant Fermi and workfunction targets
            if has_fermi_constant and has_wf_constant:
                raise ValueError(
                    "Cannot specify both target_fermi_level and target_workfunction. "
                    "Choose either constant Fermi level or constant workfunction mode."
                )

            # Disallow specifying linear endpoints for both families simultaneously
            if steps_positive and fermi_endpoints_set and wf_endpoints_set:
                raise ValueError(
                    "Cannot specify linear ramping endpoints for both Fermi level and workfunction "
                    "at the same time. Choose one family."
                )

            using_fermi_linear = steps_positive and fermi_endpoints_set and not wf_endpoints_set
            using_wf_linear = steps_positive and wf_endpoints_set and not fermi_endpoints_set

            # Disallow constant + linear in the same family
            if has_fermi_constant and using_fermi_linear:
                raise ValueError(
                    "Cannot use both target_fermi_level and Fermi-level linear ramping parameters "
                    "simultaneously. Choose one approach."
                )
            if has_wf_constant and using_wf_linear:
                raise ValueError(
                    "Cannot use both target_workfunction and workfunction linear ramping parameters "
                    "simultaneously. Choose one approach."
                )

            # Disallow mixing Fermi-level and workfunction families in any form
            if (has_fermi_constant or using_fermi_linear) and (has_wf_constant or using_wf_linear):
                raise ValueError(
                    "Cannot mix Fermi-level and workfunction target parameters. "
                    "Choose either Fermi-level control or workfunction control, not both."
                )

            # Generic linear ramping validation
            if steps_positive or fermi_endpoints_set or wf_endpoints_set:
                # Intent to use linear ramping is present
                if not steps_positive:
                    raise ValueError(
                        f"transition_steps must be positive when using linear ramping, got {steps}"
                    )

                if not (using_fermi_linear or using_wf_linear):
                    raise ValueError(
                        "When using transition_steps > 0, you must specify endpoints for either "
                        "Fermi level or workfunction (but not both)."
                    )

            # Set default values for unspecified endpoints in the active linear family
            if using_fermi_linear:
                if math.isnan(iF):
                    config["initial_target_fermi_level"] = 0.0
                if math.isnan(fF):
                    config["final_target_fermi_level"] = 0.0

            if using_wf_linear:
                if math.isnan(iW):
                    config["initial_target_workfunction"] = 0.0
                if math.isnan(fW):
                    config["final_target_workfunction"] = 0.0

            # Determine mode: default to Fermi-level control if no workfunction targets are set
            if using_wf_linear or has_wf_constant:
                config["mode"] = "workfunction"
            else:
                config["mode"] = "fermi"

            # Validate z_average_region when workfunction mode is enabled
            if config["mode"] == "workfunction":
                z_region = np.array(config["z_average_region"], dtype=float).flatten()
                if z_region.size != 2:
                    raise ValueError(
                        "z_average_region must be an array of length 2 [z_min, z_max] when "
                        "workfunction mode is enabled."
                    )

                z_min, z_max = float(z_region[0]), float(z_region[1])
                if (not np.isfinite(z_min)) or (not np.isfinite(z_max)):
                    raise ValueError("z_average_region values must be finite when workfunction mode is enabled.")
                if z_max <= z_min:
                    raise ValueError("z_average_region requires z_max > z_min when workfunction mode is enabled.")

                config["z_average_region"] = np.array([z_min, z_max], float)

        return config
    
    def create_electronic_state(self):
        """Create ElectronicState object from configuration."""
        config = self.fetch()
        
        if not config["enabled"]:
            return None
        
        # Determine control mode (Fermi level vs workfunction) and ramping mode
        import math

        mode = config.get("mode", "fermi")

        if mode == "workfunction":
            # Workfunction control
            has_linear = (
                (not math.isnan(config["initial_target_workfunction"]))
                or (not math.isnan(config["final_target_workfunction"]))
                or config["transition_steps"] > 0
            )

            if has_linear:
                electronic_state = epotentiostat.ElectronicState(
                    q_init=config["q_init"],
                    mass=config["mass"],
                    mode="workfunction",
                    initial_target_workfunction=config["initial_target_workfunction"],
                    final_target_workfunction=config["final_target_workfunction"],
                    transition_steps=config["transition_steps"],
                )
            else:
                target_wf = config["target_workfunction"]
                if math.isnan(target_wf):
                    target_wf = 0.0
                electronic_state = epotentiostat.ElectronicState(
                    q_init=config["q_init"],
                    mass=config["mass"],
                    mode="workfunction",
                    target_workfunction=target_wf,
                )
        else:
            # Fermi level control (default / backward compatible)
            has_linear = (
                (not math.isnan(config["initial_target_fermi_level"]))
                or (not math.isnan(config["final_target_fermi_level"]))
                or config["transition_steps"] > 0
            )

            if has_linear:
                electronic_state = epotentiostat.ElectronicState(
                    q_init=config["q_init"],
                    mass=config["mass"],
                    mode="fermi",
                    initial_target_ef=config["initial_target_fermi_level"],
                    final_target_ef=config["final_target_fermi_level"],
                    transition_steps=config["transition_steps"],
                )
            else:
                target_ef = config["target_fermi_level"]
                if math.isnan(target_ef):
                    target_ef = 0.0  # Default for constant mode if not specified
                electronic_state = epotentiostat.ElectronicState(
                    q_init=config["q_init"],
                    mass=config["mass"],
                    mode="fermi",
                    target_ef=target_ef,
                )
        # Attach neutral electron count to the electronic state so that
        # downstream components (forcefields, properties) can access it
        neutral_electrons = config.get("neutral_electrons", -1)
        if neutral_electrons is None or int(neutral_electrons) <= 0:
            raise ValueError(
                f"neutral_electrons must be a positive integer when electrons are enabled, got {neutral_electrons}"
            )
        try:
            electronic_state.neutral_electrons = int(neutral_electrons)
        except (TypeError, ValueError):
            raise ValueError(f"neutral_electrons must be an integer, got {neutral_electrons!r}")

        return electronic_state
