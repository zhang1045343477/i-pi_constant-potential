"""Contains the classes that connect the driver to the python code.

ForceField objects are force providers, i.e. they are the abstraction
layer for a driver that gets positions and returns forces (and energy).
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import time
import threading
import json
import sys
import subprocess
import tempfile
import os
import socket
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ipi.engine.cell import GenericCell
from ipi.utils.prng import Random
from ipi.utils.softexit import softexit
from ipi.utils.messages import info, verbosity, warning
from ipi.interfaces.sockets import InterfaceSocket, CP2KSocketServer, CP2KSocketCommunicator
from ipi.utils.depend import dstrip
from ipi.utils.io import read_file
from ipi.utils.units import unit_to_internal, unit_to_user, Constants
from ipi.utils.cp2k_cube import (
    read_cube as read_cp2k_cube,
    planar_average_z as planar_average_z_cp2k,
    z_coordinates_A as z_coords_cp2k,
    BOHR_TO_ANGSTROM,
)
from ipi.utils.distance import vector_separation
from ipi.pes import __drivers__
from ipi.utils.mathtools import (
    get_rotation_quadrature_legendre,
    get_rotation_quadrature_lebedev,
    random_rotation,
)

plumed = None


class ForceRequest(dict):
    """An extension of the standard Python dict class which only has a == b
    if a is b == True, rather than if the elements of a and b are identical.

    Standard dicts are checked for equality if elements have the same value.
    Here I only care if requests are instances of the very same object.
    This is useful for the `in` operator, which uses equality to test membership.
    """

    def __eq__(self, y):
        """Overwrites the standard equals function."""
        return self is y


class ForceField:
    """Base forcefield class.

    Gives the standard methods and quantities needed in all the forcefield
    classes.

    Attributes:
        pars: A dictionary of the parameters needed to initialize the forcefield.
            Of the form {'name1': value1, 'name2': value2, ... }.
        name: The name of the forcefield.
        latency: A float giving the number of seconds the socket will wait
            before updating the client list.
        offset: A float giving a constant value that is subtracted from the return
            value of the forcefield
        requests: A list of all the jobs to be given to the client codes.
        dopbc: A boolean giving whether or not to apply the periodic boundary
            conditions before sending the positions to the client code.
        _thread: The thread on which the socket polling loop is being run.
        _doloop: A list of booleans. Used to decide when to stop running the
            polling loop.
        _threadlock: Python handle used to lock the thread held in _thread.
    """

    def __init__(
        self,
        latency=1e-4,
        offset=0.0,
        name="",
        pars=None,
        dopbc=False,
        active=np.array([-1]),
        threaded=False,
    ):
        """Initialises ForceField.

        Args:
            latency: The number of seconds the socket will wait before updating
                the client list.
            offset: A constant offset subtracted from the energy value given by the
                client.
            name: The name of the forcefield.
            pars: A dictionary used to initialize the forcefield, if required.
                Of the form {'name1': value1, 'name2': value2, ... }.
            dopbc: Decides whether or not to apply the periodic boundary conditions
                before sending the positions to the client code.
            active: Indexes of active atoms in this forcefield
        """

        if pars is None:
            self.pars = {}
        else:
            self.pars = pars

        self.name = name
        self.latency = latency
        self.offset = offset
        self.requests = []
        self.dopbc = dopbc
        self.active = active
        self.iactive = None
        self.threaded = threaded
        self._thread = None
        self._doloop = [False]
        self._threadlock = threading.Lock()

    def bind(self, output_maker=None):
        """Binds the FF, at present just to allow for
        managed output"""

        self.output_maker = output_maker

    def queue(self, atoms, cell, reqid=-1, template=None):
        """Adds a request.

        Note that the pars dictionary need to be sent as a string of a
        standard format so that the initialisation of the driver can be done.

        Args:
            atoms: An Atoms object giving the atom positions.
            cell: A Cell object giving the system box.
            pars: An optional dictionary giving the parameters to be sent to the
                driver for initialisation. Defaults to {}.
            reqid: An optional integer that identifies requests of the same type,
               e.g. the bead index
            template: a dict giving a base model for the request item -
               e.g. to add entries that are not needed for the base class execution

        Returns:
            A dict giving the status of the request of the form {'pos': An array
            giving the atom positions folded back into the unit cell,
            'cell': Cell object giving the system box, 'pars': parameter string,
            'result': holds the result as a list once the computation is done,
            'status': a string labelling the status of the calculation,
            'id': the id of the request, usually the bead number, 'start':
            the starting time for the calculation, used to check for timeouts.}.
        """

        par_str = " "

        if self.pars is not None:
            for k, v in list(self.pars.items()):
                par_str += k + " : " + str(v) + " , "
        else:
            par_str = " "

        pbcpos = dstrip(atoms.q).copy()

        # Indexes come from input in a per atom basis and we need to make a per atom-coordinate basis
        # Reformat indexes for full system (default) or piece of system
        # active atoms do not change but we only know how to build this array once we get the positions once
        if self.iactive is None:
            if self.active[0] == -1:
                activehere = np.arange(len(pbcpos))
            else:
                activehere = np.array(
                    [[3 * n, 3 * n + 1, 3 * n + 2] for n in self.active]
                )

            # Reassign active indexes in order to use them
            activehere = activehere.flatten()

            # Perform sanity check for active atoms
            if len(activehere) > len(pbcpos) or activehere[-1] > (len(pbcpos) - 1):
                raise ValueError("There are more active atoms than atoms!")

            self.iactive = activehere

        if self.dopbc:
            cell.array_pbc(pbcpos)

        if template is None:
            template = {}
        template.update(
            {
                "id": reqid,
                "pos": pbcpos,
                "active": self.iactive,
                "cell": (dstrip(cell.h).copy(), dstrip(cell.ih).copy()),
                "pars": par_str,
                "result": None,
                "status": "Queued",
                "start": -1,
                "t_queued": time.time(),
                "t_dispatched": 0,
                "t_finished": 0,
            }
        )

        newreq = ForceRequest(template)

        with self._threadlock:
            self.requests.append(newreq)

        if not self.threaded:
            self.poll()

        return newreq

    def poll(self):
        """Polls the forcefield object to check if it has finished."""

        with self._threadlock:
            for r in self.requests:
                if r["status"] == "Queued":
                    r["t_dispatched"] = time.time()
                    r["result"] = [
                        0.0 - self.offset,
                        np.zeros(len(r["pos"]), float),
                        np.zeros((3, 3), float),
                        {"raw": ""},
                    ]
                    r["status"] = "Done"
                    r["t_finished"] = time.time()

    def _poll_loop(self):
        """Polling loop.

        Loops over the different requests, checking to see when they have
        finished.
        """

        info(
            f" @ForceField ({self.name}): Starting the polling thread main loop.",
            verbosity.low,
        )
        while self._doloop[0]:
            time.sleep(self.latency)
            if len(self.requests) > 0:
                self.poll()

    def release(self, request, lock=True):
        """Removes a request from the evaluation queue.

        Args:
            request: The id of the job to release.
            lock: whether we should apply a threadlock here
        """

        """Frees up a request."""

        with self._threadlock if lock else nullcontext():
            if request in self.requests:
                try:
                    self.requests.remove(request)
                except ValueError:
                    print("failed removing request", id(request), " ", end=" ")
                    print(
                        [id(r) for r in self.requests], "@", threading.currentThread()
                    )
                    raise

    def stop(self):
        """Dummy stop method."""

        self._doloop[0] = False
        for r in self.requests:
            r["status"] = "Exit"

    def start(self):
        """Spawns a new thread.

        Splits the main program into two threads, one that runs the polling loop
        which updates the client list, and one which gets the data.

        Raises:
            NameError: Raised if the polling thread already exists.
        """

        if self._thread is not None:
            raise NameError("Polling thread already started")

        if self.threaded:
            self._doloop[0] = True
            self._thread = threading.Thread(
                target=self._poll_loop, name="poll_" + self.name
            )
            self._thread.daemon = True
            self._thread.start()
            softexit.register_thread(self._thread, self._doloop)
        softexit.register_function(self.softexit)

    def softexit(self):
        """Takes care of cleaning up upon softexit"""

        self.stop()

    def update(self):
        """Makes updates to the potential that only need to be triggered
        upon completion of a time step."""

        pass


class FFSocket(ForceField):
    """Interface between the PIMD code and a socket for a single replica.

    Deals with an individual replica of the system, obtaining the potential
    force and virial appropriate to this system. Deals with the distribution of
    jobs to the interface.

    This class also provides optional extensions used for constant potential
    simulations. When enabled, these extensions allow FFSocket to:

        * propagate an electronic charge/NELECT to a single-endpoint driver
          (VASP-style constant potential, ``charge=True, mixing=False``);
        * delegate to an internal two-endpoint mixing backend
          (CP2K-style constant potential, ``charge=True, mixing=True``).

    In the default configuration (``charge=False`` and ``mixing=False``) the
    behaviour is identical to the original i-PI FFSocket implementation.

    Attributes:
        socket: The interface object which contains the socket through which
            communication between the forcefield and the driver is done, for
            the single-endpoint modes.
    """

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        pars=None,
        dopbc=True,
        active=np.array([-1]),
        threaded=True,
        interface=None,
        charge_enabled=False,
        mixing_enabled=False,
        Ne_doping=False,
        client=None,
    ):
        """Initialises FFSocket.

        Args:
           latency: The number of seconds the socket will wait before updating
              the client list.
           name: The name of the forcefield.
           pars: A dictionary used to initialize the forcefield, if required.
              Of the form {'name1': value1, 'name2': value2, ... }.
           dopbc: Decides whether or not to apply the periodic boundary conditions
              before sending the positions to the client code.
           interface: The object used to create the socket used to interact
              with the client codes (single-endpoint modes).
           charge_enabled: If True, enable electronic charge coupling to the
              underlying driver (constant potential / workfunction modes).
           mixing_enabled: If True together with ``charge_enabled``, enable
              two-endpoint mixing backend instead of a single socket endpoint.
        """

        # a socket to the communication library is created or linked
        super(FFSocket, self).__init__(
            latency, offset, name, pars, dopbc, active, threaded
        )

        # Flags that control constant-potential extensions. The default is the
        # plain single-endpoint socket behaviour.
        self.charge_enabled = bool(charge_enabled)
        self.mixing_enabled = bool(mixing_enabled)
        # Optional Ne-doping mode used for VASP constant-potential runs.
        # When enabled (and mixing_enabled is False) the CHGDATA payload is
        # interpreted as the Ne ZVAL value instead of the total NELECT.
        self.Ne_doping = bool(Ne_doping)

        # In the default and single-endpoint charge-coupled modes we still use
        # the standard InterfaceSocket object.
        if not self.mixing_enabled:
            if interface is None:
                self.socket = InterfaceSocket()
            else:
                self.socket = interface
            self.socket.requests = self.requests
            self.socket.offset = self.offset

        # Placeholders for electronic state and mixing backends. These will be
        # wired up by higher-level helpers in subsequent refactoring steps.
        #
        # current_nelect stores the target total electron number communicated
        # by the electronic degrees of freedom in constant-potential mode.
        self.current_nelect = None  # used when charge_enabled and not mixing
        self._mixing_backend = None  # used when charge_enabled and mixing

        # Neutral electron count (for the reference neutral system) is
        # provided by the ElectronicState via Dynamics. It is required when
        # running in Ne-doping mode so that the per-Ne ZVAL can be
        # reconstructed from the target total electron number.
        self.neutral_electrons = None

        # Cached number of Ne atoms in the simulated system, only used when
        # Ne_doping is enabled.
        self._ne_atom_count = None

        # Reference neutral valence electron count for Ne (fixed to 8).
        self.zval_neutral = 8.0

        # Backend identifier for Ne_doping constant-potential mode. This is
        # typically "vasp" or "cp2k" and is treated in a case-insensitive
        # manner. When not provided, we keep it as None and rely on the input
        # layer to enforce presence only when Ne_doping is enabled.
        try:
            self.client = str(client).lower() if client is not None else None
        except Exception:
            self.client = None

    def poll(self):
        """Function to check the status of the client calculations."""

        self.socket.poll()

    def start(self):
        """Spawns a new thread."""

        self.socket.open()
        super(FFSocket, self).start()

    def stop(self):
        """Closes the socket and the thread."""

        super(FFSocket, self).stop()
        if self._thread is not None:
            # must wait until loop has ended before closing the socket
            self._thread.join()
        self.socket.close()

    def set_electronic_state(self, q):
        """Receive current electronic charge for constant potential simulations.

        In the single-endpoint mode (``charge_enabled=True, mixing_enabled=False``)
        the value is stored and later sent to the driver as NELECT.
        For other modes this is currently a no-op.
        """

        if not self.charge_enabled:
            return

        # For now we only act in the single-endpoint mode. Mixing backends
        # will hook into this entry point in a later refactoring step.
        if self.mixing_enabled:
            return

        if q is None:
            return
        try:
            q_val = float(q)
        except Exception:
            return
        if q_val <= 0.0:
            return
        self.current_nelect = q_val

    def queue(self, atoms, cell, reqid=-1, template=None):
        """Adds a request and optionally injects charge data for constant potential.

        In the default single-endpoint constant-potential mode
        (``charge_enabled=True, mixing_enabled=False, Ne_doping=False``), this
        method appends an ``NELECT : value`` token to the free-form parameter
        string and sends the same value via the CHGDATA message.

        When ``Ne_doping=True`` (and still ``mixing_enabled=False``), the
        CHGDATA payload is reinterpreted as the Ne ZVAL value ``zval_ne``.
        The total target electron number is still provided via
        ``current_nelect``, but is combined with ``neutral_electrons`` and the
        number of Ne atoms to reconstruct ``zval_ne`` as::

            delta_q  = current_nelect - neutral_electrons
            zval_ne  = zval_neutral + delta_q / N_ne

        The INIT parameter string will in this case contain ``NELECT``,
        ``neutral_electrons`` and ``zval_ne`` tokens, which are parsed on the
        VASP side during the INITSTR phase.
        """

        # Build base request using parent implementation
        request = super(FFSocket, self).queue(atoms, cell, reqid=reqid, template=template)

        # Only act in single-endpoint charge-coupled mode.
        if not self.charge_enabled or self.mixing_enabled:
            return request

        # No electronic state available yet: nothing to inject.
        if self.current_nelect is None:
            return request

        client_id = getattr(self, "client", None)
        client_lower = str(client_id).lower() if client_id is not None else ""

        # Attempt to locate the Dynamics object and associated electrons_config
        # so that we can access solvation_stride and enforce compatibility with
        # Ne_doping as well as compute the per-step implicit-solvent flag.
        dynamics = getattr(self, "_dynamics_ref", None)
        if dynamics is None:
            dynamics = self._find_dynamics_object()
        electrons_config = getattr(dynamics, "electrons_config", None) if dynamics is not None else None

        # In Ne_doping mode we do not support implicit solvent stride control.
        # If the user attempts to activate electrons with a custom
        # solvation_stride, raise an explicit error.
        if self.Ne_doping and isinstance(electrons_config, dict) and electrons_config.get("enabled", False):
            stride_val = electrons_config.get("solvation_stride", 1)
            try:
                stride_int = int(stride_val)
            except Exception:
                stride_int = 1
            if stride_int != 1:
                raise ValueError(
                    "FFSocket: Ne_doping mode does not support solvation_stride / implicit solvent stride control."
                )

        # Compute per-step implicit-solvent enable flag for VASP single-endpoint
        # runs when electrons are enabled and Ne_doping is disabled. This flag
        # will be propagated via the CHGDATA lightweight message.
        solvation_flag = None
        if (
            not self.Ne_doping
            and client_lower == "vasp"
            and isinstance(electrons_config, dict)
            and electrons_config.get("enabled", False)
        ):
            # Default to enabling solvation every step; if we cannot access an
            # integrator step counter, log this and keep the default.
            step_index = 0
            integrator = getattr(dynamics, "integrator", None) if dynamics is not None else None
            if integrator is not None and hasattr(integrator, "_simulation_step"):
                try:
                    raw_step = int(getattr(integrator, "_simulation_step", 0))
                except Exception:
                    raw_step = 0
                # _simulation_step is incremented at the beginning of each MD
                # step on the integrator side; convert to a zero-based index.
                step_index = max(raw_step - 1, 0)
            else:
                # info(
                #     " @FFSocket: Non-NVT or unknown integrator detected; "
                #     "implicit solvent stride control falls back to enabling every step.",
                #     verbosity.medium,
                # )

            stride_val = electrons_config.get("solvation_stride", 1)
            try:
                stride_int = int(stride_val)
            except Exception:
                stride_int = 1
            if stride_int <= 0:
                solvation_flag = 0
            else:
                solvation_flag = 1 if (step_index % stride_int == 0) else 0

            request["solvation_flag"] = int(solvation_flag)

        # Standard path: no Ne-doping. Behave as before and send NELECT, and in
        # VASP workfunction mode also pass the Z-average region in Angstrom.
        if not self.Ne_doping:
            nelect_str = f"{self.current_nelect:.4f}"
            base_pars = request.get("pars", " ") or " "

            if "NELECT" not in base_pars:
                base_pars_stripped = base_pars.strip()
                if base_pars_stripped:
                    base_pars_stripped += " , "
                base_pars_stripped += f"NELECT : {nelect_str}"
                base_pars = base_pars_stripped

            # In VASP single-endpoint workfunction mode, propagate the
            # Z-average region [z_min, z_max] in Angstrom via the INIT
            # parameter string so that the driver can compute the
            # workfunction internally without LOCPOT/LOCPOT_Z files.
            if client_lower == "vasp":
                try:
                    dynamics = getattr(self, "_dynamics_ref", None)
                    if dynamics is None:
                        dynamics = self._find_dynamics_object()
                    electronic_state = getattr(dynamics, "electronic_state", None)
                    mode = getattr(electronic_state, "mode", "fermi")
                except Exception:
                    mode = "fermi"

                if mode == "workfunction":
                    try:
                        z_min_A, z_max_A = self._get_z_average_region_A()

                        if "z_avg_min_A" not in base_pars:
                            base_pars_stripped = base_pars.strip()
                            if base_pars_stripped:
                                base_pars_stripped += " , "
                            base_pars_stripped += f"z_avg_min_A : {z_min_A:.8f}"
                            base_pars = base_pars_stripped

                        if "z_avg_max_A" not in base_pars:
                            base_pars_stripped = base_pars.strip()
                            if base_pars_stripped:
                                base_pars_stripped += " , "
                            base_pars_stripped += f"z_avg_max_A : {z_max_A:.8f}"
                            base_pars = base_pars_stripped
                    except Exception:
                        pass

            request["pars"] = base_pars
            request["nelect"] = float(self.current_nelect)
            return request

        # --- Ne-doping path ---
        # We require a positive neutral_electrons value to reconstruct the
        # Ne-related doping variable (zval_ne for VASP or core_corr_ne for
        # CP2K) from the target total electron number.
        if self.neutral_electrons is None or float(self.neutral_electrons) <= 0.0:
            raise RuntimeError(
                "FFSocket: Ne_doping enabled but neutral_electrons is not set or non-positive. "
                "Ensure that ElectronicState.neutral_electrons is configured and propagated before the first step."
            )

        # Lazily determine the number of Ne atoms in the system.
        if self._ne_atom_count is None:
            names = getattr(atoms, "names", None)
            if names is None:
                raise RuntimeError(
                    "FFSocket: Ne_doping enabled but atoms.names is not available to count Ne atoms."
                )
            try:
                name_array = dstrip(names)
            except Exception:
                name_array = names

            # Ensure we iterate over a flat NumPy array; this also normalizes
            # possible depend_array views and other sequence types.
            try:
                name_array_np = np.asarray(name_array).ravel()
            except Exception:
                name_array_np = name_array

            count_ne = 0
            for nm in name_array_np:
                # Robust conversion of atomic labels to strings: handle bytes,
                # NumPy string scalars, and generic objects in a consistent way.
                try:
                    if isinstance(nm, bytes):
                        symbol = nm.decode(errors="ignore")
                    else:
                        # NumPy string scalars and generic objects
                        symbol = str(nm)
                except Exception:
                    symbol = str(nm)

                symu = symbol.strip().upper()
                # Accept standard "NE" as well as common variants such as
                # "NEON", "NE1", etc., by matching on the "NE" prefix.
                if symu == "NE" or symu.startswith("NE"):
                    count_ne += 1

            if count_ne <= 0:
                # Diagnostic logging to understand what atomic symbols are seen
                # at runtime when Ne_doping is enabled but no Ne atoms are
                # detected. This helps track issues with atoms.names typing or
                # parsing.
                try:
                    symbols_debug = []
                    try:
                        iterable = name_array_np
                    except NameError:
                        iterable = name_array
                    for nm in iterable:
                        try:
                            if isinstance(nm, bytes):
                                s = nm.decode(errors="ignore")
                            else:
                                s = str(nm)
                        except Exception:
                            s = str(nm)
                        symbols_debug.append(s)
                    unique_syms = sorted(set(symbols_debug))
                    natoms_debug = len(symbols_debug)
                    warning(
                        "FFSocket Ne_doping debug: could not find any Ne atoms. "
                        f"Unique symbols={unique_syms}, natoms={natoms_debug}",
                        verbosity.medium,
                    )
                except Exception:
                    # Best-effort debug only; do not mask the original error.
                    pass

                raise ValueError(
                    "FFSocket: Ne_doping='true' but no Ne atoms were found in the atomic names."
                )

            self._ne_atom_count = int(count_ne)

        # Compute shared quantities from the target total electron number.
        q_target = float(self.current_nelect)
        ne_ref = float(self.neutral_electrons)
        N_ne = float(self._ne_atom_count)

        delta_q = q_target - ne_ref

        # Select backend behaviour based on the client identifier. For VASP we
        # retain the existing semantics where the CHGDATA payload is zval_ne.
        # For CP2K we interpret the payload as the per-Ne CORE_CORRECTION
        # value core_corr_ne = delta_q / N_ne.
        client_id = getattr(self, "client", None)
        client_lower = str(client_id).lower() if client_id is not None else ""

        if client_lower == "vasp":
            zval_ne = self.zval_neutral + delta_q / N_ne

            base_pars = request.get("pars", " ") or " "
            base_pars_stripped = base_pars.strip()

            def _append_token(current, token_key, token_value):
                if token_key in current:
                    return current
                if current:
                    current += " , "
                current += f"{token_key} : {token_value}"
                return current

            if "NELECT" not in base_pars:
                nelect_str = f"{q_target:.4f}"
                base_pars_stripped = _append_token(base_pars_stripped, "NELECT", nelect_str)

            if "neutral_electrons" not in base_pars:
                ne_str = f"{int(ne_ref)}"
                base_pars_stripped = _append_token(base_pars_stripped, "neutral_electrons", ne_str)

            if "zval_ne" not in base_pars:
                zval_ne_str = f"{zval_ne:.6f}"
                base_pars_stripped = _append_token(base_pars_stripped, "zval_ne", zval_ne_str)

            if base_pars_stripped:
                request["pars"] = base_pars_stripped

            # In VASP Ne-doping mode the CHGDATA payload is the Ne ZVAL, not
            # NELECT.
            request["nelect"] = float(zval_ne)
            return request

        if client_lower == "cp2k":
            # For CP2K Ne-doping constant-potential simulations we interpret
            # the CHGDATA payload as the per-Ne CORE_CORRECTION value. A
            # positive delta_q corresponds to a positive core_corr_ne.
            core_corr_ne = delta_q / N_ne
            request["nelect"] = float(core_corr_ne)
            return request

        # Any other client identifier is not supported in Ne_doping mode.
        raise RuntimeError(
            f"FFSocket: Unsupported client '{client_id}' for Ne_doping; expected 'vasp' or 'cp2k'."
        )

    def _get_z_average_region_A(self):
        """Get Z-average region [z_min, z_max] in Angstrom for workfunction mode.

        The region is taken from the Dynamics.electrons_config["z_average_region"],
        provided in internal length units (Bohr), and converted to Angstrom.
        """

        # Cache to avoid repeated lookups and conversions
        if hasattr(self, "_z_average_region_A") and self._z_average_region_A is not None:
            return self._z_average_region_A

        dynamics = getattr(self, "_dynamics_ref", None)
        if dynamics is None:
            dynamics = self._find_dynamics_object()
        if dynamics is None:
            raise RuntimeError("FFCPVasp: Could not locate Dynamics object for z_average_region.")

        electrons_config = getattr(dynamics, "electrons_config", None)
        if not isinstance(electrons_config, dict):
            raise RuntimeError("FFCPVasp: Dynamics.electrons_config is not available for workfunction mode.")

        if "z_average_region" not in electrons_config:
            raise RuntimeError(
                "FFCPVasp: z_average_region must be specified in electrons configuration "
                "when using workfunction mode."
            )

        z_internal = np.array(electrons_config["z_average_region"], dtype=float).flatten()
        if z_internal.size != 2:
            raise ValueError(
                "FFCPVasp: z_average_region in electrons_config must have length 2 [z_min, z_max]."
            )

        # Convert from internal length (Bohr) to Angstrom
        z_min_A = unit_to_user("length", "angstrom", z_internal[0])
        z_max_A = unit_to_user("length", "angstrom", z_internal[1])

        if z_max_A <= z_min_A:
            raise ValueError(
                f"FFCPVasp: Invalid z_average_region after conversion to Angstrom: z_min={z_min_A}, z_max={z_max_A}"
            )

        self._z_average_region_A = (float(z_min_A), float(z_max_A))
        return self._z_average_region_A

    def _compute_workfunction_from_locpot(self, fermi_level_eV):
        """Compute workfunction for a VASP endpoint using LOCPOT.

        The workfunction is defined as the planar-averaged electrostatic
        potential in the user-specified Z region minus the Fermi level
        (both in eV)::

            workfunction = V_avg_region_eV - fermi_level_eV

        LOCPOT is assumed to be in the current working directory of the
        VASP run, with filename 'LOCPOT'.
        """

        from ipi.utils.messages import info, verbosity
        import numpy as np

        # By default, assume LOCPOT_Z lives in the current working directory of the VASP run.
        locpot_z_path = os.path.join(os.getcwd(), "LOCPOT_Z")
        if not os.path.exists(locpot_z_path):
            # Treat missing LOCPOT_Z as a hard error in workfunction mode; this exception will
            # propagate up to FFSocket.update and terminate the simulation.
            raise RuntimeError(f"LOCPOT_Z file not found at {locpot_z_path}")

        # Read planar-averaged profile z(Angstrom), V_avg(eV) from LOCPOT_Z.
        z_vals = []
        v_vals = []
        with open(locpot_z_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    z_val = float(parts[0])
                    v_val = float(parts[1])
                except ValueError:
                    continue
                z_vals.append(z_val)
                v_vals.append(v_val)

        if len(z_vals) == 0:
            raise RuntimeError(f"No valid data rows found in LOCPOT_Z file {locpot_z_path}")

        z_vals_A = np.array(z_vals, dtype=float)
        z_profile_eV = np.array(v_vals, dtype=float)

        z_min_A, z_max_A = self._get_z_average_region_A()
        mask = (z_vals_A >= z_min_A) & (z_vals_A <= z_max_A)
        if not np.any(mask):
            raise ValueError(
                f"FFCPVasp: No grid points found in z_average_region [{z_min_A}, {z_max_A}] "
                f"for LOCPOT_Z file {locpot_z_path} (nz={z_vals_A.size})."
            )

        V_avg_region_eV = float(z_profile_eV[mask].mean())
        workfunction_eV = V_avg_region_eV - float(fermi_level_eV)

        # info(
        #     f" @FFCPVasp: Workfunction from LOCPOT: V_avg={V_avg_region_eV:.6f} eV, "
        #     f"E_F={fermi_level_eV:.6f} eV, phi={workfunction_eV:.6f} eV",
        #     verbosity.medium,
        # )

        return V_avg_region_eV, workfunction_eV

    def _find_dynamics_object(self):
        """Locate the Dynamics object to access electrons_config and electronic_state.

        For constant-potential / workfunction simulations we need access to the
        ``electrons_config`` and ``electronic_state`` associated with the
        active Dynamics instance *before* any Fermi cache has been created.
        Therefore we identify candidates by the presence of an ``electrons_config``
        attribute rather than relying solely on ``_fermi_cache``.
        """

        try:
            from ipi.utils.messages import info, warning, verbosity

            if hasattr(self, "_dynamics_ref") and self._dynamics_ref is not None:
                # info(" @FFCPVasp: Using cached Dynamics reference", verbosity.debug)
                return self._dynamics_ref

            import gc

            # info(" @FFCPVasp: Searching for Dynamics object...", verbosity.debug)

            candidates = []
            for obj in gc.get_objects():
                if hasattr(obj, "__class__"):
                    class_name = str(obj.__class__)
                    # Prefer Dynamics objects that expose an electrons_config
                    # attribute, which is set as soon as the <electrons> input
                    # block is bound. This allows us to access the Z-average
                    # region for workfunction mode already during the initial
                    # INIT handshake with the driver.
                    if "Dynamics" in class_name and hasattr(obj, "electrons_config"):
                        candidates.append(obj)

            if len(candidates) > 0:
                self._dynamics_ref = candidates[0]
                # info(
                #     f" @FFCPVasp: Using Dynamics object: {type(self._dynamics_ref)}",
                #     verbosity.medium,
                # )
                return self._dynamics_ref

            warning(" @FFCPVasp: No Dynamics object found for workfunction coupling", verbosity.medium)
            return None

        except Exception:
            return None

    def update(self):
        """Update hook called at the end of a time step.

        When running in single-endpoint constant-potential mode this method:

            - reads fermi_level_eV and nelect from forces.extras;
            - optionally computes the workfunction from LOCPOT in workfunction mode;
            - updates the ElectronicState and Dynamics Fermi cache if available.
        """

        super(FFSocket, self).update()

        # Only act in single-endpoint constant-potential / workfunction mode.
        if not self.charge_enabled or self.mixing_enabled:
            return

        dynamics = self._find_dynamics_object()
        if dynamics is None:
            return

        # Access extras from the aggregated forces object
        extras = getattr(dynamics.forces, "extras", None)
        if not isinstance(extras, dict):
            return

        from ipi.utils.messages import info, warning, verbosity

        # Fermi level handling (preferred key: fermi_level_eV). Note that
        # ForceComponent.extras aggregates per-bead extras into lists/arrays,
        # so we must robustly extract a scalar value.
        fermi_level_eV = None
        if "fermi_level_eV" in extras:
            val = extras["fermi_level_eV"]
            # Unwrap list/tuple/ndarray produced by extras aggregation
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) > 0:
                    val = val[0]
                else:
                    val = None
            try:
                if val is not None:
                    fermi_level_eV = float(val)
            except Exception:
                fermi_level_eV = None

        # Optionally, read back nelect from extras if provided by VASP
        nelect_returned = extras.get("nelect", None)
        if nelect_returned is not None:
            val_n = nelect_returned
            if isinstance(val_n, (list, tuple, np.ndarray)):
                if len(val_n) > 0:
                    val_n = val_n[0]
                else:
                    val_n = None
            try:
                if val_n is not None:
                    nelect_returned = float(val_n)
                else:
                    nelect_returned = None
            except Exception:
                nelect_returned = None

        electronic_state = getattr(dynamics, "electronic_state", None)
        if electronic_state is None:
            return

        debug_mode = getattr(electronic_state, "mode", None)
        try:
            extras_keys = list(extras.keys())
        except Exception:
            extras_keys = []
        # info(
        #     f" @FFCPVasp-DEBUG: electronic_state.mode = {debug_mode}, extras keys = {extras_keys}",
        #     verbosity.medium,
        # )

        # info(
        #     f" @FFCPVasp-DEBUG: electronic_state.mode = {debug_mode}, extras keys = {extras_keys}",
        #     verbosity.medium,
        # )

        # Update Fermi cache for electronic B step: cache value in eV
        if fermi_level_eV is not None:
            dynamics._fermi_cache = {"vasp_fermi": fermi_level_eV}
            dynamics._fermi_cache_valid = True

            # info(
            #     f" @FFCPVasp: Cached Fermi level from VASP: {fermi_level_eV:.6f} eV",
            #     verbosity.medium,
            # )

        # Optionally read back workfunction (in eV) directly from the driver
        # via JSON extras, falling back to LOCPOT_Z-based computation only if
        # the driver does not provide it.
        workfunction_eV = None
        if "workfunction_eV" in extras:
            val_w = extras["workfunction_eV"]
            if isinstance(val_w, (list, tuple, np.ndarray)):
                if len(val_w) > 0:
                    val_w = val_w[0]
                else:
                    val_w = None
            try:
                if val_w is not None:
                    workfunction_eV = float(val_w)
            except Exception:
                workfunction_eV = None

        mode = getattr(electronic_state, "mode", "fermi")
        if mode == "workfunction":
            if workfunction_eV is None:
                if fermi_level_eV is None:
                    return
                _, workfunction_eV = self._compute_workfunction_from_locpot(fermi_level_eV)

            workfunction_au = workfunction_eV / Constants.EV_PER_HARTREE
            electronic_state.current_workfunction = workfunction_au
            # info(
            #     f" @FFCPVasp: Updated electronic_state.current_workfunction = {workfunction_au:.6f} Ha",
            #     verbosity.medium,
            # )

            extras["workfunction_eV"] = workfunction_eV
            extras["workfunction"] = workfunction_au

        # Optionally, mirror returned nelect into current_nelect if provided
        if nelect_returned is not None and nelect_returned > 0.0:
            self.current_nelect = nelect_returned


class FFEval(ForceField):
    """General class for models that provide a self.evaluate(request)
    to compute the potential, force and virial.
    """

    def poll(self):
        """Polls the forcefield checking if there are requests that should
        be answered, and if necessary evaluates the associated forces and energy."""

        # We have to be thread-safe, as in multi-system mode this might get
        # called by many threads at once.
        with self._threadlock:
            for r in self.requests:
                if r["status"] == "Queued":
                    r["status"] = "Running"
                    r["t_dispatched"] = time.time()
                    self.evaluate(r)
                    r["result"][0] -= self.offset  # subtract constant offset

    def evaluate(self, request):
        request["result"] = [
            0.0,
            np.zeros(len(request["pos"]), float),
            np.zeros((3, 3), float),
            {"raw": ""},
        ]
        request["status"] = "Done"


class FFDirect(FFEval):
    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        pars=None,
        dopbc=False,
        active=np.array([-1]),
        threaded=False,
        pes="dummy",
    ):
        """Initialises FFDirect.

        Args:
            latency: The number of seconds the socket will wait before updating
                the client list.
            offset: A constant offset subtracted from the energy value given by the
                client.
            name: The name of the forcefield.
            pars: A dictionary used to initialize the forcefield, if required.
                Of the form {'name1': value1, 'name2': value2, ... }.
            dopbc: Decides whether or not to apply the periodic boundary conditions
                before sending the positions to the client code.
            active: Indexes of active atoms in this forcefield

        """

        super().__init__(latency, offset, name, pars, dopbc, active, threaded)

        if pars is None:
            pars = {}  # defaults no pars
        if not "verbosity" in pars:
            pars["verbosity"] = verbosity.high
        self.pes = pes
        try:
            self.driver = __drivers__[self.pes](**pars)
        except ImportError:
            # specific errors have already been triggered
            raise
        except Exception as err:
            print(f"Error setting up PES mode {self.pes}")
            print(__drivers__[self.pes].__doc__)
            print("Error trace: ")
            raise err

    def evaluate(self, request):
        results = list(self.driver(request["cell"][0], request["pos"].reshape(-1, 3)))

        # ensure forces and virial have the correct shape to fit the results
        results[1] = results[1].reshape(-1)
        results[2] = results[2].reshape(3, 3)

        # converts the extra fields, if there are any - improved compatibility for dict/JSON
        mxtra = results[3]
        if isinstance(mxtra, dict):
            mxtradict = dict(mxtra)  # directly use if already a dict
        elif isinstance(mxtra, str) and mxtra:
            try:
                mxtradict = json.loads(mxtra)
            except:
                # if we can't parse it as a dict, issue a warning and carry on
                mxtradict = {"raw": mxtra}
        else:
            mxtradict = {}

        # Validate 'raw' field conflicts
        if "raw" in mxtradict:
            raise ValueError(
                "'raw' cannot be used as a field in a JSON-formatted extra string"
            )

        mxtradict["raw"] = mxtra
        results[3] = mxtradict

        # Minimal coupling to constant-potential infrastructure: if electrons are
        # enabled and the PES provides a Fermi level in extras (in eV), cache it
        # into the active Dynamics object so that electronic_B_step can reuse it.
        try:
            from ipi.engine.motion.dynamics import Dynamics

            # Locate Dynamics instance (if any). We only care about the case where
            # electronic DOFs are present; otherwise this is a no-op.
            dynamics_obj = None

            try:
                import gc

                for obj in gc.get_objects():
                    if isinstance(obj, Dynamics) and hasattr(obj, "electrons_config"):
                        dynamics_obj = obj
                        break
            except Exception:
                dynamics_obj = None

            if dynamics_obj is not None and getattr(dynamics_obj, "electrons_config", None) is not None:
                extras = mxtradict
                fermi_level_eV = None

                if "fermi_level_eV" in extras:
                    val = extras["fermi_level_eV"]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        if len(val) > 0:
                            val = val[0]
                        else:
                            val = None
                    try:
                        if val is not None:
                            fermi_level_eV = float(val)
                    except Exception:
                        fermi_level_eV = None

                # Fall back to Hartree-valued fermi_level if only that is present
                if fermi_level_eV is None and "fermi_level" in extras:
                    try:
                        from ipi.utils.units import Constants

                        val_h = extras["fermi_level"]
                        if isinstance(val_h, (list, tuple, np.ndarray)):
                            if len(val_h) > 0:
                                val_h = val_h[0]
                            else:
                                val_h = None
                        if val_h is not None:
                            fermi_level_eV = float(val_h) * Constants.EV_PER_HARTREE
                    except Exception:
                        fermi_level_eV = None

                if fermi_level_eV is not None:
                    dynamics_obj._fermi_cache = {"direct_fermi": fermi_level_eV}
                    dynamics_obj._fermi_cache_valid = True
                    info(
                        f" @FFDirect: Cached Fermi level from direct PES: {fermi_level_eV:.6f} eV",
                        verbosity.medium,
                    )
        except Exception:
            # Any failure in this optional coupling must not affect plain FFDirect
            # behaviour for non-electronic simulations.
            pass

        request["result"] = results
        request["status"] = "Done"
        request["t_finished"] = time.time()

    def set_electronic_state(self, q):
        """Set the current electronic charge for constant potential simulations.

        This method forwards the call to the underlying driver if it supports
        electronic state management.

        Args:
            q (float): Electronic charge (must be > 0)
        """
        if hasattr(self.driver, 'set_electronic_state'):
            self.driver.set_electronic_state(q)

    def get_fermi_level(self):
        """Get the current Fermi level from the underlying driver.

        Returns:
            float: Current Fermi level in atomic units, or 0.0 if not available
        """
        if hasattr(self.driver, 'get_fermi_level'):
            return self.driver.get_fermi_level()
        else:
            return 0.0


class FFLennardJones(FFEval):
    """Basic fully pythonic force provider.

    Computes LJ interactions without minimum image convention, cutoffs or
    neighbour lists. Parallel evaluation with threads.

    Attributes:
        parameters: A dictionary of the parameters used by the driver. Of the
            form {'name': value}.
        requests: During the force calculation step this holds a dictionary
            containing the relevant data for determining the progress of the step.
            Of the form {'atoms': atoms, 'cell': cell, 'pars': parameters,
                         'status': status, 'result': result, 'id': bead id,
                         'start': starting time}.
    """

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        pars=None,
        dopbc=True,
        active=np.array([-1]),
        threaded=True,
        interface=None,
    ):
        """Initialises FFLennardJones.

        Args:
           pars: Optional dictionary, giving the parameters needed by the driver.
        """

        # check input - PBCs are not implemented here
        if dopbc:
            raise ValueError(
                "Periodic boundary conditions are not supported by FFLennardJones."
            )

        # a socket to the communication library is created or linked
        super(FFLennardJones, self).__init__(
            latency, offset, name, pars, dopbc=dopbc, threaded=threaded, active=active
        )
        self.epsfour = float(self.pars["eps"]) * 4
        self.sixepsfour = 6 * self.epsfour
        self.sigma2 = float(self.pars["sigma"]) * float(self.pars["sigma"])

    def evaluate(self, r):
        """Just a silly function evaluating a non-cutoffed, non-pbc and
        non-neighbour list LJ potential."""

        q = r["pos"].reshape((-1, 3))
        nat = len(q)

        v = 0.0
        f = np.zeros(q.shape)
        for i in range(1, nat):
            dij = q[i] - q[:i]
            rij2 = (dij**2).sum(axis=1)

            x6 = (self.sigma2 / rij2) ** 3
            x12 = x6**2

            v += (x12 - x6).sum()
            dij *= (self.sixepsfour * (2.0 * x12 - x6) / rij2)[:, np.newaxis]
            f[i] += dij.sum(axis=0)
            f[:i] -= dij

        v *= self.epsfour

        r["result"] = [v, f.reshape(nat * 3), np.zeros((3, 3), float), {"raw": ""}]
        r["status"] = "Done"


class FFdmd(FFEval):
    """Pythonic force provider.

    Computes DMD forces as in Bowman, .., Brown JCP 2003 DOI: 10.1063/1.1578475. It is a time dependent potential.
    Here extended for periodic systems and for virial term calculation.

    Attributes:
        parameters: A dictionary of the parameters used by the driver. Of the
            form {'name': value}.
        requests: During the force calculation step this holds a dictionary
            containing the relevant data for determining the progress of the step.
            Of the form {'atoms': atoms, 'cell': cell, 'pars': parameters,
                         'status': status, 'result': result, 'id': bead id,
                         'start': starting time}.
    """

    def __init__(
        self,
        latency=1.0e-3,
        offset=0.0,
        name="",
        coupling=None,
        freq=0.0,
        dtdmd=0.0,
        dmdstep=0,
        pars=None,
        dopbc=False,
        threaded=False,
    ):
        """Initialises FFdmd.

        Args:
           pars: Optional dictionary, giving the parameters needed by the driver.
        """

        # a socket to the communication library is created or linked
        super(FFdmd, self).__init__(
            latency, offset, name, pars, dopbc=dopbc, threaded=threaded
        )

        if coupling is None:
            raise ValueError("Must provide the couplings for DMD.")
        if freq is None:
            raise ValueError(
                "Must provide a frequency for the periodically oscillating potential."
            )
        if dtdmd is None:
            raise ValueError(
                "Must provide a time step for the periodically oscillating potential."
            )
        self.coupling = coupling
        self.freq = freq
        self.dtdmd = dtdmd
        self.dmdstep = dmdstep

    def evaluate(self, r):
        """Evaluating dmd: pbc: YES,
        cutoff: NO, neighbour list: NO."""

        q = r["pos"].reshape((-1, 3))
        nat = len(q)
        cell_h, cell_ih = r["cell"]

        if len(self.coupling) != int(nat * (nat - 1) / 2):
            raise ValueError("Coupling matrix size mismatch")

        v = 0.0
        f = np.zeros(q.shape)
        vir = np.zeros((3, 3), float)
        # must think and check handling of time step
        periodic = np.sin(self.dmdstep * self.freq * self.dtdmd)
        # MR: the algorithm below has been benchmarked against explicit loop implementation
        for i in range(1, nat):
            # MR's first implementation:
            #            dij = q[i] - q[:i]
            #            rij = np.sqrt((dij ** 2).sum(axis=1))
            # KF's implementation:
            dij, rij = vector_separation(cell_h, cell_ih, q[i], q[:i])
            cij = self.coupling[i * (i - 1) // 2 : i * (i + 1) // 2]
            prefac = np.dot(
                cij, rij
            )  # for each i it has the distances to all indexes previous
            v += np.sum(prefac) * periodic
            nij = np.copy(dij)
            nij *= -(cij / rij)[:, np.newaxis]  # magic line...
            f[i] += nij.sum(axis=0) * periodic
            f[:i] -= nij * periodic  # everything symmetric
            # virial:
            fij = nij * periodic
            for j in range(i):
                for cart1 in range(3):
                    for cart2 in range(3):
                        vir[cart1][cart2] += fij[j][cart1] * dij[j][cart2]
            # MR 2021: The virial looks correct and produces stable NPT simulations. It was not bullet-proof benchmarked, though.
            #          Because this is "out of equilibrium" I still did not find a good benchmark. Change cell and look at variation of energy only for this term?

        r["result"] = [v, f.reshape(nat * 3), vir, ""]
        r["status"] = "Done"

    def dmd_update(self):
        """Updates time step when a full step is done. Can only be called after implementation goes into smotion mode..."""
        self.dmdstep += 1


class FFDebye(FFEval):
    """Debye crystal harmonic reference potential

    Computes a harmonic forcefield.

    Attributes:
       parameters: A dictionary of the parameters used by the driver. Of the
          form {'name': value}.
       requests: During the force calculation step this holds a dictionary
          containing the relevant data for determining the progress of the step.
          Of the form {'atoms': atoms, 'cell': cell, 'pars': parameters,
                       'status': status, 'result': result, 'id': bead id,
                       'start': starting time}.
    """

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        H=None,
        xref=None,
        vref=0.0,
        pars=None,
        dopbc=False,
        threaded=False,
    ):
        """Initialises FFDebye.

        Args:
           pars: Optional dictionary, giving the parameters needed by the driver.
        """

        # a socket to the communication library is created or linked
        # NEVER DO PBC -- forces here are computed without.
        super(FFDebye, self).__init__(latency, offset, name, pars, dopbc=False)

        if H is None:
            raise ValueError("Must provide the Hessian for the Debye crystal.")
        if xref is None:
            raise ValueError(
                "Must provide a reference configuration for the Debye crystal."
            )

        self.H = H
        self.xref = xref
        self.vref = vref

        eigsys = np.linalg.eigh(self.H)
        info(
            " @ForceField: Hamiltonian eigenvalues: " + " ".join(map(str, eigsys[0])),
            verbosity.medium,
        )

    def evaluate(self, r):
        """A simple evaluator for a harmonic Debye crystal potential."""

        q = r["pos"]
        n3 = len(q)
        if self.H.shape != (n3, n3):
            raise ValueError("Hessian size mismatch")
        if self.xref.shape != (n3,):
            raise ValueError("Reference structure size mismatch")

        d = q - self.xref
        mf = np.dot(self.H, d)

        r["result"] = [
            self.vref + 0.5 * np.dot(d, mf),
            -mf,
            np.zeros((3, 3), float),
            {"raw": ""},
        ]
        r["status"] = "Done"
        r["t_finished"] = time.time()


class FFPlumed(FFEval):
    """Direct PLUMED interface

    Computes forces from a PLUMED input.

    Attributes:
        parameters: A dictionary of the parameters used by the driver. Of the
            form {'name': value}.
        requests: During the force calculation step this holds a dictionary
            containing the relevant data for determining the progress of the step.
            Of the form {'atoms': atoms, 'cell': cell, 'pars': parameters,
                      'status': status, 'result': result, 'id': bead id,
                      'start': starting time}.
    """

    def __init__(
        self,
        latency=1.0e-3,
        offset=0.0,
        name="",
        pars=None,
        dopbc=False,
        threaded=False,
        init_file="",
        compute_work=True,
        plumed_dat="",
        plumed_step=0,
        plumed_extras=[],
    ):
        """Initialises FFPlumed.

        Args:
           pars: Optional dictionary, giving the parameters needed by the driver.
        """

        global plumed
        # a socket to the communication library is created or linked
        try:
            import plumed
        except ImportError:
            raise ImportError(
                "Cannot find plumed libraries to link to a FFPlumed object/"
            )
        super(FFPlumed, self).__init__(
            latency, offset, name, pars, dopbc=False, threaded=threaded
        )
        self.plumed = plumed.Plumed()
        self.plumed_dat = plumed_dat
        self.plumed_step = plumed_step
        self.plumed_extras = plumed_extras
        self.compute_work = compute_work
        self.init_file = init_file

        if self.init_file.mode == "xyz":
            infile = open(self.init_file.value, "r")
            myframe = read_file(self.init_file.mode, infile)
            myatoms = myframe["atoms"]
            mycell = myframe["cell"]
            myatoms.q *= unit_to_internal("length", self.init_file.units, 1.0)
            mycell.h *= unit_to_internal("length", self.init_file.units, 1.0)

        self.natoms = myatoms.natoms
        self.plumed.cmd("setRealPrecision", 8)  # i-PI uses double precision
        self.plumed.cmd("setMDEngine", "i-pi")
        self.plumed.cmd("setPlumedDat", self.plumed_dat)
        self.plumed.cmd("setNatoms", self.natoms)
        timeunit = 2.4188843e-05  # atomic time to ps
        self.plumed.cmd("setMDTimeUnits", timeunit)
        # given we don't necessarily call plumed once per step, so time does not make
        # sense, we set the time step so that time in plumed is a counter of the number of times
        # called
        self.plumed.cmd("setTimestep", 1 / timeunit)
        self.plumed.cmd(
            "setMDEnergyUnits", 2625.4996
        )  # Pass a pointer to the conversion factor between the energy unit used in your code and kJ mol-1
        self.plumed.cmd(
            "setMDLengthUnits", 0.052917721
        )  # Pass a pointer to the conversion factor between the length unit used in your code and nm
        self.plumedrestart = False
        if self.plumed_step > 0:
            # we are restarting, signal that PLUMED should continue
            self.plumedrestart = True
            self.plumed.cmd("setRestart", 1)
        self.plumed.cmd("init")

        self.plumed_data = {}
        for x in plumed_extras:
            rank = np.zeros(1, dtype=np.int_)
            self.plumed.cmd(f"getDataRank {x}", rank)
            if rank[0] > 1:
                raise ValueError("Cannot retrieve varibles with rank > 1")
            shape = np.zeros(rank[0], dtype=np.int_)
            if shape[0] > 1:
                raise ValueError("Cannot retrieve varibles with size > 1")
            self.plumed.cmd(f"getDataShape {x}", shape)
            self.plumed_data[x] = np.zeros(shape, dtype=np.double)
            self.plumed.cmd(f"setMemoryForData {x}", self.plumed_data[x])

        self.charges = dstrip(myatoms.q) * 0.0
        self.masses = dstrip(myatoms.m)
        self.lastq = np.zeros(3 * self.natoms)
        self.system_force = None  # reference to physical force calculator
        softexit.register_function(self.softexit)

    def softexit(self):
        """Takes care of cleaning up upon softexit"""

        self.plumed.finalize()

    def evaluate(self, r):
        """A wrapper function to call the PLUMED evaluation routines
        and return forces."""

        if self.natoms != len(r["pos"]) / 3:
            raise ValueError(
                "Size of atom array changed after initialization of FFPlumed"
            )

        v = 0.0
        f = np.zeros((self.natoms, 3))
        vir = np.zeros((3, 3))

        self.lastq[:] = r["pos"]
        # for the moment these are set to dummy values taken from an init file.
        # linking with the current value in simulations is non-trivial, as masses
        # are not expected to be the force evaluator's business, and charges are not
        # i-PI's business.
        self.plumed.cmd("setStep", self.plumed_step)
        self.plumed.cmd("setCharges", self.charges)
        self.plumed.cmd("setMasses", self.masses)

        # these instead are set properly. units conversion is done on the PLUMED side
        self.plumed.cmd("setBox", r["cell"][0].T.copy())
        pos = r["pos"].reshape(-1, 3)

        if self.system_force is not None:
            f[:] = dstrip(self.system_force.f).reshape((-1, 3))
            vir[:] = -dstrip(self.system_force.vir)
            self.plumed.cmd("setEnergy", dstrip(self.system_force.pot))

        self.plumed.cmd("setPositions", pos)
        self.plumed.cmd("setForces", f)
        self.plumed.cmd("setVirial", vir)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")

        bias = np.zeros(1, float)
        self.plumed.cmd("getBias", bias)
        v = bias[0]
        f = f.flatten()
        vir *= -1

        if self.system_force is not None:
            # plumed increments the value of the force, here we need only the correction term
            f[:] -= dstrip(self.system_force.f).flatten()
            vir[:] -= -dstrip(self.system_force.vir)

        extras = {"raw": ""}
        for x in self.plumed_data:
            extras[str(x)] = self.plumed_data[x].copy()

        # nb: the virial is a symmetric tensor, so we don't need to transpose
        r["result"] = [v, f, vir, extras]
        r["status"] = "Done"

    def mtd_update(self, pos, cell):
        """Makes updates to the potential that only need to be triggered
        upon completion of a time step."""

        # NB - this assumes this is called at the end of a step,
        # when the bias has already been computed to integrate MD
        # unexpected behavior will happen if it's called when the
        # bias force is not "freshly computed"

        self.plumed_step += 1

        bias_before = np.zeros(1, float)
        bias_after = np.zeros(1, float)

        if self.compute_work:
            self.plumed.cmd("getBias", bias_before)

        # Checks that the update is called on the right position.
        # this should be the case for most workflows - if this error
        # is triggered and your input makes sense, the right thing to
        # do is to perform a full plumed-side update (which will have a cost,
        # so see if you can avoid it)
        if np.linalg.norm(self.lastq - pos) > 1e-10:
            raise ValueError(
                "Metadynamics update is performed using an incorrect position"
            )

        # sets the step and does the actual update
        self.plumed.cmd("setStep", self.plumed_step)
        self.plumed.cmd("update")

        # recompute the bias so we can compute the work
        if self.compute_work:
            self.plumed.cmd("performCalcNoForces")
            self.plumed.cmd("getBias", bias_after)

        work = (bias_before - bias_after).item()

        return work


class FFYaff(FFEval):
    """Use Yaff as a library to construct a force field"""

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        threaded=False,
        yaffpara=None,
        yaffsys=None,
        yafflog="yaff.log",
        rcut=18.89726133921252,
        alpha_scale=3.5,
        gcut_scale=1.1,
        skin=0,
        smooth_ei=False,
        reci_ei="ewald",
        pars=None,
        dopbc=False,
    ):
        """Initialises FFYaff and enables a basic Yaff force field.

        Args:

           yaffpara: File name of the Yaff parameter file

           yaffsys: File name of the Yaff system file

           yafflog: File name to which Yaff will write some information about the system and the force field

           pars: Optional dictionary, giving the parameters needed by the driver.

           **kwargs: All keyword arguments that can be provided when generating
                     a Yaff force field; see constructor of FFArgs in Yaff code

        """

        warning(
            """
                <ffyaff> is deprecated and might be removed in a future release of i-PI.
                If you are interested in using it, please help port it to the PES
                infrastructure.
                """
        )

        from yaff import System, ForceField, log
        import codecs
        import locale
        import atexit

        # a socket to the communication library is created or linked
        super(FFYaff, self).__init__(
            latency, offset, name, pars, dopbc, threaded=threaded
        )

        # A bit weird to use keyword argument for a required argument, but this
        # is also done in the code above.
        if yaffpara is None:
            raise ValueError("Must provide a Yaff parameter file.")

        if yaffsys is None:
            raise ValueError("Must provide a Yaff system file.")

        self.yaffpara = yaffpara
        self.yaffsys = yaffsys
        self.rcut = rcut
        self.alpha_scale = alpha_scale
        self.gcut_scale = gcut_scale
        self.skin = skin
        self.smooth_ei = smooth_ei
        self.reci_ei = reci_ei
        self.yafflog = yafflog

        # Open log file
        logf = open(yafflog, "w")
        # Tell Python to close the file when the script exits
        atexit.register(logf.close)

        # Redirect Yaff log to file
        log._file = codecs.getwriter(locale.getpreferredencoding())(logf)

        self.system = System.from_file(self.yaffsys)
        self.ff = ForceField.generate(
            self.system,
            self.yaffpara,
            rcut=self.rcut,
            alpha_scale=self.alpha_scale,
            gcut_scale=self.gcut_scale,
            skin=self.skin,
            smooth_ei=self.smooth_ei,
            reci_ei=self.reci_ei,
        )

        log._active = False

    def evaluate(self, r):
        """Evaluate the energy and forces with the Yaff force field."""

        q = r["pos"]
        nat = len(q) / 3
        rvecs = r["cell"][0]

        self.ff.update_rvecs(np.ascontiguousarray(rvecs.T, dtype=np.float64))
        self.ff.update_pos(q.reshape((nat, 3)))
        gpos = np.zeros((nat, 3))
        vtens = np.zeros((3, 3))
        e = self.ff.compute(gpos, vtens)

        r["result"] = [e, -gpos.ravel(), -vtens, {"raw": ""}]
        r["status"] = "Done"


class FFsGDML(FFEval):
    """A symmetric Gradient Domain Machine Learning (sGDML) force field.
    Chmiela et al. Sci. Adv., 3(5), e1603015, 2017; Nat. Commun., 9(1), 3887, 2018.
    http://sgdml.org/doc/
    https://github.com/stefanch/sGDML
    """

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        threaded=False,
        sGDML_model=None,
        pars=None,
        dopbc=False,
    ):
        """Initialises FFsGDML

        Args:

           sGDML_model: Filename contaning the sGDML model

        """

        warning(
            """
                <ffsgdml> is deprecated and might be removed in a future release of i-PI.
                If you are interested in using it, please help port it to the PES
                infrastructure.
                """
        )

        # a socket to the communication library is created or linked
        super(FFsGDML, self).__init__(
            latency, offset, name, pars, dopbc, threaded=threaded
        )

        from ipi.utils.units import unit_to_user

        # --- Load sGDML package ---
        try:
            from sgdml.predict import GDMLPredict
            from sgdml import __version__

            info(" @ForceField: Using sGDML version " + __version__, verbosity.low)
        except ImportError:
            raise ValueError(
                "ERROR: sGDML package not located. Install it via: pip install sgdml"
            )

        # A bit weird to use keyword argument for a required argument, but this
        # is also done in the code above.
        if sGDML_model is None:
            raise ValueError("Must provide a sGDML model file.")

        if dopbc is True:
            raise ValueError("Must set PBCs to False.")

        self.sGDML_model = sGDML_model

        # --- Load sGDML model file. ---
        try:
            self.model = dict(np.load(self.sGDML_model, allow_pickle=True))
            info(
                " @ForceField: sGDML model " + self.sGDML_model + " loaded",
                verbosity.medium,
            )
        except ValueError:
            raise ValueError(
                "ERROR: Reading sGDML model " + self.sGDML_model + " file failed."
            )

        info(
            " @ForceField: IMPORTANT: It is always assumed that the units in"
            + " the provided model file are in Angstroms and kcal/mol.",
            verbosity.low,
        )

        # --- Units ---
        transl_units_names = {
            "Ang": "angstrom",
            "bohr": "atomic_unit",
            "millihartree": "milliatomic_unit",
            "eV": "electronvolt",
            "kcal/mol": "kilocal/mol",
        }
        if "r_unit" in self.model and "e_unit" in self.model:
            distanceUnits = transl_units_names["{}".format(self.model["r_unit"])]
            energyUnits = transl_units_names["{}".format(self.model["e_unit"])]
            info(
                " @ForceField: The units used in the sGDML model are: "
                "distance --> {}, energy --> {}".format(distanceUnits, energyUnits),
                verbosity.low,
            )
        else:
            info(
                " @ForceField: IMPORTANT: Since the sGDML model doesn't have units for distance and energy it is "
                "assumed that the units are in Angstroms and kilocal/mol, respectively.",
                verbosity.low,
            )
            distanceUnits = "angstrom"
            energyUnits = "kilocal/mol"

        # --- Constants ---
        self.bohr_to_distanceUnits = unit_to_user("length", distanceUnits, 1)
        self.energyUnits_to_hartree = unit_to_internal("energy", energyUnits, 1)
        self.forceUnits_to_hartreebohr = (
            self.bohr_to_distanceUnits * self.energyUnits_to_hartree
        )

        # --- Creates predictor ---
        self.predictor = GDMLPredict(self.model)

        info(
            " @ForceField: Optimizing parallelization settings for sGDML FF.",
            verbosity.medium,
        )
        self.predictor.prepare_parallel(n_bulk=1)

    def evaluate(self, r):
        """Evaluate the energy and forces."""

        E, F = self.predictor.predict(r["pos"] * self.bohr_to_distanceUnits)

        r["result"] = [
            E[0] * self.energyUnits_to_hartree,
            F.flatten() * self.forceUnits_to_hartreebohr,
            np.zeros((3, 3), float),
            {"raw": ""},
        ]
        r["status"] = "Done"
        r["t_finished"] = time.time()


class FFCommittee(ForceField):
    """Combines multiple forcefields into a single forcefield object that consolidates
    individual components. Provides the infrastructure to run a simulation based on a
    committee of potentials, and implements the weighted baseline method."""

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        pars=None,
        dopbc=True,
        active=np.array([-1]),
        threaded=True,
        fflist=[],
        ffweights=[],
        alpha=1.0,
        baseline_name="",
        baseline_uncertainty=-1.0,
        active_thresh=0.0,
        active_out=None,
        parse_json=False,
    ):
        # force threaded mode as otherwise it cannot have threaded children
        super(FFCommittee, self).__init__(
            latency=latency,
            offset=offset,
            name=name,
            pars=pars,
            dopbc=dopbc,
            active=active,
            threaded=True,  # hardcoded, otherwise won't work!
        )
        if len(fflist) == 0:
            raise ValueError(
                "Committee forcefield cannot be initialized from an empty list"
            )
        self.fflist = fflist
        self.ff_requests = {}
        self.baseline_uncertainty = baseline_uncertainty
        self.baseline_name = baseline_name
        if len(ffweights) == 0 and self.baseline_uncertainty < 0:
            ffweights = np.ones(len(fflist))
        elif len(ffweights) == 0 and self.baseline_uncertainty > 0:
            ffweights = np.ones(len(fflist) - 1)
        if len(ffweights) != len(fflist) and self.baseline_uncertainty < 0:
            raise ValueError("List of weights does not match length of committee model")
        elif len(ffweights) != len(fflist) - 1 and self.baseline_uncertainty > 0:
            raise ValueError("List of weights does not match length of committee model")
        if (self.baseline_name == "") != (self.baseline_uncertainty < 0):
            raise ValueError(
                "Name and the uncertainty of the baseline are not simultaneously defined"
            )
        self.ffweights = ffweights
        self.alpha = alpha
        self.active_thresh = active_thresh
        self.active_out = active_out
        self.parse_json = parse_json

    def bind(self, output_maker):
        super(FFCommittee, self).bind(output_maker)
        if self.active_thresh > 0:
            if self.active_out is None:
                raise ValueError(
                    "Must specify an output file if you want to save structures for active learning"
                )
            else:
                self.active_file = self.output_maker.get_output(self.active_out, "w")

    def start(self):
        for ff in self.fflist:
            ff.start()
        super(FFCommittee, self).start()

    def queue(self, atoms, cell, reqid=-1):
        # launches requests for all of the committee FF objects
        ffh = []
        for ff in self.fflist:
            ffh.append(ff.queue(atoms, cell, reqid))

        # creates the request with the help of the base class,
        # making sure it already contains a handle to the list of FF
        # requests
        req = super(FFCommittee, self).queue(
            atoms, cell, reqid, template=dict(ff_handles=ffh)
        )
        req["t_dispatched"] = time.time()
        return req

    def check_finish(self, r):
        """Checks if all sub-requests associated with a given
        request are finished"""
        for ff_r in r["ff_handles"]:
            if ff_r["status"] != "Done":
                return False
        return True

    def gather(self, r):
        """Collects results from all sub-requests, and assemble the committee of models."""

        r["result"] = [
            0.0,
            np.zeros(len(r["pos"]), float),
            np.zeros((3, 3), float),
            "",
        ]

        # list of pointers to the forcefield requests. shallow copy so we can remove stuff
        com_handles = r["ff_handles"].copy()
        if self.baseline_name != "":
            # looks for the baseline potential, store its value and drops it from the list
            names = [ff.name for ff in self.fflist]

            for i, ff_r in enumerate(com_handles):
                if names[i] == self.baseline_name:
                    baseline_pot = ff_r["result"][0]
                    baseline_frc = ff_r["result"][1]
                    baseline_vir = ff_r["result"][2]
                    baseline_xtr = ff_r["result"][3]
                    com_handles.pop(i)
                    break

        # Gathers the forcefield energetics and extras
        pots = []
        frcs = []
        virs = []
        xtrs = []

        all_have_frc = True
        all_have_vir = True

        for ff_r in com_handles:
            # if required, tries to extract multiple committe members from the extras JSON string
            if "committee_pot" in ff_r["result"][3] and self.parse_json:
                pots += ff_r["result"][3]["committee_pot"]
                if "committee_force" in ff_r["result"][3]:
                    frcs += ff_r["result"][3]["committee_force"]
                    ff_r["result"][3].pop("committee_force")
                else:
                    # if the commitee doesn't have forces, just add the mean force from this model
                    frcs.append(ff_r["result"][1])
                    warning("JSON committee doesn't have forces", verbosity.medium)
                    all_have_frc = False

                if "committee_virial" in ff_r["result"][3]:
                    virs += ff_r["result"][3]["committee_virial"]
                    ff_r["result"][3].pop("committee_virial")
                else:
                    # if the commitee doesn't have virials, just add the mean virial from this model
                    virs.append(ff_r["result"][2])
                    warning("JSON committee doesn't have virials", verbosity.medium)
                    all_have_vir = False

            else:
                pots.append(ff_r["result"][0])
                frcs.append(ff_r["result"][1])
                virs.append(ff_r["result"][2])

        pots = np.array(pots)
        if len(pots) != len(frcs) and len(frcs) > 1:
            raise ValueError(
                "If the committee returns forces, we need *all* components"
            )
        frcs = np.array(frcs).reshape(len(frcs), -1)

        if len(pots) != len(virs) and len(virs) > 1:
            raise ValueError(
                "If the committee returns virials, we need *all* components"
            )
        virs = np.array(virs).reshape(-1, 3, 3)

        xtrs.append(ff_r["result"][3])

        # Computes the mean energetics
        mean_pot = np.mean(pots, axis=0)
        mean_frc = np.mean(frcs, axis=0)
        mean_vir = np.mean(virs, axis=0)

        # Rescales the committee energetics so that their standard deviation corresponds to the error
        rescaled_pots = np.asarray(
            [mean_pot + self.alpha * (pot - mean_pot) for pot in pots]
        )
        rescaled_frcs = np.asarray(
            [mean_frc + self.alpha * (frc - mean_frc) for frc in frcs]
        )
        rescaled_virs = np.asarray(
            [mean_vir + self.alpha * (vir - mean_vir) for vir in virs]
        )

        # Calculates the error associated with the committee
        var_pot = np.var(rescaled_pots, ddof=1)
        std_pot = np.sqrt(var_pot)

        if self.baseline_name != "":
            if not (all_have_frc and all_have_vir):
                raise ValueError(
                    "Cannot use weighted baseline without a force ensemble"
                )

            # Computes the additional component of the energetics due to a position
            # dependent weight. This is based on the assumption that V_committee is
            # a correction over the baseline, that V = V_baseline + V_committe, that
            # V_baseline has an uncertainty given by baseline_uncertainty,
            # and V_committee the committee error. Then
            # V = V_baseline + s_b^2/(s_c^2+s_b^2) V_committe

            s_b2 = self.baseline_uncertainty**2

            nmodels = len(pots)
            uncertain_frc = (
                self.alpha**2
                * np.sum(
                    [
                        (pot - mean_pot) * (frc - mean_frc)
                        for pot, frc in zip(pots, frcs)
                    ],
                    axis=0,
                )
                / (nmodels - 1)
            )
            uncertain_vir = (
                self.alpha**2
                * np.sum(
                    [
                        (pot - mean_pot) * (vir - mean_vir)
                        for pot, vir in zip(pots, virs)
                    ],
                    axis=0,
                )
                / (nmodels - 1)
            )

            # Computes the final average energetics
            final_pot = baseline_pot + mean_pot * s_b2 / (s_b2 + var_pot)
            final_frc = (
                baseline_frc
                + mean_frc * s_b2 / (s_b2 + var_pot)
                - 2.0 * mean_pot * s_b2 / (s_b2 + var_pot) ** 2 * uncertain_frc
            )
            final_vir = (
                baseline_vir
                + mean_vir * s_b2 / (s_b2 + var_pot)
                - 2.0 * mean_pot * s_b2 / (s_b2 + var_pot) ** 2 * uncertain_vir
            )

            # Sets the output of the committee model.
            r["result"][0] = final_pot
            r["result"][1] = final_frc
            r["result"][2] = final_vir
        else:
            # Sets the output of the committee model.
            r["result"][0] = mean_pot
            r["result"][1] = mean_frc
            r["result"][2] = mean_vir

        r["result"][3] = {
            "committee_pot": rescaled_pots,
            "committee_uncertainty": std_pot,
        }

        if all_have_frc:
            r["result"][3]["committee_force"] = rescaled_frcs.reshape(
                len(rescaled_pots), -1
            )
        if all_have_vir:
            r["result"][3]["committee_virial"] = rescaled_virs.reshape(
                len(rescaled_pots), -1
            )

        if self.baseline_name != "":
            r["result"][3]["baseline_pot"] = (baseline_pot,)
            r["result"][3]["baseline_force"] = (baseline_frc,)
            r["result"][3]["baseline_virial"] = ((baseline_vir.flatten()),)
            r["result"][3]["baseline_extras"] = (baseline_xtr,)
            r["result"][3]["wb_mixing"] = (s_b2 / (s_b2 + var_pot),)

        # "dissolve" the extras dictionaries into a list
        for k in xtrs[0].keys():
            if ("committee_" + k) in r["result"][3].keys():
                raise ValueError(
                    "Name clash between extras key "
                    + k
                    + " and default committee extras"
                )
            r["result"][3][("committee_" + k)] = []
            for x in xtrs:
                r["result"][3][("committee_" + k)].append(x[k])

        if self.active_thresh > 0.0 and std_pot > self.active_thresh:
            dumps = json.dumps(
                {
                    "position": list(r["pos"]),
                    "cell": list(r["cell"][0].flatten()),
                    "uncertainty": std_pot,
                }
            )
            self.active_file.write(dumps)

        # releases the requests from the committee FF
        for ff, ff_r in zip(self.fflist, r["ff_handles"]):
            ff.release(ff_r)

    def poll(self):
        """Polls the forcefield object to check if it has finished."""

        with self._threadlock:
            for r in self.requests:
                if r["status"] != "Done" and self.check_finish(r):
                    r["t_finished"] = time.time()
                    self.gather(r)
                    r["result"][0] -= self.offset
                    r["status"] = "Done"


class FFRotations(ForceField):
    """Forcefield to manipulate models that are not exactly rotationally equivariant.
    Can be used to evaluate a different random rotation at each evaluation, or to average
    over a regular grid of Euler angles"""

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        pars=None,
        dopbc=True,
        active=np.array([-1]),
        threaded=True,
        prng=None,
        ffsocket=None,
        ffdirect=None,
        random=False,
        inversion=False,
        grid_order=1,
        grid_mode="lebedev",
    ):
        super(
            FFRotations, self
        ).__init__(  # force threaded execution to handle sub-ffield
            latency, offset, name, pars, dopbc, active, threaded=True
        )

        if prng is None:
            warning("No PRNG provided, will initialize one", verbosity.low)
            self.prng = Random()
        else:
            self.prng = prng
        self.ffsocket = ffsocket
        self.ffdirect = ffdirect
        if ffsocket is None or self.ffsocket.name == "__DUMMY__":
            if ffdirect is None or self.ffdirect.name == "__DUMMY__":
                raise ValueError(
                    "Must specify a non-default value for either `ffsocket` or `ffdirect` into `ffrotations`"
                )
            else:
                self.ff = self.ffdirect
        elif ffdirect is None or self.ffdirect.name == "__DUMMY__":
            self.ff = self.ffsocket
        else:
            raise ValueError(
                "Cannot specify both `ffsocket` and `ffdirect` into `ffrotations`"
            )

        self.random = random
        self.inversion = inversion
        self.grid_order = grid_order
        self.grid_mode = grid_mode

        if self.grid_mode == "lebedev":
            self._rotations = get_rotation_quadrature_lebedev(self.grid_order)
        elif self.grid_mode == "legendre":
            self._rotations = get_rotation_quadrature_legendre(self.grid_order)
        else:
            raise ValueError(f"Invalid quadrature {self.grid_mode}")
        info(
            f"""
# Generating {self.grid_mode} rotation quadrature of order {self.grid_order}.
# Grid contains {len(self._rotations)} proper rotations.
{("# Inversion is also active, doubling the number of evaluations." if self.inversion else "")}""",
            verbosity.low,
        )

    def bind(self, output_maker=None):
        super().bind(output_maker)
        self.ff.bind(output_maker)

    def start(self):
        super().start()
        self.ff.start()

    def queue(self, atoms, cell, reqid=-1):

        # launches requests for all of the rotations FF objects
        ffh = []  # this is the list of "inner" FF requests
        rots = []  # this is a list of tuples of (rotation matrix, weight)
        if self.random:
            R_random = random_rotation(self.prng, improper=True)
        else:
            R_random = np.eye(3)

        for R, w, _ in self._rotations:
            R = R @ R_random

            rot_atoms = atoms.clone()
            rot_cell = GenericCell(
                R @ dstrip(cell.h).copy()
            )  # NB we need generic cell orientation
            rot_atoms.q[:] = (dstrip(rot_atoms.q).reshape(-1, 3) @ R.T).flatten()

            rots.append((R, w))
            ffh.append(self.ff.queue(rot_atoms, rot_cell, reqid))

            if self.inversion:
                # also add a "flipped rotation" to the evaluation list
                R = R * -1

                rot_cell = GenericCell(R @ dstrip(cell.h).copy())
                rot_atoms = atoms.clone()
                rot_atoms.q[:] = (dstrip(rot_atoms.q).reshape(-1, 3) @ R.T).flatten()

                rots.append((R, w))
                ffh.append(self.ff.queue(rot_atoms, rot_cell, reqid))

        # creates the request with the help of the base class,
        # making sure it already contains a handle to the list of FF
        # requests
        req = ForceField.queue(
            self, atoms, cell, reqid, template=dict(ff_handles=ffh, rots=rots)
        )
        req["status"] = "Running"
        req["t_dispatched"] = time.time()
        return req

    def check_finish(self, r):
        """Checks if all sub-requests associated with a given
        request are finished"""

        for ff_r in r["ff_handles"]:
            if ff_r["status"] != "Done":
                return False
        return True

    def gather(self, r):
        """Collects results from all sub-requests, and assemble the committee of models."""

        r["result"] = [
            0.0,
            np.zeros(len(r["pos"]), float),
            np.zeros((3, 3), float),
            "",
        ]

        # list of pointers to the forcefield requests. shallow copy so we can remove stuff
        rot_handles = r["ff_handles"].copy()
        rots = r["rots"].copy()

        # Gathers the forcefield energetics and extras
        pots = []
        frcs = []
        virs = []
        xtrs = []
        quad_w = []
        for ff_r, (R, w) in zip(rot_handles, rots):
            pots.append(ff_r["result"][0])
            # must rotate forces and virial back into the original reference frame
            frcs.append((ff_r["result"][1].reshape(-1, 3) @ R).flatten())
            virs.append((R.T @ ff_r["result"][2] @ R))
            xtrs.append(ff_r["result"][3])
            quad_w.append(w)

        quad_w = np.array(quad_w)
        pots = np.array(pots)
        frcs = np.array(frcs).reshape(len(frcs), -1)
        virs = np.array(virs).reshape(-1, 3, 3)

        # Computes the mean energetics (using the quadrature weights)
        mean_pot = np.sum(pots * quad_w, axis=0) / quad_w.sum()
        mean_frc = np.sum(frcs * quad_w[:, np.newaxis], axis=0) / quad_w.sum()
        mean_vir = (
            np.sum(virs * quad_w[:, np.newaxis, np.newaxis], axis=0) / quad_w.sum()
        )

        # Sets the output of the committee model.
        r["result"][0] = mean_pot
        r["result"][1] = mean_frc
        r["result"][2] = mean_vir
        r["result"][3] = {
            "o3grid_pots": pots
        }  # this is the list of potentials on a grid, for monitoring

        # "dissolve" the extras dictionaries into a list
        if isinstance(xtrs[0], dict):
            for k in xtrs[0].keys():
                r["result"][3][k] = []
                for x in xtrs:
                    r["result"][3][k].append(x[k])
        else:
            r["result"][3]["raw"] = []
            for x in xtrs:
                r["result"][3]["raw"].append(x)

        for ff_r in r["ff_handles"]:
            self.ff.release(ff_r)

    def poll(self):
        """Polls the forcefield object to check if it has finished."""

        with self._threadlock:
            for r in self.requests:
                if "ff_handles" in r and r["status"] != "Done" and self.check_finish(r):
                    r["t_finished"] = time.time()
                    self.gather(r)
                    r["result"][0] -= self.offset
                    r["status"] = "Done"
                    self.release(r, lock=False)


class PhotonDriver:
    """
    Photon driver for a single cavity mode
    """

    def __init__(self, apply_photon=True, E0=1e-4, omega_c=0.01, ph_rep="loose"):
        """
        Initialise PhotonDriver

        In this implementation, the photonic masses are set as 1 a.u.

        Args:
            apply_photon: Determine if applying light-matter interactions
            E0: varepsilon in the paper (doi.org/10.1073/pnas.2009272117), light-matter coupling strength
            omega_c: cavity frequency at normal incidence
            ph_rep: 'loose' or 'dense'. In the current implementation, two energy-degenerate photon modes polarized along x and y directions
                are coupled to the molecular system. If 'loose', the cavity photons polarized along the x, y directions are represented by two 'L' atoms;
                the x dimension of the first 'L' atom is coupled to the molecules, and the y dimension of the second 'L' atom is coupled to the molecules.
                If 'dense', the cavity photons polarized along the x, y directions are represented by one 'L' atom;
                the x and y dimensions of this 'L' atom are coupled to the molecules.
        """
        self.apply_photon = apply_photon
        self.E0 = E0
        self.omega_c = omega_c

        if self.apply_photon == False:
            self.n_mode = 0
        elif self.apply_photon == True:
            self.n_mode = 1
            self.n_grid = 1
            self.ph_rep = ph_rep
            self.init_photon()

    def init_photon(self):
        """
        Initialize the photon environment parameters
        """

        if self.ph_rep == "loose":
            self.n_photon = 2 * self.n_mode
        elif self.ph_rep == "dense":
            self.n_photon = self.n_mode
        self.n_photon_3 = self.n_photon * 3
        self.pos_ph = np.zeros(self.n_photon_3)

        # construct cavity mode frequency array for all photons
        self.omega_k = np.array([self.omega_c])
        if self.ph_rep == "loose":
            self.omega_klambda = np.concatenate((self.omega_k, self.omega_k))
        elif self.ph_rep == "dense":
            self.omega_klambda = self.omega_k
        self.omega_klambda3 = np.reshape(
            np.array([[x, x, x] for x in self.omega_klambda]), -1
        )

        # construct varepsilon array for all photons
        self.varepsilon_k = self.E0
        self.varepsilon_klambda = (
            self.E0 * self.omega_klambda / np.min(self.omega_klambda)
        )
        self.varepsilon_klambda3 = (
            self.E0 * self.omega_klambda3 / np.min(self.omega_klambda3)
        )

        # cavity mode function acting on the dipole moment
        self.ftilde_kx = np.array([1.0])
        self.ftilde_ky = np.array([1.0])
        self.ftilde_kx3 = np.reshape(np.array([[x, x, x] for x in self.ftilde_kx]), -1)
        self.ftilde_ky3 = np.reshape(np.array([[x, x, x] for x in self.ftilde_ky]), -1)

    def split_atom_ph_coord(self, pos):
        """
        Split atomic and photonic coordinates and update our photonic coordinates

        Args:
            pos: A 3*N position numpy array, [1x, 1y, 1z, 2x, ...]

        Returns:
            Atomic coordinates, Photonic coordinates
        """
        if self.apply_photon:
            pos_at = pos[: -self.n_photon_3]
            pos_ph = pos[-self.n_photon_3 :]
            self.pos_ph = pos_ph
        else:
            pos_at = pos
            pos_ph = pos[0:0]
        return pos_at, pos_ph

    def get_ph_energy(self, dx_array, dy_array):
        """
        Calculate the total photonic potential energy, including the light-matter
        interaction and dipole self energy

        Args:
            dx_array: x-direction dipole array of molecular subsystems
            dy_array: y-direction dipole array of molecular subsystems

        Returns:
            total energy of photonic system
        """
        # calculate the photonic potential energy
        e_ph = np.sum(0.5 * self.omega_klambda3**2 * self.pos_ph**2)

        # calculate the dot products between mode functions and dipole array
        d_dot_f_x = np.dot(self.ftilde_kx, dx_array)
        d_dot_f_y = np.dot(self.ftilde_ky, dy_array)

        # calculate the light-matter interaction
        if self.ph_rep == "loose":
            e_int_x = np.sum(
                self.varepsilon_k * d_dot_f_x * self.pos_ph[: self.n_mode * 3 : 3]
            )
            e_int_y = np.sum(
                self.varepsilon_k * d_dot_f_y * self.pos_ph[1 + self.n_mode * 3 :: 3]
            )
        elif self.ph_rep == "dense":
            e_int_x = np.sum(self.varepsilon_k * d_dot_f_x * self.pos_ph[::3])
            e_int_y = np.sum(self.varepsilon_k * d_dot_f_y * self.pos_ph[1::3])

        # calculate the dipole self-energy term
        dse = np.sum(
            (self.varepsilon_k**2 / 2.0 / self.omega_k**2)
            * (d_dot_f_x**2 + d_dot_f_y**2)
        )

        e_tot = e_ph + e_int_x + e_int_y + dse

        return e_tot

    def get_ph_forces(self, dx_array, dy_array):
        """
        Calculate the photonic forces

        Args:
            dx_array: x-direction dipole array of molecular subsystems
            dy_array: y-direction dipole array of molecular subsystems

        Returns:
            force array of all photonic dimensions (3*nphoton) [1x, 1y, 1z, 2x..]
        """
        # calculat the bare photonic contribution of the force
        f_ph = -self.omega_klambda3**2 * self.pos_ph

        # calculate the dot products between mode functions and dipole array
        d_dot_f_x = np.dot(self.ftilde_kx, dx_array)
        d_dot_f_y = np.dot(self.ftilde_ky, dy_array)

        # calculate the force due to light-matter interactions
        if self.ph_rep == "loose":
            f_ph[: self.n_mode * 3 : 3] -= self.varepsilon_k * d_dot_f_x
            f_ph[self.n_mode * 3 + 1 :: 3] -= self.varepsilon_k * d_dot_f_y
        elif self.ph_rep == "dense":
            f_ph[::3] -= self.varepsilon_k * d_dot_f_x
            f_ph[1::3] -= self.varepsilon_k * d_dot_f_y
        return f_ph

    def get_nuc_cav_forces(self, dx_array, dy_array, charge_array_bath):
        """
        Calculate the photonic forces on nuclei from MM partial charges

        Args:
            dx_array: x-direction dipole array of molecular subsystems
            dy_array: y-direction dipole array of molecular subsystems
            charge_array_bath: partial charges of all atoms in a single bath

        Returns:
            force array of all nuclear dimensions (3*natoms) [1x, 1y, 1z, 2x..]
        """

        # calculate the dot products between mode functions and dipole array
        d_dot_f_x = np.dot(self.ftilde_kx, dx_array)
        d_dot_f_y = np.dot(self.ftilde_ky, dy_array)

        # cavity force on x direction
        if self.ph_rep == "loose":
            Ekx = self.varepsilon_k * self.pos_ph[: self.n_mode * 3 : 3]
            Eky = self.varepsilon_k * self.pos_ph[self.n_mode * 3 + 1 :: 3]
        elif self.ph_rep == "dense":
            Ekx = self.varepsilon_k * self.pos_ph[::3]
            Eky = self.varepsilon_k * self.pos_ph[1::3]
        Ekx += self.varepsilon_k**2 / self.omega_k**2 * d_dot_f_x
        Eky += self.varepsilon_k**2 / self.omega_k**2 * d_dot_f_y

        # dimension of independent baths (xy grid points)
        coeff_x = np.dot(np.transpose(Ekx), self.ftilde_kx)
        coeff_y = np.dot(np.transpose(Eky), self.ftilde_ky)
        fx = -np.kron(coeff_x, charge_array_bath)
        fy = -np.kron(coeff_y, charge_array_bath)
        return fx, fy


class FFCavPhSocket(FFSocket):
    """
    Socket for dealing with cavity photons interacting with molecules by
    Tao E. Li @ 2023-02-25
    Check https://doi.org/10.1073/pnas.2009272117 for details

    Independent bath approximation will be made to communicate with many sockets
    """

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        pars=None,
        dopbc=False,
        active=np.array([-1]),
        threaded=True,
        interface=None,
        charge_array=None,
        apply_photon=True,
        E0=1e-4,
        omega_c=0.01,
        ph_rep="loose",
    ):
        """Initialises FFCavPhFPSocket.

        Args:
           latency: The number of seconds the socket will wait before updating
              the client list.
           name: The name of the forcefield.
           pars: A dictionary used to initialize the forcefield, if required.
              Of the form {'name1': value1, 'name2': value2, ... }.
           dopbc: Decides whether or not to apply the periodic boundary conditions
              before sending the positions to the client code.
           interface: The object used to create the socket used to interact
              with the client codes.
           charge_array: An N-dimensional numpy array for fixed point charges of all atoms
           apply_photon: If add photonic degrees of freedom in the dynamics
           E0: Effective light-matter coupling strength
           omega_c: Cavity mode frequency
           ph_rep: A string to control how to represent the photonic coordinates: 'loose' or 'dense'.
                In the current implementation, two energy-degenerate photon modes polarized along x and
                y directions are coupled to the molecular system. If 'loose', the cavity photons polarized
                along the x, y directions are represented by two 'L' atoms; the x dimension of the first
                'L' atom is coupled to the molecules, and the y dimension of the second 'L' atom is coupled
                to the molecules. If 'dense', the cavity photons polarized along the x, y directions are
                represented by one 'L' atom; the x and y dimensions of this 'L' atom are coupled to the molecules.
        """

        # a socket to the communication library is created or linked
        super(FFCavPhSocket, self).__init__(
            latency, offset, name, pars, dopbc, active, threaded, interface
        )

        # definition of independent baths
        self.n_independent_bath = 1
        self.charge_array = charge_array

        # store photonic variables
        self.apply_photon = apply_photon
        self.E0 = E0
        self.omega_c = omega_c
        self.ph_rep = ph_rep
        # define the photon environment
        self.ph = PhotonDriver(
            apply_photon=apply_photon, E0=E0, omega_c=omega_c, ph_rep=ph_rep
        )

        self._getallcount = 0

    def calc_dipole_xyz_mm(self, pos, n_bath, charge_array_bath):
        """
        Calculate the x, y, and z components of total dipole moment for a single molecular subsystem (bath)

        Args:
            pos: position of all atoms (3*n) in all subsystems
            n_bath: total number of molecular subsystems (baths)
            charge_array_bath: charge_array of all atoms (n) in a single subsystem (bath)

        Returns:
            dx_array, dy_array, dz_array: total dipole moment array along x, y, and z directions
        """
        ndim_tot = np.size(pos)
        ndim_local = int(ndim_tot // n_bath)

        dx_array, dy_array, dz_array = [], [], []
        for idx in range(n_bath):
            pos_bath = pos[ndim_local * idx : ndim_local * (idx + 1)]
            # check the dimension of charge array
            if np.size(pos_bath[::3]) != np.size(charge_array_bath):
                softexit.trigger(
                    "The size of charge array = {}  does not match the size of atoms = {} ".format(
                        np.size(charge_array_bath), np.size(pos_bath[::3])
                    )
                )
            dx = np.sum(pos_bath[::3] * charge_array_bath)
            dy = np.sum(pos_bath[1::3] * charge_array_bath)
            dz = np.sum(pos_bath[2::3] * charge_array_bath)
            dx_array.append(dx)
            dy_array.append(dy)
            dz_array.append(dz)
        dx_array = np.array(dx_array)
        dy_array = np.array(dy_array)
        dz_array = np.array(dz_array)
        return dx_array, dy_array, dz_array

    def queue(self, atoms, cell, reqid=-1):
        """Adds a request.

        Note that the pars dictionary need to be sent as a string of a
        standard format so that the initialisation of the driver can be done.

        Args:
            atoms: An Atoms object giving the atom positions.
            cell: A Cell object giving the system box.
            pars: An optional dictionary giving the parameters to be sent to the
                driver for initialisation. Defaults to {}.
            reqid: An optional integer that identifies requests of the same type,
               e.g. the bead index

        Returns:
            A list giving the status of the request of the form {'pos': An array
            giving the atom positions folded back into the unit cell,
            'cell': Cell object giving the system box, 'pars': parameter string,
            'result': holds the result as a list once the computation is done,
            'status': a string labelling the status of the calculation,
            'id': the id of the request, usually the bead number, 'start':
            the starting time for the calculation, used to check for timeouts.}.
        """

        par_str = " "

        if not self.pars is None:
            for k, v in list(self.pars.items()):
                par_str += k + " : " + str(v) + " , "
        else:
            par_str = " "

        pbcpos = dstrip(atoms.q).copy()

        # Indexes come from input in a per atom basis and we need to make a per atom-coordinate basis
        # Reformat indexes for full system (default) or piece of system
        # active atoms do not change but we only know how to build this array once we get the positions once
        if self.iactive is None:
            if self.active[0] == -1:
                activehere = np.arange(len(pbcpos))
            else:
                activehere = np.array(
                    [[3 * n, 3 * n + 1, 3 * n + 2] for n in self.active]
                )

            # Reassign active indexes in order to use them
            activehere = activehere.flatten()

            # Perform sanity check for active atoms
            if len(activehere) > len(pbcpos) or activehere[-1] > (len(pbcpos) - 1):
                raise ValueError("There are more active atoms than atoms!")

            self.iactive = activehere

        newreq_lst = []

        # 1. split coordinates to atoms and photons
        pbcpos_atoms, pbcpos_phs = self.ph.split_atom_ph_coord(pbcpos)
        ndim_tot = np.size(pbcpos_atoms)
        ndim_local = int(ndim_tot // self.n_independent_bath)

        # 2. for atomic coordinates, we now evaluate their atomic forces
        for idx in range(self.n_independent_bath):
            pbcpos_local = pbcpos_atoms[
                ndim_local * idx : ndim_local * (idx + 1)
            ].copy()
            iactive_local = self.iactive[0:ndim_local]
            # Let's try to do PBC for the small regions
            if self.dopbc:
                cell.array_pbc(pbcpos_local)
            newreq_local = ForceRequest(
                {
                    "id": int(reqid * self.n_independent_bath) + idx,
                    "pos": pbcpos_local,
                    "active": iactive_local,
                    "cell": (dstrip(cell.h).copy(), dstrip(cell.ih).copy()),
                    "pars": par_str,
                    "result": None,
                    "status": "Queued",
                    "start": -1,
                    "t_queued": time.time(),
                    "t_dispatched": 0,
                    "t_finished": 0,
                }
            )
            newreq_lst.append(newreq_local)

        with self._threadlock:
            for newreq in newreq_lst:
                self.requests.append(newreq)
                self._getallcount += 1

        if not self.threaded:
            self.poll()

        # sleeps until all the new requests have been evaluated
        for self.request in newreq_lst:
            while self.request["status"] != "Done":
                if self.request["status"] == "Exit" or softexit.triggered:
                    # now, this is tricky. we are stuck here and we cannot return meaningful results.
                    # if we return, we may as well output wrong numbers, or mess up things.
                    # so we can only call soft-exit and wait until that is done. then kill the thread
                    # we are in.
                    softexit.trigger(" @ FORCES : cannot return so will die off here")
                    while softexit.exiting:
                        time.sleep(self.latency)
                    sys.exit()
                time.sleep(self.latency)

            """
            with self._threadlock:
                self._getallcount -= 1

            # releases just once, but wait for all requests to be complete
            if self._getallcount == 0:
                self.release(self.request)
                self.request = None
            else:
                while self._getallcount > 0:
                    time.sleep(self.latency)
            """
            self.release(self.request)
            self.request = None

        # ...atomic forces have been calculated at this point

        # 3. At this moment, we combine the small requests to a big mega request (updated results)
        result_tot = [0.0, np.zeros(len(pbcpos), float), np.zeros((3, 3), float), {}]
        for idx, newreq in enumerate(newreq_lst):
            u, f, vir, extra = newreq["result"]
            result_tot[0] += u
            result_tot[1][ndim_local * idx : ndim_local * (idx + 1)] = f
            result_tot[2] += vir
            result_tot[3][idx] = extra

        if self.ph.apply_photon:
            # 4. calculate total dipole moment array for N baths
            dx_array, dy_array, dz_array = self.calc_dipole_xyz_mm(
                pos=pbcpos_atoms,
                n_bath=self.n_independent_bath,
                charge_array_bath=self.charge_array,
            )
            # check the size of photon modes + molecules to match the total number of particles
            if (
                self.ph.n_photon + self.n_independent_bath * self.charge_array.size
                != int(len(pbcpos) // 3)
            ):
                softexit.trigger(
                    "Total number of photons + molecules does not match total number of particles"
                )
            # info("mux = %.6f muy = %.6f muz = %.6f [units of a.u.]" %(dipole_x_tot, dipole_y_tot, dipole_z_tot), verbosity.medium)
            # 5. calculate photonic contribution of total energy
            e_ph = self.ph.get_ph_energy(dx_array=dx_array, dy_array=dy_array)
            # 6. calculate photonic forces
            f_ph = self.ph.get_ph_forces(dx_array=dx_array, dy_array=dy_array)
            # 7. calculate cavity forces on nuclei
            fx_cav, fy_cav = self.ph.get_nuc_cav_forces(
                dx_array=dx_array,
                dy_array=dy_array,
                charge_array_bath=self.charge_array,
            )
            # 8. add cavity effects to our output
            result_tot[0] += e_ph
            result_tot[1][:ndim_tot:3] += fx_cav
            result_tot[1][1:ndim_tot:3] += fy_cav
            result_tot[1][ndim_tot:] = f_ph

        result_tot[0] -= self.offset

        # At this moment, we have sucessfully gathered the CavMD forces
        newreq = ForceRequest(
            {
                "id": reqid,
                "pos": pbcpos,
                "active": self.iactive,
                "cell": (dstrip(cell.h).copy(), dstrip(cell.ih).copy()),
                "pars": par_str,
                "result": result_tot,
                "status": newreq_lst[-1]["status"],
                "start": newreq_lst[0]["start"],
                "t_queued": newreq_lst[0]["t_queued"],
                "t_dispatched": newreq_lst[0]["t_dispatched"],
                "t_finished": newreq_lst[-1]["t_finished"],
            }
        )

        return newreq


class CP2KEndpoint:
    """Manages a single CP2K endpoint with specific charge.

    This class handles the lifecycle of a CP2K process, including:
    - Template rendering and input file generation
    - Process startup and shutdown
    - Socket communication
    - Health monitoring and error recovery
    """

    def __init__(self, charge, host, port, cp2k_template, cp2k_exe, timeout=300.0, cp2k_env="", cp2k_run_cmd=None, neutral_electrons=None):
        """Initialize CP2K endpoint.

        Args:
            charge (int): Integer charge for this endpoint
            host (str): Host for socket communication
            port (int): Port number for socket communication
            cp2k_template (str): CP2K input template with placeholders
            cp2k_exe (str): Path to CP2K executable
            timeout (float): Timeout in seconds for operations
        """
        self.charge = charge
        self.host = host
        self.port = port
        self.cp2k_template = cp2k_template
        self.cp2k_exe = cp2k_exe
        self.timeout = timeout
        self.cp2k_env = cp2k_env
        self.cp2k_run_cmd = cp2k_run_cmd
        self.neutral_electrons = neutral_electrons

        # Project name (from &GLOBAL PROJECT) used to locate V_HARTREE_CUBE files
        self.project_name = None

        # Process and communication state
        self.process = None
        self.socket_path = None
        self.input_file = None
        self.output_file = None
        self.error_file = None
        self.is_running = False
        self.last_health_check = 0.0

        # Communication interface using new socket classes
        self.socket_server = CP2KSocketServer(host=host, port=port, timeout=timeout)
        self.socket_communicator = CP2KSocketCommunicator(self.socket_server)
        self.last_result = None

        info(f" @CP2KEndpoint: Initialized endpoint for charge {charge}, inet://{host}:{port}", verbosity.medium)

    def render_template(self):
        """Render CP2K input template with current parameters.

        Returns:
            str: Rendered CP2K input content
        """
        if not self.cp2k_template:
            raise ValueError("CP2K template is empty")

        # Template substitution
        content = self.cp2k_template
        substitutions = {
            "{{CHARGE}}": str(self.charge),
            "{{HOST}}": self.host,
            "{{PORT}}": str(self.port),
        }

        for placeholder, value in substitutions.items():
            content = content.replace(placeholder, value)

        # Auto-adjust spin settings (LSD / MULTIPLICITY) based on electron-count parity
        # N_electrons = neutral_electrons - CHARGE, and parity(N) == parity(neutral_electrons + CHARGE)
        try:
            ne = getattr(self, "neutral_electrons", None)
            if ne is not None:
                total_e_parity = (int(ne) + int(self.charge)) % 2
                is_even = (total_e_parity == 0)

                lines = content.splitlines()

                # Only apply spin auto-adjust when OT algorithm is enabled.
                # Detect OT by the presence of an "&OT" block in the SCF section.
                has_ot = False
                for line in lines:
                    stripped = line.lstrip()
                    upper = stripped.upper()
                    if upper.startswith("&OT"):
                        has_ot = True
                        break

                if not has_ot:
                    # Diagonalization (no &OT): respect user-provided LSD / MULTIPLICITY settings.
                    content = "\n".join(lines)
                else:
                    lsd_idx = None
                    mult_idx = None
                    charge_idx = None

                    for i, line in enumerate(lines):
                        stripped = line.lstrip()
                        upper = stripped.upper()
                        if upper.startswith("CHARGE"):
                            charge_idx = i
                        if upper.startswith("LSD") or upper.startswith("UKS"):
                            lsd_idx = i
                        if upper.startswith("MULTIPLICITY"):
                            mult_idx = i

                    def _update_mult(idx, value):
                        """Helper to overwrite MULTIPLICITY line while preserving indentation and comments."""
                        line = lines[idx]
                        indent = line[: len(line) - len(line.lstrip())]
                        body = line.lstrip()
                        parts = body.split("#", 1)
                        comment = parts[1].strip() if len(parts) > 1 else None
                        new_line = f"{indent}MULTIPLICITY    {value}"
                        if comment:
                            new_line += f"  # {comment}"
                        lines[idx] = new_line

                    if is_even:
                        # Closed-shell case: no LSD/UKS, MULTIPLICITY = 1
                        if lsd_idx is not None:
                            # Comment out existing LSD/UKS line (if not already commented)
                            if not lines[lsd_idx].lstrip().startswith("#"):
                                indent = lines[lsd_idx][: len(lines[lsd_idx]) - len(lines[lsd_idx].lstrip())]
                                lines[lsd_idx] = indent + "# " + lines[lsd_idx].lstrip()

                        if mult_idx is not None:
                            _update_mult(mult_idx, 1)
                        elif charge_idx is not None:
                            # Insert a new MULTIPLICITY line just after CHARGE
                            indent = lines[charge_idx][: len(lines[charge_idx]) - len(lines[charge_idx].lstrip())]
                            insert_idx = charge_idx + 1
                            lines.insert(insert_idx, f"{indent}MULTIPLICITY    1")
                    else:
                        # Open-shell (odd electrons): ensure LSD/UKS is active and MULTIPLICITY = 2
                        if mult_idx is not None:
                            _update_mult(mult_idx, 2)
                        elif charge_idx is not None:
                            indent = lines[charge_idx][: len(lines[charge_idx]) - len(lines[charge_idx].lstrip())]
                            insert_idx = charge_idx + 1
                            lines.insert(insert_idx, f"{indent}MULTIPLICITY    2")
                            mult_idx = insert_idx

                        if lsd_idx is None:
                            # Insert LSD just above MULTIPLICITY if present, otherwise after CHARGE
                            target_idx = None
                            if mult_idx is not None:
                                target_idx = mult_idx
                            elif charge_idx is not None:
                                target_idx = charge_idx + 1

                            if target_idx is not None:
                                indent = lines[target_idx][: len(lines[target_idx]) - len(lines[target_idx].lstrip())]
                                lines.insert(target_idx, f"{indent}LSD")

                    content = "\n".join(lines)

        except Exception as e:
            warning(f" @CP2KEndpoint: Failed to auto-adjust LSD/MULTIPLICITY: {e}", verbosity.medium)

        # Extract PROJECT name from &GLOBAL section for later cube file identification
        try:
            project_name = None
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("PROJECT"):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        project_name = parts[1]
                        break
            if project_name:
                self.project_name = project_name
        except Exception:
            # If we cannot determine the project name, leave it as None and fall back later
            pass

        # Ensure a V_HARTREE_CUBE print block exists under &DFT
        upper = content.upper()
        if "&V_HARTREE_CUBE" not in upper:
            # Work line-wise within the &DFT ... &END DFT block
            lines = content.splitlines(True)  # keep line endings
            dft_start_idx = None
            dft_end_idx = None
            for i, line in enumerate(lines):
                stripped = line.strip().upper()
                if dft_start_idx is None and stripped.startswith("&DFT"):
                    dft_start_idx = i
                if stripped.startswith("&END DFT"):
                    dft_end_idx = i
                    break

            if dft_start_idx is not None and dft_end_idx is not None:
                # Look for existing PRINT blocks inside &DFT
                print_start_indices = []
                for i in range(dft_start_idx, dft_end_idx + 1):
                    if lines[i].lstrip().upper().startswith("&PRINT"):
                        print_start_indices.append(i)

                if print_start_indices:
                    # Insert V_HARTREE_CUBE into the last &PRINT before its &END PRINT
                    print_start = print_start_indices[-1]
                    end_print_idx = None
                    for j in range(print_start + 1, dft_end_idx + 1):
                        if lines[j].lstrip().upper().startswith("&END PRINT"):
                            end_print_idx = j
                            break

                    if end_print_idx is not None:
                        end_line = lines[end_print_idx]
                        indent = end_line[: len(end_line) - len(end_line.lstrip())]
                        inner_indent = indent + "  "
                        vh_lines = [
                            f"{indent}&V_HARTREE_CUBE\n",
                            f"{inner_indent}STRIDE 1\n",
                            f"{indent}&END V_HARTREE_CUBE\n",
                        ]
                        lines = lines[:end_print_idx] + vh_lines + lines[end_print_idx:]
                        content = "".join(lines)
                else:
                    # No existing PRINT block under DFT: append a new one before &END DFT
                    end_line = lines[dft_end_idx]
                    indent = end_line[: len(end_line) - len(end_line.lstrip())]
                    vh_lines = [
                        f"{indent}&PRINT\n",
                        f"{indent}  &V_HARTREE_CUBE\n",
                        f"{indent}    STRIDE 1\n",
                        f"{indent}  &END V_HARTREE_CUBE\n",
                        f"{indent}&END PRINT\n",
                    ]
                    lines = lines[:dft_end_idx] + vh_lines + lines[dft_end_idx:]
                    content = "".join(lines)

        info(f" @CP2KEndpoint: Rendered template for charge {self.charge}", verbosity.debug)
        return content

    def create_input_file(self):
        """Create temporary CP2K input file from template.

        Returns:
            str: Path to the created input file
        """
        content = self.render_template()

        # Create input file in current working directory
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        self.input_file = f"cp2k_{unique_id}_q{self.charge}.inp"

        with open(self.input_file, 'w') as f:
            f.write(content)

        info(f" @CP2KEndpoint: Created input file {self.input_file} for charge {self.charge}", verbosity.debug)
        return self.input_file

    def create_socket_server(self):
        """Create socket server for i-PI communication.

        Returns:
            bool: True if server created successfully
        """
        return self.socket_server.create_server()

    def start_process(self):
        """Start the CP2K process and create socket server.

        Returns:
            bool: True if process started successfully
        """
        if self.is_running:
            warning(f"CP2K endpoint for charge {self.charge} is already running", verbosity.medium)
            return True

        try:
            # Step 1: Create socket server first
            if not self.create_socket_server():
                raise RuntimeError(f"Failed to create socket server for charge {self.charge}")

            # Step 2: Create input file
            self.create_input_file()

            # Create output and error files in current working directory instead of /tmp
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            self.output_file = f"{base_name}.out"
            self.error_file = f"{base_name}.err"

            # Set up environment for CP2K
            env = os.environ.copy()

            # Determine working directory for CP2K input
            input_dir = os.path.dirname(self.input_file) if os.path.dirname(self.input_file) else os.getcwd()

            # Build run command using full cp2k_run_cmd prefix
            if not self.cp2k_run_cmd or not str(self.cp2k_run_cmd).strip():
                raise ValueError("cp2k_run_cmd must be provided and non-empty.")
            run_part = f"{self.cp2k_run_cmd.strip()} {self.cp2k_exe} {os.path.basename(self.input_file)}"

            # Build full shell command: optional env setup, then cd, then run
            cmd_parts = []
            if getattr(self, "cp2k_env", ""):
                env_cmd = self.cp2k_env.strip()
                if env_cmd:
                    cmd_parts.append(env_cmd)
            cmd_parts.append(f"cd {input_dir}")
            cmd_parts.append(f"{run_part} 1>{self.output_file} 2>{self.error_file}")
            cmd_str = " && ".join(cmd_parts)

            cmd = [
                "bash", "-c",
                cmd_str,
            ]

            # Start the process
            self.process = subprocess.Popen(
                cmd,
                env=env,
                cwd=input_dir,
            )

            self.is_running = True
            self.last_health_check = time.time()

            info(f" @CP2KEndpoint: Started CP2K process (PID: {self.process.pid}) for charge {self.charge}", verbosity.medium)

            # Give the process some time to initialize
            time.sleep(2.0)

            # Check if process is still alive after startup
            if not self.health_check():
                error_msg = f"CP2K process for charge {self.charge} failed during startup"
                # Try to read error message from output file
                if self.output_file and os.path.exists(self.output_file):
                    try:
                        with open(self.output_file, 'r') as f:
                            output_content = f.read()
                            if output_content.strip():
                                error_msg += f"\nCP2K output:\n{output_content}"
                    except Exception:
                        pass
                raise RuntimeError(error_msg)

            return True

        except Exception as e:
            error_msg = f"Failed to start CP2K process for charge {self.charge}: {e}"
            warning(error_msg, verbosity.low)
            self.cleanup()
            return False

    def stop_process(self):
        """Stop the CP2K process gracefully."""
        if not self.is_running or self.process is None:
            return

        try:
            info(f" @CP2KEndpoint: Stopping CP2K process (PID: {self.process.pid}) for charge {self.charge}", verbosity.medium)

            # Try graceful termination first
            self.process.terminate()

            # Wait for termination with timeout
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination failed
                warning(f"Force killing CP2K process for charge {self.charge}", verbosity.medium)
                self.process.kill()
                self.process.wait()

        except Exception as e:
            warning(f"Error stopping CP2K process for charge {self.charge}: {e}", verbosity.medium)
        finally:
            self.is_running = False
            self.process = None
            self.cleanup()

    def accept_client_connection(self, timeout=30.0):
        """Accept connection from CP2K client.

        Args:
            timeout: Timeout in seconds for connection

        Returns:
            bool: True if connection established successfully
        """
        return self.socket_server.accept_connection(timeout)

    def close_client_connection(self):
        """Close client socket connection."""
        self.socket_server.close_connection()

    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            # Close socket connections using the new socket classes
            self.socket_server.cleanup()

            # Clean up temporary files
            if self.input_file and os.path.exists(self.input_file):
                os.unlink(self.input_file)
                self.input_file = None

            if self.output_file and os.path.exists(self.output_file):
                # Keep output file for debugging, just note its location
                info(f" @CP2KEndpoint: CP2K output saved to {self.output_file}", verbosity.debug)

        except Exception as e:
            warning(f"Error during cleanup for charge {self.charge}: {e}", verbosity.medium)

    def health_check(self):
        """Check if the CP2K process is healthy.

        Returns:
            bool: True if process is healthy
        """
        if not self.is_running or self.process is None:
            return False

        # Check if process is still alive
        poll_result = self.process.poll()
        if poll_result is not None:
            warning(f"CP2K process for charge {self.charge} has terminated with code {poll_result}", verbosity.medium)
            self.is_running = False
            return False

        self.last_health_check = time.time()
        return True

    def get_status(self):
        """Get current endpoint status.

        Returns:
            dict: Status information
        """
        return {
            "charge": self.charge,
            "is_running": self.is_running,
            "pid": self.process.pid if self.process else None,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "last_health_check": self.last_health_check,
            "socket_info": f"inet://{self.host}:{self.port}",
        }


class FFMixTwoSockets(FFEval):
    """Mixed two-endpoint forcefield for constant potential simulations.

    This forcefield manages two CP2K endpoints with different integer charges
    and performs λ-mixing of energies and forces based on the continuous
    electronic charge. Supports automatic thermal switching and chemical
    potential calculation using linear mixing.

    The λ-mixing formula is: λ = (q_ele - q1_ele) / (q2_ele - q1_ele)
    where q_ele is the current electron number, q1_ele and q2_ele are the
    electron numbers of the two endpoints.

    Endpoint ordering: q1 > q2 (q1_ele < q2_ele in electron numbers)
    Switching thresholds: λ > 1.05 (switch up), λ < -0.05 (switch down)

    Attributes:
        endpoint1: First CP2K endpoint (higher charge, lower electron number)
        endpoint2: Second CP2K endpoint (lower charge, higher electron number)
        q1: Electron number of endpoint 1 (q1_ele = neutral_electrons - q1_relative)
        q2: Electron number of endpoint 2 (q2_ele = neutral_electrons - q2_relative)
        q1_relative: Net charge of endpoint 1 (for CP2K CHARGE parameter)
        q2_relative: Net charge of endpoint 2 (for CP2K CHARGE parameter)
        current_q: Current continuous electronic number
        lambda_val: Current λ value for mixing
        mixing_mode: Chemical potential calculation mode (fixed to 'linear')
        cp2k_template: Template string for CP2K input with placeholders
        auto_switch: Whether to enable automatic thermal switching
        neutral_electrons: Total electron number for the neutral reference system.
            This must be set to a positive integer before the first call to
            set_electronic_state / set_electronic_charge, and is typically
            provided by the electronic degrees of freedom (InputElectrons).
    """

    def __init__(
        self,
        latency=1.0,
        offset=0.0,
        name="",
        pars=None,
        dopbc=True,
        active=np.array([-1]),
        threaded=True,
        switch_threshold=0.05,
        cp2k_template="",
        cp2k_template_path=None,
        cp2k_exe=None,
        auto_switch=True,
        host="localhost",
        port_base=12345,
        cp2k_env="",
        cp2k_run_cmd=None,
        cp2k_run_cmd1=None,
        cp2k_run_cmd2=None,
    ):
        """Initialize FFMixTwoSockets.

        Args:
            latency: The number of seconds between polling cycles
            offset: Constant energy offset
            name: Name of the forcefield
            pars: Parameters dictionary
            dopbc: Whether to apply periodic boundary conditions
            active: Active atom indices
            threaded: Whether to use threading
            q1: Integer charge of first endpoint (relative to neutral system)
            q2: Integer charge of second endpoint (relative to neutral system)
            initial_q: Initial continuous electronic charge (relative to neutral system)
            mixing_mode: Chemical potential calculation mode (fixed to 'linear')
            switch_threshold: Hysteresis threshold ε for switching
            cp2k_template: CP2K input template with placeholders
            cp2k_exe: Path to CP2K executable
            auto_switch: Enable automatic thermal switching
            host: Host for socket communication
            port_base: Base port number (endpoint1=port_base, endpoint2=port_base+1)
        """

        # Initialize base class - force threaded mode for concurrent endpoints
        super(FFMixTwoSockets, self).__init__(
            latency=latency,
            offset=offset,
            name=name,
            pars=pars,
            dopbc=dopbc,
            active=active,
            threaded=True,  # Required for concurrent endpoint management
        )

        # Neutral electron count will be provided later (typically from the
        # electronic_state created by InputElectrons). It must be set to a
        # positive integer before charges are initialized.
        self.neutral_electrons = None

        # Initialize charge-related members. These are now always inferred lazily
        # from the electronic charge q via _initialize_charges_from_q.
        self.q1_relative = None
        self.q2_relative = None
        self.initial_q_relative = None
        self.q1 = None
        self.q2 = None
        self.current_q = None
        self._charges_initialized = False

        # Mixing and switching parameters
        self.mixing_mode = "linear"
        self.switch_threshold = float(switch_threshold)
        self.auto_switch = auto_switch

        # CP2K configuration
        self.cp2k_template = cp2k_template
        self.cp2k_template_path = cp2k_template_path
        self.cp2k_exe = cp2k_exe
        self.host = host
        self.port_base_initial = port_base
        self.port_base = port_base
        self.port_rotation = 0  # how many times ports have been bumped
        self.cp2k_env = cp2k_env
        if cp2k_run_cmd1 is None and cp2k_run_cmd2 is None:
            self.cp2k_run_cmd1 = cp2k_run_cmd
            self.cp2k_run_cmd2 = cp2k_run_cmd
        else:
            self.cp2k_run_cmd1 = cp2k_run_cmd1 if cp2k_run_cmd1 is not None else cp2k_run_cmd
            self.cp2k_run_cmd2 = cp2k_run_cmd2 if cp2k_run_cmd2 is not None else cp2k_run_cmd
        self.cp2k_run_cmd = self.cp2k_run_cmd1

        # Validate CP2K configuration
        if not cp2k_template:
            raise ValueError("cp2k_template is required. Dummy endpoints are no longer supported.")
        if not cp2k_exe:
            raise ValueError("cp2k_exe must be provided for FFMixTwoSockets; it is a required parameter.")
        if not os.path.exists(cp2k_exe):
            warning(f"CP2K executable not found at {cp2k_exe}", verbosity.medium)

        # Validate launcher commands are provided for both endpoints
        def _is_non_empty(val):
            return bool(val) and bool(str(val).strip())

        if not _is_non_empty(self.cp2k_run_cmd1) or not _is_non_empty(self.cp2k_run_cmd2):
            raise ValueError("Launcher commands must be provided and non-empty for both CP2K endpoints.")

        # Initialize CP2K endpoints (will be started in start())
        self.endpoint1 = None
        self.endpoint2 = None

        # State tracking
        if self._charges_initialized and self.q1 is not None and self.q2 is not None and self.current_q is not None:
            self.lambda_val = self._calculate_lambda()
        else:
            # Will be set on first call to set_electronic_state / set_electronic_charge
            self.lambda_val = 0.0
        self.last_results = {"endpoint1": None, "endpoint2": None}
        self.current_mu = 0.0  # Current chemical potential
        self.endpoints_started = False  # Track if CP2K processes have been started
        self.handshake_completed = False  # Track if initial handshake is done

        # Statistics and monitoring
        self.switch_count = 0
        self.evaluation_count = 0

        # Cache mechanism to avoid redundant calculations
        self._cached_result = None
        self._cached_positions = None
        self._cached_cell = None
        # 注意：不再缓存电荷，因为在恒电势模拟中电荷应该动态变化

        info(f" @FFMixTwoSockets: Initialized with neutral_electrons={self.neutral_electrons}", verbosity.medium)
        if self._charges_initialized:
            info(
                f" @FFMixTwoSockets: Relative charges: q1={self.q1_relative}, q2={self.q2_relative}, initial_q={self.initial_q_relative}",
                verbosity.medium,
            )
            info(
                f" @FFMixTwoSockets: Absolute charges (i-PI): q1={self.q1}, q2={self.q2}, current_q={self.current_q}",
                verbosity.medium,
            )
        else:
            info(
                " @FFMixTwoSockets: Endpoint charges will be inferred from neutral_electrons and electronic q_init at first electronic_state sync",
                verbosity.medium,
            )
        info(f" @FFMixTwoSockets: Mode={self.mixing_mode}, threshold={switch_threshold}, auto_switch={auto_switch}", verbosity.medium)

    def _ensure_neutral_electrons(self):
        """Validate that neutral_electrons has been set to a positive integer.

        This helper should be called before any logic that relies on the
        neutral electron count (e.g. charge initialization). It raises a
        clear error if the attribute has not been configured yet.
        """

        if getattr(self, "neutral_electrons", None) is None:
            raise ValueError(
                "FFMixTwoSockets.neutral_electrons is not set. "
                "neutral_electrons must be provided by the electronic degrees of freedom (InputElectrons) "
                "before constant potential simulations can start."
            )
        try:
            ne = int(self.neutral_electrons)
        except (TypeError, ValueError):
            raise ValueError(
                f"FFMixTwoSockets.neutral_electrons must be an integer, got {self.neutral_electrons!r}"
            )
        if ne <= 0:
            raise ValueError(
                f"FFMixTwoSockets.neutral_electrons must be a positive integer, got {ne}"
            )
        self.neutral_electrons = ne

    def _initialize_charges_from_q(self, q):
        """Infer endpoint charges (q1,q2,initial_q) from neutral_electrons and electronic charge.

        This is the automatic mode requested for constant potential simulations.
        Given the continuous electronic charge q (in electrons) and the neutral
        electron count, we define the relative initial charge and endpoint
        charges as:

            initial_q_rel = neutral_electrons - q
            q2 = floor(initial_q_rel)
            q1 = q2 + 1

        Args:
            q (float): Current continuous electronic charge (electron number).
        """

        # Ensure neutral_electrons has been configured
        self._ensure_neutral_electrons()

        q_elec = float(q)
        if q_elec <= 1.0:
            raise ValueError(
                f"Electronic charge must be > 1.0 to define mixing endpoints, got {q_elec}"
            )

        # Relative initial charge: neutral_electrons - q_elec
        initial_rel = self.neutral_electrons - q_elec
        q2_rel = int(np.floor(initial_rel))
        q1_rel = q2_rel + 1

        # Store relative and absolute charges
        self.q1_relative = q1_rel
        self.q2_relative = q2_rel
        self.initial_q_relative = initial_rel

        self.q1 = self.neutral_electrons - self.q1_relative
        self.q2 = self.neutral_electrons - self.q2_relative
        self.current_q = q_elec

        # Validate that electron numbers are positive
        if self.q1 <= 1.0 or self.q2 <= 1.0 or self.current_q <= 1.0:
            raise ValueError(
                f"All electron numbers must be > 1.0. Got q1_ele={self.q1}, q2_ele={self.q2}, current_q_ele={self.current_q}"
            )

        self.lambda_val = self._calculate_lambda()
        self._charges_initialized = True

        info(
            f" @FFMixTwoSockets: Inferred endpoint charges from q={q_elec:.6f}: "
            f"q1_rel={self.q1_relative}, q2_rel={self.q2_relative}, "
            f"q1_ele={self.q1}, q2_ele={self.q2}, λ={self.lambda_val:.6f}",
            verbosity.medium,
        )

    def _calculate_lambda(self):
        """Calculate λ value from current electronic charge.

        Returns:
            float: λ value (can be outside [0, 1] to trigger endpoint switching)
        """
        if self.q2 == self.q1:
            return 0.5  # Fallback for edge case
        lambda_val = (self.current_q - self.q1) / (self.q2 - self.q1)
        return lambda_val  # Don't clamp - allow values outside [0, 1] for switching detection

    def set_electronic_state(self, q):
        """Set the current electronic state (interface for i-PI integration).

        This method is called by i-PI's electronic thermostat and forwards
        the call to set_electronic_charge to trigger λ-mixing updates and
        automatic thermal switching.

        Args:
            q (float): New electronic charge
        """
        info(f" @FFMixTwoSockets: set_electronic_state called with q={q:.6f}", verbosity.medium)
        self.set_electronic_charge(q)

    def set_electronic_charge(self, q):
        """Set the current electronic charge and update λ.

        Args:
            q (float): New electronic charge
        """
        q_val = float(q)

        # Lazy initialization path: if charges have not been set yet, infer them
        # from the initial electronic charge and neutral_electrons.
        if not getattr(self, "_charges_initialized", False) or self.q1 is None or self.q2 is None:
            self._initialize_charges_from_q(q_val)
            return

        old_q = self.current_q
        self.current_q = q_val
        old_lambda = self.lambda_val
        self.lambda_val = self._calculate_lambda()

        info(
            f" @FFMixTwoSockets: Charge updated from {old_q:.6f} to {q_val:.6f}, λ: {old_lambda:.6f} → {self.lambda_val:.6f}",
            verbosity.debug,
        )

        # Check if thermal switching is needed
        if self.auto_switch:
            self._check_thermal_switch()

    def _check_thermal_switch(self):
        """Check if thermal switching should occur based on λ value."""
        # Switch when λ drifts outside [-ε, 1+ε]
        lower = -self.switch_threshold
        upper = 1.0 + self.switch_threshold

        if self.lambda_val < lower:
            info(f" @FFMixTwoSockets: λ={self.lambda_val:.6f} < {lower:.4f}, triggering thermal switch down", verbosity.medium)
            # Thermal switch down: (q1,q2) → (q1+1,q1)
            self._perform_thermal_switch(direction="down")
        elif self.lambda_val > upper:
            info(f" @FFMixTwoSockets: λ={self.lambda_val:.6f} > {upper:.4f}, triggering thermal switch up", verbosity.medium)
            # Thermal switch up: (q1,q2) → (q2,q2-1)
            self._perform_thermal_switch(direction="up")

    def _perform_thermal_switch(self, direction):
        """Perform thermal switching of endpoints.

        Args:
            direction (str): "up" or "down" indicating switch direction
        """
        # Store old values for rollback
        old_q1_abs, old_q2_abs = self.q1, self.q2
        old_q1_rel, old_q2_rel = self.q1_relative, self.q2_relative
        # If endpoints have not been created yet (before start()), avoid spawning CP2K here
        lazy_only = (self.endpoint1 is None and self.endpoint2 is None and not self.endpoints_started)

        try:
            # Determine new endpoint charges (work with relative charges)
            # Note: q1 > q2, λ > 1.05 means shift range up (decrease charges), λ < -0.05 means shift range down (increase charges)
            if direction == "down":
                # Switch down: λ < -0.05, shift range down (q1_rel,q2_rel) → (q1_rel+1,q1_rel)
                new_q1_rel = self.q1_relative + 1
                new_q2_rel = self.q1_relative
            elif direction == "up":
                # Switch up: λ > 1.05, shift range up (q1_rel,q2_rel) → (q2_rel,q2_rel-1)
                new_q1_rel = self.q2_relative
                new_q2_rel = self.q2_relative - 1
            else:
                raise ValueError(f"Invalid switch direction: {direction}")

            # Ensure q1 > q2 is maintained
            if new_q1_rel <= new_q2_rel:
                raise ValueError(f"Switch would violate q1 > q2 requirement: new_q1_rel={new_q1_rel}, new_q2_rel={new_q2_rel}")

            # Update to new electron numbers (neutral_electrons - charge)
            new_q1_ele = self.neutral_electrons - new_q1_rel
            new_q2_ele = self.neutral_electrons - new_q2_rel

            info(f" @FFMixTwoSockets: Switching relative charges from ({old_q1_rel},{old_q2_rel}) to ({new_q1_rel},{new_q2_rel})", verbosity.medium)
            info(f" @FFMixTwoSockets: Switching electron numbers from ({old_q1_abs},{old_q2_abs}) to ({new_q1_ele},{new_q2_ele})", verbosity.medium)

            # Stop current endpoints
            self._stop_current_endpoints()

            # Update both relative charges and electron numbers
            self.q1_relative = new_q1_rel
            self.q2_relative = new_q2_rel
            self.q1 = new_q1_ele
            self.q2 = new_q2_ele

            # Recalculate λ with new endpoints
            old_lambda = self.lambda_val
            self.lambda_val = self._calculate_lambda()

            info(f" @FFMixTwoSockets: λ updated from {old_lambda:.6f} to {self.lambda_val:.6f} after switching", verbosity.medium)

            # Start new endpoints unless we are still before start()
            if not lazy_only:
                # Bump ports to avoid TIME_WAIT collisions when rapidly restarting endpoints
                self._start_new_endpoints(bump_ports=True)
            else:
                # Defer endpoint creation to start(); make sure flags reflect that nothing is running
                self.endpoints_started = False
                self.handshake_completed = False

            # Update switch counter
            self.switch_count += 1

            info(f" @FFMixTwoSockets: Thermal switch #{self.switch_count} completed successfully", verbosity.medium)

        except Exception as e:
            error_msg = f"Thermal switching failed: {e}"
            warning(error_msg, verbosity.low)

            # Restore original endpoint charges on failure (both relative and absolute)
            self.q1_relative, self.q2_relative = old_q1_rel, old_q2_rel
            self.q1, self.q2 = old_q1_abs, old_q2_abs
            self.lambda_val = self._calculate_lambda()

            # Try to restart original endpoints
            try:
                self._start_new_endpoints(bump_ports=True)
                warning("Restored original endpoints after failed switch", verbosity.medium)
            except:
                warning("Failed to restore original endpoints - system may be unstable", verbosity.low)

    def _stop_current_endpoints(self):
        """Stop current CP2K endpoints."""
        if self.endpoint1:
            try:
                self.endpoint1.stop_process()
                info(f" @FFMixTwoSockets: Stopped endpoint 1 (q={self.endpoint1.charge})", verbosity.debug)
            except Exception as e:
                warning(f"Error stopping endpoint 1: {e}", verbosity.medium)
            finally:
                self.endpoint1 = None

        if self.endpoint2:
            try:
                self.endpoint2.stop_process()
                info(f" @FFMixTwoSockets: Stopped endpoint 2 (q={self.endpoint2.charge})", verbosity.debug)
            except Exception as e:
                warning(f"Error stopping endpoint 2: {e}", verbosity.medium)
            finally:
                self.endpoint2 = None

    def _ensure_endpoints_started(self):
        """Ensure CP2K endpoints are started and handshaked (eager initialization)."""
        if self.endpoints_started:
            return  # Already started and handshaked

        try:
            info(" @FFMixTwoSockets: Starting CP2K endpoint processes with full handshake...", verbosity.medium)

            # Step 1: Start processes
            if self.endpoint1.start_process():
                info(f" @FFMixTwoSockets: Started endpoint 1 (CP2K_CHARGE={self.q1_relative}, i-PI_q={self.q1})", verbosity.medium)
            else:
                raise RuntimeError(f"Failed to start endpoint 1 (CP2K_CHARGE={self.q1_relative}, i-PI_q={self.q1})")

            if self.endpoint2.start_process():
                info(f" @FFMixTwoSockets: Started endpoint 2 (CP2K_CHARGE={self.q2_relative}, i-PI_q={self.q2})", verbosity.medium)
            else:
                raise RuntimeError(f"Failed to start endpoint 2 (CP2K_CHARGE={self.q2_relative}, i-PI_q={self.q2})")

            # Step 2: Wait for socket servers to be ready - PARALLEL HANDSHAKE
            info(" @FFMixTwoSockets: Performing initial handshake with both endpoints...", verbosity.medium)
            with ThreadPoolExecutor(max_workers=2) as executor:
                info(" @FFMixTwoSockets: Starting parallel endpoint handshake", verbosity.medium)

                # Submit both socket ready tasks in parallel
                future1 = executor.submit(self._wait_for_socket_ready, self.endpoint1)
                future2 = executor.submit(self._wait_for_socket_ready, self.endpoint2)

                # Wait for both to complete
                socket_ready1 = future1.result()
                socket_ready2 = future2.result()

            if socket_ready1 and socket_ready2:
                info(" @FFMixTwoSockets: ✓ Both endpoints handshaked successfully during startup", verbosity.medium)
                self.endpoints_started = True
                self.handshake_completed = True
            else:
                raise RuntimeError(f"Handshake failed (EP1: {socket_ready1}, EP2: {socket_ready2})")

            info(" @FFMixTwoSockets: CP2K endpoints started and handshaked successfully", verbosity.medium)

        except Exception as e:
            # Clean up on failure
            if self.endpoint1:
                self.endpoint1.stop_process()
            if self.endpoint2:
                self.endpoint2.stop_process()
            raise RuntimeError(f"Failed to start and handshake endpoints: {e}")

    def _start_new_endpoints(self, bump_ports=False):
        """Start new CP2K endpoints with current q1, q2 values.

        Args:
            bump_ports (bool): If True, shift port_base by +2 to avoid TIME_WAIT collisions.
        """
        if bump_ports:
            self.port_rotation += 1
            self.port_base = self.port_base_initial + 2 * self.port_rotation
            info(f" @FFMixTwoSockets: Bumping port_base to {self.port_base} for new endpoints", verbosity.medium)
        try:
            # Create new endpoints - use negative relative charges for CP2K CHARGE parameter
            # Physical relation: CP2K_CHARGE = q_relative (net charge of the system)
            # q_relative = -1 means net charge -1 (1 extra electron), CP2K_CHARGE = -1
            # q_relative = 0 means neutral system, CP2K_CHARGE = 0
            self.endpoint1 = CP2KEndpoint(
                charge=self.q1_relative,
                host=self.host,
                port=self.port_base,
                cp2k_template=self.cp2k_template,
                cp2k_exe=self.cp2k_exe,
                cp2k_env=self.cp2k_env,
                cp2k_run_cmd=self.cp2k_run_cmd1,
                neutral_electrons=self.neutral_electrons,
            )

            self.endpoint2 = CP2KEndpoint(
                charge=self.q2_relative,
                host=self.host,
                port=self.port_base + 1,
                cp2k_template=self.cp2k_template,
                cp2k_exe=self.cp2k_exe,
                cp2k_env=self.cp2k_env,
                cp2k_run_cmd=self.cp2k_run_cmd2,
                neutral_electrons=self.neutral_electrons,
            )

            # Start processes
            if self.endpoint1.start_process():
                info(f" @FFMixTwoSockets: Started new endpoint 1 (CP2K_CHARGE={self.q1_relative}, i-PI_q={self.q1})", verbosity.debug)
            else:
                raise RuntimeError(f"Failed to start new endpoint 1 (CP2K_CHARGE={self.q1_relative}, i-PI_q={self.q1})")

            if self.endpoint2.start_process():
                info(f" @FFMixTwoSockets: Started new endpoint 2 (CP2K_CHARGE={self.q2_relative}, i-PI_q={self.q2})", verbosity.debug)
            else:
                raise RuntimeError(f"Failed to start new endpoint 2 (CP2K_CHARGE={self.q2_relative}, i-PI_q={self.q2})")

            # CRITICAL FIX: Perform handshake with new endpoints (same as in _ensure_endpoints_started)
            info(" @FFMixTwoSockets: Performing handshake with new endpoints after switching...", verbosity.medium)
            with ThreadPoolExecutor(max_workers=2) as executor:
                info(" @FFMixTwoSockets: Starting parallel handshake for switched endpoints", verbosity.medium)

                # Submit both socket ready tasks in parallel
                future1 = executor.submit(self._wait_for_socket_ready, self.endpoint1)
                future2 = executor.submit(self._wait_for_socket_ready, self.endpoint2)

                # Wait for both to complete
                socket_ready1 = future1.result()
                socket_ready2 = future2.result()

            if socket_ready1 and socket_ready2:
                info(" @FFMixTwoSockets: ✓ Both new endpoints handshaked successfully after switching", verbosity.medium)
                self.endpoints_started = True
                self.handshake_completed = True
            else:
                raise RuntimeError(f"New endpoint handshake failed after switching (EP1: {socket_ready1}, EP2: {socket_ready2})")

            info(f" @FFMixTwoSockets: New endpoints started and handshaked successfully after thermal switch", verbosity.medium)

        except Exception as e:
            # Clean up on failure
            if self.endpoint1:
                self.endpoint1.stop_process()
                self.endpoint1 = None
            if self.endpoint2:
                self.endpoint2.stop_process()
                self.endpoint2 = None
            raise RuntimeError(f"Failed to start new endpoints: {e}")

    def can_switch_up(self):
        """Check if upward thermal switching is possible.

        Returns:
            bool: True if switch up is allowed
        """
        # Could add constraints like maximum charge, computational cost, etc.
        return True  # For now, allow unlimited switching

    def can_switch_down(self):
        """Check if downward thermal switching is possible.

        Returns:
            bool: True if switch down is allowed
        """
        # Could add constraints like minimum charge, etc.
        return True  # For now, allow unlimited switching

    def start(self):
        """Start the forcefield and prepare endpoints for lazy initialization."""
        info(f" @FFMixTwoSockets: Starting with endpoints q1={self.q1}, q2={self.q2}", verbosity.low)

        # If endpoints were started earlier (e.g., during pre-start thermal switch), stop and reset
        if self.endpoints_started or self.endpoint1 or self.endpoint2:
            try:
                self._stop_current_endpoints()
            finally:
                self.endpoints_started = False
                self.handshake_completed = False

        # Prepare CP2K endpoint configuration (but don't start processes yet)
        try:
            # Create endpoint 1 (lower charge) - use negative relative charge for CP2K CHARGE parameter
            # Physical relation: CP2K_CHARGE = q_relative (net charge of the system)
            # q_relative = -1 means net charge -1 (1 extra electron), CP2K_CHARGE = -1
            # q_relative = 0 means neutral system, CP2K_CHARGE = 0
            self.endpoint1 = CP2KEndpoint(
                charge=self.q1_relative,
                host=self.host,
                port=self.port_base,
                cp2k_template=self.cp2k_template,
                cp2k_exe=self.cp2k_exe,
                cp2k_env=self.cp2k_env,
                cp2k_run_cmd=self.cp2k_run_cmd,
                neutral_electrons=self.neutral_electrons,
            )

            # Create endpoint 2 (higher charge) - use relative charge for CP2K CHARGE parameter
            self.endpoint2 = CP2KEndpoint(
                charge=self.q2_relative,
                host=self.host,
                port=self.port_base + 1,
                cp2k_template=self.cp2k_template,
                cp2k_exe=self.cp2k_exe,
                cp2k_env=self.cp2k_env,
                cp2k_run_cmd=self.cp2k_run_cmd,
                neutral_electrons=self.neutral_electrons,
            )

            info(" @FFMixTwoSockets: CP2K endpoints configured (processes will start on first evaluation)", verbosity.medium)

        except Exception as e:
            error_msg = f"Failed to configure CP2K endpoints: {e}"
            warning(error_msg, verbosity.low)
            raise RuntimeError(error_msg)

        # Call parent start method
        super(FFMixTwoSockets, self).start()

        info(" @FFMixTwoSockets: Started successfully (lazy initialization mode)", verbosity.medium)
        
    def evaluate(self, request):
        """Evaluate request using two-endpoint λ-mixing.

        This method implements the FFEval interface by calculating forces
        and energy from two CP2K endpoints and performing λ-mixing.

        Args:
            request: Standard i-PI request dictionary with pos, cell, etc.
        """
        # Calculate λ from current electronic charge (need to get from somewhere)
        lambda_val = self._calculate_lambda()
        request["lambda"] = lambda_val

        # Evaluate both endpoints
        result1, result2 = self._evaluate_endpoints(request)

        # Perform λ-mixing and format result
        mixed_result = self._perform_lambda_mixing(result1, result2, request)

        # Set result in standard i-PI format
        request["result"] = mixed_result
        request["status"] = "Done"


    def _directly_cache_fermi_level(self, extras):
        """直接在FFMixTwoSockets评估后设置费米能级缓存，避免后续通过depend机制访问forces.extras"""
        try:
            from ipi.utils.messages import info, warning, verbosity
            info(" @FFMixTwoSockets: Attempting to cache Fermi level", verbosity.medium)

            # 尝试找到Dynamics对象
            dynamics = self._find_dynamics_object()
            if dynamics is None:
                warning(" @FFMixTwoSockets: Could not find Dynamics object for caching", verbosity.medium)
                return

            # 从extras中提取费米能级
            if isinstance(extras, dict) and 'fermi_level_eV' in extras:
                fermi_level = float(extras['fermi_level_eV'])

                # 直接设置缓存
                dynamics._fermi_cache = {"ffmix_fermi": fermi_level}
                dynamics._fermi_cache_valid = True

                info(f" @FFMixTwoSockets: ✓ Successfully cached Fermi level: {fermi_level:.6f} eV", verbosity.medium)

                # 如果处于恒功函数模式，同时缓存当前功函数（Hartree）
                electronic_state = getattr(dynamics, 'electronic_state', None)
                mode = getattr(electronic_state, 'mode', 'fermi') if electronic_state is not None else 'fermi'
                if mode == 'workfunction' and 'workfunction_eV' in extras:
                    wf_eV = float(extras['workfunction_eV'])
                    wf_au = wf_eV / Constants.EV_PER_HARTREE
                    electronic_state.current_workfunction = wf_au
                    info(
                        f" @FFMixTwoSockets: ✓ Cached workfunction for workfunction mode: {wf_eV:.6f} eV ({wf_au:.6f} Ha)",
                        verbosity.medium,
                    )
            else:
                warning(f" @FFMixTwoSockets: Invalid extras for caching - missing fermi_level_eV: {type(extras)} {extras}", verbosity.medium)

        except Exception as e:
            from ipi.utils.messages import warning, verbosity
            warning(f" @FFMixTwoSockets: ERROR: Could not directly cache Fermi level: {e}", verbosity.medium)
            import traceback
            warning(f" @FFMixTwoSockets: Traceback: {traceback.format_exc()}", verbosity.low)

    def _find_dynamics_object(self):
        """找到Dynamics对象"""
        try:
            from ipi.utils.messages import info, warning, verbosity

            if hasattr(self, '_dynamics_ref') and self._dynamics_ref is not None:
                info(" @FFMixTwoSockets: Using cached Dynamics reference", verbosity.debug)
                return self._dynamics_ref

            # 通过多种方式查找Dynamics对象
            import gc

            info(" @FFMixTwoSockets: Searching for Dynamics object...", verbosity.debug)

            # 方法1: 查找所有Dynamics对象
            candidates = []
            dynamics_count = 0
            for obj in gc.get_objects():
                if hasattr(obj, '__class__'):
                    class_name = str(obj.__class__)
                    if 'Dynamics' in class_name:
                        dynamics_count += 1
                        if hasattr(obj, '_fermi_cache'):
                            candidates.append(obj)
                            info(f" @FFMixTwoSockets: Found candidate: {class_name}", verbosity.debug)

            info(f" @FFMixTwoSockets: Found {dynamics_count} Dynamics objects, {len(candidates)} with _fermi_cache", verbosity.debug)

            if len(candidates) > 0:
                self._dynamics_ref = candidates[0]  # 使用第一个找到的
                info(f" @FFMixTwoSockets: ✓ Using Dynamics object: {type(self._dynamics_ref)}", verbosity.medium)
                return self._dynamics_ref

            # 方法2: 通过类型查找
            try:
                from ipi.engine.motion.dynamics import Dynamics
                for obj in gc.get_objects():
                    if isinstance(obj, Dynamics) and hasattr(obj, '_fermi_cache'):
                        self._dynamics_ref = obj
                        info(f" @FFMixTwoSockets: ✓ Found via isinstance: {type(self._dynamics_ref)}", verbosity.medium)
                        return self._dynamics_ref
            except ImportError:
                warning(" @FFMixTwoSockets: Could not import Dynamics class", verbosity.low)

            warning(f" @FFMixTwoSockets: ✗ Could not find Dynamics object with _fermi_cache (found {dynamics_count} Dynamics objects total)", verbosity.medium)
            return None
        except Exception as e:
            from ipi.utils.messages import warning, verbosity
            warning(f" @FFMixTwoSockets: ERROR in _find_dynamics_object: {e}", verbosity.medium)
            import traceback
            warning(f" @FFMixTwoSockets: Traceback: {traceback.format_exc()}", verbosity.low)
            return None

    def stop(self):
        """Stop the forcefield and clean up endpoints."""
        info(" @FFMixTwoSockets: Stopping and cleaning up endpoints", verbosity.low)

        # Stop CP2K endpoint processes
        if self.endpoint1:
            try:
                self.endpoint1.stop_process()
                info(f" @FFMixTwoSockets: Endpoint 1 (q={self.q1}) stopped", verbosity.medium)
            except Exception as e:
                warning(f"Error stopping endpoint 1: {e}", verbosity.medium)
            finally:
                self.endpoint1 = None

        if self.endpoint2:
            try:
                self.endpoint2.stop_process()
                info(f" @FFMixTwoSockets: Endpoint 2 (q={self.q2}) stopped", verbosity.medium)
            except Exception as e:
                warning(f"Error stopping endpoint 2: {e}", verbosity.medium)
            finally:
                self.endpoint2 = None

        # Call parent stop method
        super(FFMixTwoSockets, self).stop()

        info(" @FFMixTwoSockets: Stopped successfully", verbosity.medium)

    def queue(self, atoms, cell, reqid=-1):
        """Add a force evaluation request.

        Args:
            atoms: Atoms object with positions
            cell: Cell object with system box
            reqid: Request identifier

        Returns:
            ForceRequest object
        """
        self.evaluation_count += 1

        # Create base request using parent method
        request = super(FFMixTwoSockets, self).queue(atoms, cell, reqid, template={
            "lambda": self.lambda_val,
            "current_q": self.current_q,
            "q1": self.q1,
            "q2": self.q2,
            "mixing_mode": self.mixing_mode,
            "evaluation_count": self.evaluation_count
        })

        info(f" @FFMixTwoSockets: Queued evaluation #{self.evaluation_count}, λ={self.lambda_val:.6f}, q={self.current_q:.6f}", verbosity.debug)

        return request

    def poll(self):
        """Poll endpoint evaluations and perform λ-mixing."""
        with self._threadlock:
            for r in self.requests:
                if r["status"] == "Queued":
                    r["t_dispatched"] = time.time()
                    r["status"] = "Running"

                    # Try to evaluate endpoints
                    if self._evaluate_request(r):
                        r["status"] = "Done"
                        r["t_finished"] = time.time()

    def _evaluate_request(self, request):
        """Evaluate a request using both endpoints and perform λ-mixing.

        Args:
            request: The force request to evaluate

        Returns:
            bool: True if evaluation completed successfully
        """
        try:
            # Step 1: Evaluate both endpoints
            result1, result2 = self._evaluate_endpoints(request)

            # Step 2: Perform λ-mixing
            mixed_result = self._perform_lambda_mixing(result1, result2, request)

            # Step 3: Store final result
            request["result"] = mixed_result

            # Step 4: 直接缓存费米能级到Dynamics对象，避免后续通过forces.extras访问
            try:
                from ipi.utils.messages import info, verbosity
                info(f" @FFMixTwoSockets: About to cache fermi level, extras type: {type(mixed_result[3])}", verbosity.medium)
                if isinstance(mixed_result[3], dict):
                    info(f" @FFMixTwoSockets: Extras keys: {list(mixed_result[3].keys())}", verbosity.medium)
                    if 'fermi_level_eV' in mixed_result[3]:
                        info(f" @FFMixTwoSockets: fermi_level_eV = {mixed_result[3]['fermi_level_eV']}", verbosity.medium)
                self._directly_cache_fermi_level(mixed_result[3])
            except Exception as e:
                from ipi.utils.messages import warning, verbosity
                warning(f" @FFMixTwoSockets: Error in caching call: {e}", verbosity.medium)

            info(f" @FFMixTwoSockets: Completed evaluation #{request['evaluation_count']}, λ={request['lambda']:.6f}, μ={self.current_mu:.6f} eV", verbosity.debug)

            return True

        except Exception as e:
            from ipi.utils.messages import warning, verbosity
            warning(f"Endpoint evaluation failed: {e}", verbosity.medium)
            self._abort_due_to_cp2k_failure("Endpoint evaluation failed; aborting to preserve last checkpoint", e)

    def _abort_due_to_cp2k_failure(self, reason, exc=None):
        """Trigger a clean stop when CP2K data cannot be obtained."""
        from ipi.utils.messages import warning, verbosity

        message = f" @FFMixTwoSockets: {reason}"
        if exc is not None:
            message = f"{message}: {exc}"

        # Emit a single low-verbosity warning to make the failure explicit
        warning(message, verbosity.low)

        # Request a soft exit so the checkpoint writer is invoked
        try:
            from ipi.utils.softexit import softexit
            softexit.trigger(status="bad", message=message)
        except SystemExit:
            # Propagate the termination to stop the run immediately
            raise
        except Exception as trigger_exc:
            warning(f" @FFMixTwoSockets: Failed to trigger soft exit cleanly: {trigger_exc}", verbosity.low)

        # If soft exit did not terminate (e.g., in a controlled test), raise a fatal error
        if exc is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from exc

    def _is_cache_valid(self, request):
        """Check if the cached result is valid for the current request.

        Args:
            request: The force request to check

        Returns:
            bool: True if cache is valid, False otherwise
        """
        from ipi.utils.messages import info, verbosity

        # ALWAYS print cache check for debugging
        info(" @FFMixTwoSockets: [CACHE CHECK] Starting cache validation", verbosity.low)

        if self._cached_result is None:
            info(" @FFMixTwoSockets: [CACHE MISS] no cached result", verbosity.low)
            return False

        # Check if positions changed
        pos = request["pos"]
        if self._cached_positions is None:
            info(" @FFMixTwoSockets: [CACHE MISS] no cached positions", verbosity.low)
            return False

        if not np.allclose(pos, self._cached_positions, rtol=1e-12):
            pos_diff = np.linalg.norm(pos - self._cached_positions)
            info(f" @FFMixTwoSockets: [CACHE MISS] positions changed (diff={pos_diff:.2e})", verbosity.low)
            return False

        # Check if cell changed
        cell = request["cell"]
        if self._cached_cell is None:
            info(" @FFMixTwoSockets: [CACHE MISS] no cached cell", verbosity.low)
            return False

        # Need to compare cell matrices properly
        cell_h = cell[0] if isinstance(cell, tuple) else cell
        cached_cell_h = self._cached_cell[0] if isinstance(self._cached_cell, tuple) else self._cached_cell

        if not np.allclose(cell_h, cached_cell_h, rtol=1e-12):
            cell_diff = np.linalg.norm(cell_h - cached_cell_h)
            info(f" @FFMixTwoSockets: [CACHE MISS] cell changed (diff={cell_diff:.2e})", verbosity.low)
            return False

        # NOTE: 不再检查电荷变化！
        # 在恒电势模拟中，电荷本来就应该动态变化
        # 只要位置和晶胞相同，就可以使用缓存的力和能量
        # 电荷变化只会影响λ混合权重，但端点的原始结果是相同的

        info(" @FFMixTwoSockets: [CACHE HIT] using cached result", verbosity.low)
        return True

    def _update_cache(self, request, result):
        """Update the cache with current request and result.

        Args:
            request: The force request
            result: The computed result
        """
        from ipi.utils.messages import info, verbosity

        self._cached_result = result
        self._cached_positions = np.copy(request["pos"])
        self._cached_cell = np.copy(request["cell"])
        # 不再缓存电荷：在恒电势模拟中电荷应该动态变化
        # self._cached_q = self.current_q

        info(f" @FFMixTwoSockets: Cache UPDATED - saved result for reuse", verbosity.medium)

    def _evaluate_endpoints(self, request):
        """Evaluate both CP2K endpoints for the given request.

        Args:
            request: The force request containing positions and cell

        Returns:
            tuple: (result1, result2) from endpoints 1 and 2
        """
        # Check cache first - avoid expensive computation if result is already available
        if self._is_cache_valid(request):
            info(" @FFMixTwoSockets: Cache HIT - using cached result", verbosity.medium)
            return self._cached_result

        import traceback
        stack = traceback.extract_stack()
        caller_info = f"{stack[-2].filename}:{stack[-2].lineno} in {stack[-2].name}"

        # 尝试获取更准确的步数信息
        if hasattr(self, 'request_counter'):
            self.request_counter += 1
        else:
            self.request_counter = 1

        info(f"[DEBUG-EVAL] Request #{self.request_counter}, caller={caller_info}", verbosity.medium)

        # AAA文档建议：监控每步的端点评估次数
        step_id = getattr(self, '_current_step', 0)
        info(f"[EVAL] step={step_id} starting endpoint evaluation", verbosity.medium)
        # Ensure CP2K endpoints are started (lazy initialization)
        try:
            self._ensure_endpoints_started()
        except Exception as e:
            self._abort_due_to_cp2k_failure("Failed to start CP2K endpoints", e)

        # Check if endpoints are available
        if not self.endpoint1 or not self.endpoint2:
            self._abort_due_to_cp2k_failure("No active CP2K endpoints after startup")

        # Check endpoint health
        if not (self.endpoint1.health_check() and self.endpoint2.health_check()):
            self._abort_due_to_cp2k_failure("One or more CP2K endpoints failed health check")

        # Try direct socket communication - endpoints should be handshaked already
        info(" @FFMixTwoSockets: Starting parallel endpoint communication (handshake completed)", verbosity.medium)

        # Direct parallel communication - PARALLEL
        with ThreadPoolExecutor(max_workers=2) as executor:
            info(" @FFMixTwoSockets: Starting parallel endpoint communication", verbosity.medium)

            # Submit both communication tasks in parallel
            future1 = executor.submit(self._communicate_with_endpoint, self.endpoint1, request)
            future2 = executor.submit(self._communicate_with_endpoint, self.endpoint2, request)

            # Wait for both to complete
            result1 = future1.result()
            result2 = future2.result()

        # Check if we got real data
        if result1.get("is_real_data", False) and result2.get("is_real_data", False):
            info(" @FFMixTwoSockets: ✓ SUCCESS: Got real data from both CP2K endpoints (PARALLEL)", verbosity.low)
            # Update cache with successful real data results
            self._update_cache(request, (result1, result2))
            return result1, result2
        else:
            self._abort_due_to_cp2k_failure("Socket communication failed for both endpoints. No CP2K data received.")

    def _wait_for_socket_ready(self, endpoint, timeout=60):
        """Wait for CP2K endpoint to complete SCF and start socket server.

        Args:
            endpoint: CP2KEndpoint instance
            timeout: Maximum wait time in seconds

        Returns:
            bool: True if socket server is ready
        """
        import socket
        import time
        import select

        if not endpoint:
            return False

        start_time = time.time()
        last_file_check = 0.0
        file_check_interval = 0.5  # Check files every 500ms instead of every loop

        # Monitor CP2K output for completion with non-blocking approach
        while (time.time() - start_time) < timeout:
            current_time = time.time()

            # Check if process is still running
            if not endpoint.health_check():
                warning(f" @FFMixTwoSockets: Endpoint {endpoint.charge} process died", verbosity.medium)
                return False

            # Check file output only at intervals, not every loop iteration
            if (current_time - last_file_check) >= file_check_interval:
                last_file_check = current_time

                if endpoint.output_file and os.path.exists(endpoint.output_file):
                    try:
                        with open(endpoint.output_file, 'r') as f:
                            content = f.read()
                            # Look for signs that CP2K entered driver mode
                            if "Waiting for client" in content or "DRIVER|" in content or "Driver mode" in content:
                                info(f" @FFMixTwoSockets: Endpoint {endpoint.charge} entered driver mode", verbosity.medium)
                                break
                            elif "ABORT" in content or "ERROR" in content:
                                warning(f" @FFMixTwoSockets: Endpoint {endpoint.charge} encountered error", verbosity.medium)
                                return False
                    except:
                        pass

            # Non-blocking short sleep
            time.sleep(0.05)  # 50ms poll interval

        # Accept connection from CP2K client
        try:
            info(f" @FFMixTwoSockets: CP2K endpoint {endpoint.charge} ready, accepting client connection", verbosity.medium)
            # Call accept_client_connection to accept the CP2K client connection
            if endpoint.accept_client_connection(timeout=timeout):
                info(f" @FFMixTwoSockets: Successfully accepted connection from CP2K endpoint {endpoint.charge}", verbosity.medium)
                return True
            else:
                warning(f" @FFMixTwoSockets: Failed to accept connection from CP2K endpoint {endpoint.charge}", verbosity.medium)
                return False
        except Exception as e:
            warning(f" @FFMixTwoSockets: Error accepting connection from CP2K endpoint {endpoint.charge}: {e}", verbosity.medium)
            return False


    def _communicate_with_endpoint(self, endpoint, request):
        """Communicate with CP2K endpoint using the new socket communication classes.

        Args:
            endpoint: CP2KEndpoint instance with socket communication components
            request: Force request with positions

        Returns:
            dict: Result with energy, forces, virial, fermi_level
        """
        try:
            # Prepare position and cell data - keep positions as 1D array for socket communication
            positions = np.array(request["pos"], dtype=np.float64)  # Keep as 1D array

            # Prepare cell matrix
            cell_h = request["cell"][0]
            if isinstance(cell_h, np.ndarray) and cell_h.shape == (3, 3):
                h = cell_h.astype(np.float64)
            else:
                h = np.array(cell_h, dtype=np.float64).reshape((3, 3))

            # Use the new socket communicator to handle all communication
            result = endpoint.socket_communicator.send_positions_and_get_forces(
                positions=positions,
                cell_h=h,
                charge=endpoint.charge
            )

            # Handle the case where the communicator failed and returned no data
            if result is None:
                last_err = getattr(endpoint.socket_communicator, "last_error", None)
                msg = "CP2K socket communicator returned no data"
                if last_err:
                    msg += f" (last_error={last_err})"
                raise RuntimeError(msg)

            info(f" @FFMixTwoSockets: Endpoint {endpoint.charge} - Energy: {result['energy']:.6f} Hartree, Fermi: {result.get('fermi_level', 'N/A')} eV", verbosity.medium)

            # Add endpoint-specific metadata
            result.update({
                "charge": endpoint.charge,
                "is_real_data": True,
                "data_source": "cp2k_socket_refactored"
            })

            return result

        except Exception as e:
            warning(f" @FFMixTwoSockets: Socket communication failed for endpoint {endpoint.charge}: {e}", verbosity.medium)
            # Do not use dummy fallback - let the error propagate
            raise RuntimeError(f"Socket communication failed for endpoint {endpoint.charge}: {e}")

    def _get_z_average_region_A(self):
        """Get Z-average region [z_min, z_max] in Angstrom for workfunction mode.

        This is derived from the Dynamics.electrons_config["z_average_region"]
        (stored in internal length units, Bohr) using the unit conversion
        helpers in ipi.utils.units.
        """

        # Cache to avoid repeated lookups and conversions
        if hasattr(self, "_z_average_region_A") and self._z_average_region_A is not None:
            return self._z_average_region_A

        dynamics = getattr(self, "_dynamics_ref", None)
        if dynamics is None:
            dynamics = self._find_dynamics_object()
        if dynamics is None:
            raise RuntimeError("FFMixTwoSockets: Could not locate Dynamics object for z_average_region.")

        electrons_config = getattr(dynamics, "electrons_config", None)
        if not isinstance(electrons_config, dict):
            raise RuntimeError("FFMixTwoSockets: Dynamics.electrons_config is not available for workfunction mode.")

        if "z_average_region" not in electrons_config:
            raise RuntimeError(
                "FFMixTwoSockets: z_average_region must be specified in electrons configuration "
                "when using workfunction mode."
            )

        z_internal = np.array(electrons_config["z_average_region"], dtype=float).flatten()
        if z_internal.size != 2:
            raise ValueError(
                "FFMixTwoSockets: z_average_region in electrons_config must have length 2 [z_min, z_max]."
            )

        # Convert from internal length (Bohr) to Angstrom
        z_min_A = unit_to_user("length", "angstrom", z_internal[0])
        z_max_A = unit_to_user("length", "angstrom", z_internal[1])

        if z_max_A <= z_min_A:
            raise ValueError(
                f"FFMixTwoSockets: Invalid z_average_region after conversion to Angstrom: z_min={z_min_A}, z_max={z_max_A}"
            )

        self._z_average_region_A = (float(z_min_A), float(z_max_A))
        return self._z_average_region_A

    def _compute_endpoint_workfunction_eV(self, endpoint, fermi_level_eV):
        """Compute workfunction for a single CP2K endpoint in eV.

        The workfunction is defined as the planar-averaged electrostatic
        potential in the user-specified Z region minus the endpoint Fermi
        level (both in eV):

            workfunction = V_avg_region_eV - fermi_level_eV
        """

        from ipi.utils.messages import info, warning, verbosity

        if endpoint is None:
            raise RuntimeError("FFMixTwoSockets: Endpoint is None while computing workfunction.")

        # Determine cube file directory
        if endpoint.input_file:
            base_dir = os.path.dirname(endpoint.input_file) or os.getcwd()
        else:
            base_dir = os.getcwd()

        project = getattr(endpoint, "project_name", None)
        if not project:
            warning(
                f" @FFMixTwoSockets: Endpoint {endpoint.charge} has no project_name; "
                "cannot locate V_HARTREE_CUBE file for workfunction.",
                verbosity.medium,
            )
            raise RuntimeError("Missing project_name for CP2K endpoint; cannot compute workfunction.")

        cube_filename = f"{project}-v_hartree-1.cube"
        cube_path = os.path.join(base_dir, cube_filename)

        # Read cube file and compute planar-averaged potential along Z
        cube_data, origin_bohr, dx, dy, dz, shape = read_cp2k_cube(cube_path)
        nx, ny, nz = shape

        z_profile_eV = planar_average_z_cp2k(cube_data, to_eV=True)
        z_vals_A = z_coords_cp2k(origin_bohr, dz, nz)

        z_min_A, z_max_A = self._get_z_average_region_A()
        mask = (z_vals_A >= z_min_A) & (z_vals_A <= z_max_A)
        if not np.any(mask):
            raise ValueError(
                f"FFMixTwoSockets: No grid points found in z_average_region [{z_min_A}, {z_max_A}] "
                f"for cube file {cube_path} (nz={nz})."
            )

        V_avg_region_eV = float(z_profile_eV[mask].mean())
        workfunction_eV = V_avg_region_eV - float(fermi_level_eV)

        info(
            f" @FFMixTwoSockets: Endpoint {endpoint.charge} workfunction: V_avg={V_avg_region_eV:.6f} eV, "
            f"E_F={fermi_level_eV:.6f} eV, phi={workfunction_eV:.6f} eV",
            verbosity.medium,
        )

        return V_avg_region_eV, workfunction_eV

    def _perform_lambda_mixing(self, result1, result2, request):
        """Perform λ-mixing of results from two endpoints.

        Args:
            result1: Result from endpoint 1 (charge q1)
            result2: Result from endpoint 2 (charge q2)
            request: The original request

        Returns:
            list: Mixed result in standard format [energy, forces, virial, extras]
        """
        lambda_val = request["lambda"]

        # Energy mixing: E_mix = (1-λ)·E1 + λ·E2
        mixed_energy = (1.0 - lambda_val) * result1["energy"] + lambda_val * result2["energy"]
        mixed_energy -= self.offset

        # Force mixing: F_mix = (1-λ)·F1 + λ·F2
        mixed_forces = (1.0 - lambda_val) * result1["forces"] + lambda_val * result2["forces"]

        # Virial mixing: Vir_mix = (1-λ)·Vir1 + λ·Vir2
        mixed_virial = (1.0 - lambda_val) * result1["virial"] + lambda_val * result2["virial"]

        # Calculate chemical potential using linear mixing of endpoint Fermi levels
        mu1 = result1.get("fermi_level", 0.0)
        mu2 = result2.get("fermi_level", 0.0)
        self.current_mu = (1.0 - lambda_val) * mu1 + lambda_val * mu2

        # Calculate mixed Fermi level for i-PI electronic thermostat
        # Extract Fermi levels from results, ensuring they exist
        fermi1 = result1.get("fermi_level")  # in eV
        fermi2 = result2.get("fermi_level")  # in eV

        if fermi1 is None or fermi2 is None:
            raise RuntimeError(f"Missing Fermi level data: endpoint1={fermi1}, endpoint2={fermi2}")

        fermi1 = float(fermi1)
        fermi2 = float(fermi2)
        mixed_fermi_level_eV = (1.0 - lambda_val) * fermi1 + lambda_val * fermi2

        # Optionally compute workfunctions if workfunction mode is active. In workfunction
        # mode the availability of the CP2K V_HARTREE_CUBE files is essential, so any
        # failure to read or process them should terminate the simulation rather than
        # being silently downgraded to a warning.
        mixed_workfunction_eV = None
        endpoint1_workfunction_eV = None
        endpoint2_workfunction_eV = None

        dynamics = getattr(self, "_dynamics_ref", None)
        if dynamics is None:
            dynamics = self._find_dynamics_object()

        electronic_state = getattr(dynamics, "electronic_state", None) if dynamics is not None else None
        mode = getattr(electronic_state, "mode", "fermi") if electronic_state is not None else "fermi"

        if mode == "workfunction":
            # Compute endpoint workfunctions from V_HARTREE_CUBE and mix linearly.
            _, endpoint1_workfunction_eV = self._compute_endpoint_workfunction_eV(self.endpoint1, fermi1)
            _, endpoint2_workfunction_eV = self._compute_endpoint_workfunction_eV(self.endpoint2, fermi2)
            mixed_workfunction_eV = (
                (1.0 - lambda_val) * endpoint1_workfunction_eV
                + lambda_val * endpoint2_workfunction_eV
            )

        # Convert mixed Fermi level from eV to atomic units (Hartree) for i-PI internal use
        mixed_fermi_level_au = mixed_fermi_level_eV / Constants.EV_PER_HARTREE  # eV to Hartree conversion

        # Prepare extras dictionary
        extras = {
            "raw": "",
            "fermi_level": mixed_fermi_level_au,  # In atomic units (Hartree) for i-PI electronic thermostat
            "fermi_level_eV": mixed_fermi_level_eV,  # Also keep eV version for debugging/output
            "lambda": lambda_val,
            "current_q": request["current_q"],
            "chemical_potential": self.current_mu,
            "endpoint_charges": [self.q1, self.q2],
            "mixing_mode": self.mixing_mode,
            "evaluation_count": request["evaluation_count"],
            "endpoint1_energy": result1["energy"],
            "endpoint2_energy": result2["energy"],
            "endpoint1_fermi": fermi1,
            "endpoint2_fermi": fermi2,
            "endpoint1_converged": result1.get("converged", True),
            "endpoint2_converged": result2.get("converged", True),
            "endpoint1_is_real": result1.get("is_real_data", False),
            "endpoint2_is_real": result2.get("is_real_data", False),
            "endpoint1_data_source": result1.get("data_source", "unknown"),
            "endpoint2_data_source": result2.get("data_source", "unknown"),
            "using_real_cp2k_data": result1.get("is_real_data", False) and result2.get("is_real_data", False),
        }

        # Attach workfunction information if available
        if mixed_workfunction_eV is not None:
            extras["workfunction_eV"] = mixed_workfunction_eV
            extras["workfunction"] = mixed_workfunction_eV / Constants.EV_PER_HARTREE
            extras["endpoint1_workfunction_eV"] = endpoint1_workfunction_eV
            extras["endpoint2_workfunction_eV"] = endpoint2_workfunction_eV

        return [mixed_energy, mixed_forces, mixed_virial, extras]


    def get_chemical_potential(self):
        """Get the current chemical potential.

        Returns:
            float: Chemical potential in atomic units
        """
        return self.current_mu

    def get_mixing_info(self):
        """Get current mixing state information.

        Returns:
            dict: Dictionary with mixing parameters and state
        """
        return {
            "q1": self.q1,
            "q2": self.q2,
            "current_q": self.current_q,
            "lambda": self.lambda_val,
            "mixing_mode": self.mixing_mode,
            "chemical_potential": self.current_mu,
            "switch_threshold": self.switch_threshold,
            "switch_count": self.switch_count,
            "evaluation_count": self.evaluation_count,
            "auto_switch": self.auto_switch
        }
