"""Deals with the socket communication between the i-PI and drivers.

Deals with creating the socket, transmitting and receiving data, accepting and
removing different driver routines and the parallelization of the force
calculation.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import os
import socket
import select
import time
import threading

import numpy as np
import json

from ipi.utils.messages import verbosity, warning, info
from ipi.utils.softexit import softexit
from ipi.utils.units import Constants

from concurrent.futures import ThreadPoolExecutor


__all__ = ["InterfaceSocket"]


HDRLEN = 12
UPDATEFREQ = 10
TIMEOUT = 0.1
SERVERTIMEOUT = 5.0 * TIMEOUT
NTIMEOUT = 20
SELECTTIMEOUT = 60


def Message(mystr):
    """Returns a header of standard length HDRLEN."""

    # convert to bytestream since we'll be sending this over a socket
    return str.ljust(str.upper(mystr), HDRLEN).encode()


MESSAGE = {
    msg: Message(msg)
    for msg in [
        "exit",
        "status",
        "ready",
        "havedata",
        "init",
        "needinit",
        "posdata",
        "getforce",
        "forceready",
        "chgdata",
    ]
}


class Disconnected(Exception):
    """Disconnected: Raised if client has been disconnected."""

    pass


class InvalidSize(Exception):
    """Disconnected: Raised if client returns forces with inconsistent number of atoms."""

    pass


class InvalidStatus(Exception):
    """InvalidStatus: Raised if client has the wrong status.

    Shouldn't have to be used if the structure of the program is correct.
    """

    pass


class Status(object):
    """Simple class used to keep track of the status of the client.

    Uses bitwise or to give combinations of different status options.
    i.e. Status.Up | Status.Ready would be understood to mean that the client
    was connected and ready to receive the position and cell data.

    Attributes:
       Disconnected: Flag for if the client has disconnected.
       Up: Flag for if the client is running.
       Ready: Flag for if the client has ready to receive position and cell data.
       NeedsInit: Flag for if the client is ready to receive forcefield
          parameters.
       HasData: Flag for if the client is ready to send force data.
       Busy: Flag for if the client is busy.
       Timeout: Flag for if the connection has timed out.
    """

    Disconnected = 0
    Up = 1
    Ready = 2
    NeedsInit = 4
    HasData = 8
    Busy = 16
    Timeout = 32


class DriverSocket(socket.socket):
    """Deals with communication between the client and driver code.

    Deals with sending and receiving the data between the client and the driver
    code. This class holds common functions which are used in the driver code,
    but can also be used to directly implement a python client.
    Basically it's just a wrapper around socket to simplify some of the
    specific needs of i-PI communication pattern.

    Attributes:
       _buf: A string buffer to hold the reply from the other connection.
    """

    def __init__(self, sock):
        """Initialises DriverSocket.

        Args:
           socket: A socket through which the communication should be done.
        """

        super(DriverSocket, self).__init__(
            sock.family, sock.type, sock.proto, fileno=socket.dup(sock.fileno())
        )
        self.settimeout(sock.gettimeout())

        self._buf = np.zeros(0, np.byte)
        if socket:
            self.peername = self.getpeername()
        else:
            self.peername = "no_socket"

    def send_msg(self, msg):
        """Send the next message through the socket.

        Args:
           msg: The message to send through the socket.
        """
        return self.sendall(MESSAGE[msg])

    def recv_msg(self, length=HDRLEN):
        """Get the next message send through the socket.

        Args:
           l: Length of the accepted message. Defaults to HDRLEN.
        """
        return self.recv(length)

    def recvall(self, dest):
        """Gets the potential energy, force and virial from the driver.

        Args:
           dest: Object to be read into.

        Raises:
           Disconnected: Raised if client is disconnected.

        Returns:
           The data read from the socket to be read into dest.
        """

        blen = dest.itemsize * dest.size
        if blen > len(self._buf):
            self._buf = np.zeros(blen, np.byte)
        bpos = 0
        ntimeout = 0

        while bpos < blen:
            timeout = False

            try:
                bpart = 0
                bpart = self.recv_into(self._buf[bpos:], blen - bpos)
            except socket.timeout:
                # warning(" @SOCKET:   Timeout in recvall, trying again!", verbosity.low)
                timeout = True
                ntimeout += 1
                if ntimeout > NTIMEOUT:
                    warning(
                        " @SOCKET:  Couldn't receive within %5d attempts. Time to give up!"
                        % (NTIMEOUT),
                        verbosity.low,
                    )
                    raise Disconnected()
                pass

            if not timeout and bpart == 0:
                raise Disconnected()
            bpos += bpart

        if dest.ndim > 0:
            dest[:] = np.frombuffer(self._buf[0:blen], dest.dtype).reshape(dest.shape)
            return dest  # tmp.copy() #np.frombuffer(self._buf[0:blen], dest.dtype).reshape(dest.shape).copy()
        else:
            return np.frombuffer(self._buf[0:blen], dest.dtype)[0]


class Driver(DriverSocket):
    """Deals with communication between the client and driver code.

    Deals with sending and receiving the data from the driver code. Keeps track
    of the status of the driver. Initialises the driver forcefield, sends the
    position and cell data, and receives the force data.

    Attributes:
       waitstatus: Boolean giving whether the driver is waiting to get a status answer.
       status: Keeps track of the status of the driver.
       lastreq: The ID of the last request processed by the client.
       locked: Flag to mark if the client has been working consistently on one image.
    """

    def __init__(self, sock):
        """Initialises Driver.

        Args:
           socket: A socket through which the communication should be done.
        """

        super(Driver, self).__init__(sock)
        self.waitstatus = False
        self.status = Status.Up
        self.lastreq = None
        self.locked = False
        self.exit_on_disconnect = False

    def shutdown(self, how=socket.SHUT_RDWR):
        """Tries to send an exit message to clients to let them exit gracefully."""

        if self.exit_on_disconnect:
            trd = threading.Thread(
                target=softexit.trigger, kwargs={"message": "Client shutdown."}
            )
            trd.daemon = True
            trd.start()

        self.send_msg("exit")
        self.status = Status.Disconnected

        super(DriverSocket, self).shutdown(how)

    def _getstatus_select(self):
        """Gets driver status. Uses socket.select to make sure one can read/write on the socket.

        Returns:
           An integer labelling the status via bitwise or of the relevant members
           of Status.
        """

        if not self.waitstatus:
            try:
                # This can sometimes hang with no timeout.
                readable, writable, errored = select.select(
                    [], [self], [], SELECTTIMEOUT
                )
                if self in writable:
                    self.send_msg("status")
                    self.waitstatus = True
            except socket.error:
                return Status.Disconnected

        try:
            readable, writable, errored = select.select([self], [], [], SELECTTIMEOUT)
            if self in readable:
                reply = self.recv_msg(HDRLEN)
                self.waitstatus = False  # got some kind of reply
            else:
                # This is usually due to VERY slow clients.
                warning(
                    f" @SOCKET: Couldn't find readable socket in {SELECTTIMEOUT}s, will try again",
                    verbosity.low,
                )
                return Status.Busy
        except socket.timeout:
            warning(" @SOCKET:   Timeout in status recv!", verbosity.debug)
            return Status.Up | Status.Busy | Status.Timeout
        except:
            warning(
                " @SOCKET:   Other socket exception. Disconnecting client and trying to carry on.",
                verbosity.debug,
            )
            return Status.Disconnected

        if not len(reply) == HDRLEN:
            return Status.Disconnected
        elif reply == MESSAGE["ready"]:
            return Status.Up | Status.Ready
        elif reply == MESSAGE["needinit"]:
            return Status.Up | Status.NeedsInit
        elif reply == MESSAGE["havedata"]:
            return Status.Up | Status.HasData
        else:
            warning(" @SOCKET:    Unrecognized reply: " + str(reply), verbosity.low)
            return Status.Up

    def _getstatus_direct(self):
        """Gets driver status. Relies on blocking send/recv, which might lead to
        timeouts with slow networks.

        Returns:
           An integer labelling the status via bitwise or of the relevant members
           of Status.
        """

        if not self.waitstatus:
            try:
                self.send_msg("status")
                self.waitstatus = True
            except socket.error:
                return Status.Disconnected
        try:
            reply = self.recv_msg(HDRLEN)
            self.waitstatus = False  # got some kind of reply
        except socket.timeout:
            warning(" @SOCKET:   Timeout in status recv!", verbosity.debug)
            return Status.Up | Status.Busy | Status.Timeout
        except:
            warning(
                " @SOCKET:   Other socket exception. Disconnecting client and trying to carry on.",
                verbosity.debug,
            )
            return Status.Disconnected

        if not len(reply) == HDRLEN:
            return Status.Disconnected
        elif reply == MESSAGE["ready"]:
            return Status.Up | Status.Ready
        elif reply == MESSAGE["needinit"]:
            return Status.Up | Status.NeedsInit
        elif reply == MESSAGE["havedata"]:
            return Status.Up | Status.HasData
        else:
            warning(" @SOCKET:    Unrecognized reply: " + str(reply), verbosity.low)
            return Status.Up

    # depending on the system either _select or _direct can be slightly faster
    # if you're network limited it might be worth experimenting changing this
    _getstatus = _getstatus_select

    def get_status(self):
        """Sets (and returns) the client internal status. Wait for an answer if
        the client is busy."""
        status = self._getstatus()
        while status & Status.Busy:
            status = self._getstatus()
        self.status = status
        return status

    def initialize(self, rid, pars):
        """Sends the initialisation string to the driver.

        Args:
           rid: The index of the request, i.e. the replica that
              the force calculation is for.
           pars: The parameter string to be sent to the driver.

        Raises:
           InvalidStatus: Raised if the status is not NeedsInit.
        """

        if self.status & Status.NeedsInit:
            try:
                # combines all messages in one to reduce latency
                self.sendall(
                    MESSAGE["init"]
                    + np.int32(rid)
                    + np.int32(len(pars))
                    + pars.encode()
                )
            except:
                self.get_status()
                return
        else:
            raise InvalidStatus("Status in init was " + self.status)

    def sendpos(self, pos, h_ih):
        """Sends the position and cell data to the driver.

        Args:
           pos: An array containing the atom positions.
           cell: A cell object giving the system box.

        Raises:
           InvalidStatus: Raised if the status is not Ready.
        """
        global TIMEOUT  # we need to update TIMEOUT in case of sendall failure

        if self.status & Status.Ready:
            try:
                # reduces latency by combining all messages in one
                self.sendall(
                    MESSAGE["posdata"]  # header
                    + h_ih[0].tobytes()  # cell
                    + h_ih[1].tobytes()  # inverse cell
                    + np.int32(len(pos) // 3).tobytes()  # length of position array
                    + pos.tobytes()  # positions
                )
                self.status = Status.Up | Status.Busy
            except socket.timeout:
                warning(
                    f"Timeout in sendall after {TIMEOUT}s: resetting status and increasing timeout",
                    verbosity.quiet,
                )
                self.status = Status.Timeout
                TIMEOUT *= 2
                return
            except Exception as exc:
                warning(
                    f"Other exception during posdata receive: {exc}", verbosity.quiet
                )
                raise exc
        else:
            raise InvalidStatus("Status in sendpos was " + self.status)

    def sendchg(self, nelect, solvation_flag=1):
        """Sends updated electronic charge (NELECT) and solvation flag to the driver.

        Uses the CHGDATA header followed by:
            - float64: nelect
            - int32: solvation_flag

        This is a lightweight message that does not change the driver's status
        flags; it is intended to be sent while the client is Ready, before
        sending POSDATA for the next force evaluation.
        """

        if not (self.status & Status.Ready):
            raise InvalidStatus("Status in sendchg was " + str(self.status))

        try:
            payload_nelect = np.float64(nelect)
            payload_solv = np.int32(solvation_flag)
            self.sendall(
                MESSAGE["chgdata"] + payload_nelect.tobytes() + payload_solv.tobytes()
            )
        except Exception as exc:
            warning(
                f" @SOCKET:   Exception during CHGDATA send: {exc}",
                verbosity.quiet,
            )

    def getforce(self):
        """Gets the potential energy, force and virial from the driver.

        Raises:
           InvalidStatus: Raised if the status is not HasData.
           Disconnected: Raised if the driver has disconnected.

        Returns:
           A list of the form [potential, force, virial, extra].
        """

        if self.status & Status.HasData:
            self.send_msg("getforce")
            reply = ""
            while True:
                try:
                    reply = self.recv_msg()
                except socket.timeout:
                    warning(
                        " @SOCKET:   Timeout in getforce, trying again!", verbosity.low
                    )
                    continue
                except:
                    warning(
                        " @SOCKET:   Error while receiving message: %s" % (reply),
                        verbosity.low,
                    )
                    raise Disconnected()
                if reply == MESSAGE["forceready"]:
                    break
                else:
                    warning(
                        " @SOCKET:   Unexpected getforce reply: %s" % (reply),
                        verbosity.low,
                    )
                if reply == "":
                    raise Disconnected()
        else:
            raise InvalidStatus("Status in getforce was " + str(self.status))

        mu = np.float64()
        mu = self.recvall(mu)

        mlen = np.int32()
        mlen = self.recvall(mlen)

        mf = np.zeros(3 * mlen, np.float64)
        mf = self.recvall(mf)

        mvir = np.zeros((3, 3), np.float64)
        mvir = self.recvall(mvir)

        # Machinery to return a string as an "extra" field.
        # Comment if you are using a ancient patched driver that does not return anything!
        # Actually, you should really update your driver, you're like a decade behind.
        mlen = np.int32()
        mlen = self.recvall(mlen)
        if mlen > 0:
            mxtra = np.zeros(mlen, np.dtype("S1"))
            mxtra = self.recvall(mxtra)
            mxtra = bytearray(mxtra).decode("utf-8")
        else:
            mxtra = ""
        mxtradict = {}
        if mxtra:
            try:
                mxtradict = json.loads(mxtra)
                # info(
                #     "@driver.getforce: Extra string JSON has been loaded.",
                #     verbosity.debug,
                # )
            except:
                # if we can't parse it as a dict, issue a warning and carry on
                # info(
                #     "@driver.getforce: Extra string could not be loaded as a dictionary. Extra="
                #     + mxtra,
                #     verbosity.debug,
                # )
                mxtradict = {}
                pass
            if "raw" in mxtradict:
                raise ValueError(
                    "'raw' cannot be used as a field in a JSON-formatted extra string"
                )

            mxtradict["raw"] = mxtra
        return [mu, mf, mvir, mxtradict]

    def dispatch(self, r):
        """Dispatches a request r and looks after it setting results
        once it has been evaluated. This is meant to be launched as a
        separate thread, and takes care of all the communication related to
        the request.
        """

        if not self.status & Status.Up:
            warning(
                " @SOCKET:   Inconsistent client state in dispatch thread! (I)",
                verbosity.low,
            )
            return

        r["t_dispatched"] = time.time()

        self.get_status()
        if self.status & Status.NeedsInit:
            self.initialize(r["id"], r["pars"])
            self.status = self.get_status()

        if not (self.status & Status.Ready):
            warning(
                " @SOCKET:   Inconsistent client state in dispatch thread! (II)",
                verbosity.low,
            )
            return

        # If the request carries an updated electronic charge (NELECT), send it
        # to the driver before sending positions, using the lightweight CHGDATA
        # message.
        #
        # The CHGDATA payload is:
        #   - float64: nelect
        #   - int32: solvation_flag
        #
        # solvation_flag defaults to 1 (enable every step) unless overridden.
        nelect = r.get("nelect", None)
        solvation_flag = r.get("solvation_flag", 1)
        if nelect is not None:
            try:
                self.sendchg(nelect, int(solvation_flag))
            except Exception as exc:
                warning(
                    f" @SOCKET:   Failed to send CHGDATA for request {r['id']}: {exc}",
                    verbosity.low,
                )

        r["start"] = time.time()
        self.sendpos(r["pos"][r["active"]], r["cell"])

        self.get_status()
        if not (self.status & Status.HasData):
            warning(
                " @SOCKET:   Inconsistent client state in dispatch thread! (III)",
                verbosity.low,
            )
            return

        try:
            r["result"] = self.getforce()
        except Disconnected:
            self.status = Status.Disconnected
            return

        r["result"][0] -= r["offset"]

        if len(r["result"][1]) != len(r["pos"][r["active"]]):
            raise InvalidSize

        # If only a piece of the system is active, resize forces and reassign
        if len(r["active"]) != len(r["pos"]):
            rftemp = r["result"][1]
            r["result"][1] = np.zeros(len(r["pos"]), dtype=np.float64)
            r["result"][1][r["active"]] = rftemp
        r["t_finished"] = time.time()
        self.lastreq = r["id"]  #

        # updates the status of the client before leaving
        self.get_status()

        # marks the request as done as the very last thing
        r["status"] = "Done"


class InterfaceSocket(object):
    """Host server class.

    Deals with distribution of all the jobs between the different client servers
    and both initially and as clients either finish or are disconnected.
    Deals with cleaning up after all calculations are done. Also deals with the
    threading mechanism, and cleaning up if the interface is killed.

    Attributes:
       address: A string giving the name of the host network.
       port: An integer giving the port the socket will be using.
       slots: An integer giving the maximum allowed backlog of queued clients.
       mode: A string giving the type of socket used.
       latency: A float giving the number of seconds the interface will wait
          before updating the client list.
       timeout: A float giving a timeout limit for considering a calculation dead
          and dropping the connection.
       server: The socket used for data transmission.
       clients: A list of the driver clients connected to the server.
       requests: A list of all the jobs required in the current PIMD step.
       jobs: A list of all the jobs currently running.
       _poll_thread: The thread the poll loop is running on.
       _prev_kill: Holds the signals to be sent to clean up the main thread
          when a kill signal is sent.
       _poll_true: A boolean giving whether the thread is alive.
       _poll_iter: An integer used to decide whether or not to check for
          client connections. It is used as a counter, once it becomes higher
          than the pre-defined number of steps between checks the socket will
          update the list of clients and then be reset to zero.
    """

    def __init__(
        self,
        address="localhost",
        port=31415,
        slots=4,
        mode="unix",
        timeout=1.0,
        match_mode="auto",
        exit_on_disconnect=False,
        max_workers=128,
        sockets_prefix="/tmp/ipi_",
    ):
        """Initialises interface.

        Args:
           address: An optional string giving the name of the host server.
              Defaults to 'localhost'.
           port: An optional integer giving the port number. Defaults to 31415.
           slots: An optional integer giving the maximum allowed backlog of
              queueing clients. Defaults to 4.
           mode: An optional string giving the type of socket. Defaults to 'unix'.
           timeout: Length of time waiting for data from a client before we assume
              the connection is dead and disconnect the client.
            max_workers: Maximum number of threads launched concurrently

        Raises:
           NameError: Raised if mode is not 'unix' or 'inet'.
        """

        self.address = address
        self.port = port
        self.slots = slots
        self.mode = mode
        self.timeout = timeout
        self.sockets_prefix = sockets_prefix
        self.poll_iter = UPDATEFREQ  # triggers pool_update at first poll
        self.prlist = []  # list of pending requests
        self.match_mode = match_mode  # heuristics to match jobs and active clients
        self.requests = None  # these will be linked to the request list of the FFSocket object using the interface
        self.exit_on_disconnect = exit_on_disconnect
        self.max_workers = max_workers
        self.offset = 0.0  # a constant energy offset added to the results returned by the driver (hacky but simple)

    def open(self):
        """Creates a new socket.

        Used so that we can create a interface object without having to also
        create the associated socket object.
        """

        if self.mode == "unix":
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                self.server.bind(self.sockets_prefix + self.address)
                info(
                    " @interfacesocket.open: Created unix socket with address "
                    + self.address,
                    verbosity.medium,
                )
            except socket.error:
                raise RuntimeError(
                    "Error opening unix socket. Check if a file "
                    + (self.sockets_prefix + self.address)
                    + " exists, and remove it if unused."
                )

        elif self.mode == "inet":
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            try:
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # TCP_NODELAY is set because Nagle's algorithm slows down a lot
                # the communication pattern of i-PI
                self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError as e:
                warning(f"Error setting socket options {e}")

            self.server.bind((self.address, self.port))
            info(
                " @interfacesocket.open: Created inet socket with address "
                + self.address
                + " and port number "
                + str(self.port),
                verbosity.medium,
            )
        else:
            raise NameError(
                "InterfaceSocket mode "
                + self.mode
                + " is not implemented (should be unix/inet)"
            )

        self.server.listen(self.slots)
        self.server.settimeout(SERVERTIMEOUT)

        # these are the two main objects the socket interface should worry about and manage
        self.clients = []  # list of active clients (working or ready to compute)
        self.jobs = []  # list of jobs
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def close(self):
        """Closes down the socket."""

        info(
            " @interfacesocket.close: Shutting down the driver interface.",
            verbosity.low,
        )

        for c in self.clients:
            try:
                c.shutdown(socket.SHUT_RDWR)
                c.close()
            except:
                pass

        # flush it all down the drain
        self.clients = []
        self.jobs = []

        try:
            self.server.shutdown(socket.SHUT_RDWR)
            self.server.close()
        except:
            info(
                " @interfacesocket.close: Problem shutting down the server socket. Will just continue and hope for the best.",
                verbosity.low,
            )
        if self.mode == "unix":
            os.unlink(self.sockets_prefix + self.address)

    def poll(self):
        """Called in the main thread loop.

        Runs until either the program finishes or a kill call is sent. Updates
        the pool of clients every UPDATEFREQ loops and loops every latency seconds.
        The actual loop is in the associated forcefield class.
        """

        # makes sure to remove the last dead client as soon as possible -- and to get clients if we are dry
        if (
            self.poll_iter >= UPDATEFREQ
            or len(self.clients) == 0
            or (len(self.clients) > 0 and not (self.clients[0].status & Status.Up))
        ):
            self.poll_iter = 0
            self.pool_update()

        self.poll_iter += 1
        self.pool_distribute()

    def pool_update(self):
        """Deals with keeping the pool of client drivers up-to-date during a
        force calculation step.

        Deals with maintaining the client list. Clients that have
        disconnected are removed and their jobs removed from the list of
        running jobs and new clients are connected to the server.
        """

        # check for disconnected clients
        for c in self.clients[:]:
            if not (c.status & Status.Up):
                try:
                    warning(
                        " @SOCKET:   Client "
                        + str(c.peername)
                        + " died or got unresponsive(C). Removing from the list.",
                        verbosity.low,
                    )
                    c.shutdown(socket.SHUT_RDWR)
                    c.close()
                except socket.error:
                    pass
                c.status = Status.Disconnected
                self.clients.remove(c)
                # requeue jobs that have been left hanging
                for [k, j, tc] in self.jobs[:]:
                    tc.result()
                    if j is c:
                        self.jobs = [
                            w for w in self.jobs if not (w[0] is k and w[1] is j)
                        ]  # removes pair in a robust way

                        k["status"] = "Queued"
                        k["start"] = -1

        if len(self.clients) == 0:
            searchtimeout = SERVERTIMEOUT
        else:
            searchtimeout = 0.0

        keepsearch = True
        while keepsearch:
            readable, writable, errored = select.select(
                [self.server], [], [], searchtimeout
            )
            if self.server in readable:
                client, address = self.server.accept()
                client.settimeout(TIMEOUT)
                driver = Driver(client)
                info(
                    " @interfacesocket.pool_update:   Client asked for connection from "
                    + str(address)
                    + ". Now hand-shaking.",
                    verbosity.low,
                )
                driver.get_status()
                if driver.status | Status.Up:
                    driver.exit_on_disconnect = self.exit_on_disconnect
                    self.clients.append(driver)
                    info(
                        " @interfacesocket.pool_update:   Handshaking was successful. Added to the client list.",
                        verbosity.low,
                    )
                    self.poll_iter = UPDATEFREQ  # if a new client was found, will try again harder next time
                    searchtimeout = SERVERTIMEOUT
                else:
                    warning(
                        " @SOCKET:   Handshaking failed. Dropping connection.",
                        verbosity.low,
                    )
                    client.shutdown(socket.SHUT_RDWR)
                    client.close()
            else:
                keepsearch = False

    def pool_distribute(self):
        """Deals with keeping the list of jobs up-to-date during a force
        calculation step.

        Deals with maintaining the jobs list. Gets data from drivers that have
        finished their calculation and removes that job from the list of running
        jobs, adds jobs to free clients and initialises the forcefields of new
        clients.
        """

        ttotal = tdispatch = tcheck = 0
        ttotal -= time.time()

        # get clients that are still free
        freec = self.clients[:]
        for [r2, c, ct] in self.jobs:
            freec.remove(c)

        # fills up list of pending requests if empty, or if clients are abundant
        if len(self.prlist) == 0 or len(freec) > len(self.prlist):
            self.prlist = [r for r in self.requests if r["status"] == "Queued"]

        if self.match_mode == "auto":
            match_seq = ["match", "none", "free", "any"]
        elif self.match_mode == "any":
            match_seq = ["any"]
        elif self.match_mode == "lock":
            match_seq = ["match", "none"]

        # first: dispatches jobs to free clients (if any!)
        # tries first to match previous replica<>driver association, then to get new clients, and only finally send the a new replica to old drivers
        ndispatch = 0
        tdispatch -= time.time()
        while len(freec) > 0 and len(self.prlist) > 0:
            for match_ids in match_seq:
                for fc in freec[:]:
                    if self.dispatch_free_client(fc, match_ids):
                        freec.remove(fc)
                        ndispatch += 1

                    if len(self.prlist) == 0:
                        break
            # if using lock mode, check that there is a at least one client-replica match in the lists freec and prlist.
            # If not, we break out of the while loop
            if self.match_mode == "lock":
                break

            if len(freec) > 0:
                self.prlist = [r for r in self.requests if r["status"] == "Queued"]
        tdispatch += time.time()

        # now check for client status
        if len(self.jobs) == 0:
            for c in self.clients:
                if (
                    c.status == Status.Disconnected
                ):  # client disconnected. force a pool_update
                    self.poll_iter = UPDATEFREQ
                    return

        # check for finished jobs
        nchecked = 0
        nfinished = 0
        tcheck -= time.time()
        for [r, c, ct] in self.jobs[:]:
            chk = self.check_job_finished(r, c, ct)
            if chk == 1:
                nfinished += 1
            elif chk == 0:
                self.poll_iter = UPDATEFREQ  # client disconnected. force a pool_update
            nchecked += 1
        tcheck += time.time()

        ttotal += time.time()
        # info("POLL TOTAL: %10.4f  Dispatch(N,t):  %4i, %10.4f   Check(N,t):   %4i, %10.4f" % (ttotal, ndispatch, tdispatch, nchecked, tcheck), verbosity.debug)

        if nfinished > 0:
            # don't wait, just try again to distribute
            self.pool_distribute()

    def dispatch_free_client(self, fc, match_ids="any", send_threads=[]):
        """
        Tries to find a request to match a free client.
        """

        # first, makes sure that the client is REALLY free
        if not (fc.status & Status.Up):
            return False
        if fc.status & Status.HasData:
            return False
        if not (fc.status & (Status.Ready | Status.NeedsInit | Status.Busy)):
            warning(
                " @SOCKET: Client "
                + str(fc.peername)
                + " is in an unexpected status "
                + str(fc.status)
                + " at (1). Will try to keep calm and carry on.",
                verbosity.low,
            )
            return False

        for r in self.prlist[:]:
            if match_ids == "match" and fc.lastreq is not r["id"]:
                continue
            elif match_ids == "none" and fc.lastreq is not None:
                continue
            elif (
                self.match_mode == "lock"
                and match_ids == "none"
                and (r["id"] in [c.lastreq for c in self.clients])
            ):
                # if using lock mode and the user connects more clients than there are replicas, do not allow this client to
                # be matched with a pending request.
                continue

            elif match_ids == "free" and fc.locked:
                continue

            # makes sure the request is marked as running and the client included in the jobs list
            fc.locked = fc.lastreq is r["id"]

            r["offset"] = (
                self.offset
            )  # transmits with the request an offset value for the energy (typically zero)

            r["status"] = "Running"
            self.prlist.remove(r)
            info(
                " @interfacesocket.dispatch_free_client: %s Assigning [%5s] request id %4s to client with last-id %4s (% 3d/% 3d : %s)"
                % (
                    time.strftime("%y/%m/%d-%H:%M:%S"),
                    match_ids,
                    str(r["id"]),
                    str(fc.lastreq),
                    self.clients.index(fc),
                    len(self.clients),
                    str(fc.peername),
                ),
                verbosity.high,
            )
            # fc_thread = threading.Thread(
            #    target=fc.dispatch, name="DISPATCH", kwargs={"r": r}
            # )
            fc_thread = self.executor.submit(fc.dispatch, r=r)
            self.jobs.append([r, fc, fc_thread])
            # fc_thread.daemon = True
            # fc_thread.start()
            return True

        return False

    def check_job_finished(self, r, c, ct):
        """
        Checks if a job has been completed, and retrieves the results
        """

        if r["status"] == "Done":
            ct.result()
            self.jobs = [
                w for w in self.jobs if not (w[0] is r and w[1] is c)
            ]  # removes pair in a robust way
            return 1

        if (
            self.timeout > 0
            and r["start"] > 0
            and time.time() - r["start"] > self.timeout
        ):
            warning(
                " @SOCKET:  Timeout! request has been running for "
                + str(time.time() - r["start"])
                + " sec.",
                verbosity.low,
            )
            warning(
                " @SOCKET:   Client "
                + str(c.peername)
                + " died or got unresponsive(A). Disconnecting.",
                verbosity.low,
            )
            try:
                c.shutdown(socket.SHUT_RDWR)
            except socket.error:
                pass
            c.close()
            c.status = Status.Disconnected
            return 0  # client will be cleared and request resuscitated in poll_update

        return -1


class CP2KSocketServer:
    """Dedicated CP2K socket server for constant potential simulations.

    This class manages a single TCP server socket for communication with
    one CP2K process in constant potential molecular dynamics simulations.
    It implements the i-PI socket protocol with CP2K-specific extensions
    for Fermi level extraction and charge state management.
    """

    def __init__(self, host="localhost", port=12345, timeout=30.0):
        """Initialize CP2K socket server.

        Args:
            host: Host address for socket binding
            port: Port number for socket binding
            timeout: Timeout in seconds for operations
        """
        self.host = host
        self.port = port
        self.timeout = timeout

        # Server socket for accepting connections
        self.server_socket = None
        self.client_socket = None
        self.is_connected = False
        self.is_listening = False

        info(f" @CP2KSocketServer: Initialized server for {host}:{port}", verbosity.medium)

    def create_server(self):
        """Create and bind TCP server socket.

        Returns:
            bool: True if server created successfully
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.is_listening = True

            info(f" @CP2KSocketServer: Created server on {self.host}:{self.port}", verbosity.medium)
            return True

        except Exception as e:
            warning(f" @CP2KSocketServer: Failed to create server: {e}", verbosity.medium)
            self.cleanup()
            return False

    def accept_connection(self, timeout=None):
        """Accept connection from CP2K client.

        Args:
            timeout: Timeout in seconds (None for default)

        Returns:
            bool: True if connection accepted successfully
        """
        if not self.server_socket or not self.is_listening:
            warning(" @CP2KSocketServer: No server socket available", verbosity.medium)
            return False

        if self.is_connected and self.client_socket:
            info(" @CP2KSocketServer: Already connected to client", verbosity.debug)
            return True

        try:
            timeout = timeout or self.timeout
            self.server_socket.settimeout(timeout)

            info(f" @CP2KSocketServer: Waiting for CP2K client connection on port {self.port}", verbosity.medium)
            self.client_socket, addr = self.server_socket.accept()

            # Disable timeout after connection established
            self.client_socket.settimeout(None)
            self.is_connected = True

            info(f" @CP2KSocketServer: Accepted connection from {addr}", verbosity.medium)
            return True

        except Exception as e:
            warning(f" @CP2KSocketServer: Failed to accept connection: {e}", verbosity.medium)
            return False

    def is_connection_healthy(self):
        """Check if the socket connection is still healthy.

        Returns:
            bool: True if connection is healthy
        """
        try:
            import select

            if not self.is_connected or not self.client_socket:
                return False

            # Use select to check if socket is readable (might indicate disconnection)
            ready, _, _ = select.select([self.client_socket], [], [], 0.001)
            if ready:
                try:
                    data = self.client_socket.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    if not data:
                        # Connection closed by peer
                        return False
                except socket.error as e:
                    if e.errno in (socket.EWOULDBLOCK, socket.EAGAIN):
                        # No data available, connection is healthy
                        return True
                    else:
                        # Connection error
                        return False

            return True

        except Exception:
            return False

    def close_connection(self):
        """Close client connection."""
        if self.client_socket:
            try:
                self.client_socket.close()
                info(" @CP2KSocketServer: Closed client connection", verbosity.debug)
            except Exception as e:
                warning(f" @CP2KSocketServer: Error closing client: {e}", verbosity.medium)
            finally:
                self.client_socket = None
                self.is_connected = False

    def cleanup(self):
        """Clean up all socket resources."""
        self.close_connection()

        if self.server_socket:
            try:
                self.server_socket.close()
                info(" @CP2KSocketServer: Closed server socket", verbosity.debug)
            except Exception as e:
                warning(f" @CP2KSocketServer: Error closing server: {e}", verbosity.medium)
            finally:
                self.server_socket = None
                self.is_listening = False


class CP2KSocketCommunicator:
    """CP2K socket communication handler for i-PI protocol.

    This class implements the complete i-PI socket communication protocol
    specifically for CP2K interactions in constant potential simulations.
    It handles data serialization, message passing, and error recovery.
    """

    def __init__(self, socket_server):
        """Initialize CP2K socket communicator.

        Args:
            socket_server: CP2KSocketServer instance
        """
        self.server = socket_server
        self.last_error = None

    def send_positions_and_get_forces(self, positions, cell_h, charge=None):
        """Send positions to CP2K and receive energy, forces, and Fermi level.

        This method implements the complete i-PI communication cycle:
        STATUS → READY/HAVEDATA → [POSDATA] → GETFORCE → FORCEREADY + data

        Args:
            positions: Atomic positions array (3*natoms)
            cell_h: Cell matrix (3x3)
            charge: Optional charge parameter for endpoint identification

        Returns:
            dict: Results with keys 'energy', 'forces', 'virial', 'fermi_level', etc.
                 Returns None on failure.
        """
        if not self.server.is_connected or not self.server.client_socket:
            warning(" @CP2KSocketCommunicator: No active connection", verbosity.medium)
            return None

        if not self.server.is_connection_healthy():
            warning(" @CP2KSocketCommunicator: Connection unhealthy, attempting reconnect", verbosity.medium)
            if not self.server.accept_connection(timeout=30.0):
                warning(" @CP2KSocketCommunicator: Failed to reconnect", verbosity.medium)
                return None

        try:
            return self._execute_ipi_protocol(positions, cell_h, charge)

        except Exception as e:
            error_msg = str(e).lower()
            self.last_error = str(e)

            # Smart error classification
            connection_error_keywords = ['connection', 'socket', 'broken', 'closed', 'reset', 'refused', 'timeout']
            is_connection_error = any(keyword in error_msg for keyword in connection_error_keywords)

            if is_connection_error:
                warning(f" @CP2KSocketCommunicator: Connection error, marking for reconnection: {e}", verbosity.medium)
                self.server.is_connected = False
                self.server.close_connection()
            else:
                # info(f" @CP2KSocketCommunicator: Recoverable error, keeping connection: {e}", verbosity.medium)

            return None

    def _execute_ipi_protocol(self, positions, cell_h, charge):
        """Execute the complete i-PI socket protocol with CP2K.

        Args:
            positions: Atomic positions array
            cell_h: Cell matrix
            charge: Endpoint charge for logging

        Returns:
            dict: CP2K calculation results
        """
        client_socket = self.server.client_socket
        natoms = len(positions) // 3

        # Prepare data arrays
        positions_array = np.array(positions, dtype=np.float64)
        h = np.array(cell_h, dtype=np.float64).reshape((3, 3))
        h_ih = np.linalg.inv(h)
        h_flat = h.flatten()
        ih_flat = h_ih.flatten()

        # info(f" @CP2KSocketCommunicator: Starting i-PI protocol for charge {charge}", verbosity.medium)

        # Step 1: Send STATUS to query CP2K state
        client_socket.sendall(MESSAGE["status"])
        # info(f" @CP2KSocketCommunicator: Sent STATUS to CP2K client {charge}", verbosity.debug)

        # Step 2: Receive READY or HAVEDATA
        header = client_socket.recv(HDRLEN)
        header_str = header.decode('utf-8').strip()
        # info(f" @CP2KSocketCommunicator: Received '{header_str}' from CP2K client {charge}", verbosity.debug)

        if header_str == "READY":
            # Send position data
            posdata_msg = (
                MESSAGE["posdata"] +
                h_flat.tobytes() +
                ih_flat.tobytes() +
                np.int32(natoms).tobytes() +
                positions_array.tobytes()
            )
            client_socket.sendall(posdata_msg)
            # info(f" @CP2KSocketCommunicator: Sent POSDATA to CP2K client {charge}", verbosity.debug)

        elif header_str == "HAVEDATA":
            # info(f" @CP2KSocketCommunicator: CP2K client {charge} already has data", verbosity.debug)

        else:
            raise ValueError(f"Expected 'READY' or 'HAVEDATA', got '{header_str}'")

        # Step 3: Send GETFORCE to request results
        client_socket.sendall(MESSAGE["getforce"])
        # info(f" @CP2KSocketCommunicator: Sent GETFORCE to CP2K client {charge}", verbosity.debug)

        # Step 4: Wait for FORCEREADY
        reply = client_socket.recv(HDRLEN)
        if reply != MESSAGE["forceready"]:
            raise ValueError(f"Expected 'forceready', got {reply}")
        # info(f" @CP2KSocketCommunicator: Received FORCEREADY from CP2K client {charge}", verbosity.debug)

        # Step 5: Receive calculation results
        return self._receive_cp2k_results(client_socket, natoms, charge)

    def _receive_cp2k_results(self, client_socket, natoms, charge):
        """Receive and parse CP2K calculation results.

        Args:
            client_socket: Connected socket
            natoms: Number of atoms
            charge: Endpoint charge for logging

        Returns:
            dict: Parsed results
        """
        def recv_all(sock, dest):
            """Receive all data for a numpy array."""
            blen = dest.itemsize * dest.size
            buf = np.zeros(blen, np.byte)
            bpos = 0

            while bpos < blen:
                chunk = sock.recv(blen - bpos)
                if not chunk:
                    raise socket.error("Socket connection broken")
                buf[bpos:bpos+len(chunk)] = np.frombuffer(chunk, dtype=np.byte)
                bpos += len(chunk)

            return np.frombuffer(buf, dtype=dest.dtype).reshape(dest.shape)

        # Receive energy (1 float64)
        energy_array = np.zeros(1, dtype=np.float64)
        energy_array = recv_all(client_socket, energy_array)
        energy = float(energy_array[0])

        # Receive natoms confirmation (1 int32)
        mlen_array = np.zeros(1, dtype=np.int32)
        mlen_array = recv_all(client_socket, mlen_array)
        received_natoms = int(mlen_array[0])

        if received_natoms != natoms:
            raise ValueError(f"Atom count mismatch: expected {natoms}, got {received_natoms}")

        # Receive forces (3*natoms float64)
        forces = np.zeros(3 * natoms, dtype=np.float64)
        forces = recv_all(client_socket, forces)

        # Receive virial (9 float64)
        virial_flat = np.zeros(9, dtype=np.float64)
        virial_flat = recv_all(client_socket, virial_flat)
        virial = virial_flat.reshape((3, 3))

        # Receive extra string length (1 int32)
        extra_len_array = np.zeros(1, dtype=np.int32)
        extra_len_array = recv_all(client_socket, extra_len_array)
        extra_len = int(extra_len_array[0])

        # Parse CP2K extras (JSON format)
        fermi_level = None
        converged = True
        extra_dict = {}

        if extra_len > 0:
            extra_bytes = np.zeros(extra_len, dtype=np.dtype('S1'))
            extra_bytes = recv_all(client_socket, extra_bytes)
            extra_string = bytearray(extra_bytes).decode('utf-8')

            try:
                import json
                extra_dict = json.loads(extra_string)

                # Extract Fermi level from CP2K
                if 'fermi_level_eV' in extra_dict:
                    fermi_level = float(extra_dict['fermi_level_eV'])
                elif 'fermi_level' in extra_dict:
                    fermi_level = float(extra_dict['fermi_level'])

                # Extract convergence status
                if 'scf_converged' in extra_dict:
                    converged = bool(extra_dict['scf_converged'])
                elif 'converged' in extra_dict:
                    converged = bool(extra_dict['converged'])

                # info(f" @CP2KSocketCommunicator: Parsed CP2K extras {charge}: fermi={fermi_level} eV, converged={converged}", verbosity.debug)

            except json.JSONDecodeError:
                warning(f" @CP2KSocketCommunicator: Could not parse extra JSON: {extra_string[:100]}...", verbosity.medium)

        # Convert Fermi level from eV to atomic units (Hartree) for i-PI
        fermi_level_au = fermi_level / Constants.EV_PER_HARTREE if fermi_level is not None else None

        # info(f" @CP2KSocketCommunicator: Results from charge {charge} - Energy: {energy:.6f} Ha, Fermi: {fermi_level} eV", verbosity.medium)

        return {
            "energy": energy,           # in Hartree
            "forces": forces,           # in Hartree/Bohr
            "virial": virial,          # in Hartree
            "fermi_level": fermi_level, # in eV
            "fermi_level_au": fermi_level_au,  # in Hartree
            "charge": charge,
            "converged": converged,
            "is_real_data": True,
            "data_source": "cp2k_socket",
            "extras": extra_dict
        }
