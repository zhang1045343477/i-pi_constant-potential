# i-PI 3.1.5.1 — Constant-Potential Molecular Dynamics Framework

This project is an experimental development version based on i-PI 3.1.5.1, implementing a flexible and general constant-potential molecular dynamics framework.
Related manuscript (preprint):
https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000187/v1

## Main Idea

The framework is built upon the distinctive server–client architecture of i-PI. Its central idea is to introduce an additional electronic degree of freedom and couple it to a target potential, such as the Fermi level or work function. The propagation of the electronic DOF, the potentiostat control, and the exchange of electronic information such as electron number and Fermi level/work function are all handled on the i-PI side. As a result, the method is not tied to a specific electronic-structure package, but can instead be interfaced with different DFT clients through minimal and non-intrusive modifications.

This design makes the framework both flexible and general. It is applicable not only to conventional constant-potential simulations based on implicit solvent, but also to a Ne-electrode (Ne counter-electrode) based constant-potential scheme, extending constant-potential simulations beyond the implicit-solvent setting.

At the same time, the framework is compatible with two broad classes of electronic-structure backends.
For DFT clients that support fractional electron numbers, constant-potential sampling can be performed directly.
For DFT clients that are restricted to integer electron numbers, the framework employs a linear interpolation (two-endpoint mixing) strategy between two neighboring constant-charge states. By constructing an equivalent mixed-Hamiltonian description, this approach bypasses the integer-electron-number constraint and enables flexible constant-potential sampling.

More broadly, this linear-interpolation strategy provides a natural bridge between constant-charge and constant-potential simulations. Therefore, the framework is not only applicable to conventional DFT-based simulations, but also provides a natural route toward constant-potential machine-learning potentials.

## Key Features

Constant-potential framework built on top of i-PI by introducing an additional electronic degree of freedom.

General and portable implementation enabled by the i-PI server–client communication architecture, allowing coupling to multiple DFT clients with minimal modifications.

Two stochastic potentiostats (CSVR && Langevin) are provided. Compared with deterministic schemes (e.g., Nosé–Hoover), stochastic thermostats typically offer better dynamical properties and ergodicity for a single electronic DOF; in tests, the electronic number exhibits a well-behaved distribution.

Supports both conventional implicit-solvent constant-potential simulations and a Ne-electrode-based constant-potential scheme.

Establishes a bridge between constant-charge and constant-potential simulations through linear interpolation between two neighboring constant-charge endpoints, making the framework applicable to DFT clients subject to integer-electron-number constraints.

Naturally extendable to constant-potential machine-learning potentials.

## Usage

To use this version of i-PI, the external electronic-structure clients must first be patched accordingly. The current implementation relies on the patches provided in examples/clients/vasp and examples/clients/cp2k. Therefore, before running simulations, the corresponding VASP or CP2K client needs to be recompiled and reinstalled with these patches applied.

Once the patched clients are installed, this framework can be used in essentially the same way as standard i-PI. Users can prepare the i-PI input files as usual, launch the i-PI server, and connect the patched client to perform constant-potential molecular dynamics simulations.

Example input files can be found in:

examples/constant-potential/

These examples provide typical setups for constant-potential simulations and can serve as starting points for practical calculations.
