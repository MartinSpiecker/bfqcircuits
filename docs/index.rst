Black Forest Quantum Circuits documentation
============================================

This package allows the numerical exploration of the most commonly used superconducting artificial atoms,
the transmon and the fluxonium. Even though these circuit names are typically associated with certain parameter regimes,
all regimes of these circuits can efficiently be analysed.

The package can be used to design the spectrum of the atoms, their dispersive readout, as well as to extract various
properties. A fluxonium fit routine is included, to extract the circuit parameters from a measured spectrum.
The transmon Hamiltonian can be set up with an arbitrary number of Josephson junction harmonics.

The package includes the celebrated quantum Rabi model to showcase the complex behavior of the dispersive shift
with increasing readout power.

The managable source code may serve as a starting point to implement more complex artificial atoms and their readout.

Created at the foot of the Black Forest in Karlsruhe, Germany.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/bfqcircuits
   examples/index


Quick start
--------------

To install bfqcircuits, run the following command in your terminal:

    pip install bfqcircuits

or download the package and execute in the main directory::

    pip install .

In case you wish to edit the package, install it via::

    pip install --editable .

Use it in your project::

    from bfqcircuits.core import fluxonium

To get started, you may have a look at the examples.

Documentation
--------------

The documentation for bfqcircuits is available at: https://black-forest-quantum-circuits.readthedocs.io