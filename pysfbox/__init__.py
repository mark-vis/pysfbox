"""
PySFBox -- the Scheutjens-Fleer self-consistent-field (SF-SCF) lattice theory
for polymers and surfactants at interfaces, in pure Python/NumPy. It reads and
writes the input/output file format of the classic sfbox and Namics programs.

Scope (v1): one gradient direction (planar / cylindrical / spherical,
including FJC_choices lattice refinement) plus the Namics two- and
three-gradient lattices (2D flat, 2D cylindrical, 3D flat); linear
(multi-block) and branched chains, monomeric solvents, frozen surfaces,
pinned (grafted) segments, Flory-Huggins chi interactions; charged systems
(fixed charges, electrodes, and weak/multistate charges via
state/reaction); var scans and search/super-iteration. Unknown keywords are
reported and ignored; unknown output properties print "NiN", exactly like
Namics; unsupported features raise a clear NotImplementedError.

Every supported feature is validated against the compiled Namics at or near
machine precision (regression suite in tests/); the pure-NumPy source doubles
as a readable reference implementation of the SF-SCF machinery.

Usage:
    python -m pysfbox path/to/input.in

The architecture mirrors Namics (Lattice / Segment / Molecule / System /
Solver / Output) so that extensions slot in where they live in the C++
original.
"""

from .system import System
from .inputreader import read_input
from .runner import run_file

__version__ = "1.1.0"
__all__ = ["System", "read_input", "run_file", "__version__"]
