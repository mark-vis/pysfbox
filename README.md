# PySFBox — Scheutjens–Fleer SCF lattice theory in pure Python/NumPy

[![PyPI](https://img.shields.io/pypi/v/pysfbox.svg)](https://pypi.org/project/pysfbox/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21338367.svg)](https://doi.org/10.5281/zenodo.21338367)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Scheutjens–Fleer lattice SCF in pure Python/NumPy that **reads Namics input
files and writes Namics output files**, so existing input files, analysis and
plotting scripts, and course material keep working — no C++ toolchain required.

**Install** (Python ≥ 3.9; the only dependency is NumPy):

```bash
pip install pysfbox            # adds a `pysfbox` command to your PATH
pysfbox my_input.in            # writes my_input.kal / my_input.pro
```

Or `pipx install pysfbox` for an isolated, always-on-PATH command-line tool.
`pysfbox` runs from any folder and writes its `.kal`/`.pro` next to each input
file. Without installing at all — just clone the repo (or drop the `pysfbox/`
folder next to your inputs) and run the module directly:

```bash
pip install numpy
python -m pysfbox my_input.in        # run from the folder that contains pysfbox/
```

GPL-3.0-or-later (see `LICENSE` and `NOTICE`; `pysfbox/sfnewton.py` carries its
own Wageningen copyright notice and governing terms).

Regression tests: `python tests/run_tests.py`. The `tests/` folder doubles as
an example collection — every supported feature has a small, commented `.in`
file there (brushes, adsorption, charged walls, weak polyelectrolytes,
branched molecules, 2D/3D grids, searches).

Keyword reference: `REFERENCE.md` documents every input keyword — allowed
values, defaults, and what each computes — as a lookup manual.

## Validation

PySFBox is validated against the compiled Namics: dozens of regression cases
agree with the C++ oracle at or near machine precision, a 28-case exercise set
reproduces the compiled Namics output case for case, and thermodynamic
observables (free energy, grand potential) agree to the shared convergence
floor. A depletion profile additionally cross-validates against an independent
SF-SCF implementation to ~4e-13, and the physics has passed two full
first-principles reviews with adversarial verification. Where PySFBox departs
from Namics it does so deliberately — implementing the corrected physics where
Namics has confirmed bugs (the fixed-potential Poisson factor 2, the
refined-lattice bond length, the weak-charge free-energy/chemical-potential
index terms) — each noted in the relevant section below.

## What it supports (v1)

- **Input**: the Namics `key : name : parameter : value` format, `//`
  comments, `start` blocks (settings accumulate, like Namics), `alias`
  (`#N#` substitution), and `var` parameter scans (`scan`/`step`/`end_value`),
  including alias scans (`scan : N-value`).
- **Lattice**: one, **two, or three gradient directions** (`gradients : 1/2/3`),
  as in Namics. 1-gradient: planar, cylindrical, spherical, `simple_cubic`
  (λ=1/6) or `hexagonal` (λ=1/4), lattice refinement `FJC_choices : 5, 7, …`
  (spherical/cylindrical), `offset_first_layer` shifts the radial origin.
  2-gradient (`simple_cubic`, `FJC 3`, neutral/linear): flat (x,y) and
  cylindrical (r,z); 3-gradient: flat (x,y,z). N-D uses `n_layers_x/y/z` and
  per-axis bounds `lowerbound_x`, `upperbound_y`, … (`mirror` / `surface` /
  `periodic`); one unified finite-volume lattice handles every geometry.
- **mon**: `freedom` free / frozen / pinned, `frozen_range` / `pinned_range`
  (1-gradient `1;3`, `lowerbound`, `upperbound`, `firstlayer`, `lastlayer`;
  2D/3D box grammar `xlo,ylo[,zlo];xhi,yhi[,zhi]`), `chi_X` interaction
  parameters.
- **mol**: linear multi-block compositions `(X)1(A)200(G)1` AND **branched
  architectures** — bracket side chains `(C)18[(O)1](O)1...` (nestable;
  several `[..][..]` on one node), symmetric dendrimers/stars
  `@dend(X,(A)200(G)1,5)`, combs `@comb((B)10;B,(A)25,(B)9,100;(B)1)`, and
  group-repeat sugar `(C)12((O)1(C)2)5(O)1`, on any 1-gradient geometry;
  `freedom` free (`phibulk`), restricted (`n` or `theta`), solvent.
- **search / super-iteration**: `var : mol-X : search : theta` (or `n` /
  `phibulk`) paired with a target — `var : sys-NN : grand_potential : 0`,
  `... : free_energy : V`, or `var : mol-Y : mu / theta / n / phibulk : V` —
  runs an outer scalar root-find (regula falsi) that adjusts the searched
  quantity until the target observable hits its value, each step a full
  warm-started SCF solve. (The `constraint:delta` / `Laplace_pressure`
  membrane-balance search of the Namics examples is not supported.)
- **Electrostatics**: `mon : valence` (fixed charges, strong electrolytes),
  `mon : epsilon` (relative permittivity, default 80), `mon : e.psi0/kT`
  (fixed surface potential on a frozen electrode layer), `mol : freedom :
  neutralizer` (auto-adjusts its bulk fraction for an electroneutral bulk),
  `lat : bondlength` (meters). The dimensionless potential psi joins the
  iteration; profiles via `pro : sys : X : psi` and `q`. Works on **planar,
  cylindrical and spherical** lattices (the curved Poisson uses
  face-area-weighted fluxes; `tests/charged_sphere.in`,
  `tests/charged_cylinder.in` match the compiled Namics at the convergence
  floor, and the spherical potential reproduces analytic Debye–Hückel
  screening). Validated against the compiled Namics at machine precision
  (`tests/polyelectrolyte_wall.in`); fixed-potential systems
  (`tests/edl_fixed_psi.in`) validate against Debye theory instead, because
  PySFBox deliberately corrects a factor-2 bug in the Namics fixed-potential
  Poisson branch (Debye lengths come out √2 too short there). Refined curved
  charged lattices (`FJC_choices > 3`) are supported and refinement-consistent:
  the Poisson equation keeps the physical bond length, where Namics' Debye
  lengths scale spuriously with the refinement — so expect a deviation from an
  unpatched Namics on that combination. Curved fixed-potential electrodes
  raise a clear message.
- **Weak (multistate) charges**: `state : AH : mon : A` + `state : AH :
  valence : 0` attach annealed internal states to a segment;
  `reaction : weak : equation : 1(AH) + 1(H2O) = 1(AM) + 1(H3O)` +
  `reaction : weak : pK : 7` set their equilibria; `state : H3O :
  alphabulk : 1e-7` anchors the pH (site-fraction based, as in Namics:
  "pH" = −log10 alphabulk_H3O). The bulk ionisation is solved to machine
  precision (Henderson–Hasselbalch exactly, in site fractions with the
  water activity included); locally every state equilibrates with the
  fields — charge regulation comes out automatically. Titration scans:
  `var : state-H3O : scan : alphabulk` with `scale : exponential`
  (`steps` = steps per decade, as in Namics). Per-state outputs: kal
  `theta_AM`, `alphabulk_AM`, `phibulk_AM`, `theta_exc_AM` (on the parent
  mon), `mol : X : mu-AM`; pro `phi-AM`, `alpha-AM`, `u-AM`. States
  inherit the mon's chi (state-level chi overrides); the mon-level
  valence is ignored once states exist. Validated against the compiled
  Namics at the convergence floor on the Namics `polE.in` example and a
  73-step titration (`tests/weak_*.in`); PySFBox implements corrected
  physics in a few spots where Namics has known bugs, so F/mu comparisons
  with an unpatched Namics differ whenever state–state chi is nonzero.
- **sys**: `initial_guess` — `polymer_adsorption` / `membrane` / `micelle`
  analytic starting potentials (the Namics formulas, e.g. u = −λ·χ at
  solid-adjacent sites), honored on the first calculation only, exactly
  like Namics; `previous_result`/`none` accepted; guess files and
  `membrane_torus` raise a clear message.
- **Solver**: the Namics pseudohessian quasi-Newton (a licensed Python
  translation of `sfnewton.cpp`) as primary, with Anderson-accelerated
  Picard and two rescue stages as fallbacks (reads `newton : ... :
  tolerance / iterationlimit / deltamax / m / method`; the Namics `method`
  choices `pseudohessian` / `DIIS` / `Picard` / `LBFGS` / `hessian` map onto
  this cascade — `hessian` is the Namics full-Hessian mode, recomputing a
  finite-difference Hessian every iteration). Propagators are renormalised
  with log-domain bookkeeping, so long or strongly adsorbing chains cannot
  overflow. Large 2D/3D grids (beyond the dense pseudohessian) are handled
  by the Anderson stage.
- **Output**: `.kal` (header + one row per calculation/var step, `%.16e`)
  and `.pro` (x at half-integer layers + `%.20g` columns), with Namics file
  numbering. Unknown output properties print `NiN` plus a warning — exactly
  like Namics. Implemented kal properties: `sys` grand_potential,
  free_energy (and `free_energy (po)`), iterations, residual; `lat`
  n_layers / volume; `mol` theta, theta_exc, phibulk, Mu/mu (chemical
  potential), mu-STATE, n, N, chainlength, GN, phiM, phiMax, phiMin,
  `<alias>-value`; `mon` theta, theta_exc, phibulk, chi_X,
  1st_M_phi_z, 2nd_M_phi_z, RMS, fluctuations, and per-state
  alphabulk_S / valence_S / phibulk_S / theta_S / theta_exc_S. Profiles
  (`pro`): `mol : X : phi`, `mon : X : phi`, `mon : X : u`,
  `mon : X : phi-S / alpha-S / u-S`, `sys : X : alpha / psi / q`.

## Beyond this release

Full-Namics features that this release does not implement raise a clear
`NotImplementedError` naming the feature, rather than silently producing
wrong numbers — so an input either runs correctly or tells you exactly what
it needs. Currently outside scope: ring and asymmetric-dendrimer
architectures, the `constraint:delta` / `Laplace_pressure` membrane-balance
search, initial-guess files, and mesodyn/cleng/teng calculations.

The **one-gradient** path (planar / cylindrical / spherical, including
`FJC_choices` refinement) covers the full feature set above. The **two- and
three-gradient** lattices — 2D flat, 2D cylindrical (r,z), 3D flat, as in
Namics — currently cover neutral linear chains on `simple_cubic`; charged and
branched systems there need `gradients : 1`. Curved fixed-potential
electrodes raise. For weak charges, unbalanced reactions and states on frozen
monomers are rejected (the latter also in Namics).

## Extending (where things live)

The module layout mirrors the Namics classes, so the C++ original is the
roadmap for extensions:

| file | contents | Namics counterpart |
|---|---|---|
| `inputreader.py` | input parsing, alias, var | `input.cpp` |
| `lattice.py` | 1-gradient geometry, ghost layers, site averages, moments | `lattice.cpp`, `LGrad1.cpp` |
| `latticend.py` | 2- and 3-gradient finite-volume lattice | `LGrad2.cpp`, `LGrad3.cpp` |
| `model.py` | segments, molecules (linear + branched trees), renormalised propagators | `segment.cpp`, `molecule.cpp`, `mol_linear.cpp`, `mol_branched.cpp`, `mol_comb.cpp`, `mol_dendrimer.cpp` |
| `reactions.py` | internal states, reactions, bulk ionisation solve | `state.cpp`, `reaction.cpp` |
| `system.py` | residual, masks, observables, solver cascade, initial guesses | `system.cpp`, `solve_scf.cpp` |
| `sfnewton.py` | pseudohessian quasi-Newton (licensed translation) | `sfnewton.cpp` |
| `output.py`, `runner.py` | .kal/.pro writers, start/var loops, search | `output.cpp`, `namics.cpp`, `variate.cpp` |

Examples: new output properties = extend the tables in `System.get_value` /
`System.get_profile`; new chain architectures = generalise the tree walk in
`Molecule` (see `mol_branched.cpp`).

## Convergence envelope

The primary solver IS the Namics quasi-Newton (pseudohessian): `sfnewton.py`
is a faithful, licensed Python translation of `sfnewton.cpp`, run from the
warm-start guess with Namics-like iteration counts. Anderson-accelerated
Picard (with a damped-Picard warmup on cold starts) is the fallback, plus
two rescue stages (extended-budget pseudohessian, one-shot full-Hessian
anchor) for the stiffest cases. This converges the entire shipped
validation set, including the extreme corner — very strong adsorption of very long
chains (the shipped `homopolymer_adsorption.in`, N = 10000, chi_Si = -6)
converges via the extended rescue and matches the compiled Namics exactly.
Genuine non-convergence still fails *gracefully* (a clear "no convergence"
message with actionable advice, never a wrong number). `var` scans
warm-start each step from the previous solution — ramping a parameter
(e.g. a co-solvent) is the intended way into hard regimes.

## Speed

The default path is deliberately pure NumPy — one dependency, every equation
inspectable — running the same pseudohessian quasi-Newton as Namics, with
Namics-like iteration counts. On a 28-case benchmark set the compiled Namics
C++ takes ~18 s where PySFBox takes ~2 minutes, and every one of those 28 cases
matches the compiled Namics output: typical single problems solve in seconds,
the stiffest shipped case (`homopolymer_adsorption.in`, N = 10000, chi_Si = −6)
in a few minutes. For throughput-bound parameter sweeps, the same input files
run unchanged on the compiled Namics — compatibility works in both directions.

## Caveats for quantitative use

- The primary solver is a translation of Namics' quasi-Newton; iteration
  counts are Namics-like but not identical (fallback stages differ), and
  converged solutions agree to the shared convergence floor.
- The grand potential and free energy implement the standard uncharged SF
  expressions; cross-validated against compiled Namics on a two-brush case
  (free energy 6e-9, grand potential 5e-8 relative — the shared convergence
  floor; see `tests/two_brushes_quick.in`). For research use, spot-check
  your own geometry the same way. Profiles, theta, theta_exc, and moments
  follow the Namics definitions exactly (`LGrad1::Moment`, `WeightedSum`).
- Default tolerance 1e-7, iterationlimit 1000, deltamax 0.1 (as in Namics).

## Citing PySFBox

If you use PySFBox in your research, please cite it — GitHub's "Cite this
repository" button uses `CITATION.cff`, and each release is archived on Zenodo
with a DOI: [10.5281/zenodo.21338367](https://doi.org/10.5281/zenodo.21338367)
(concept DOI, always resolving to the latest version). Please also cite the
underlying sfbox/Namics work by Leermakers and co-workers.

## Acknowledgements & lineage

PySFBox is a reimplementation of the Scheutjens–Fleer self-consistent-field
lattice theory as realised in **sfbox/Namics** by F.A.M. Leermakers and the
Physical Chemistry group at Wageningen University — the input/output formats,
property names, and numerical conventions all follow Namics so that existing
material keeps working. `pysfbox/sfnewton.py` is a Python translation of
Namics' `sfnewton.cpp` and carries the original Wageningen University
copyright and reproduction notice in its file header, which governs that
file. Many thanks to Frans Leermakers for sfbox and Namics — the theory and
the code that PySFBox builds on — and for discussions along the way.

PySFBox itself is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later) — see `LICENSE` and `NOTICE`.
