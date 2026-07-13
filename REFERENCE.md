# PySFBox — keyword reference

A lookup reference for the PySFBox input language: every `key : name : parameter : value`
keyword the program accepts, its allowed values, defaults, and what it computes.
PySFBox reads the Namics/sfbox input format and writes Namics `.kal`/`.pro` output,
so this doubles as a compatibility map. Modelled on the sfbox manual, but generated
from — and verified against — the PySFBox source.

For narrative introductions to the physics and the solver see `README.md`.
Unsupported features raise a clear `NotImplementedError` naming the feature;
unknown output properties print `NiN` — never a silently wrong number.

**Contents**

1. [Introduction, the input file, and tips](#introduction-the-input-file-and-tips)
2. [Lattice (`lat`)](#lattice-lat)
3. [Monomers (`mon`)](#monomers-mon)
4. [Molecules (`mol`)](#molecules-mol)
5. [Weak/multistate charges (`state`, `reaction`)](#weakmultistate-charges-state-reaction)
6. [System (`sys`) and solver (`newton`)](#system-sys-and-solver-newton)
7. [Output (`output`, `kal`, `pro`) and what it means](#output-output-kal-pro-and-what-it-means)
8. [Scans, search, ranges, and aliases (`var`, `search`, ranges, `alias`)](#scans-search-ranges-and-aliases-var-search-ranges-alias)
9. [Example inputs](#example-inputs)

---

## Introduction, the input file, and tips

### What PySFBox is

PySFBox is a pure-Python/NumPy reimplementation of the Scheutjens–Fleer
self-consistent-field (SF-SCF) lattice theory for polymers, surfactants and
(weak/strong) electrolytes at interfaces. It **reads the Namics/sfbox
`key : name : parameter : value` input format and writes the same `.kal` and
`.pro` output files**, so existing input files, analysis scripts and course
material keep working without a C++ toolchain. The only dependency on the
default path is NumPy (Python ≥ 3.9).

It is validated against the compiled Namics — dozens of regression cases agree
with the C++ oracle at or near machine precision, and a 28-case teaching set
reproduces Namics output case for case. Where the C++ code has a confirmed bug,
PySFBox implements the corrected physics instead.

### Running

```bash
python -m pysfbox my_input.in                 # writes my_input.kal / my_input.pro
python -m pysfbox a.in b.in c.in              # several files, ONE process
python -m pysfbox -h                          # short usage message
```

- Output files are written **next to each input file** (same directory, base
  name with `.kal` / `.pro`), exactly as Namics does.
- Passing **several files runs them in one process** (`for path in args: run_file(path)`),
  each announced with `== <path>`.

### The input file

An input file is a sequence of lines, each of the form

```
key : name : parameter : value
```

parsed by splitting on `:`; a data line needs **at least four colon-separated
fields** (a line with fewer raises `cannot parse input line: '…'`). The `name`
is the object's label (a monomer name, molecule name, lattice name, …); the
`value` is everything after the third colon (so a value may itself contain a
colon). The order of lines within one calculation is irrelevant.

The top-level keys:

| key | governs | reference section |
|---|---|---|
| `lat` | lattice: geometry, gradients, layers, boundaries, bondlength | Lattice |
| `mon` | segments: freedom, ranges, chi, valence, epsilon, states | Monomers |
| `state` | internal states of a segment (weak charges) | States |
| `reaction` | equilibria between states (weak charges) | Reactions |
| `mol` | molecules: composition/architecture, freedom, phibulk/theta | Molecules |
| `sys` | global options (e.g. `initial_guess`) | System |
| `newton` | solver controls (tolerance, iterationlimit, deltamax, method) | Newton/solver |
| `output` | output files and what they contain | Output |
| `var` | parameter scans and `search`/super-iteration | Varying parameters |
| `alias` | `#N#` text substitution | (below) |

**Comments** — `//` starts a comment: everything from `//` to the end of the
line is ignored, whether the line is all comment or the `//` appears mid-line.
Blank lines are ignored.

```
// this whole line is a comment
mon : A : chi - B : 0.5   // and this trailing part is ignored
```

**`start` blocks** — the word `start` on its own line (case-insensitive) closes
one calculation and begins the next. **Settings accumulate across `start`
blocks**: a later block adds to, or overrides (last value wins), the settings of
earlier ones — so a scan of one parameter is just that line repeated with a new
value before each `start`. A trailing `start` is optional; the final block is
run whether or not it is present.

```
mol : water : theta : 0.25
start
mol : water : theta : 0.5     // everything else carried over from above
start
```

Internally each parameter stores a **list** of values (a parameter may legally
repeat, e.g. several output-spec lines); normal settings use override semantics
(the last value), output specs use the whole list.

**`alias` (`#N#` substitution)** — `alias : X : value : v` defines a
textual substitution: every `#X#` (in a composition string, or an `n_layers`
value) is replaced by `v` before that field is parsed. Useful for a chain
length used in several places, and an alias can itself be the scanned quantity
(`var : alias-X : scan : value`, or the `scan : X-value` shorthand on any `var`
block, e.g. `var : mol-pol : scan : N-value`).

```
alias : N : value : 100
mol : poly : composition : (A)#N#      // becomes (A)100
```

**Entering numbers** — a field is read as numeric only if it starts with a
digit (or `-` then a digit); scientific notation (`1e-6`, `6.02E23`) is
accepted, but a leading dot is not (`.34` falls back to the default). A field
that does not start that way (e.g. a stray `e-10` typo) is treated as
unparseable and the parameter falls back to its default — matching Namics'
`Input::Get_Real`.

### Output file naming

`.kal` rows accumulate one line per calculation / var step (reals as `%.16e`,
ints as `%i`). `.pro` files are numbered like Namics from the input base name:

| situation | `.pro` file name |
|---|---|
| single calculation, no scan | `base.pro` |
| single calculation, var scan | `base_<step>.pro` |
| multiple `start` blocks, no scan | `base_<start>.pro` |
| multiple `start` blocks + scan | `base_<start>_<step>.pro` |

### Tips & troubleshooting

**Achieving / speeding up convergence.** The SF equations are highly nonlinear
and no method guarantees convergence. The most effective tool is a
**warm-started `var` scan**: within a scan (and across `start` blocks) each step
starts from the previous converged solution, so ramping a hard parameter (a
co-solvent, chi, wall separation) gently into a stiff regime is the documented
way in. When only `n_layers` changes, the previous potentials are remapped onto
the new grid; from the third scan step a secant extrapolation predicts the next
step. These only shape the initial guess and cannot change the converged
solution. For self-assembling structures an analytic starting field helps:
`sys : <name> : initial_guess : polymer_adsorption / membrane / micelle`
(honored on the first calculation only, as in Namics). If the solver stalls,
lower `newton : … : deltamax` (the largest step it may take; default 0.1).

**Overflows can't happen.** Every propagator runs in the log domain with lazy
renormalisation, so long or strongly adsorbing chains cannot overflow to
`NaN` — there is **no `overflow_protection` switch to set** (it is always on).
This is a hard project invariant; a chain that overflowed in classic sfbox
simply propagates finitely here.

**Unknown keywords, unknown output, non-convergence.**

- Keys and parameters PySFBox does not consume are simply ignored (the parser
  stores every line; only the ones it understands are read). An **unknown
  `newton` `method` or `engine`** prints a `note:` and falls back to the default
  solver / NumPy residual.
- An **unknown output property** is written as `NiN` in the `.kal` (and skipped
  in the `.pro`), with a `warning:` on the console — like Namics.
- A **genuinely unsupported feature raises `NotImplementedError`** naming the
  feature and the alternative (e.g. "use the full Namics"), never a silently
  wrong number.
- **Non-convergence raises `RuntimeError`** with actionable advice, e.g.
  `no convergence in <N> iterations (max|g| = …); try a smaller deltamax`. A run
  whose restricted `theta` values exceed what the box can hold in equilibrium
  raises `converged to a state with negative solvent bulk fraction …`. In a
  `search` (super-iteration), a failed inner solve reports that the target is
  likely unreachable or non-monotone in the search variable.

Convergence quality is also visible in the run log, which prints
`converged in <it> iterations, max|g| = …, max|phi_T-1| = …` per calculation.

---

## Lattice (`lat`)

The `lat` block defines the discretised space in which the SCF equations are
solved: the number of gradients, the geometry, the lattice grid and its
boundary conditions. **Exactly one `lat` block is required** (more than one
raises `exactly one 'lat' block is required`). The block name is free; input
lines look like `lat : name : param : value`.

```
lat : L : gradients : 1
lat : L : geometry : planar
lat : L : n_layers : 100
```

### Common keywords

| keyword | values | default | meaning |
|---|---|---|---|
| `gradients` | `1`, `2`, `3` | `1` | number of spatial gradients; selects the 1-gradient `Lattice1D` or the N-gradient `LatticeND`. Any other value raises. |
| `geometry` | see tables below | `planar` (1D) / `flat` (N-D) | coordinate system. `flat` and `planar` are synonyms. |
| `lattice_type` | `simple_cubic`, `hexagonal` | `simple_cubic` | sets the neighbour-transition weight λ (`simple_cubic` → 1/6, `hexagonal` → 1/4). N-D accepts `simple_cubic` only. |
| `n_layers` | integer | *(required, 1D)* | number of physical layers z = 1..n_layers (see FJC refinement). Missing raises. |
| `FJC_choices` | `3`, `5`, `7`, … | `3` | lattice refinement (freely-jointed-chain sub-layers). `3` = unrefined (fjc = 1). |
| `offset_first_layer` | float ≥ 0 | `0.0` | radial offset of the first layer from the axis/centre (curved geometries). |
| `bondlength` | float in 1e-12..1e-8 (m) | `5e-10` | segment bond length; **only consumed by charged systems** (Poisson prefactor). Out-of-range raises. |
| `Markov` | `1` | `1` | chain statistics: `1` = fully flexible. |

### 1-gradient geometry (`gradients : 1`)

Layers z = 1..MX with one ghost layer per side (fjc ghosts when refined) for
the boundary conditions.

| `geometry` | meaning |
|---|---|
| `planar` / `flat` | flat (Cartesian) layers; site volume 1 |
| `cylindrical` | concentric shells; site average curvature-weighted |
| `spherical` | concentric spherical shells |

An unrecognised 1D geometry raises `unknown geometry '...'`.

### Lattice refinement (`FJC_choices`)

`FJC_choices = 3 + 2i` gives `fjc = (FJC_choices−1)/2` sub-layers per bond
length, so the internal grid is `MX = fjc · n_layers` and the propagator
becomes a `(2·fjc+1)`-point stencil with curvature-weighted, position-dependent
coefficients. A value that is not `3, 5, 7, …` raises
`FJC_choices must be 3 + 2*i`.

- `FJC_choices : 3` (fjc = 1) keeps the familiar three-point lattice.
- fjc > 1 is supported for **`spherical` and `cylindrical` only**; planar
  fjc > 1 raises `FJC_choices > 1 with geometry 'planar' is not supported`.
- fjc > 1 with `gradients > 1` raises (`LatticeND` is fjc = 1 only).

Note: refined charged runs use the base bond length (refinement-consistent),
which **intentionally deviates** from Namics' fjc-scaled Debye length.

### Boundaries (1-gradient)

| keyword | values | default | meaning |
|---|---|---|---|
| `lowerbound` | `mirror`, `surface` | `mirror` | condition at z = 0 (below layer 1) |
| `upperbound` | `mirror`, `surface` | `mirror` | condition at z = MX+1 (above the last layer) |

- `mirror`: reflect the interior into the ghost layer (zero-flux wall).
- `surface`: the ghost is a solid/impenetrable wall (density 0 for
  propagators). Frozen surface segments (`mon : S : freedom : frozen`) pin the
  ghost density explicitly.

Only `mirror` and `surface` are meaningful in 1-gradient; any other value
(e.g. `periodic`) is treated as `mirror`. `periodic` is a genuine option only
in the N-D lattice (below).

### Chain statistics (`Markov`)

In the public tree only `Markov : 1` is accepted; any `Markov : N` with N ≠ 1
raises `... (semiflexible chains) is not supported`, and any `k_stiff` line
raises.

### Two and three gradients (`gradients : 2` / `gradients : 3`)

For `gradients ≥ 2` the unified finite-volume `LatticeND` replaces the per-axis
`n_layers`/`lowerbound` keys with axis-suffixed ones. Profiles are laid out on
a padded grid with one ghost layer per side per axis.

**Grid size** — one required key per axis (missing raises
`gradients : N needs 'lat : ... : n_layers_<axis>'`):

| keyword | required for | meaning |
|---|---|---|
| `n_layers_x` | 2D, 3D | cells along axis 1 |
| `n_layers_y` | 2D, 3D | cells along axis 2 |
| `n_layers_z` | 3D | cells along axis 3 |

**Geometry** (`geometry`, default `flat`; axis roles: `x`/`z` Cartesian, `r`
radial, `theta` colatitude, `phi` azimuth):

| gradients | `geometry` | axis roles |
|---|---|---|
| 2 | `flat` / `planar` | (x, x) |
| 2 | `cylindrical` | (r, z) |

| 3 | `flat` / `planar` | (x, x, x) |

The public tree supports exactly the geometries Namics has: **2D flat, 2D
cylindrical (r,z), 3D flat**. An unsupported geometry raises with the list of
supported choices.

**Per-axis boundaries** — `lowerbound_x/y/z` and `upperbound_x/y/z`, values
`mirror`, `surface`, `periodic`. Defaults are role-aware: a full-2π azimuthal
(`phi`) axis defaults to `periodic`, every other axis to `mirror`. Set only the
axes you want to override (an unset axis keeps its role-aware default — do not
force `mirror` on a full-2π seam).

**Box ranges** — frozen/pinned segment ranges on the N-D grid use the
coordinate grammar `xlo,ylo[,zlo];xhi,yhi[,zhi]` (1-based interior indices per
axis), or the single-axis idioms `lowerbound` / `upperbound` (a whole wall on
axis 1). A wrong coordinate count raises `range '...' needs N coordinates per
corner`. These appear on `mon`/`mol` range lines, not in the `lat` block
itself. A single-layer box flush against a boundary that spans the *full* face
is treated as a solid wall; a partial patch stays an interior mask.

Other N-D keys: `lattice_type` must be `simple_cubic` (`hexagonal` raises);
`FJC_choices` must be `3` (refinement > 3 raises); `offset_first_layer` offsets
a radial axis from its axis/centre. An innermost angular arc shorter than one
bond gives a negative stay-weight and raises — offset the domain from the axis
or use fewer angular cells.

`LatticeND` reduces bit-for-bit to `Lattice1D` when the extra axes are uniform;
the flat/cylindrical N-D field solve is oracle-validated against Namics
(LG2Planar / LGrad3), the angular/curved N-D geometries by dimensional
reduction and the continuum Laplacian.

### Notes

- The N-D lattice path covers **neutral, linear, flexible** chains only.
  Charged (`valence`/`e.psi0`), branched
  molecules require `gradients : 1` and raise otherwise.
- `bondlength` matters only when the system is charged; for neutral systems it
  is ignored.

---

## Monomers (`mon`)

A `mon` block declares one **segment type** (monomer). Every distinct segment
that appears in a `mol` composition must have its own `mon` block. At least one
`mon` block is required (`no 'mon' blocks found` is raised otherwise). Lines take
the form:

```
mon : NAME : param : value
```

`NAME` is the segment name used in compositions and `chi` cross-references.
Parameters are read with override (last-wins) semantics; only `freedom`, the
electrostatic and range parameters below, plus any `chi_*` interactions are
interpreted at the `mon` level. Segment output properties (`phi`, `u`, moments,
…) are requested via `kal`/`pro` output lines, not set here.

### Freedom

| param | values | default | meaning |
|---|---|---|---|
| `freedom` | `free`, `frozen`, `pinned` | `free` | where the segment may sit on the lattice |

- **`free`** — the segment may be anywhere; it is one of the iterated fields.
- **`frozen`** — a solid/fixed segment (a wall). Its density is set to 1 on the
  range and it is *removed* from the free sites (`ksam`); it is not iterated but
  it still contributes `chi` to its neighbours. Requires `frozen_range`.
- **`pinned`** — the segment is confined to a region (its single-segment weight
  is masked to the range) but is still an iterated field. Requires
  `pinned_range`.

A `frozen`/`pinned` segment without the matching range raises
`freedom <f> requires a <f>_range`. Internal `state` blocks may not be attached
to a `frozen` mon (`states on frozen mons are not allowed … use a pinned mon
instead`).

### Range specification (`frozen_range`, `pinned_range`)

Set with `mon : NAME : frozen_range : …` (or `pinned_range`). One-gradient forms
(case-insensitive; a trailing `;` is tolerated, e.g. `upperbound;`):

| range value | region selected |
|---|---|
| `lo;hi` | interior layers `lo` … `hi` inclusive (1-based physical layers) |
| `lowerbound` | the lower ghost/boundary face (a wall below layer 1) |
| `upperbound` | the upper ghost/boundary face (a wall above the last layer) |
| `firstlayer` / `first_layer` | the first interior layer |
| `lastlayer` / `last_layer` | the last interior layer |

A frozen `lowerbound`/`upperbound` fills the corresponding ghost layer(s) to
density 1 (the surface a `chi_NAME` affinity couples to). With refined lattices
(`FJC_choices > 3`, `fjc > 1`), physical layers are expanded to refined sites
automatically: `lo;hi` maps to refined `[lo·fjc, (hi+1)·fjc − 1]`, matching
Namics `segment.cpp`.

```
mon : S  : freedom : frozen
mon : S  : frozen_range : lowerbound
mon : pg : freedom : pinned
mon : pg : pinned_range : 1;1
```

**2D/3D box grammar.** On an N-gradient lattice (`gradients > 1`), a range is a
box given as one corner per axis:

```
xlo,ylo[,zlo] ; xhi,yhi[,zhi]
```

Coordinates are 1-based interior indices per axis; each corner must list exactly
`gradients` coordinates (`range '…' needs N coordinates per corner` otherwise). A
single corner (no `;`) selects one cell. `lowerbound`/`upperbound` are also
accepted and refer to the lower/upper face of the **first** axis. A box that is
one layer thick against a boundary **and spans the full face on every other axis**
is treated as a solid wall face (its ghost hyperplane is filled). A *partial*
boundary patch is deliberately kept as an interior box and is **not** promoted to
a full wall (a physics-review fix that prevents faking solid contact along the
whole boundary).

### Interaction parameters (`chi_X`)

Flory–Huggins χ between this segment and segment `X` is set as
`mon : NAME : chi_X : value`. The parameter name is normalised by stripping all
spaces and accepting either separator, so the following are equivalent:

```
mon : A : chi_S    : -6
mon : A : chi-S    : -6
mon : A : chi - S  : -6
```

The χ table is **symmetric** and defaults to 0: a value given on either partner
(`mon : A : chi_S` or `mon : S : chi_A`) is used for both; if neither names the
other, χ = 0. `X` may be another monomer, a frozen wall segment, or (when weak
charges are in play) a named internal **state** — a mon-level `chi` may name a
state explicitly, and states inherit their parent mon's χ unless overridden.

### Electrostatics

| param | values | default | meaning |
|---|---|---|---|
| `valence` | real (units of e) | `0` | fixed charge per segment |
| `epsilon` | real | `80` | relative permittivity of this segment (water-like default) |
| `e.psi0/kT` | real | — | fixed dimensionless surface potential (electrode); requires `freedom : frozen` |

- A system is treated as **charged** if any segment (or any of its states) has a
  nonzero `valence`, or any segment sets `e.psi0/kT`. Charged systems require
  `gradients : 1` (`charged systems (valence / e.psi0/kT) need gradients : 1`)
  and a `planar`, `cylindrical`, or `spherical` geometry.
- `epsilon` stays a mon-level property even when internal states exist (states
  carry charge, not permittivity). Differing `epsilon` values enable the
  dielectric-contrast (variable-ε) Poisson solve.
- **`e.psi0/kT`** sets a fixed-potential electrode. It requires
  `freedom : frozen` (else `e.psi0/kT requires freedom : frozen`) and is
  **mutually exclusive** with `valence` (`valence and e.psi0/kT are mutually
  exclusive`). A fixed surface potential on a **curved** (cylindrical/spherical)
  lattice raises `fixed surface potential (e.psi0/kT) on a curved lattice is not
  supported yet`.
- **`valence` is ignored once states exist.** If a multistate segment (weak
  charge, via `state`/`reaction` blocks) also carries a mon-level `valence`, the
  mon-level value is dropped with a warning — the per-state valences carry the
  charge. The segment's effective bulk charge is then the annealed average
  `Σ_s alphabulk_s · valence_s`.

```
mon : Na : valence : 1
mon : Cl : valence : -1
mon : X  : epsilon : 40
mon : E  : freedom : frozen
mon : E  : frozen_range : lowerbound
mon : E  : e.psi0/kT : 0.5
```

> Note: sfbox's second-order `Pf - monomername2` transition-probability keyword
> is **not** a `mon` parameter in PySFBox.

---

## Molecules (`mol`)

A `mol` block defines one molecular species: its chain architecture
(`composition`), how much of it is present (`freedom` + `phibulk`/`theta`/`n`),
and optional per-molecule solver/model settings. Every input needs at least one
`mol`, and exactly one of them must be the `solvent` (see *Freedom*).

```
mol : <name> : composition : <architecture>
mol : <name> : freedom : <free|restricted|solvent|neutralizer>
```

Segments referenced in a composition must be declared as `mon` blocks first;
an unknown monomer raises `mol <name>: unknown mon '<mon>'`.

### Parameters

| keyword | values | default | meaning |
|---|---|---|---|
| `composition` | architecture string (see below) | — (required) | chain sequence/topology |
| `freedom` | `free`, `restricted`, `solvent`, `neutralizer` | `free` | how the amount is fixed (see below) |
| `phibulk` | real ≥ 0 | `0.0` | bulk volume fraction (used by `free`) |
| `theta` | real ≥ 0 | `n * N` | total amount ∑<sub>z</sub>L(z)φ(z) (used by `restricted`) |
| `n` | real ≥ 0 | `0.0` | number of chains; only sets the `theta` default (`theta = n·N`) |
| `ring` | `true`/`false` | `false` | ring topology — **raises** `NotImplementedError` if true |

`N` is the total number of segments in the chain (the sum over the composition).
Whitespace inside a composition is ignored, and the `#X#` alias mechanism is
expanded before parsing (e.g. `composition : (X)1(A)#N#(G)1`).

### Freedom — how the amount is set

| freedom | amount fixed by | notes |
|---|---|---|
| `free` | `phibulk` | bulk-equilibrium chain; density normalised so far-field φ = `phibulk` |
| `restricted` | `theta` (or `n`) | fixed total amount in the box; required for grafted/pinned chains |
| `solvent` | (computed) | fills the remaining volume: φbulk<sub>solvent</sub> = 1 − ∑(others). **Exactly one** required, else `ValueError`. |
| `neutralizer` | (computed) | bulk fraction set by electroneutrality; **requires a charged system** and there may be **at most one** |

Details and gotchas:

- A molecule containing a `pinned` (or otherwise grafted) segment must be
  `freedom : restricted`, otherwise: `mol <name>: contains pinned segment(s),
  which requires freedom : restricted`.
- For `restricted`, the effective density is `theta/GN`; a grafted/pinned chain
  has an implied bulk density of 0, a free-floating restricted (dissolved) chain
  has implied bulk φ = θ/GN (refreshed every iteration). Setting `theta`/`phibulk`
  to 0 deletes the molecule's density cleanly (exactly zero, not NaN).
- `neutralizer` on an uncharged system raises `freedom : neutralizer requires a
  charged system`; two neutralizers raise `at most one mol with freedom :
  neutralizer`.
- `n` is only a convenience: it sets the `theta` default via `theta = n·N`. If
  both `theta` and `n` are given, `theta` wins (override semantics).
- Any other `freedom` value raises `mol <name>: freedom '<value>' not supported`.

### Composition — linear multiblock

The linear grammar is a sequence of blocks `(<mon>)<count>`:

```
mol : pol : composition : (A)200
mol : pol : composition : (X)1(A)200(G)1
```

`(A)200` is 200 A-segments; blocks concatenate left to right. A block count is a
positive integer. Anything the block grammar cannot consume raises `cannot parse
composition '<comp>'`.

**Group-repeat sugar** `((...)...)n` expands textually (linear only), innermost
group first — e.g. `((O)1(C)2)5` → `(O)1(C)2` repeated 5 times (the surfactant
`C12E5` idiom).

### Composition — branched architectures

Branched syntax triggers the tree parser (`parse_architecture`). Branched
molecules require `gradients : 1` (1-gradient lattice); on a 2D/3D lattice they
raise `branched architectures need gradients : 1`.

**Bracket side chains** — `[...]` attaches a side chain to the *preceding*
segment; the backbone continues after the bracket. Brackets nest, and multiple
consecutive `[...][...]` attach to the same node (degree-4 junctions):

```
mol : surf : composition : (A)4(B)1[(C)2(B)1[(C)1](B)1](A)4
mol : water : composition : (W)2[(W)1][(W)1](W)1
```

A `[` with nothing before it, or an unbalanced `[`, raises a clear `ValueError`.

**`@dend` — symmetric dendrimer.** Body is `;`-separated generations, each
`junction, arm_composition, functionality`:

```
mol : star : composition : @dend(X,(A)40(G)1,6)
mol : pol  : composition : @dend(X,(A)24,2;A,(B)24,2;B,(D)25(D)5,2)
```

Generation *g+1*'s junction is appended to the end of every generation-*g* arm.
Only one arm type per generation is supported; several arm types per generation
(**asymmetric dendrimers**) raise `asymmetric dendrimers ... are not supported
yet -- needs the full Namics`.

**`@comb` — comb polymer.** Body is exactly three `;`-separated parts:
`lead-in ; junction,tooth,spacer,n_repeat ; lead-out`:

```
mol : pol : composition : @comb((B)5;B,(A)10,(B)4,12;(B)1)
```

= lead-in `(B)5`, then 12 repeats of [junction `B` + tooth `(A)10` + spacer
`(B)4`], then lead-out `(B)1`. A malformed repeat part raises `@comb repeat part
'...' must be junction,tooth,spacer,n`.

Any `@`-generator other than `@dend`/`@comb` raises `composition '<comp>' not
supported (supported generators: @dend, @comb)`.

**Ring molecules** (`mol : X : ring : true`) change the propagator topology and
are **not supported** in PySFBox (raises) — use the C++ Namics.

### See also

- `mon` — segment declarations (`freedom`, `chi_*`, `valence`, `pinned_range`),
  which the composition references.
- `lat` — `gradients`, `geometry`, `FJC_choices`.
- `newton` — solver settings that determine how the resulting field is converged.

---

## Weak/multistate charges (`state`, `reaction`)

A segment type (`mon`) can carry **internal states** that interconvert through
**reactions** with equilibrium constants `K = 10^-pK`. This is how PySFBox
models weak (annealed) charges — a weak acid `A = {AH, AM}`, water
`W = {H2O, H3O, OH}`, a titrating surface silanol `SiO = {SIOH, SIO}`. States
are annealed: locally equilibrated with the fields at every lattice site, so
the degree of ionisation `alpha_s(z)` varies through the profile.

The propagators never see the states: a multistate mon `X` propagates with the
**annealed weight** `G_X = Σ_s alphabulk_s · exp(-u_s)`; the local state
fraction `alpha_s(z) = alphabulk_s · exp(-u_s) / G_X` splits the segment
density afterwards (`phi_s = alpha_s(z)·phi_X`).

Available in **both** the public and dev trees (`reactions.py`; `_Species` in
`system.py`). Charged systems require `gradients : 1` (weak charges on 2D/3D
lattices raise).

### `state` block — one internal state of a mon

```
state : AM : mon : A
state : AM : valence : -1
state : H3O : mon : W
state : H3O : alphabulk : 1e-7
```

| param | values | default | meaning |
|---|---|---|---|
| `mon` | an existing `mon` name | — (required) | the segment type this state belongs to |
| `valence` | real | `0.0` | state charge in units of `e` (this is what carries the charge) |
| `alphabulk` | real | unset | anchors the bulk fraction of this state; unset states are solved from the reactions |
| `chi_Y` / `chi-Y` | real | inherit mon's | per-state χ override with partner `Y` (a stateless mon or another state) |

States accept **only** those parameters. Anything else — notably `epsilon` or
`e.psi0/kT` — **raises** `ValueError` ("states take mon, valence, alphabulk,
and chi_X only"): a state uses its parent mon's `epsilon`, and states cannot be
frozen or held at fixed potential.

State-level errors (all in `System`):

| condition | raises |
|---|---|
| `state` with no `mon` | `ValueError` "no 'mon' given" |
| `mon` not defined | `ValueError` "unknown mon" |
| state name equals a `mon` name | `ValueError` "name collides with a mon" (names must differ, as in Namics) |
| state on a `freedom : frozen` mon | `ValueError` "states on frozen mons are not allowed" (use a pinned mon) |
| χ between a mon and its **own** state | `ValueError` |
| a mon with **exactly one** state | `NotImplementedError` (the single-state corner is inconsistent in Namics too — give ≥2 states or drop the block) |

If a mon has states, any **mon-level `valence`** on it is ignored (states carry
the charge): PySFBox prints a one-time warning and zeroes it. `epsilon` stays
mon-level.

### `reaction` block — one equilibrium

```
reaction : weak : equation : 1(AH) + 1(H2O) = 1(AM) + 1(H3O)
reaction : weak : pK : 7
reaction : auto : equation : 2(H2O) = 1(OH) + 1(H3O)
reaction : auto : pK : 14
```

| param | values | default | meaning |
|---|---|---|---|
| `equation` | grammar below | — (required) | mass-action reaction over state names |
| `pK` | real | — (required) | `-log10 K`; the equilibrium is `Π_products α^ν / Π_reactants α^ν = 10^-pK` |

**Equation grammar.** Terms are `ν(StateName)` where the integer coefficient
`ν` acts as a **power** on that state's site fraction (`2(H2O)` contributes
`α_H2O²`). Terms are joined by `+`, and the two sides are separated by exactly
one `=`. Whitespace is free (`1(AH)+1(H2O)` and `1 ( AH ) + 1 ( H2O )` parse
the same). Anything that is not a clean sum of `ν(name)` terms is rejected
rather than silently dropped.

Reaction-level errors:

| condition | raises |
|---|---|
| no `equation` / no `pK` | `ValueError` |
| not exactly one `=`, or an unparseable side | `ValueError` |
| a state name not defined anywhere | `ValueError` "unknown state" |
| `reaction` blocks but **no** `state` blocks | `ValueError` |
| net stoichiometry per mon ≠ 0 (unbalanced) | `NotImplementedError` — the bulk ionisation would then couple to composition; "use the C++ Namics" |
| net valence change ≠ 0 (not electroneutral) | `ValueError` |

### The bulk solve

At every System build (once per calculation / `var` step — the Namics cadence)
PySFBox solves all bulk state fractions `alpha_s^b` to **machine precision** via
a log-space Newton iteration (`solve_bulk_alphas`). The equations are one
normalisation `Σ_s alpha_s = 1` per multistate mon plus one equilibrium per
reaction:

```
Σ_products ν·ln(alpha) − Σ_reactants ν·ln(alpha) = −pK·ln(10)
```

This is the **site-fraction** convention (water activity is included as a
regular term — `1(H2O)` etc.), validated against the compiled Namics (which
instead pre-solves iteratively to ~1e-6 relative). The count of free
(un-anchored) states must equal `#mons + #reactions`; supply exactly enough
`state : X : alphabulk` anchors to close the system, or it raises
(`ValueError`, "give exactly enough 'state : X : alphabulk' anchors … e.g. one
for H3O to set the pH"). A singular system or non-convergence also raise with
actionable messages.

### pH and pK are in lattice units (not molar)

PySFBox has **no `pH` keyword**, and — this is the part that trips people up —
pH, pK, and every `alphabulk` are **lattice (site-fraction) quantities, exactly
as in sfbox and Namics, not molar.** pH is set implicitly by anchoring the
hydronium (`H3O`) bulk fraction:

```
"pH" = −log10( alphabulk_H3O )      (alphabulk_H3O is a bulk SITE FRACTION)
```

A site fraction is a volume fraction; it converts to molarity through the
lattice site density

```
D = 1 / (N_A · bondlength³)   ≈ 61.5 M  for the default bondlength 3e-10 m
                             (scales as bondlength⁻³)
```

so a molar concentration `c` corresponds to the site fraction `c / D`. Two
consequences you must respect:

- `−log10(alphabulk_H3O)` is a **lattice** pH. The real (molar) pH is
  `pH_molar = −log10(alphabulk_H3O) − log10 D` (subtract ≈ 1.79 for a 3 Å bond).
  So `alphabulk : 1e-7` is "pH 7" **only in lattice units** — in molar terms it
  is pH ≈ 5.2.
- **pK must be entered in the same units.** Convert a molar pK by adding
  `log10 D` once for each dissolved ion the equilibrium nets out: a monoprotic
  acid `AH + H2O = A⁻ + H3O⁺` uses `pK_lattice = pKa_molar + log10 D`; water
  autoionization `2 H2O = H3O⁺ + OH⁻` uses `pKw_lattice = pKw_molar + 2·log10 D`.

Worked, bondlength 3 Å (`D ≈ 61.5 M`, `log10 D ≈ 1.79`):

| real / molar value | lattice value you actually enter |
|---|---|
| pH 4.5 → `alphabulk_H3O` | `5.14e-7` |
| water `pKw = 14` | `17.58` |
| carboxylic `pKa ≈ 4.76` | `6.55` |

The `pK : 14` and `alphabulk : 1e-7` in the examples on this page are
illustrative **lattice-unit** numbers (a lattice pKw of 14 and a lattice pH of
7); for a quantitative titration, convert your molar pH and pK as above.
PySFBox uses the lattice-unit pK directly (no activity correction).

**State fraction vs volume fraction.** You set the pH through
`alphabulk_H3O`, which is the bulk fraction of the water segment that is in the
`H3O` state — a **state fraction** (Σ over a mon's states = 1), *not* the bulk
volume fraction of H3O. The actual bulk H3O volume fraction is
`phibulk_H3O = phibulk_water · alphabulk_H3O` (available as the `phibulk_H3O`
output). Anchoring the *state* fraction is deliberate: the acid ionization
responds to the ratio `α_H2O / α_H3O`, in which `phibulk_water` cancels, so
`alphabulk_H3O` fixes the degree of ionization regardless of how much water the
bulk holds. A pH electrode, though, reads `−log10[H3O⁺]`, so the real molar pH
carries the solvent-dilution term as well:

```
pH_molar = −log10(alphabulk_H3O) − log10(phibulk_water) − log10 D
```

When the bulk is essentially pure water (`phibulk_water ≈ 1`, the usual case)
the middle term vanishes and `alphabulk_H3O ≈ phibulk_H3O`; it only matters when
a co-solvent, high salt, or crowding displaces water in the **bulk**.

Titrate by scanning the anchor (exponential scale — `steps` means steps
*per decade*):

```
var : state-H3O : scan : alphabulk
var : state-H3O : scale : exponential
var : state-H3O : steps : 2
var : state-H3O : end_value : 1e-5
```

### Outputs

State-resolved quantities are exposed on the parent mon and on the `state`
object (see the output-properties section for the full tables):

- `kal : mon : X : alphabulk_S | valence_S | phibulk_S | theta_S | theta_exc_S`
  (with `phibulk_S = phibulk_X · alphabulk_S`)
- `kal : state : S : alphabulk | valence`
- `kal : mol : M : mu-S` — per-state chemical potential `Mu + ln(alphabulk_s)`,
  chainlength-1 molecules only
- `pro : mon : X : phi-S | alpha-S | u-S` — state density / local fraction /
  potential profiles

### Example (bulk water + weak acid at pH 7)

```
mon : W : freedom : free
mon : A : freedom : free
state : H2O : mon : W
state : H2O : valence : 0
state : H3O : mon : W
state : H3O : valence : 1
state : OH  : mon : W
state : OH  : valence : -1
state : AH  : mon : A
state : AH  : valence : 0
state : AM  : mon : A
state : AM  : valence : -1
reaction : weak : equation : 1(AH) + 1(H2O) = 1(AM) + 1(H3O)
reaction : weak : pK : 7
reaction : auto : equation : 2(H2O) = 1(OH) + 1(H3O)
reaction : auto : pK : 14
state : H3O : alphabulk : 1e-7      // pH 7 in lattice units (see above)
```

---

## System (`sys`) and solver (`newton`)

The `sys` block holds whole-system settings (initial guess); the `newton`
block controls the SCF solver (tolerances, iteration budget, method, engine).
Both are optional — with neither block present, PySFBox cold-starts and runs
its default solver cascade with the defaults below.

### `sys : <name> : ...`

The system name is a label, not a functional key (Namics default: `NN`).
PySFBox is name-agnostic: system-level output properties are looked up by the
`sys` key regardless of the name, so `kal : sys : NN : grand_potential` and any
other name resolve the same. Use the name your output-spec lines reference.

```
sys : NN : initial_guess : polymer_adsorption
```

| param | values | default | meaning |
|---|---|---|---|
| `initial_guess` | `polymer_adsorption`, `membrane`, `micelle`, `previous_result`, `none` | `none` | starting potentials for the **first** calculation only |
| `overflow_protection` | `true` / `false` | (ignored) | accepted for Namics input compatibility, but a **no-op** — see below |

**`initial_guess`.** Applied only to the first calculation of a run (exactly
like Namics: after the first solve the type resets to `previous_result`, and
`var`/`start` steps warm-start from the previous solution).

- `polymer_adsorption` — analytic adsorption seed: `u = -lambda*chi` at sites
  adjacent to each frozen (solid) mask. Honored on 1-gradient lattices.
- `membrane` / `micelle` — `u = -log(1.8)` on the first `4*fjc` interior
  layers for segments that repel the solvent (`chi_solvent > 0.8`). If no such
  segment exists, a note is printed and the guess is a no-op (as in Namics).
- `previous_result` / `none` (or omitted) — cold start (potentials from zero,
  or the warm-started previous solution inside a scan).
- On N-D lattices (`gradients > 1`) the analytic guesses are skipped (the run
  cold-starts; scans still warm-start).
- **Raises `NotImplementedError`:** any other value — in particular reading a
  guess **from a file**, and `membrane_torus`. Message names the supported
  set and points at "the C++ Namics".

**`overflow_protection`.** PySFBox uses log-domain / renormalised propagation
everywhere (long and strongly-adsorbing chains cannot overflow), so no overflow
guard is needed. The key is accepted and ignored — put it in inputs shared with
the compiled Namics (whose recent builds nag without it) at no cost here.

### `newton : <name> : ...`

```
newton : NN : method : pseudohessian
newton : NN : tolerance : 1e-7
newton : NN : deltamax : 0.1
```

| param | values | default | meaning |
|---|---|---|---|
| `tolerance` | float | `1e-7` | convergence threshold on `max|g|` (the residual sup-norm) |
| `iterationlimit` | int | `1000` | per-stage iteration budget |
| `deltamax` | float | `0.1` | trust-region cap: max change in any potential per Newton step |
| `m` | int | `8` | Anderson history depth (used by the Anderson fallback / `DIIS`/`Picard`/`LBFGS`) |
| `method` | see table | `pseudohessian` | primary solver (see below) |
| `engine` | see table | `numpy` | residual/kernel backend |

Unknown `method` or `engine` values do **not** raise: a note is printed and
PySFBox falls back to the default cascade / NumPy residual. This keeps
Namics inputs that name a solver PySFBox does not implement runnable.

#### `method`

| value | maps to | notes |
|---|---|---|
| `pseudohessian` | pseudohessian quasi-Newton (primary) | **default**; licensed translation of Namics `sfnewton.cpp` (Scheutjens variable-metric), Namics-like iteration counts |
| `hessian` | full Hessian every iteration | Namics full-Hessian mode; faithful but usually slower (the trust region, not Hessian quality, limits stiff cases) |
| `DIIS`, `Picard`, `LBFGS` | Anderson-accelerated Picard | Namics alternative methods all map onto PySFBox's Anderson mixing (type II) |

#### `engine`

| value | notes |
|---|---|
| `numpy` (`np`) | **default**; the pure-NumPy residual and solver kernels |

#### Solver cascade (brief)

`newton` values feed `System.solve`, which tries stages in order and keeps the
best iterate; a propagator death (`FloatingPointError`) in one stage falls
through to the next:

0. **Pseudohessian primary** from the guess, budget `iterationlimit` — for
   small problems (`n_var <= 4000`) and default methods.

1. **Anderson** (type II, with a 200-iteration damped-Picard warmup on cold
   starts; adaptive Tikhonov regularisation) — the fallback, the primary for
   non-default `method` values, and the primary for large problems (`n_var >
   4000`) in the public tree.
2. **Extended-budget pseudohessian** from the original guess
   (`max(iterationlimit, 3000)`), for `n_var <= 4000`.
3. **One-shot full-Hessian anchor** (gated on `max|g| < 0.5` and
   `n_var <= 1000`).

Non-convergence raises `RuntimeError` with actionable advice ("try a smaller
`deltamax`"); converging to a negative solvent bulk fraction also raises (the
restricted `theta` values exceed what an equilibrium box can hold).

#### Super-iteration (search) sub-options

When a `var : ... : search` block drives an outer scalar root-find, its inner
tolerances are read from the same `newton` block (each SCF residual is a full
warm-started solve). These are inert without a `search` block.

| param | default |
|---|---|
| `super_tolerance` | `10 * tolerance` |
| `super_deltamax` | `0.5` |
| `super_iterationlimit` | `max(iterationlimit/10, 30)` |

---

## Output (`output`, `kal`, `pro`) and what it means

PySFBox writes two Namics-compatible output files next to each input: a
tabular `.kal` (scalar values, one row per calculation/var step) and one or
more `.pro` profile dumps (arrays over the lattice). Output is driven by two
kinds of line:

- **Declaration lines** `output : <file> : <param> : <value>` — set per-file
  options.
- **Spec lines** `kal : <key> : <name> : <prop>` and
  `pro : <key> : <name> : <prop>` — request one scalar (kal) or one profile
  (pro).

A `.kal` file is written whenever at least one `kal :` spec line is present;
a `.pro` file whenever at least one `pro :` spec line is present. (Unlike
sfbox, PySFBox does not require the `output :` declaration to *register* the
file — it keys off the `kal`/`pro` spec lines directly — but keep the
declaration for Namics compatibility; see the ordering gotchas.)

```
output : kal : append : false
output : pro : append : false
kal : sys : NN : grand_potential
kal : mol : brush : GN
pro : mol : brush : phi
pro : sys : NN : psi
```

### `output` declaration parameters

| param | values | default | meaning |
|---|---|---|---|
| `write_bounds` | true / false | false | On `pro` only: also emit the `fjc` ghost (boundary) layers, filled per the boundary condition (Namics `LGrad1::PutProfiles`). false = interior layers only. Ignored for N-D (2D/3D) lattices. |
| `append` | true / false | — | **Accepted but never read.** PySFBox always writes a fresh `.kal` per run (see file semantics below) and a fresh `.pro` per dump. Kept only so Namics-style inputs parse; the value has no effect. |

### Spec-line details

- **`*` wildcard** — `kal : mol : * : phi` / `pro : mon : * : phi` expands to
  every molecule / every monomer in definition order (Namics behaviour).
- **Repeated props** — the same `key:name` may carry several props on separate
  lines (e.g. `kal : mon : A : 1st_M_phi_z` then `kal : mon : A : 2nd_M_phi_z`);
  each becomes its own column, in input order.
- **Alias echo** — any prop ending in `-value` whose prefix is a defined
  `alias` returns that alias's numeric value (int if integral, else real). Use
  it to echo a scanned alias into the `.kal`, e.g. `kal : sys : NN : chi-value`.
- **Unknown props** — a kal prop with no match prints the literal `NiN` and a
  `warning: kal property ... unknown -> NiN` line; an unknown pro prop is
  skipped with a warning. This mirrors Namics exactly.

### File formats

| file | header | value format | coordinate |
|---|---|---|---|
| `.kal` | tab-joined `key:name:prop` labels, once | ints `%d`, reals `%.16e`, unknown `NiN` | — |
| `.pro` (1-gradient) | `x` + tab-joined labels | reals `%.20g` | leading `x` column = cell-centre position `(x − fjc + 0.5)/fjc` in bond lengths (i.e. `z − 0.5` at `fjc = 1`); coord `%e` |
| `.pro` (2D/3D) | `x y[ z]` + labels | reals `%.20g` | one leading column per gradient (Cartesian/radial in bond lengths); one row per interior cell in C order; coord `%e` |

`.kal` semantics: the file is created fresh (overwrite) on the first
calculation of a run; the header is written once and every subsequent
calculation or var step **appends** one row. Running the same input again
overwrites (the `append` setting is not consulted).

### File numbering (`.pro`)

`.pro` files are numbered exactly like Namics (`j` = 0-based var-step index,
`start` = 1-based `start`-block index):

| condition | filename |
|---|---|
| single start, no `var` scan | `base.pro` |
| single start, with `var` scan | `base_<j>.pro` |
| multiple starts, no `var` | `base_<start>.pro` |
| multiple starts, with `var` | `base_<start>_<j>.pro` |

### Ordering gotchas (Namics compatibility)

- Put every `output : <file> : ... : ...` declaration **before** any spec line
  for that file (Namics registers output keywords dynamically from the
  declarations; a spec before its declaration is a Namics load error).
- Within `.kal`, a `kal : mol : ...` or `kal : mon : ...` line must precede any
  `kal : sys : <name> : ...` line, or Namics raises "Error in Load() in output".
- `kal : lat : ...` — PySFBox accepts it, but the current Namics build does
  not; leave it out of inputs meant to run in both engines.
- The default system name is `NN` in the current Namics (older/teaching inputs
  use `noname`); the `<name>` in `sys : <name> : ...` must match your system.

### `kal` scalar properties (`System.get_value`)

Types: `int` → `%d`, `real` → `%.16e`, no match → `NiN`.

| key | prop | type | meaning |
|---|---|---|---|
| `lat` | `n_layers` | int | number of interior layers (1-D `MX`, or ∏ dims for N-D) |
| `lat` | `volume` | real | total lattice volume (Σ site volumes) |
| `sys` | `grand_potential` | real | grand potential Ω (per unit area / normalised as in Namics) |
| `sys` | `free_energy`, `free_energy(po)` | real | Helmholtz free energy F (both spellings map to F; `(po)` is Namics' Ω + Σnμ route to the same value) |
| `sys` | `iterations` | int | solver iteration count (solver-dependent; ignored in regression diffs) |
| `sys` | `residual` | real | final residual norm (solver-dependent) |
| `state` | `alphabulk` | real | bulk fraction of this internal state |
| `state` | `valence` | real | state valence (charge) |
| `mol` | `theta` | real | total amount Σ L·φ of the molecule |
| `mol` | `theta_exc` | real | excess amount θ − V·φ_bulk |
| `mol` | `phibulk` | real | bulk volume fraction |
| `mol` | `Mu`, `MU`, `mu` | real | chemical potential μ |
| `mol` | `mu-<state>` | real | per-state μ = μ + ln(α_bulk,state), for chain-length-1 molecules only |
| `mol` | `n` | real | number of molecules θ/N |
| `mol` | `N`, `chainlength` | int | chain length (segments) |
| `mol` | `GN` | real | single-chain partition function G_N |
| `mol` | `phiMax` | real | max φ over interior layers |
| `mol` | `phiMin` | real | min φ over interior layers |
| `mol` | `phiM` | real | φ at the last interior layer (the bulk/reservoir side) — the Namics `phiM`, **not** the maximum |
| `mon` | `theta` | real | total amount of this monomer |
| `mon` | `theta_exc` | real | excess amount θ − V·φ_bulk |
| `mon` | `phibulk` | real | bulk volume fraction of the monomer |
| `mon` | `chi_<X>`, `chi-<X>` | real | Flory χ between this monomer and monomer `X` |
| `mon` | `1st_M_phi_z` | real | first moment ⟨z⟩ of the excess profile (M₁/θ_exc) |
| `mon` | `2nd_M_phi_z` | real | second moment M₂/θ_exc |
| `mon` | `RMS` | real | √(second moment) |
| `mon` | `fluctuations` | real | √(M₂/θ_exc − ⟨z⟩²) |
| `mon` | `alphabulk_<state>` | real | per-state bulk fraction, pushed on the parent monomer |
| `mon` | `valence_<state>` | real | per-state valence |
| `mon` | `phibulk_<state>` | real | per-state bulk volume fraction |
| `mon` | `theta_<state>` | real | per-state total amount |
| `mon` | `theta_exc_<state>` | real | per-state excess amount |
| any | `<alias>-value` | int/real | value of the named `alias` (echo of a scanned parameter) |

### `pro` profile properties (`System.get_profile`)

| key | prop | meaning |
|---|---|---|
| `mol` | `phi` | volume-fraction profile of the molecule |
| `mon` | `phi` | volume-fraction profile of the monomer |
| `mon` | `u` | self-consistent potential field u(z) of the monomer |
| `mon` | `phi-<state>` | per-state density profile (parent-monomer output) |
| `mon` | `alpha-<state>` | per-state degree-of-dissociation profile α(z) |
| `mon` | `u-<state>` | per-state potential field |
| `sys` | `alpha` | incompressibility (Lagrange) field α(z) |
| `sys` | `psi` | electrostatic potential ψ(z) — charged systems only |
| `sys` | `q` | charge-density profile q(z) — charged systems only |

`psi`/`q` return nothing (skipped with a warning) on a neutral system. With
`write_bounds : true`, potentials/intensive fields (`psi`, `u`, `alpha`,
`u-`, `alpha-`) get mirror-filled ghosts; densities (`phi`, `q`) get
surface-zero or mirror ghosts, with a frozen wall segment overriding its
ghost to the wall density.

### What the output means

- **theta (θ)** — total amount of a species, Σ over layers of site-volume ×
  volume fraction (`Σ L·φ`). For a grafted layer θ = σ·N per unit area.
- **theta_exc** — the *excess* over the bulk reservoir, θ − V·φ_bulk; the
  adsorbed/depleted amount. It is what the moments normalise by.
- **phibulk (φ_bulk)** — the volume fraction far from any surface (the
  reservoir composition); 0 for a strictly grafted/pinned molecule.
- **GN** — the single-chain partition function in the converged field;
  n = θ/N = φ_bulk·GN/N for a free chain, and GN = V for an ideal chain
  (the normalisation anchor).
- **grand_potential (Ω)** — the surface/interfacial grand potential; the
  quantity minimised at equilibrium and the natural output for interfacial
  tension and for `search` targets. Excludes frozen-wall χ partners.
- **free_energy (F)** — the Helmholtz free energy; F = Ω + Σ n μ. Keeps the
  full (un-halved) χ against frozen walls, unlike Ω (the two conventions
  differ only in how the frozen-surface coupling is booked).
- **mu (μ)** — the molecular chemical potential; dF/dθ. Drives `search` on
  `mol : ... : mu` and equals the μ from the exchange/free-energy accounting.
- **psi (ψ)** — the dimensionless electrostatic potential (units of kT/e) from
  the lattice Poisson solve; ≈ 0 over neutral regions.
- **q** — the local charge density feeding the Poisson equation.
- **moments** (`1st_M_phi_z`, `2nd_M_phi_z`, `RMS`, `fluctuations`) — moments
  of the *excess* profile measured from the first layer, normalised by
  θ_exc: `1st_M_phi_z` is the mean position ⟨z⟩ (brush/layer height proxy),
  `2nd_M_phi_z` the mean-square position, `RMS = √M₂`, and `fluctuations` the
  standard deviation √(⟨z²⟩ − ⟨z⟩²) (layer width).

---

## Scans, search, ranges, and aliases (`var`, `search`, ranges, `alias`)

PySFBox drives parameter sweeps (`var` scans), constraint-based root-finds
(`search`/super-iteration), spatial masks (frozen/pinned ranges), and text
substitution (`alias`) from the Namics keywords. Scans and searches are driven
from `runner.py` (`_var_roles`, `_var_plan`, `_regula_falsi`), which rebuilds the
`System` for each step; frozen/pinned ranges are parsed in
`model.py`/`latticend.py` (`_set_range`/`parse_range`) and aliases in
`inputreader.py` (`substitute_aliases`).

### The `var` block

A `var` line names its target in the **second** field as `<item>-<name>`, and
the **third** field is the role:

```
var : <item>-<name> : <role> : <value>
```

`<item>-<name>` is the settings block to change, e.g. `mol-poly`, `lat-flat`,
`sys-NN`, `mon-P`, `state-H3O`. (Namics writes a stray space, `lat- flat`; it is
stripped.) A single block may carry more than one role. PySFBox recognises
these roles: `scan` (a sweep), `search` (a super-iteration variable), and a
**target property** (paired with a search). At most **one** search and **one**
target are allowed per calculation — extra ones raise; multiple `scan` blocks do
not raise, but only the **last** one is used.

### `var` scans

Mark a block as a scan with `scan : <parameter>`; the swept parameter's name is
the *value* of `scan`. The **start value is the parameter's current setting**
(from the ordinary keyword line, defaulting to 0 if unset), not given in the
`var` block; the scan runs from there to `end_value` inclusive.

```
mol : poly : theta : 500
var : mol-poly : scan : theta
var : mol-poly : step : -50
var : mol-poly : end_value : 100
```

| param | values | default | meaning |
|---|---|---|---|
| `scan` | a parameter name | — (required) | marks the block a scan; value = the parameter to sweep (`theta`, `phibulk`, `n`, `n_layers`, `chi_X`, `alphabulk`, …, or `NAME-value` for an alias) |
| `step` | real (may be negative) | `1` | linear increment per step |
| `end_value` | real | — (required) | last value of the sweep |
| `scale` | `linear` / `exponential` | `linear` | `linear` = constant step; `exponential` = log-spaced (see below) |
| `steps` | int | `1` | **exponential only**: number of steps per decade |

- **Linear** produces `round((end − start)/step) + 1` values (both ends
  included). If `step` has the wrong sign for the direction, the count collapses
  to a single step at the start value.
- **Exponential** (`scale : exponential`) interpolates in `log10`; `steps` is the
  number of steps **per decade**, so it takes as many steps from 10→100 as from
  100→1000. Both start and `end_value` must be positive (else it raises). This is
  the idiom for `state` `alphabulk` titrations, e.g. `steps : 12` over 6 decades
  gives 73 rows.
- The `var` block may carry sfbox's `type : integer/real` and `output_name`
  params; PySFBox **ignores** them (integer vs real is auto-detected;
  `.pro` files are numbered by step index, not renamed).
- Any settings key/name can be scanned (`set_value` writes it back before each
  rebuild). Each scan step is **warm-started** from the previous converged
  solution; `n_layers` scans additionally remap the potential onto the new grid.
  One `.kal` row is written per step; `.pro` files are numbered like Namics.

### Scanning an alias

A scan can drive an `alias` value. Two equivalent forms:

```
var : alias-N : scan : value       // block named alias-N, sweep its value
var : mol-pol : scan : N-value      // any block; "NAME-value" sweeps alias NAME
```

Both write `alias : N : value : <v>` before each rebuild, so the alias's
`#N#` substitutions (see below) update per step. The current alias value is
available as an output property `NAME-value` (e.g. `kal : mol : pol : N-value`).

### `search` / super-iteration

A `search` block turns a molecule quantity into the unknown of an **outer
scalar root-find**: it adjusts the search variable until a paired **target
observable** hits its requested value. Each residual evaluation is a full,
warm-started SCF solve.

```
var : mol-pol : search : theta            // the search variable
var : sys-NN  : grand_potential : -0.010  // the target
```

A mol may target one of *its own* quantities in the same block, e.g. search
`theta` to hit a target `phibulk`:

```
var : mol-pol : search : theta
var : mol-pol : phibulk : 0.1
```

**Search variable** (must be a molecule):

| `search : <prop>` | notes |
|---|---|
| `theta` | grafted/adsorbed amount; linked to `n` (setting one clears the other) |
| `n` | number of chains; linked to `theta` |
| `phibulk` | bulk volume fraction |

Any other item (`lat`/`mon`/`state`/`reaction` search) or other property raises
`NotImplementedError`. The molecule must **declare an initial value** of the
searched quantity (the search starts from it; `theta` may be implied by
`n × chainlength`).

**Target observable:**

| item | `<prop>` | meaning |
|---|---|---|
| `sys` | `grand_potential` | drive Ω to the value |
| `sys` | `free_energy` | drive F to the value |
| `mol` | `mu` | drive chemical potential to a **numeric** value |
| `mol` | `theta` / `n` / `phibulk` | drive that molecule quantity to the value |

Rejected (raise): `sys : Laplace_pressure` (needs the `sys:constraint:delta`
membrane-balance machinery PySFBox lacks); a `mu` target naming *another*
molecule (`eq_to_mu` equilibration) or the equate-to-solvent / balance-membrane
searches. A `search` without a target, or a target without a `search`, raises a
"lonely" error.

**Super-iteration controls** (read from the `newton` block):

| param | default | meaning |
|---|---|---|
| `super_tolerance` | `10 × tolerance` | target-error tolerance |
| `super_deltamax` | `0.5` | base growth factor for the bracket expansion |
| `super_iterationlimit` | `max(iterationlimit // 10, 30)` | max super-iterations (each = one full SCF) |

The outer solver is a robust **bracket-then-Illinois false position**
(`_regula_falsi`), deliberately replacing Namics' bare damped secant: it expands
a positive-definite bracket geometrically (the search variables are positive),
then does Illinois-weighted false position. The root is unique, so it lands on
the same solution; only the path differs. Failure to bracket, or a
non-monotone/unreachable target, raises with actionable advice.

### Frozen / pinned ranges

Ranges pin a `mon`'s segments to lattice sites; a segment with
`freedom : frozen` or `freedom : pinned` **requires** a range (else it raises).
Give it with `mon : X : frozen_range : ...` or `mon : X : pinned_range : ...`
(cross-ref `mon`).

**1-gradient** tokens (`model.py:_set_range`; coordinates are 1-based *physical*
layers, auto-expanded to refined sites when `fjc > 1`; a trailing `;` is
tolerated):

| token | meaning |
|---|---|
| `lowerbound` | the lower ghost layer (frozen wall at the low end) |
| `upperbound` | the upper ghost layer (frozen wall at the high end) |
| `firstlayer` (or `first_layer`) | interior layer 1 |
| `lastlayer` (or `last_layer`) | interior layer MX |
| `lo;hi` | interior layers `lo` through `hi`, inclusive |

```
mon : S  : freedom : frozen
mon : S  : frozen_range : lowerbound
mon : pp : freedom : pinned
mon : pp : pinned_range : 5;5
```

**2D/3D box grammar** (`latticend.py:parse_range`; available for whichever
geometries the N-D lattice supports — cross-ref `lat`/`mon`):

- `lowerbound` / `upperbound` — a wall on the first axis' low/high face.
- `xlo,ylo[,zlo];xhi,yhi[,zhi]` — an inclusive box in 1-based per-axis interior
  coordinates. Each corner must supply `gradients` coordinates (else it raises);
  omitting the second corner (`;xhi,…`) selects a single layer.
- A one-layer-thick box that spans a **full** boundary face is treated as a wall
  face (it fills the ghost hyperplane); a **partial** patch stays an interior
  mask (it will *not* fake solid contact along the whole boundary).

### `alias`

```
alias : N : value : 100
```

defines the substitution `#N#`. `substitute_aliases` replaces every `#N#`
occurrence in **composition strings** (`mol : X : composition`) and in the
lattice **`n_layers`** fields (`n_layers`, `n_layers_x/y/z`) before parsing, so
one alias can size the box and the chain at once:

```
alias : N : value : 100
mol : pol : composition : (X)1(A)#N#(G)1
```

Combined with a scan (`scan : N-value`), the alias — and every field that
references it — updates each step; the value is also exposable as the output
property `N-value`.

---

## Example inputs

Three small, runnable inputs adapted from the regression suite
(`tests/*.in`); the inline `//` comments use the comment syntax Namics accepts.
Each is a single self-contained calculation; run any of them with
`python -m pysfbox <file>.in`, which writes the `.kal`/`.pro` files next to the
input. All features shown are in both the public and dev trees.

Every input follows the same block order: `lat` (geometry), `mon` (segment
types and their χ/charge/freedom), `mol` (molecules built from segments),
`newton` (solver settings), then `output`/`kal`/`pro` (what to write). Names
after the keyword (`flat`, `S`, `pol`, `isaac`, …) are user-chosen labels; the
default system name is `NN` (the `sys` name is not enforced, so `noname` also
works).

### 1. Grafted brush in a Θ-solvent

A polymer end-grafted to a wall, swollen against solvent. The chain is pinned
to the first layer by a single `X` segment; `restricted` + `theta` fixes the
grafted amount (surface coverage). `chi_W : 0.5` makes the solvent a Θ-solvent.
From `tests/brush_in_theta_solvent.in`:

```
lat : flat : n_layers : 500              // 500-layer 1D grid
lat : flat : lattice_type : simple_cubic // step weights λ±1=1/6, λ0=2/3
lat : flat : geometry : planar           // flat (Cartesian) lattice
lat : flat : gradients : 1               // 1 gradient direction
lat : flat : lowerbound : surface        // solid wall at z=0 (ghost density 0)

mon : S : freedom : frozen               // the wall material
mon : S : frozen_range : lowerbound      // pinned into the lower ghost layer

mon : W : freedom : free                 // solvent segment
mon : A : freedom : free                 // brush backbone segment
mon : G : freedom : free                 // free chain end
mon : X : freedom : pinned               // the grafting segment...
mon : X : pinned_range : firstlayer      // ...confined to layer 1 (the graft point)

mon : X : chi_W : 0.5                     // Flory χ of each segment with solvent W
mon : A : chi_W : 0.5                     // χ = 0.5 ⇒ Θ-solvent
mon : G : chi_W : 0.5

mol : pol : composition : (X)1(A)200(G)1 // chain: 1 graft + 200 backbone + 1 end
mol : pol : freedom : restricted         // fixed amount, floating bulk (grafted)
mol : pol : theta : 0.5                  // grafted coverage θ = 0.5

mol : water : freedom : solvent          // fills the remaining volume (1−Σφ)
mol : water : composition : (W)1

output : kal : append : false            // overwrite the .kal each run
kal : mol : pol : theta                  // scalar outputs: coverage,
kal : mol : pol : n                      //   number of chains,
kal : mol : pol : chainlength            //   N,
kal : mon : G : 1st_M_phi_z              //   1st/2nd moments of the end density,
kal : mon : G : 2nd_M_phi_z
kal : mon : G : fluctuations             //   brush-height fluctuation

var : mol-pol : scan : theta             // sweep coverage θ = 0.5, 1.0, …, 20
var : mol-pol : step : 0.5               //   (warm-started; one .kal row per step)
var : mol-pol : end_value : 20
```

Computes the brush density profile φ(z) at each coverage; the `.kal` records
how the brush height (moments of the chain-end density) grows with θ. Drop the
three `var` lines for a single-coverage run.

### 2. Homopolymer adsorption from bulk

A strongly wall-attracted homopolymer adsorbing from a dilute solution. The
chain is `free` with a fixed `phibulk`, so it exchanges with a reservoir; the
`chi_Si : -6` makes segment `A` stick to the frozen wall `Si`. From
`tests/adsorption_small.in`:

```
lat : flat : n_layers : 100
lat : flat : lattice_type : simple_cubic
lat : flat : geometry : planar
lat : flat : gradients : 1

mon : Si : freedom : frozen              // wall material
mon : Si : frozen_range : 1;1            // explicit layer range (layer 1 only)
mon : A : freedom : free                 // polymer segment
mon : W : freedom : free                 // solvent segment
mon : A : chi_Si : -6                    // strong A–wall attraction (adsorption)

mol : water : composition : (W)1
mol : water : freedom : solvent

mol : pol : composition : (A)200         // N = 200 homopolymer
mol : pol : freedom : free               // exchanges with a bulk reservoir
mol : pol : phibulk : 1e-4               // reservoir volume fraction 1e-4

output : pro : append : true             // z-resolved profiles
pro : mol : pol : phi                    // φ_pol(z)
pro : mol : water : phi                  // φ_water(z)

kal : mol : pol : theta_exc              // surface excess Γ = Σ(φ−φbulk)
kal : sys : noname : grand_potential     // Ω (adsorption free energy)
kal : sys : noname : iterations          // solver iteration count

newton : isaac : deltamax : 0.1          // max field step (stabilises the stiff χ=-6)
newton : isaac : iterationlimit : 5000
```

Computes the adsorbed layer: `theta_exc` is the surface excess Γ, and the `.pro`
gives the decaying φ_pol(z) profile. The strong χ makes this a stiff solve; the
small `deltamax` keeps the field step bounded.

### 3. Polyelectrolyte adsorption with salt

A strong polyelectrolyte (charged backbone) adsorbing on a neutral wall, with
added salt and an automatic neutralizer. This exercises the electrostatics
block: per-segment `valence` and `epsilon` (relative permittivity, default 80),
the free-ψ Poisson branch, and `freedom : neutralizer` (a molecule whose bulk
fraction is set each iteration to enforce bulk electroneutrality). From
`tests/polyelectrolyte_wall.in`:

```
lat : flat : gradients : 1
lat : flat : geometry : planar
lat : flat : n_layers : 80
lat : flat : lattice_type : simple_cubic
lat : flat : bondlength : 3e-10          // sets the Poisson/Debye length scale (m)
lat : flat : lowerbound : surface

mon : S : freedom : frozen               // neutral wall
mon : S : frozen_range : lowerbound
mon : S : epsilon : 5                     // wall permittivity
mon : S : chi_A : -2                      // wall attracts the polymer segment A

mon : W : valence : 0                     // water: neutral
mon : W : epsilon : 80                    //   high permittivity
mon : A : valence : -0.5                  // polyelectrolyte segment: charge −0.5 e
mon : A : epsilon : 20
mon : Na : valence : 1                    // cation
mon : Na : epsilon : 20
mon : Cl : valence : -1                   // anion
mon : Cl : epsilon : 20

mol : water : composition : (W)1
mol : water : freedom : solvent
mol : pol : composition : (A)50           // 50-segment strong polyelectrolyte
mol : pol : freedom : free
mol : pol : phibulk : 1e-3
mol : Na : composition : (Na)1
mol : Na : freedom : neutralizer          // bulk φ set by electroneutrality
mol : Cl : composition : (Cl)1
mol : Cl : freedom : free
mol : Cl : phibulk : 0.005                // fixes the salt concentration

newton : isaac : iterationlimit : 2000
newton : isaac : tolerance : 1e-9

output : kal : append : false
kal : mol : * : phibulk                   // '*' = every molecule
kal : mol : * : theta_exc
kal : sys : NN : grand_potential
output : pro : append : false
pro : mol : * : phi                       // φ(z) for every molecule
pro : sys : NN : psi                      // electrostatic potential ψ(z)
pro : sys : NN : q                        // charge density q(z)
```

Computes the diffuse electric double layer and the adsorbed polyelectrolyte:
the `.pro` gives ψ(z) and q(z) alongside the density profiles. The `neutralizer`
Na⁺ has no fixed `phibulk`; its bulk value is solved from bulk electroneutrality
each iteration, so only the salt (`Cl` `phibulk`) and polymer reservoir are set
by hand.
