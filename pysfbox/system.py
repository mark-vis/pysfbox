"""The SCF system: assembles lattice, segments, and molecules from parsed
input; provides the Namics-style residual and the observables used by the
kal/pro output machinery.

Residual (cf. Namics System::Classical_residual): the iteration variable is
the stack of segment potentials u_i(z) for all non-frozen segment types;

    g_i = u_i - u_int_i ;  g_i -= mean_i(g_i) ;  g_i += 1/phi_T - 1

masked to free lattice sites. The incompressibility field alpha is the mean
over components, eliminated analytically inside the residual.
"""

import re

import numpy as np

from . import progress
from .inputreader import get_blocks, last, substitute_aliases
from .lattice import Lattice1D, curved_face_radii
from .latticend import LatticeND
from .model import Molecule, Segment, U_CLIP
from .reactions import Reaction, State, solve_bulk_alphas
from .sfnewton import SFNewton


# warnings already printed this process (avoid per-scan-step spam)
_WARN_NOTES = set()

# physical constants for electrostatics, copied VERBATIM from Namics
# (namics.cpp:45-49) so converged charged states agree with the oracle
E_CHARGE = 1.60217e-19          # elementary charge (C)
K_BOLTZMANN = 1.38065e-23       # J/K
T_ABS = 298.15                  # K
EPS0 = 8.85418e-12              # F/m
K_BT = K_BOLTZMANN * T_ABS


# the characteristic function X (Namics `sys : NN : X`, system.cpp:1104-1206
# parse, :1588-1627 evaluation): the help text Namics prints on a malformed
# definition or on `X : ?`, in its own words
_X_HELP = (
    "X is the characteristic function specified by the user. Examples of "
    "how a characteristic function is defined:\n"
    "  Case 1, no internal states: 'F-molname_1-molname_2-...'\n"
    "  Case 2, internal states: 'F-molname_1-(statename_i,statename_j,n)-...'"
    "\n"
    "F is the Helmholtz energy (sys : free_energy); '-molname' subtracts "
    "n*mu of that molecule; '(statename_i,statename_j,n)' subtracts "
    "n * theta_i * mu_j, with theta_i the amount of state i (sum over sites) "
    "and mu_j the chemical potential of state j, which must belong to a "
    "monomeric molecule (mol : X : mu-statename_j). Add as many entries as "
    "you wish. Example: X : F-water-Na-Cl is canonical in every molecule "
    "not listed and grand in the listed ones.")


def parse_characteristic_X(spec, mol_names, states, monomer_of_state):
    """Parse a Namics characteristic-function definition (`sys : NN : X`).

    spec              the value string, e.g. 'F-water-Na-(PH,H3O,1)'
                      (whitespace is free, as Namics strips it)
    mol_names         the declared molecule names
    states            {state name: State}, every declared state
    monomer_of_state  {state name: [molecule names]}: the MONOMERIC
                      molecules whose segment carries that state

    Returns (mols, state_terms): mols = list of molecule names to subtract
    n*mu for (repeats allowed, as in Namics), state_terms = list of
    (state_i, state_j, molecule of state_j, n). Raises ValueError with the
    Namics help text on every malformed entry (Namics sets success = false
    and stops); `X : ?` prints the help the same way."""
    s = re.sub(r"\s+", "", str(spec))
    sub = s.split("-")
    if sub[0] != "F":
        raise ValueError(
            "sys : X : " + _X_HELP + "\n"
            + ("(help requested with X : ?)" if s == "?" else
               f"Error found: the first item of '{spec}' is not the "
               "expected 'F'"))
    mols, terms = [], []
    for item in sub[1:]:
        if item == "":
            raise ValueError(
                f"In characteristic function X, '{spec}' has an empty entry "
                "(a doubled or trailing '-'?)\n" + _X_HELP)
        parts = item.split(",")
        if len(parts) == 1:
            if item not in mol_names:
                from .inputreader import _suggest
                raise ValueError(
                    f"In characteristic function X, the entry '{item}' is "
                    f"not a molecule name{_suggest(item, mol_names)}\n"
                    + _X_HELP)
            mols.append(item)
            continue
        if (len(parts) != 3 or not parts[0].startswith("(")
                or not parts[2].endswith(")")):
            raise ValueError(
                f"In characteristic function X, the entry '{item}' is not "
                "recognised as '(statename_1,statename_2,n_1)'\n" + _X_HELP)
        si, sj, ns = parts[0][1:], parts[1], parts[2][:-1]
        try:
            n = int(ns)
        except ValueError:
            n = -1
        if n < 0:
            raise ValueError(
                f"In characteristic function X, the entry '{item}' does not "
                "include a non-negative integer at the third argument")
        missing = [nm for nm in (si, sj) if nm not in states]
        if missing:
            raise ValueError(
                f"In characteristic function X, the entry '{item}' is not "
                "coding for '(statename_1,statename_2,n_1)': state name(s) "
                f"{', '.join(missing)} do not exist")
        hosts = monomer_of_state.get(sj, [])
        if len(hosts) != 1:
            # Namics prints "Failed to find chemical potential for state
            # ...: (not a monomer?) ... mu is set to zero" and goes on (and
            # from the second state entry on silently reuses the previous
            # entry's mu: its -999 sentinel is set once, before the loop,
            # system.cpp:1598). A number built on mu = 0 is not X, so
            # PySFBox refuses instead (Invariant 2).
            why = ("no monomeric molecule carries it" if not hosts else
                   "several monomeric molecules carry it ("
                   + ", ".join(hosts) + ")")
            raise ValueError(
                f"In characteristic function X, the entry '{item}': the "
                f"chemical potential of state {sj} is not defined -- {why} "
                "(a state chemical potential mu-<state> exists only for a "
                "state of a single monomeric molecule; Namics would set "
                "mu = 0 here)")
        terms.append((si, sj, hosts[0], n))
    return mols, terms



class _Species:
    """One iteration/interaction species: an ordinary segment type, or one
    internal state of a multistate segment (weak charges). Namics iterates
    states exactly like mons ((itmon+itstate)*M layout); PySFBox promotes
    each state of every non-frozen multistate mon to a full species. NOTE:
    Namics additionally merges species with identical chi rows into shared
    iteration blocks (IsUnique, system.cpp:1246); PySFBox deliberately
    iterates every species separately -- the fixed point is identical, only
    transient trajectories (and iteration counts vs the oracle) differ, and
    the charge-regulation identity alpha_s/alpha_t = (b-ratio)*exp(-dv*psi)
    becomes a genuine convergence test instead of a structural one."""

    def __init__(self, seg, state=None):
        self.seg = seg
        self.state = state
        self.name = state.name if state is not None else seg.name

    @property
    def valence(self):
        return self.state.valence if self.state is not None \
            else self.seg.valence

    @property
    def alphabulk(self):
        return self.state.alphabulk if self.state is not None else 1.0

    @property
    def phi(self):
        # state-resolved density phi_s = alpha_s * phi_X (set by the state
        # split in compute_phis), or the plain segment density
        if self.state is not None:
            return self.state.phi
        return self.seg.phi


def _species_chi(a, b):
    """chi between two species, with the Namics inheritance rules
    (system.cpp:1854-1950, oracle-verified): explicit state-level chi
    overrides; states of one mon default to 0 among themselves; otherwise
    states inherit the parent mons' chi."""
    if a.state is not None and b.name in a.state.chi:
        return a.state.chi[b.name]
    if b.state is not None and a.name in b.state.chi:
        return b.state.chi[a.name]
    # a mon-side chi may name a state explicitly (mon : S : chi_AM : ...)
    if a.state is None and b.state is not None and b.name in a.seg.chi:
        return a.seg.chi[b.name]
    if b.state is None and a.state is not None and a.name in b.seg.chi:
        return b.seg.chi[a.name]
    if a.state is not None and b.state is not None and a.seg is b.seg:
        return 0.0
    return a.seg.chi_with(b.seg)


class System:
    def __init__(self, settings):
        self.settings = settings
        self.warnings = []

        # Namics calculation types beyond equilibrium SCF: refuse loudly.
        # Silently ignoring a `mesodyn` block would run the input as a plain
        # SCF calculation and produce equilibrium numbers where dynamics were
        # asked for -- never silently wrong. (micro = microemulsion, bate =
        # balanced tensionless: both run an outer Doit() loop toward a
        # tensionless state -- review 6 Oct 2026, #46)
        for block in ("mesodyn", "cleng", "teng", "micro", "bate"):
            if get_blocks(settings, block):
                raise NotImplementedError(
                    f"'{block}' calculations are not supported in PySFBox "
                    "(equilibrium SCF only); use the C++ Namics for those")

        # ---- lattice -----------------------------------------------------
        lats = get_blocks(settings, "lat")
        if len(lats) != 1:
            raise ValueError("exactly one 'lat' block is required")
        lname, lp = lats[0]
        gradients = int(last(lp, "gradients", 1))
        markov = int(float(last(lp, "Markov", 1)))
        if markov != 1:
            raise NotImplementedError(
                f"lat : Markov : {markov} (semiflexible chains) is not "
                "supported in PySFBox; only Markov : 1 (flexible chains) "
                "is available (the compiled Namics binary also disables "
                "Markov, forcing Markov : 1)")
        if last(lp, "k_stiff") is not None:
            raise NotImplementedError(
                "lat : k_stiff (chain stiffness, part of Markov : 2 "
                "semiflexibility) is not supported in PySFBox")
        # bound keys of the other dimensionality used to be ignored
        # silently (Namics refuses them; review 6 Oct 2026, #30)
        wrong = ([k for k in lp if re.fullmatch(r"(lower|upper)bound_[xyz]",
                                                k)]
                 if gradients == 1 else
                 [k for k in lp if k in ("lowerbound", "upperbound")]
                 + [k for k in lp if gradients == 2
                    and k in ("lowerbound_z", "upperbound_z")])
        if wrong:
            raise ValueError(
                f"lat : {wrong[0]} does not apply with gradients : "
                f"{gradients} (1 gradient: lowerbound/upperbound; 2 or 3: "
                "lowerbound_x, upperbound_y, ...)")
        if last(lp, "lambda") is not None and (
                gradients > 1 or int(float(last(lp, "FJC_choices", 3))) > 3):
            # only the fjc = 1 three-point stencil has a lambda to override;
            # the refined FJC weights and the N-D finite-volume stencil
            # ignored it silently (review 6 Oct 2026, #31)
            raise NotImplementedError(
                "lat : lambda (a custom step weight) is supported on the "
                "1-gradient lattice with FJC_choices 3 only (the refined "
                "and N-D stencils have no lambda to replace); drop the line")
        if gradients == 1:
            # FJC_choices = 3 + 2*i -> fjc = (FJC-1)/2 sub-layers/bond (Namics)
            FJC = int(float(last(lp, "FJC_choices", 3)))
            if FJC < 3 or (FJC - 3) % 2 != 0:
                raise ValueError(
                    "FJC_choices must be 3 + 2*i (i.e. 3, 5, 7, ...); "
                    f"got {FJC}")
            if FJC > 3 and last(lp, "lattice_type") not in (None,
                                                           "hexagonal"):
                # the refined weights are fixed by fjc alone (the uniform
                # bond projection, whose fjc = 1 member is the hexagonal
                # stencil): omitting lattice_type is the natural input and
                # stays silent; an explicit other type promises something
                # that does not happen (Namics refuses it outright)
                note = ("FJC_choices > 3 ignores lattice_type : "
                        f"{last(lp, 'lattice_type')}: the refined stencil is "
                        "the hexagonal FJC family (step variance 1/3 + "
                        "1/(6 fjc^2) b^2); omit the line, or write hexagonal "
                        "for inputs that must also run on Namics")
                self.warnings.append(note)
                if note not in _WARN_NOTES:     # once per process
                    _WARN_NOTES.add(note)
                    print(f"  warning: {note}")
            self.lat = Lattice1D(
                n_layers=int(float(substitute_aliases(
                    last(lp, "n_layers"), settings))),
                geometry=last(lp, "geometry", "planar"),
                lattice_type=last(lp, "lattice_type", "simple_cubic"),
                lowerbound=last(lp, "lowerbound", "mirror"),
                upperbound=last(lp, "upperbound", "mirror"),
                offset_first_layer=float(last(lp, "offset_first_layer", 0.0)),
                fjc=(FJC - 1) // 2,
                lam=(float(last(lp, "lambda")) if last(lp, "lambda") is not None
                     else None))
        elif gradients in (2, 3):
            self.lat = self._build_latticeND(lp, gradients, settings)
        else:
            raise NotImplementedError(
                "gradients must be 1, 2, or 3")
        lat = self.lat

        # ---- segments ----------------------------------------------------
        self.segments = {}
        for name, params in get_blocks(settings, "mon"):
            self.segments[name] = Segment(name, params, lat)
        if not self.segments:
            raise ValueError("no 'mon' blocks found")

        # ---- internal states + reactions (weak charges) --------------------
        # `state : NAME : mon : X` attaches annealed internal states to a
        # segment type; `reaction : R : equation/pK` fixes their bulk
        # fractions (see pysfbox/reactions.py)
        self.reactions = [Reaction(name, params)
                          for name, params in get_blocks(settings, "reaction")]
        states = [State(name, params)
                  for name, params in get_blocks(settings, "state")]
        for st in states:
            if st.mon is None:
                raise ValueError(f"state {st.name}: no 'mon' given")
            if st.mon not in self.segments:
                raise ValueError(f"state {st.name}: unknown mon '{st.mon}'")
            if st.name in self.segments:
                raise ValueError(
                    f"state {st.name}: name collides with a mon (as in "
                    "Namics, state and mon names must differ)")
            seg = self.segments[st.mon]
            if seg.freedom == "frozen":
                raise ValueError(
                    f"state {st.name}: states on frozen mons are not "
                    "allowed (as in Namics); use a pinned mon instead")
            if st.name in seg.chi or st.mon in st.chi:
                raise ValueError(
                    f"chi between mon {st.mon} and its own state "
                    f"{st.name} is not allowed (as in Namics)")
            st.phi = np.zeros(lat.M)        # state-resolved density
            st.alpha_prof = np.zeros(lat.M)  # local state fraction alpha_s(z)
            seg.states.append(st)
        self.has_states = bool(states)
        if self.has_states:
            for seg in self.segments.values():
                if len(seg.states) == 1:
                    raise NotImplementedError(
                        f"mon {seg.name} has exactly one state; the "
                        "single-state corner is internally inconsistent in "
                        "Namics and not supported -- give two or more "
                        "states, or drop the state block")
                if seg.states and seg.valence != 0.0:
                    note = (f"mon {seg.name}: mon-level valence is ignored "
                            "once states exist (states carry the charge, "
                            "as in Namics)")
                    self.warnings.append(note)
                    if note not in _WARN_NOTES:     # once per process
                        _WARN_NOTES.add(note)
                        print(f"  warning: {note}")
                    seg.valence = 0.0
            # bulk ionisation: solve every alphabulk to machine precision
            # (Namics pre-solves iteratively to ~1e-6 relative; theory note
            # section 8.2) -- once per System build, i.e. per calculation /
            # var step, exactly the Namics cadence
            solved = solve_bulk_alphas(
                {seg.name: seg.states for seg in self.segments.values()
                 if seg.states},
                self.reactions)
            for seg in self.segments.values():
                for st in seg.states:
                    st.alphabulk = solved[st.name]
        elif self.reactions:
            raise ValueError("reaction blocks given but no state blocks")

        # ---- molecules ---------------------------------------------------
        self.molecules = {}
        for name, params in get_blocks(settings, "mol"):
            comp = substitute_aliases(last(params, "composition"), settings)
            self.molecules[name] = Molecule(name, params, self.segments,
                                            lat, comp)
        if not self.molecules:
            raise ValueError("no 'mol' blocks found")

        # ---- 2D/3D scope guards ------------------------------------------
        # the N-D lattice path (LatticeND) covers NEUTRAL, LINEAR, FLEXIBLE
        # chains; the tree (branched) and charged-Poisson machinery is
        # 1-gradient only for now, so refuse those combinations rather than
        # run them on the wrong stencil.
        if lat.gradients > 1:
            for m in self.molecules.values():
                if getattr(m, "tree", None) is not None:
                    raise NotImplementedError(
                        f"mol {m.name}: branched architectures need gradients "
                        ": 1 (the N-D tree propagator is not implemented)")

        # solvent fills the bulk
        solvents = [m for m in self.molecules.values()
                    if m.freedom == "solvent"]
        if len(solvents) != 1:
            raise ValueError("exactly one mol with freedom : solvent "
                             "is required")
        self.solvent = solvents[0]
        self.solvent.phibulk = 1.0 - sum(
            m.phibulk for m in self.molecules.values()
            if m is not self.solvent)

        # ---- masks ---------------------------------------------------------
        self.frozen = [s for s in self.segments.values()
                       if s.freedom == "frozen"]
        # one ghost wall per face: two would fill the same ghost layer
        # twice (contact density 2; Namics: "Overpopulated 'surface'",
        # system.cpp:1227-1232)
        faces = {}
        for s in self.frozen:
            if s.in_ghost:
                for face in ((s.surface_face,) if lat.gradients > 1 else
                             [f for f, on in (("lower", s.on_lower_surface),
                                              ("upper", s.on_upper_surface))
                              if on]):
                    faces.setdefault(face, []).append(s.name)
        for face, names in faces.items():
            if len(names) > 1:
                raise ValueError(
                    f"frozen walls {names} all sit in the same boundary "
                    f"(ghost) layer {face}; only one segment may occupy a "
                    "surface (Namics: 'Overpopulated surface')")
        interior = lat.interior.astype(float)
        solid = sum((s.range_mask for s in self.frozen), np.zeros(lat.M))
        self.ksam = interior * (1.0 - np.minimum(solid, 1.0))  # free sites
        for s in self.frozen:
            s.phi = s.range_mask.astype(float).copy()
            if lat.gradients > 1:
                # a frozen wall flush to a boundary face fills that ghost face
                if s.surface_face is not None:
                    g = s.phi.reshape(lat.pdims)
                    axis, end = s.surface_face
                    sl = [slice(None)] * lat.gradients
                    sl[axis] = 0 if end == "lo" else lat.dims[axis] + 1
                    g[tuple(sl)] = 1.0
                    s.phi = g.ravel()
            else:
                if s.on_lower_surface:
                    s.phi[:lat.fjc] = 1.0        # all lower ghost layers
                if s.on_upper_surface:
                    s.phi[-lat.fjc:] = 1.0        # all upper ghost layers

        # iterated segment types: everything not frozen
        self.it_segs = [s for s in self.segments.values()
                        if s.freedom != "frozen"]
        # iteration species: one per stateless non-frozen segment type, one
        # per STATE of a multistate segment (cf. Namics (itmon+itstate)*M;
        # see _Species for the deliberate no-dedup deviation). Identical to
        # it_segs when no states exist.
        self.it_species = []
        for s in self.it_segs:
            if s.states:
                self.it_species.extend(_Species(s, st) for st in s.states)
            else:
                self.it_species.append(_Species(s))
        # interaction partners for the chi terms: stateless segment types
        # (INCLUDING frozen walls) plus every state -- multistate mons never
        # appear as mon-level partners (Namics system.cpp:2146 ns<2 gates)
        self.partners = []
        for s in self.segments.values():
            if s.states:
                self.partners.extend(_Species(s, st) for st in s.states)
            else:
                self.partners.append(_Species(s))
        self._check_chi_table()
        # per-segment site mask for the single-segment weights
        self.gmask = {}
        for s in self.it_segs:
            mask = self.ksam.copy()
            if s.freedom == "pinned":
                mask = mask * s.range_mask
            self.gmask[s.name] = mask

        # ---- electrostatics (cf. Namics system.cpp/LG1Planar.cpp) ---------
        # charged mode: any nonzero valence or a fixed surface potential.
        # psi (dimensionless e*psi/kT) joins the iteration stack as one
        # extra block of M unknowns; its residual is a Jacobi sweep of the
        # discrete variable-coefficient Poisson equation.
        self.charged = (any(s.valence != 0.0 for s in self.segments.values())
                        or any(st.valence != 0.0
                               for s in self.segments.values()
                               for st in s.states)
                        or any(s.fixed_psi0 for s in self.segments.values()))
        if self.charged and lat.gradients > 1:
            raise NotImplementedError(
                "charged systems (valence / e.psi0/kT) need gradients : 1 "
                "(the N-D Poisson solver is not implemented yet)")
        if self.charged:
            self.geom = lat.geometry
            if self.geom not in ("planar", "cylindrical", "spherical"):
                raise NotImplementedError(
                    f"charged systems on geometry '{self.geom}' need the "
                    "full Namics")
            self.bondlength = float(last(lp, "bondlength", 5e-10))
            if not (1e-12 <= self.bondlength <= 1e-8):
                raise ValueError("lat : bondlength out of range 1e-12..1e-8 m")
            # C = e^2/(eps0 kT b): the dimensionless Poisson prefactor
            self.C_psi = E_CHARGE**2 / (EPS0 * K_BT * self.bondlength)
            # field-energy prefactor: base = 0.5*eps0*b/kT*(kT/e)^2
            # (LG1Planar/LGrad1::UpdateEE); planar carries an extra /2*fjc^2,
            # curved carries the geometry pi-factor instead (see _field_energy)
            pf_base = (0.5 * EPS0 * self.bondlength / K_BT
                       * (K_BT / E_CHARGE)**2)
            self.pf_ee = pf_base / 2.0 * lat.fjc**2
            self.pf_base = pf_base
            self.grad_epsilon = len({s.epsilon
                                     for s in self.segments.values()}) > 1
            self.fixedPsi0 = any(s.fixed_psi0
                                 for s in self.segments.values())
            # face radii for the curved Poisson/field-energy (refined units;
            # fjc = 1: the Namics shells; fjc > 1: centred on the site, the
            # propagator's channel surfaces -- see curved_face_radii)
            if self.geom != "planar":
                self.r_plus, self.r_minus = curved_face_radii(
                    lat.offset, lat.fjc, lat.M)
                if self.fixedPsi0:
                    raise NotImplementedError(
                        "fixed surface potential (e.psi0/kT) on a curved "
                        "lattice is not supported yet (the curved electrode "
                        "condition is not yet ported to PySFBox)")
            # diagnostic hazard (oracle-tested 21 Jul 2026): for epsilon
            # outside 1..250 the compiled Namics PRINTS "Default value 80
            # used instead" but does NOT substitute -- it runs with the
            # declared value (segment.cpp:1096 only prints; verified: eps=0
            # output differs from eps=80). Both engines therefore use the
            # declared value; warn so the false Namics message does not
            # mislead a cross-engine comparison, and because eps<1 is
            # physically dubious (vacuum is 1).
            for seg in self.segments.values():
                if not (1.0 <= seg.epsilon <= 250.0):
                    note = (f"mon {seg.name}: epsilon {seg.epsilon:g} is "
                            "outside 1..250; PySFBox uses it as declared -- "
                            "and so does the compiled Namics, DESPITE its "
                            "misleading 'Default value 80 used instead' "
                            "message (it only prints, never substitutes; "
                            "oracle-verified)")
                    self.warnings.append(note)
                    if note not in _WARN_NOTES:     # once per process
                        _WARN_NOTES.add(note)
                        print(f"  warning: {note}")
            self.psiMask = np.zeros(lat.M, dtype=bool)
            self.psi0_profile = np.zeros(lat.M)
            for seg in self.frozen:
                if seg.fixed_psi0:
                    m = seg.phi > 0.5
                    self.psiMask |= m
                    self.psi0_profile[m] = seg.psi0
            neut = [m for m in self.molecules.values()
                    if m.freedom == "neutralizer"]
            if len(neut) > 1:
                raise ValueError("at most one mol with freedom : neutralizer")
            self.neutralizer = neut[0] if neut else None
            if self.neutralizer is None:
                # psi = 0 is the bulk reference only for an electroneutral
                # bulk. Without a neutralizer nothing enforces that: a
                # non-neutral declared bulk relaxes to a different (Donnan)
                # reservoir while theta_exc/Omega/mu still refer to the
                # declared one, and a floating charged restricted molecule
                # has a field-dependent implied bulk charge (review 6 Oct
                # 2026, #6; Namics' rule, system.cpp:705-730)
                floating = [m.name for m in self.molecules.values()
                            if m.freedom == "restricted" and not m.has_pinned
                            and any(sg.valence != 0.0
                                    or any(st.valence != 0.0
                                           for st in sg.states)
                                    for sg in m.seq)]
                net = sum(m.phibulk * m.charge_per_seg()
                          for m in self.molecules.values()
                          if m.freedom in ("free", "solvent"))
                if floating or abs(net) > 1e-4:
                    why = (f"the restricted molecule(s) {floating} are "
                           "charged and not pinned" if floating else
                           f"the declared bulk carries net charge {net:.3g} "
                           "per site")
                    raise ValueError(
                        f"a neutralizer is needed: {why}. Declare one charged "
                        "molecule (e.g. a counter-ion) with freedom : "
                        "neutralizer -- its bulk fraction is then set by "
                        "electroneutrality (as in Namics)")
            self.psi = np.zeros(lat.M)
            self.q = np.zeros(lat.M)
            self.q_electrode = np.zeros(lat.M)   # set by _psi_residual
            self.EE = np.zeros(lat.M)
            self.eps_prof = np.full(lat.M, 80.0)
        else:
            if any(m.freedom == "neutralizer"
                   for m in self.molecules.values()):
                raise ValueError(
                    "freedom : neutralizer requires a charged system")
            self.neutralizer = None

        # ---- delta constraint (sys : constraint : delta) -----------------
        # Pins an interface: at the delta_range sites the two
        # delta_molecules' total densities are held at
        # phitot_A - phitot_B = (r-1)/(r+1) with r = phi_ratio (at
        # incompressibility, phitot_A + phitot_B ~= 1, that is the ratio
        # phi_A/phi_B = r). Mechanics (Namics system.cpp:735-908 parsing,
        # solve_scf.cpp:86 extra block, system.cpp:2113 PutU,
        # system.cpp:2240-2247 residual, molecule.cpp:2654 ComputePhi):
        # a Lagrange-multiplier field beta(z) joins the iteration stack as
        # one extra M-block after psi; every segment of molecule A feels
        # +beta (G1 *= exp(-beta)), of molecule B -beta; the constraint
        # residual (phitot_B - phitot_A + R)*mask lives on the masked sites
        # only. Off-mask beta entries are excluded from the solver AND
        # masked out of the physics (self.beta is stored pre-masked), so a
        # stale warm-start value at a moved mask site is inert -- Namics
        # instead carries them as dead unknowns. The classic use (Namics
        # example nucleation_barrier.in) fixes a free molecule's phibulk
        # and walks delta_range to map Omega(R) away from equilibrium.
        def sys_last(param, default=None):
            v = default
            for _, params in get_blocks(settings, "sys"):
                v = last(params, param, v)
            return v

        constraint = sys_last("constraint")
        self.constraintfields = constraint is not None
        if self.constraintfields:
            if constraint != "delta":
                raise NotImplementedError(
                    f"sys : constraint : {constraint} is not supported "
                    "(only 'delta')")
            if lat.gradients > 1:
                raise NotImplementedError(
                    "the delta constraint is 1-gradient only in PySFBox for "
                    "now (Namics supports 2/3 gradients; use the C++ Namics)")
            rng = sys_last("delta_range")
            if rng is None:
                raise ValueError(
                    "sys : constraint : delta needs a 'delta_range'")
            if rng.strip() == "file":
                raise NotImplementedError(
                    "delta_range : file (delta_inputfile) is not ported to "
                    "PySFBox; list the sites as (z);(z2);... instead")
            # fjc>1: coordinates are given in bondlength units (z scaled by
            # fjc) or gritsize units (raw refined index), Namics
            # system.cpp:756-780; mask index fjc-1+z*units (LGrad1::FillMask)
            units = 1
            if lat.fjc > 1:
                runits = sys_last("delta_range_units")
                if runits is None:
                    raise ValueError(
                        "FJC_choices > 3 with a delta constraint needs "
                        "'sys : ... : delta_range_units : bondlength' (z in "
                        "layers) or 'gritsize' (z in refined sub-layers)")
                if runits == "bondlength":
                    units = lat.fjc
                elif runits != "gritsize":
                    raise ValueError(
                        f"delta_range_units : {runits} not recognized "
                        "(use 'bondlength' or 'gritsize')")
            self.delta_mask = np.zeros(lat.M)
            for part in rng.split(";"):
                part = part.strip()
                if not (part.startswith("(") and part.endswith(")")):
                    raise ValueError(
                        f"delta_range entry '{part}': expected '(z)' "
                        "(1-gradient)")
                z = int(part[1:-1]) * units
                if not 1 <= z <= lat.MX:   # MX is refined (= Namics FillMask)
                    raise ValueError(
                        f"delta_range site {part} out of bounds")
                self.delta_mask[lat.fjc - 1 + z] = 1.0
            dmols = sys_last("delta_molecules")
            if dmols is None:
                raise ValueError(
                    "sys : constraint : delta needs 'delta_molecules : A;B' "
                    "(two molecule names)")
            names = [p.strip() for p in dmols.split(";")]
            if len(names) != 2 or names[0] == names[1]:
                raise ValueError(
                    "delta_molecules must name two DIFFERENT molecules "
                    "separated by ';'")
            for nm in names:
                if nm not in self.molecules:
                    raise ValueError(
                        f"delta_molecules: molecule '{nm}' not found")
            self.delta_molecules = tuple(self.molecules[nm] for nm in names)
            ratio = sys_last("phi_ratio")
            if ratio is None:
                raise ValueError(
                    "sys : constraint : delta needs 'phi_ratio' (typically "
                    "1, or the keyword 'critical_ratio')")
            if ratio == "critical_ratio":
                # the FH critical composition of an A/B pair: phi_A/phi_B =
                # sqrt(N_B/N_A) (f''' = 0). Namics (system.cpp:899-901) uses
                # sqrt(N_A/N_B) -- the inverse; deliberate deviation (review
                # 6 Oct 2026, #74)
                molA, molB = self.delta_molecules
                self.phi_ratio = float(np.sqrt(molB.N / molA.N))
            else:
                self.phi_ratio = float(ratio)
                if self.phi_ratio <= 0:
                    raise ValueError("phi_ratio must be positive")
            self.delta_R = (self.phi_ratio - 1.0) / (self.phi_ratio + 1.0)
        self.beta = np.zeros(lat.M)

        # per-segment-type bulk fractions (for u_int reference and moments);
        # refreshed every iteration by _update_bulk (restricted molecules
        # contribute their implied bulk density, which depends on GN)
        self.phibulk_seg = {name: 0.0 for name in self.segments}
        self._update_bulk()
        if self.neutralizer is not None and not any(
                m.freedom == "restricted" and not m.has_pinned
                for m in self.molecules.values()):
            # every bulk fraction is declared: the neutralizer's sign is
            # known now -- refuse before solving rather than after
            try:
                self._check_bulk_signs()
            except RuntimeError as e:
                raise ValueError(str(e)) from None

        # ---- the characteristic function X (Namics sys : X) --------------
        # parsed here so a malformed definition stops the run before the
        # solve, like Namics' CheckInput (evaluated at output time,
        # characteristic_X)
        self.X_mols, self.X_states = None, None
        xspec = sys_last("X")
        if xspec is not None:
            all_states = {st.name: st for sg in self.segments.values()
                          for st in sg.states}
            mono = {}
            for m in self.molecules.values():
                if m.N == 1 and len(m.seq) == 1:
                    for st in m.seq[0].states:
                        mono.setdefault(st.name, []).append(m.name)
            self.X_mols, self.X_states = parse_characteristic_X(
                xspec, list(self.molecules), all_states, mono)
        if sys_last("compute_kJ0") is not None:
            # Namics' compute_kJ0 pushes kal kJ0 = -(first moment) and
            # kbar = second moment of the planar grand-potential density
            # about a midpoint (system.cpp:1561-1586, GetSpontaneousCurvature
            # / GetKBar :3229-3241). Those bare moments are not the Helfrich
            # constants of an SF film, so the column is not ported.
            raise NotImplementedError(
                "sys : compute_kJ0 is not supported in PySFBox. Namics' "
                "kJ0 and kbar are the bare first and second moments of the "
                "planar grand-potential density (taken about z = 0 because "
                "of a Namics bug), not the Helfrich constants kappa*J0 and "
                "kbar of a self-consistent film: they miss the chain-end, "
                "bulk-jump and chi site-average contributions. Remove the "
                "line. The Helfrich constants follow from curved ladders "
                "(fit the grand potential of cylinders and spheres against "
                "1/R); the PySFBox development version also computes them "
                "from one flat film.")
        self.alpha = np.zeros(lat.M)
        self.iterations = 0
        self.residual_norm = np.inf

    @staticmethod
    def _build_latticeND(lp, gradients, settings):
        """Construct a 2- or 3-gradient LatticeND from the lat block (Namics
        keys: n_layers_x/y/z, geometry, lattice_type, lowerbound_x/upperbound_x
        ...). FJC_choices > 3 is rejected (LatticeND is fjc = 1)."""
        FJC = int(float(last(lp, "FJC_choices", 3)))
        if FJC != 3:
            raise NotImplementedError(
                "FJC_choices > 3 (refined lattice) is not supported with "
                "gradients > 1 yet (needs the refined N-D stencil)")

        def nl(axis):
            v = last(lp, f"n_layers_{axis}")
            if v is None:
                raise ValueError(
                    f"gradients : {gradients} needs 'lat : ... : n_layers_"
                    f"{axis}'")
            return int(float(substitute_aliases(v, settings)))

        axes = ["x", "y"] + (["z"] if gradients == 3 else [])
        dims = tuple(nl(a) for a in axes)
        # Namics' 2-D lattices default to `stencil_full : true`, a 9-point
        # product stencil; PySFBox's 2-D stencil is the 5-point reduction of
        # the 3-D 7-point lattice (= Namics' stencil_full : false). Accept
        # `false` (or no line), refuse an explicit `true` rather than run a
        # different stencil silently (review 7 Oct 2026, #49). LGrad3 has
        # no stencil_full branch, so 3-D ignores the keyword like Namics.
        sf = last(lp, "stencil_full")
        if (gradients == 2 and sf is not None
                and str(sf).strip().lower() not in ("false", "0", "no")):
            raise NotImplementedError(
                "lat : stencil_full : true (Namics' 9-point 2-D product "
                "stencil) is not ported to PySFBox; PySFBox's 2-D lattice is "
                "the 5-point stencil (Namics' stencil_full : false), the "
                "exact reduction of the 3-D lattice. Remove the line or set "
                "it to false.")
        # only override the lattice's own default for axes the user actually
        # set (else pass None -> LatticeND keeps its role-aware default, which
        # is PERIODIC for a full-2pi azimuthal axis; blanket 'mirror' here
        # silently put a reflecting wall at the phi=0/2pi seam -- HIGH bug,
        # physics review 7 Jul 2026)
        bounds = []
        for a in axes:
            lo = last(lp, f"lowerbound_{a}")
            hi = last(lp, f"upperbound_{a}")
            if lo is None and hi is None:
                bounds.append(None)
            else:
                bounds.append((lo or "mirror", hi or "mirror"))
        if (last(lp, "theta_range") is not None
                or last(lp, "phi_range") is not None):
            raise NotImplementedError(
                "lat : theta_range / phi_range (angular polar/spherical "
                "lattices) are not supported in PySFBox; the supported "
                "multi-gradient geometries are 2D flat, 2D cylindrical "
                "(r,z), and 3D flat, as in Namics")
        return LatticeND(
            dims, geometry=last(lp, "geometry", "flat"),
            lattice_type=last(lp, "lattice_type", "simple_cubic"),
            bounds=bounds,
            offset_first_layer=float(last(lp, "offset_first_layer", 0.0)))

    # ---- core SCF ----------------------------------------------------------
    def n_var(self):
        n = len(self.it_species) * self.lat.M
        if self.charged:
            n += self.lat.M                 # the psi block
        if self.constraintfields:
            n += self.lat.M                 # the beta block (last)
        return n

    def unpack(self, x):
        S = len(self.it_species)
        return x[:S * self.lat.M].reshape(S, self.lat.M)

    def var_mask(self):
        """Boolean mask over the full variable vector selecting what the
        solver actually iterates: free interior sites for the potentials,
        and interior non-fixed sites for psi (fixed-surface-potential sites
        are constants, cf. Namics' g==0 filter)."""
        m = np.tile(self.ksam > 0, len(self.it_species))
        if self.charged:
            pm = np.zeros(self.lat.M, dtype=bool)
            pm[self.lat.iv] = True
            m = np.concatenate([m, pm])
        if self.constraintfields:
            # only the delta_range sites are real beta unknowns (Namics
            # carries the full M block with g=0 dead entries instead)
            m = np.concatenate([m, self.delta_mask > 0])
        return m

    def _check_chi_table(self):
        """Validate the chi table once per build (review 6 Oct 2026, #7/#61).
        chi is a SYMMETRIC exchange parameter; a pair given on both
        partners with different values used to run with the caller's own
        entry on each side -- an asymmetric table whose F, Omega and mu no
        longer belong to one functional (Namics refuses it,
        system.cpp:1778-1792). A value given on one side only, or equally
        on both, is fine. A chi that names a MULTISTATE mon from a state
        (or a state from a multistate mon's own block) is never used --
        multistate mons couple through their states only, in Namics as
        here -- so it gets a warning naming the per-state alternative."""
        segs = list(self.segments.values())
        for i, X in enumerate(segs):
            for Y in segs[i + 1:]:
                a, b = X.chi.get(Y.name), Y.chi.get(X.name)
                if a is not None and b is not None and a != b:
                    raise ValueError(
                        f"mon {X.name} : chi_{Y.name} : {a:g} and mon "
                        f"{Y.name} : chi_{X.name} : {b:g} disagree -- chi "
                        "is symmetric; give it once (or equally on both)")
        sp = self.partners
        for i, a in enumerate(sp):
            for b in sp[i + 1:]:
                if a.state is None and b.state is None:
                    continue                   # the mon-level check above
                given = []
                if a.state is not None and b.name in a.state.chi:
                    given.append((f"state {a.name}", b.name,
                                  a.state.chi[b.name]))
                if b.state is not None and a.name in b.state.chi:
                    given.append((f"state {b.name}", a.name,
                                  b.state.chi[a.name]))
                if a.state is None and b.name in a.seg.chi:
                    given.append((f"mon {a.name}", b.name, a.seg.chi[b.name]))
                if b.state is None and a.name in b.seg.chi:
                    given.append((f"mon {b.name}", a.name, b.seg.chi[a.name]))
                if len({v for _, _, v in given}) > 1:
                    raise ValueError(
                        "conflicting chi values for one pair: "
                        + "; ".join(f"{who} : chi_{p} : {v:g}"
                                    for who, p, v in given)
                        + " -- chi is symmetric; give it once")
        multistate = {s.name for s in segs if s.states}
        state_names = {st.name for s in segs for st in s.states}
        unused = []
        for s in segs:
            for st in s.states:
                unused += [f"state {st.name} : chi_{k}" for k in st.chi
                           if k in multistate]
            if s.states:
                unused += [f"mon {s.name} : chi_{k}" for k in s.chi
                           if k in state_names]
        for u in unused:
            note = (f"{u} is never used: a multistate mon couples only "
                    "through its states (as in Namics) -- give the chi per "
                    "state instead")
            self.warnings.append(note)
            if note not in _WARN_NOTES:            # once per process
                _WARN_NOTES.add(note)
                print(f"  warning: {note}")

    def _update_bulk(self):
        """Bulk composition, refreshed every iteration exactly like Namics
        (System::ComputePhis, system.cpp:2463-2602): a free-floating
        `restricted` molecule carries an IMPLIED bulk density
        phibulk = theta/GN (what a reservoir in equilibrium with the
        constrained amount would hold; ~0 for grafted chains, whose GN is
        huge) -- pinned/grafted molecules get 0; the solvent fills up the
        remainder, phibulk_solvent = 1 - sum(others). The per-segment-type
        references used by the residual's chi terms and by the thermodynamic
        outputs follow from these."""
        B = 0.0
        A = 0.0                       # net bulk charge of the others
        for m in self.molecules.values():
            if m is self.solvent or m is self.neutralizer:
                continue
            if m.freedom == "restricted":
                if m.has_pinned \
                        or not np.isfinite(m.lnGN) or m.theta <= 0:
                    m.phibulk = 0.0
                else:
                    # cap: any implied bulk >> 1 is equally unphysical, and a
                    # tighter cap keeps the downstream sums overflow-free
                    m.phibulk = float(np.exp(
                        min(np.log(m.theta) - m.lnGN, 300.0)))
            B += m.phibulk
            if self.charged:
                A += m.phibulk * m.charge_per_seg()
        if self.neutralizer is not None:
            # electroneutral bulk (Namics system.cpp:2560-2592): solvent and
            # neutralizer jointly fill the remaining volume AND cancel the
            # net charge A of everything else
            zn = self.neutralizer.charge_per_seg()
            zs = self.solvent.charge_per_seg()
            if zn == zs:
                raise ValueError(
                    "neutralizer charge equals solvent charge; cannot "
                    "neutralize the bulk")
            self.neutralizer.phibulk = ((B - 1.0) * zs - A) / (zn - zs)
            B += self.neutralizer.phibulk
        self.solvent.phibulk = 1.0 - B     # may go negative on a wild
        # transient (Namics aborts there; we refuse only converged states)
        for name in self.phibulk_seg:
            self.phibulk_seg[name] = 0.0
        for m in self.molecules.values():
            for mon, cnt in m.blocks:
                self.phibulk_seg[mon] += m.phibulk * (cnt / m.N)

    def compute_phis(self, u, psi=None, EE=None):
        """u: (n_it_species, M). Computes all molecule and segment densities.
        In charged systems the full species potential adds the
        electrostatic energy and the dielectric (Born-like) self-energy
        (cf. Namics PutU): u_i + valence_i*psi - epsilon_i*EE.

        A multistate segment's propagator weight is the ANNEALED sum over
        its states, G_X = sum_s alphabulk_s * exp(-u_s) (Namics
        segment.cpp:862-874) -- the chain machinery
        downstream is completely state-blind. The local state fractions
        alpha_s(z) = alphabulk_s exp(-u_s)/G_X (eq 2.2) are stored per
        state and split the segment density after propagation."""
        self._u_current = u
        G1 = {}
        i = 0
        for s in self.it_segs:
            if not s.states:
                # clip: transiently wild potentials must not overflow exp;
                # the converged fields are far inside this range. U_CLIP is
                # shared with the composition-law cap in model.compute_phi.
                u_tot = u[i]
                if psi is not None:
                    if s.valence != 0.0:
                        u_tot = u_tot + s.valence * psi
                    u_tot = u_tot - s.epsilon * EE
                G1[s.name] = np.exp(-np.clip(u_tot, -U_CLIP, U_CLIP)) \
                    * self.gmask[s.name]
                i += 1
                continue
            W = np.zeros(self.lat.M)
            weights = []
            for st in s.states:
                u_tot = u[i]
                if psi is not None:
                    if st.valence != 0.0:
                        u_tot = u_tot + st.valence * psi
                    u_tot = u_tot - s.epsilon * EE   # eps is mon-level
                w = st.alphabulk * np.exp(-np.clip(u_tot, -U_CLIP, U_CLIP))
                weights.append(w)
                W += w
                i += 1
            W_safe = np.where(W > 0, W, 1.0)
            for st, w in zip(s.states, weights):
                st.alpha_prof = np.where(W > 0, w / W_safe, st.alphabulk)
            G1[s.name] = W * self.gmask[s.name]
        # delta constraint: molecule A's segments all feel +beta, molecule
        # B's -beta (Namics Molecule::ComputePhi(BETA,+-1), molecule.cpp:
        # 2654-2692 -- there G1 is multiplied in place and divided back out;
        # here each delta molecule simply propagates a modified G1 dict,
        # which the composition law divides by consistently). self.beta is
        # pre-masked (zero off the delta_range sites) and clipped like u.
        if self.constraintfields:
            b = np.clip(self.beta, -U_CLIP, U_CLIP)
            molA, molB = self.delta_molecules
            _dfac = {molA.name: np.exp(-b), molB.name: np.exp(b)}

            def G1_for(m):
                f = _dfac.get(m.name)
                if f is None:
                    return G1
                return {k: v * f for k, v in G1.items()}
        else:
            def G1_for(m):
                return G1
        # non-solvent, non-constrained molecules first: their normalisation
        # (phibulk- or theta-based) does not depend on the solvent's bulk
        # fraction ...
        for m in self.molecules.values():
            if m is not self.solvent and m is not self.neutralizer:
                m.compute_phi(G1_for(m))
        # ... which is refreshed from their current GN before the solvent
        # AND the neutralizer are evaluated -- both have their bulk fraction
        # SET by _update_bulk (the neutralizer by electroneutrality), so like
        # the solvent they must be computed AFTER it, with the current value.
        # This MATCHES Namics, which normalises the neutralizer density with
        # the freshly-computed norm in the same ComputePhis call
        # (system.cpp:2633-2650) -- there is NO lag there.
        self._update_bulk()
        if self.neutralizer is not None:
            self.neutralizer.compute_phi(G1_for(self.neutralizer))
        self.solvent.compute_phi(G1_for(self.solvent))
        # total per segment type (solution species)
        for s in self.it_segs:
            s.phi = sum((m.phi_per_seg.get(s.name, 0.0)
                         for m in self.molecules.values()),
                        np.zeros(self.lat.M))
            s.phibulk = self.phibulk_seg[s.name]
            # state split (Namics SetPhiSide, segment.cpp:1353-1375)
            for st in s.states:
                st.phi = st.alpha_prof * s.phi
                st.phibulk = s.phibulk * st.alphabulk
        return G1

    def side_phi(self, seg, phi=None):
        """Site average of a segment density with correct ghost values.
        `phi` overrides the profile (used for state-resolved densities,
        which inherit the parent segment's boundary handling)."""
        lat = self.lat
        prof = seg.phi if phi is None else phi
        if lat.gradients > 1:
            values = None
            if (seg.freedom == "frozen"
                    and getattr(seg, "surface_face", None) is not None):
                values = {seg.surface_face: 1.0}
            return lat.site_average(lat.set_bounds(prof, values=values))
        lower = 1.0 if seg.on_lower_surface else None
        upper = 1.0 if getattr(seg, "on_upper_surface", False) else None
        f = lat.set_bounds(prof, lower_value=lower, upper_value=upper)
        return lat.site_average(f)

    def residual(self, x):
        lat = self.lat
        u = self.unpack(x)
        S = len(self.it_species)
        if self.constraintfields:
            # beta is the LAST M-block; store it pre-masked so off-mask
            # entries (dead unknowns) can never leak into the physics
            nb = (S + (1 if self.charged else 0)) * lat.M
            self.beta = x[nb:nb + lat.M] * self.delta_mask
        if self.charged:
            psi_raw = x[S * lat.M:(S + 1) * lat.M]
            # At an electrode site the raw unknown is the auxiliary
            # behind-electrode value (see _psi_residual); the physical
            # potential there is psi0. The Poisson rows, the segment
            # weights AND the field energy EE all use this EFFECTIVE psi:
            # the Born term -eps*EE and the Poisson equation are both
            # derivatives of the same bond energy -eps(dpsi)^2/2, so they
            # must see the same potential. (Namics PutU builds EE from the
            # raw psi instead -- with free species of different epsilon
            # the contact layer then feels (psi_bw - psi_2)^2 rather than
            # (psi0 - psi_2)^2, -15% Na at contact on edl_fixed_psi;
            # review finding 2, 7 Oct 2026. Away from electrodes raw and
            # effective psi coincide.)
            psi = psi_raw.copy()
            if self.fixedPsi0:
                psi[self.psiMask] = self.psi0_profile[self.psiMask]
            psib = lat.set_mirror_bounds(psi)   # psi ghosts ALWAYS mirror
            EE = self._field_energy(psib)
            self.psi, self.EE = psi, EE
            self.compute_phis(u, psi=psi, EE=EE)
        else:
            self.compute_phis(u)
        phitot = sum((s.phi for s in self.segments.values()),
                     np.zeros(lat.M))
        # chi couples to the side-averaged *volume fraction* phi_side/phitot,
        # not the raw side density (cf. Namics H_PutAlpha: g -= chi*(phi_side/
        # phitot - phibulk)). The two agree at incompressibility (phitot=1) but
        # differ on the way there -- getting this right is what lets the
        # pseudohessian follow Namics on stiff, transiently-compressible states.
        phitot_safe = np.where(phitot > 0, phitot, 1.0)
        # interaction partners: stateless mons at mon level, multistate mons
        # through their states (state-resolved phi_side and phibulk;
        # cf. Namics system.cpp:2144-2181 + SetPhiSide)
        sides = {p.name: self.side_phi(p.seg, p.phi) / phitot_safe
                 for p in self.partners}
        pbulk = {p.name: (self.phibulk_seg.get(p.seg.name, 0.0)
                          * p.alphabulk)
                 for p in self.partners}

        g = np.empty_like(u)
        for i, si in enumerate(self.it_species):
            u_int = np.zeros(lat.M)
            for p in self.partners:
                chi = _species_chi(si, p)
                if chi:
                    u_int += chi * (sides[p.name] - pbulk[p.name])
            g[i] = u[i] - u_int
        self.alpha = g.mean(axis=0)
        g -= self.alpha
        with np.errstate(divide="ignore"):
            incompr = np.where(phitot > 0, 1.0 / phitot - 1.0, 0.0)
        g += incompr
        g *= self.ksam
        self.phitot = phitot
        parts = [g.ravel()]
        if self.charged:
            parts.append(self._psi_residual(psi_raw, psib, phitot))
        if self.constraintfields:
            # constraint residual (Namics system.cpp:2240-2246): at the
            # masked sites drive phitot_A - phitot_B to R = (r-1)/(r+1)
            molA, molB = self.delta_molecules
            parts.append((molB.phi - molA.phi + self.delta_R)
                         * self.delta_mask)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    # ---- electrostatics (planar, fjc = 1; cf. LG1Planar.cpp) ---------------
    def _field_energy(self, psib):
        """Electric-field energy density EE(z) (Namics UpdateEE). Planar
        (LG1Planar): EE = pf_ee*[(dpsi_left)^2 + (dpsi_right)^2]. Curved
        (LGrad1): each squared field difference is weighted by its FACE
        radius (cyl: r; sph: r^2; the radii of `curved_face_radii`, the
        same faces as `_psi_residual`, centred on the site at fjc > 1) and
        divided by the shell volume L, with
        pf = pf_base*pi (cyl) or pf_base*2pi/fjc (sph) -- so EE is an
        energy DENSITY per unit volume and L*eps*EE sums to the bond
        energies under the L-weighted weighted_sum."""
        lat = self.lat
        iv = slice(lat.fjc, lat.M - lat.fjc)
        EE = np.zeros(lat.M)
        if self.geom == "planar":
            # interior rows only (the refined lattice has fjc ghost layers;
            # a [1:-1] form would write garbage equations into the ghost
            # region at fjc > 1)
            d = np.diff(psib)                # d[x] = psi[x+1]-psi[x]
            EE[iv] = self.pf_ee * (d[lat.fjc - 1:lat.M - lat.fjc - 1] ** 2
                                   + d[lat.fjc:lat.M - lat.fjc] ** 2)
            return EE
        dl = (psib[iv] - psib[lat.fjc - 1:lat.M - lat.fjc - 1]) ** 2  # left^2
        dr = (psib[iv] - psib[lat.fjc + 1:lat.M - lat.fjc + 1]) ** 2  # right^2
        rp, rm, L = self.r_plus[iv], self.r_minus[iv], lat.L[iv]
        if self.geom == "cylindrical":
            pf = self.pf_base * np.pi
            EE[iv] = pf * (rm * dl + rp * dr) / L
        else:                                # spherical
            pf = self.pf_base * 2.0 * np.pi / lat.fjc
            EE[iv] = pf * (rm ** 2 * dl + rp ** 2 * dr) / L
        return EE

    def _electrostatics(self, phitot):
        """Charge density q(z) and permittivity profile eps(z), both
        phi-weighted and divided by phi_T (Namics DoElectrostatics):
        q = sum_i valence_i phi_i / phi_T; eps = sum_i eps_i phi_i / phi_T.
        eps is assembled with segment-profile ghost fills (wall segments
        keep their boundary values), because eps at the ghost enters the
        first interior Poisson coefficient."""
        lat = self.lat
        M = lat.M
        q = np.zeros(M)
        eps = np.zeros(M)
        phitot_b = np.zeros(M)
        for seg in self.segments.values():
            lower = 1.0 if seg.on_lower_surface else None
            upper = 1.0 if getattr(seg, "on_upper_surface", False) else None
            phib = lat.set_bounds(seg.phi, lower_value=lower,
                                  upper_value=upper)
            if seg.states:
                # multistate mons: charge from the LOCAL annealed state
                # fractions (Namics DoElectrostatics state loop); the
                # mon-level valence is dead. eps stays mon-level (no
                # per-state epsilon exists in Namics either).
                for st in seg.states:
                    if st.valence != 0.0:
                        q += st.valence * st.phi
            elif seg.valence != 0.0:
                q += seg.valence * seg.phi
            eps += seg.epsilon * phib
            phitot_b += phib
        pt_safe = np.where(phitot > 0, phitot, 1.0)
        q = np.where(phitot > 0, q / pt_safe, 0.0)
        ptb_safe = np.where(phitot_b > 0, phitot_b, 1.0)
        eps = np.where(phitot_b > 0, eps / ptb_safe, 80.0)
        return q, eps

    def _psi_residual(self, psi_raw, psib, phitot):
        """The psi block of the residual: one JACOBI sweep of the
        flux-conservative discrete Poisson equation,

            X = (epsm*psi[x-1] + (2C/fjc^2) q + epsp*psi[x+1])/(epsm+epsp),
            epsm = eps[x-1]+eps[x],  epsp = eps[x]+eps[x+1]

        (Namics' free branch, LG1Planar::UpdatePsi; Gauss-Seidel there,
        same fixed point). This form is used for ALL free sites, also when
        a fixed surface potential is present: Namics' dedicated fixedPsi0
        branch solves a Poisson equation with a DOUBLED source (plus
        inconsistent grad-eps/fjc factors), giving Debye lengths a factor
        sqrt(2) too short — a real upstream bug, reported in
        a bug report shared with the Namics authors. Per project policy PySFBox
        implements the CORRECT equation; fixed-potential results therefore
        intentionally deviate from unpatched Namics (validated against
        Debye theory instead; the port mechanics themselves were verified
        bug-compatibly against the oracle at machine precision first).

        Electrode sites (fixed psi0): as in Namics, the raw psi unknown
        there is the behind-electrode value; it is anchored to the
        zero-net-charge condition of the electrode site — in closed form
        (unit diagonal) and with the flux-conservative coefficients.

        Curved geometries (cylindrical/spherical, Namics LGrad1::UpdatePsi
        free branch, all fjc) use the SAME 3-point flux balance with the
        coefficients weighted by the face radius (cyl: r; sph: r^2) and the
        source by the shell volume L: cm*psi[x-1] + cp*psi[x+1]
        + C_geo*q[x]*L[x] - (cm+cp)*psi[x] = 0. The face radii are
        `curved_face_radii`: at fjc = 1 the Namics shells (r_minus = 0 at
        the origin); at fjc > 1 the faces r_site -+ 1/2 centred on the site
        (the propagator's channel surfaces, geometric areas; Namics puts
        them half a refined cell inward). psi ghosts always mirror, so the
        innermost/outermost faces carry no flux in either frame. This is the
        correct flux-conservative equation (the factor-2 bug lives only in
        the fixedPsi0 branch, which curved charged systems do not use)."""
        lat = self.lat
        M = lat.M
        fjc = lat.fjc
        q, eps = self._electrostatics(phitot)
        g_psi = np.zeros(M)
        if self.geom != "planar":
            iv = slice(fjc, M - fjc)
            e0 = eps[iv]
            em = eps[fjc - 1:M - fjc - 1]        # eps[x-1]
            ep = eps[fjc + 1:M - fjc + 1]        # eps[x+1]
            rm, rp = self.r_minus[iv], self.r_plus[iv]
            # The face coefficients use the refined radius r = fjc x the
            # lattice radius r_lat (Namics r++ per site; at fjc > 1 the faces
            # sit at r_site -+ 1/2, centred on the site). The correct
            # flux-conservative source C_geo*q*L is fixed by refinement
            # consistency: the physical gradient across a refined bond carries
            # one fjc, so the flux term scales as the coefficient radius (r^1
            # for cyl -> fjc; r^2 for sph -> fjc^2) divided by that one fjc.
            # With L ~ 1/fjc, this leaves cyl needing NO extra fjc in C_geo and
            # sph needing exactly ONE -- matching the Namics UpdatePsi
            # coefficients themselves (LGrad1.cpp:675 C/PIE; :690 C/(2PIE)*fjc).
            # BUT the compiled Namics is still fjc-too-strong at fjc>1 because
            # lattice.cpp:337 does `bond_length/=fjc`, and bond_length feeds
            # only C0 = e^2/(eps0 kT bond_length) -- so the oracle's C0 is fjc
            # too large, its Debye length scales as kappa ~ sqrt(fjc) (measured
            # 1.04/1.60/2.35 at fjc=1/2/4) and refining a charged run changes
            # the physics. PySFBox uses the base bondlength (self.C_psi, no
            # /fjc), which is refinement-convergent to the continuum Debye value
            # (method-of-manufactured-solutions residual -> 0 as O(h^2); see
            # a manufactured-solution convergence study). Per project
            # policy this is the correct physics, INTENTIONALLY deviating from
            # unpatched Namics at fjc>1 (bug reported to the Namics authors).
            if self.geom == "cylindrical":
                cm, cp = rm * (em + e0), rp * (e0 + ep)
                C_geo = self.C_psi / np.pi
            else:                                # spherical
                cm, cp = rm ** 2 * (em + e0), rp ** 2 * (e0 + ep)
                C_geo = self.C_psi / (2.0 * np.pi) * fjc
            denom = cm + cp
            X = ((cm * psib[fjc - 1:M - fjc - 1]
                  + cp * psib[fjc + 1:M - fjc + 1]
                  + C_geo * q[iv] * lat.L[iv]) / denom)
            g_psi[iv] = psi_raw[iv] - X
            self.q, self.eps_prof = q, eps
            return g_psi
        # interior rows only (LG1Planar loops x = fjc .. MX+fjc-1): [1:-1]
        # slices would assume fjc = 1 and write garbage equations into the
        # ghost region at planar FJC_choices > 3 (unmasked rows: the solve
        # is unaffected, but the reported max|g| would be polluted)
        C2 = self.C_psi * 2.0 / fjc**2
        piv = slice(fjc, M - fjc)
        pm1 = slice(fjc - 1, M - fjc - 1)
        pp1 = slice(fjc + 1, M - fjc + 1)
        epsm = eps[pm1] + eps[piv]
        epsp = eps[piv] + eps[pp1]
        X = ((epsm * psib[pm1] + C2 * q[piv] + epsp * psib[pp1])
             / (epsm + epsp))
        if self.fixedPsi0:
            el = np.where(self.psiMask)[0]
            if not (len(el) and
                    all(i in (1, lat.M - 2) for i in el)):
                raise NotImplementedError(
                    "fixed surface potentials are supported on single "
                    "boundary-adjacent layers only (as in the Namics "
                    "examples); interior electrodes are not yet ported to PySFBox")
            x_target = np.zeros(lat.M)
            # the charge the electrode takes up from its reservoir (the
            # battery), per site in the units of q: Gauss's law at the
            # electrode site with the free-side flux, minus any charge
            # the electrode segment carries itself. Booked by the
            # electrode work term in grand_potential_density.
            self.q_electrode = np.zeros(lat.M)
            for i in el:
                nb = i + 1 if i == 1 else i - 1     # the free-side neighbour
                gh = i - 1 if i == 1 else i + 1     # the ghost-side cell
                psi0 = self.psi0_profile[i]
                # zero net charge on the electrode site, flux form:
                # (eps_gh+eps_el)(psi_bw - psi0)
                #   + (eps_el+eps_nb)(psi_nb - psi0) + C2*q_el = 0
                a_gh = eps[gh] + eps[i]
                a_nb = eps[i] + eps[nb]
                x_target[i] = psi0 - (a_nb * (psi_raw[nb] - psi0)
                                      + C2 * q[i]) / a_gh
                self.q_electrode[i] = (-a_nb * (psi_raw[nb] - psi0) / C2
                                       - q[i])
            free = ~self.psiMask[piv]
            g_psi[piv] = np.where(free, psi_raw[piv] - X,
                                  (psi_raw - x_target)[piv])
        else:
            g_psi[piv] = psi_raw[piv] - X
        self.q, self.eps_prof = q, eps
        return g_psi

    # ---- thermodynamics ------------------------------------------------------
    def grand_potential(self):
        """Grand potential Omega (per kT), cf. Namics GetGrandPotential, incl.
        the electrostatic tail for charged systems. Frozen chi partners are
        excluded from Omega (unlike free_energy). Validated against the compiled
        Namics at the shared convergence floor (~5e-8 relative,
        tests/two_brushes_quick.in); spot-check a new geometry the same way."""
        return self.lat.weighted_sum(self.grand_potential_density())

    def laplace_pressure(self):
        """Two-sided Laplace pressure Delta p = p_in - p_out =
        -omega(first interior) + omega(last interior) of a 1-gradient
        state (omega = the grand-potential density, a local -pressure;
        first = site fjc, last = site M - 2 fjc, the first refined site
        of the last physical layer, mirroring the first side). Returns
        (dp, omega_last). The kal `sys : Laplace_pressure` column AND the
        `Laplace_pressure` search target both read this (review finding
        58, 7 Oct 2026): Namics prints the one-sided -omega(first), which
        is the pressure difference only when the far side sits at
        omega = 0 (a reservoir-terminated box). With restricted phase
        formers the implied bulk can land on a phase or an absent state,
        omega(last) != 0, and the one-sided column is no pressure
        difference at all. At a droplet's surface of tension R_s,
        Delta p = 2 gamma/R_s (sphere), gamma/R_s (cylinder), 0 (flat)."""
        lat = self.lat
        gpd = self.grand_potential_density()
        w_last = float(gpd[lat.M - 2 * lat.fjc])
        return float(-gpd[lat.fjc]) + w_last, w_last

    def grand_potential_density(self):
        """Per-layer grand-potential density: the integrand whose lattice
        weighted_sum is grand_potential(). Exposed as the
        `sys : <name> : grand_potential_density` profile (the scalar split by
        site; the total is oracle-validated, the per-site split is not
        separately validated column-by-column)."""
        lat = self.lat
        omega = np.zeros(lat.M)
        for m in self.molecules.values():
            # translational term -(phi/N - phibulk/N): the chain number
            # density minus its bulk value, in the same floating-point
            # order as the development tree, so the two trees' outputs
            # stay byte-identical
            omega -= m.phi / m.N - m.phibulk / m.N
        omega -= self.alpha
        # chi pairs at species level: stateless mons + states (state chi
        # inheritance/overrides via _species_chi; state-resolved phi/side/
        # phibulk). FROZEN walls are EXCLUDED from BOTH indices (Namics
        # GetGrandPotential, system.cpp:3293/3301): the free-frozen chi is
        # carried implicitly by the shaped profile (the field term and the
        # explicit full-chi frozen term cancel in F), so adding it here
        # double-counts AND makes Omega depend on monomer declaration order
        # (the wall survives ksam only as the later partner). Found in the
        # 5 Jul 2026 physics review; free_energy's frozen handling is correct
        # and stays. NOTE: this differs from free_energy on purpose.
        sp = [p for p in self.partners if p.seg.freedom != "frozen"]
        for a in range(len(sp)):
            pb_a = (self.phibulk_seg.get(sp[a].seg.name, 0.0)
                    * sp[a].alphabulk)
            for b in range(a + 1, len(sp)):
                chi = _species_chi(sp[a], sp[b])
                if chi:
                    # SYMMETRIC per-site booking, (phi_a<phi_b> +
                    # phi_b<phi_a>)/2, matching Namics' full double loop
                    # with chi/2 (GetGrandPotential): the L-weighted TOTAL
                    # is identical either way (the site average is
                    # self-adjoint under the volume weights -- detailed
                    # balance), but the one-sided phi_a<phi_b> booking put
                    # the contact energy of a steep interface on the wrong
                    # layers wherever l_-1 != l_+1, deviating ~3-5% from
                    # the oracle's grand_potential_density at a spherical
                    # center (oracle-checked 21 Jul 2026; the fjc=1 micelle
                    # matches the oracle's per-site value to 3e-9 after
                    # symmetrising).
                    sa = self.side_phi(sp[b].seg, sp[b].phi)
                    sb = self.side_phi(sp[a].seg, sp[a].phi)
                    pb_b = (self.phibulk_seg.get(sp[b].seg.name, 0.0)
                            * sp[b].alphabulk)
                    omega -= chi * (0.5 * (sp[a].phi * sa + sp[b].phi * sb)
                                    - pb_a * pb_b)
        if self.constraintfields:
            # constraint-field work term (Namics GetGrandPotential,
            # system.cpp:3288-3292: GP_accum -= log(BETA)*(phiA-phiB) with
            # the accumulated sign flipped at the end, Norm(GP,-1) -- so the
            # final Omega density gets -beta*(phitot_A - phitot_B))
            molA, molB = self.delta_molecules
            omega -= self.beta * (molA.phi - molB.phi)
        if self.charged:
            # Namics GetGrandPotential charged tail: add EE*eps - q*psi/2
            # inside the KSAM mask, plus q*psi/2 at the masked (surface)
            # sites — the surface-charge work term
            omega += self.EE * self.eps_prof - 0.5 * self.q * self.psi
            omega *= self.ksam
            omega += (1.0 - self.ksam) * 0.5 * self.q * self.psi
            # electrode work term: at a fixed potential the electrode
            # exchanges charge q_el with its reservoir. Integrated by
            # parts, the field-energy tail above already holds the
            # boundary term +q_el*psi0/2 (the fixed-charge booking of that
            # charge); the constant-potential grand potential subtracts
            # the reservoir work q_el*psi0: net -q_el*psi0/2 at the
            # electrode site. Then dOmega/dpsi0 = -sigma (Lippmann). Absent
            # in Namics GetGrandPotential (review finding 3, 7 Oct 2026).
            omega -= 0.5 * self.q_electrode * self.psi
            return omega
        omega *= self.ksam
        return omega

    def free_energy(self):
        """SF Helmholtz free energy (per kT), cf. System::GetFreeEnergy, incl.
        the electrostatic field terms and the multistate (weak-charge)
        contributions. Validated against the compiled Namics (~6e-9 relative,
        tests/two_brushes_quick.in) and by dF/dtheta = mu (finite difference);
        spot-check a new geometry the same way."""
        return self.lat.weighted_sum(self.free_energy_density())

    def free_energy_density(self):
        """Per-layer Helmholtz free-energy density: the integrand whose lattice
        weighted_sum is free_energy(). Exposed as the
        `sys : <name> : free_energy_density` profile (the scalar split by site;
        the total is oracle-validated, the per-site split is not separately
        validated column-by-column)."""
        lat = self.lat
        F = np.zeros(lat.M)
        # translational entropy: sum_mol phi_mol * log(N n / GN) / N
        for m in self.molecules.values():
            theta = lat.weighted_sum(m.phi)            # = N * n
            if theta <= 0:
                continue
            constant = (np.log(theta) - m.lnGN) / m.N  # log(N n / GN)/N
            F += m.phi * constant
        # field term: -sum_{free species} phi_i * u_i. For multistate mons
        # this is -sum_s phi_s u_s, which carries the state-mixing entropy
        # implicitly: -sum_s phi_s u_s = phi_X ln G_X
        # + sum_s phi_s ln(alpha_s/alphabulk_s) (there
        # is NO additional explicit state term -- in the semi-grand
        # ensemble the bath exchange compensates it, verified against the
        # oracle via dF/dtheta = mu on a bulk weak acid, 4 Jul 2026)
        u = self._u_current
        for i, sp in enumerate(self.it_species):
            F -= sp.phi * u[i]
        # state-redistribution term (review finding 5, 7 Oct 2026). u_s is
        # bulk-referenced: u_s = sum_t chi_st (<phi_t> - phi_t^b) + alpha,
        # so the annealed weight alphabulk_s exp(-u_s) carries
        # c_s = sum_t chi_st phi_t^b as if it were an intrinsic state
        # energy, and the field term above books +sum_s c_s phi_s too
        # much. The part c_s alphabulk_s phi_X is linear in the molecule
        # amounts and is matched in mu; the rest, c_s (phi_s - alphabulk_s
        # phi_X), depends on the local state fractions and is removed
        # here. Zero for stateless species, when all states of a mon share
        # c_s, and in a uniform bulk -- which is why the state-independent-
        # chi validations never saw it. Restores dF/dn = mu and
        # F = Omega + sum n mu with state-dependent chi.
        if self.has_states:
            pbulk = {p.name: (self.phibulk_seg.get(p.seg.name, 0.0)
                              * p.alphabulk) for p in self.partners}
            for sp in self.it_species:
                if sp.state is None:
                    continue
                c_s = sum(_species_chi(sp, p) * pbulk[p.name]
                          for p in self.partners)
                if c_s:
                    F -= c_s * (sp.phi - sp.alphabulk * sp.seg.phi)
        if self.constraintfields:
            # the -phi*u accounting for the constraint field: molecule A's
            # segments felt u+beta, B's u-beta, and beta is not in the
            # per-species u above (Namics GetFreeEnergy, system.cpp:3057-3061:
            # F += log(BETA)*(phitot_A - phitot_B) with log(BETA) = -beta)
            molA, molB = self.delta_molecules
            F -= self.beta * (molA.phi - molB.phi)
        if self.charged:
            # the field term uses the FULL potential, like Namics' phi*ln(G1)
            # with G1 = exp(-(u + v*psi - eps*EE)): add the electrostatic
            # parts (on free sites raw and effective psi coincide, and the
            # KSAM mask below removes the electrode/surface sites). This was
            # missing from the fixed-charge port (GP was oracle-exact, F was
            # off by the field terms; found in the weak-charge validation).
            for sp in self.it_species:
                if sp.valence != 0.0:
                    F -= sp.phi * sp.valence * self.psi
                F += sp.phi * sp.seg.epsilon * self.EE
        # chi term: sum_{j free} sum_k chi'_jk phi_j <phi_k>,
        # chi halved unless k is frozen (double-counting convention);
        # species-resolved (states as partners, mons never when multistate)
        sides = {p.name: self.side_phi(p.seg, p.phi) for p in self.partners}
        for sj in self.it_species:                     # j must be non-frozen
            for p in self.partners:
                chi = _species_chi(sj, p)
                if not chi:
                    continue
                chi_eff = chi if p.seg.freedom == "frozen" else 0.5 * chi
                F += chi_eff * sj.phi * sides[p.name]
        # per-molecule INTRAMOLECULAR chi self-energy reference (Namics
        # GetFreeEnergy, system.cpp:3166-3196): the SF free energy uses a
        # pure-component reference, so each molecule subtracts its own
        # intramolecular contacts -1/2 sum_jk chi_jk f_j f_k with f_j the
        # fraction of the molecule that is species j (block fraction *
        # alphabulk for states). Identically 0 for homopolymers and chi-free
        # systems (why chi-free F-validations pass); required so that
        # F = Omega + sum n*mu (the mu double-sum carries the matching f_j
        # f_k piece). Found in the 5 Jul 2026 physics review.
        for m in self.molecules.values():
            frac = {}
            for mon, cnt in m.blocks:
                frac[mon] = frac.get(mon, 0.0) + cnt / m.N
            const = 0.0
            for sj in self.partners:
                fj = frac.get(sj.seg.name, 0.0) * sj.alphabulk
                if fj == 0.0:
                    continue
                for sk in self.partners:
                    chi = _species_chi(sj, sk)
                    if chi:
                        fk = frac.get(sk.seg.name, 0.0) * sk.alphabulk
                        const -= 0.5 * chi * fj * fk
            if const:
                F += m.phi * const
        F *= self.ksam
        if self.charged:
            # Namics GetFreeEnergy charged term: + q*psi/2 (added after the
            # KSAM cleanup, so it includes the surface sites)
            F = F + 0.5 * self.q * self.psi
            # the electrode work term, same booking as in
            # grand_potential_density (F = Omega + sum n mu keeps closing)
            F = F - 0.5 * self.q_electrode * self.psi
        return F

    def chemical_potential(self, mol):
        """SF molecule chemical potential (per kT), bulk reference; cf. Namics
        System::ComputeMu (uncharged, single-state, pos==M case):

            mu_i = ln(theta_i / GN_i) + 1
                   - N_i * sum_k  phibulk_k / N_k                  (over molecules k)
                   - N_i * sum_{j,k} (chi_jk/2)
                                    * (phibulk_j - f_ij) (phibulk_k - f_ik)

        where f_ij is the fraction of molecule i made of segment type j, phibulk_j
        the bulk fraction of segment j, and the segment double-sum runs over all
        segment types (frozen surfaces drop out: their bulk fraction and molecule
        fraction are both zero)."""
        N = mol.N
        theta = mol.get_theta()
        if theta <= 0 or not np.isfinite(mol.lnGN):
            return 0.0
        mu = np.log(theta) - mol.lnGN + 1.0
        mu -= N * sum(m.phibulk / m.N for m in self.molecules.values())
        frac = {}
        for mon, cnt in mol.blocks:
            frac[mon] = frac.get(mon, 0.0) + cnt / N
        # species-level double sum: a state contributes its bulk-fraction
        # share of both the segment bulk density and the molecule fraction
        # (phibulk_s = phibulk_X*alphabulk_s, f_s = f_X*alphabulk_s; cf.
        # Namics CreateMu state blocks, with the partner-index bug at
        # system.cpp:3577 fixed)
        chi_sum = 0.0
        sp = self.partners
        for sj in sp:
            pbj = (self.phibulk_seg.get(sj.seg.name, 0.0) * sj.alphabulk)
            fj = frac.get(sj.seg.name, 0.0) * sj.alphabulk
            for sk in sp:
                chi = _species_chi(sj, sk)
                if chi:
                    pbk = (self.phibulk_seg.get(sk.seg.name, 0.0)
                           * sk.alphabulk)
                    fk = frac.get(sk.seg.name, 0.0) * sk.alphabulk
                    chi_sum += 0.5 * chi * (pbj - fj) * (pbk - fk)
        mu -= N * chi_sum
        return mu

    # ---- the characteristic function X (Namics sys : X) ---------------------
    def characteristic_X(self):
        """kal `sys : NN : X`: the user-composed potential (Namics
        system.cpp:1588-1627)

            X = F - sum_listed n_i mu_i - sum_(s_i,s_j,n) n theta_si mu_sj

        F = free_energy, n_i = kal mol n, mu_i = kal mol mu, theta_si = the
        amount of state s_i (sum over sites, all molecules carrying its
        segment; Namics state_theta), mu_sj = mu-<s_j> of the monomeric
        molecule carrying s_j. Every term is read through get_value, so a
        column that prints NiN makes X NiN.
        Deliberate deviation: Namics' mu-<state> accumulates ln(alphabulk)
        across output events (bug #3), so its X is wrong from the second
        kal row on whenever a state term is used; PySFBox computes mu-<s>
        fresh. 'X : F' (no entries) prints F, where Namics prints nothing.
        Returns ('real', X), or (None, None) -> NiN."""
        if self.X_mols is None:
            if "X_undeclared" not in _WARN_NOTES:
                _WARN_NOTES.add("X_undeclared")
                print("  note: kal sys : X needs its definition, e.g. "
                      "sys : NN : X : F-water-Na-Cl (X : ? prints the "
                      "help) -> NiN")
            return None, None
        _t, X = self.get_value("sys", "", "free_energy")
        if X is None:
            return None, None
        for nm in self.X_mols:
            _t, n = self.get_value("mol", nm, "n")
            _t, mu = self.get_value("mol", nm, "mu")
            if n is None or mu is None:
                return None, None
            X -= n * mu
        for si, sj, host, n in self.X_states:
            st = next(st for sg in self.segments.values()
                      for st in sg.states if st.name == si)
            _t, mu = self.get_value("mol", host, "mu-" + sj)
            if mu is None:
                return None, None
            X -= n * self.lat.weighted_sum(st.phi) * mu
        return "real", float(X)

    # ---- output property lookup (kal) -----------------------------------------
    def get_value(self, key, name, prop):
        """Returns ('int'|'real'|None, value). None -> NiN, like Namics."""
        lat = self.lat
        if prop.endswith("-value"):
            alias = prop[:-6]
            v = last(self.settings.get(("alias", alias), {}), "value")
            if v is not None:
                try:
                    fv = float(v)
                except ValueError:
                    # a composition alias (e.g. '(A)10(B)5'): no number to
                    # echo. NiN instead of a ValueError AFTER the solve,
                    # which lost the row and every later start (review
                    # 6 Oct 2026, #78)
                    print(f"  note: alias '{alias}' has the non-numeric "
                          f"value '{v}'; kal {key}:{name}:{prop} -> NiN")
                    return None, None
                return ("int", int(fv)) if fv == int(fv) else ("real", fv)
        if key == "lat":
            # the INPUT layer counts, like Namics (MX/fjc, MY/fjc, ...):
            # refined cells are an internal detail (review 7 Oct 2026, #63)
            layers = (lat.dims if lat.gradients > 1
                      else (lat.MX // lat.fjc,))
            axis = {"n_layers": 0, "n_layers_x": 0, "n_layers_y": 1,
                    "n_layers_z": 2}.get(prop)
            if axis is not None and axis < len(layers):
                return "int", int(layers[axis])
            if prop == "volume":
                return "real", self.lat.volume
        if key == "sys":
            if prop == "grand_potential":
                return "real", self.grand_potential()
            # Laplace pressure, TWO-SIDED: -omega(first) + omega(last)
            # (laplace_pressure(); the same quantity the Laplace_pressure
            # search drives). DELIBERATE deviation from Namics PushOutput,
            # which prints -GrandPotentialDensity[fjc] only (review
            # finding 58, 7 Oct 2026): identical whenever the far side sits
            # at omega = 0, a note otherwise. The first-site omega matches
            # the oracle at the solver floor (3e-9, fjc=1 spherical
            # micelle) SINCE the symmetric chi booking in
            # grand_potential_density (21 Jul 2026).
            if prop == "Laplace_pressure" and self.lat.gradients == 1:
                dp, w_last = self.laplace_pressure()
                if (abs(w_last) > max(1e-7, 1e-4 * abs(dp))
                        and "laplace_two_sided" not in _WARN_NOTES):
                    _WARN_NOTES.add("laplace_two_sided")
                    print(f"  note: sys : Laplace_pressure: the far side "
                          f"is not at omega = 0 (omega(last layer) = "
                          f"{w_last:.3e}); PySFBox prints the two-sided "
                          f"-omega(first) + omega(last) = {dp:.6e}, while "
                          f"Namics prints -omega(first) only "
                          f"(= {dp - w_last:.6e} here). Note printed once")
                return "real", dp
            if prop == "phi_ratio" and self.constraintfields:
                return "real", self.phi_ratio
            # free_energy (po) is Namics' "GP + n*mu" route to the same F
            if prop == "free_energy" or prop.replace(" ", "") == "free_energy(po)":
                return "real", self.free_energy()
            if prop == "iterations":
                return "int", self.iterations
            if prop == "residual":
                return "real", self.residual_norm
            if prop == "X":
                return self.characteristic_X()
            if prop == "kJ0":
                # Namics' compute_kJ0 column (a bare moment of the
                # grand-potential density): NiN, as in Namics without
                # compute_kJ0, plus a one-time note
                if "kJ0" not in _WARN_NOTES:
                    _WARN_NOTES.add("kJ0")
                    print("  note: sys : kJ0 is Namics' compute_kJ0 column "
                          "(the bare first moment of the grand-potential "
                          "density about z = 0, not the Helfrich kappa*J0 "
                          "of a self-consistent film; not supported) -> "
                          "NiN. The Helfrich constants follow from curved "
                          "ladders (grand potential of cylinders and "
                          "spheres against 1/R)")
                return None, None
        if key == "state":
            for seg in self.segments.values():
                for st in seg.states:
                    if st.name == name:
                        if prop == "alphabulk":
                            return "real", st.alphabulk
                        if prop == "valence":
                            return "real", st.valence
                        if prop == "phibulk":
                            return "real", st.phibulk
                        if prop == "theta":
                            return "real", lat.weighted_sum(st.phi)
                        if prop == "theta_exc":
                            return "real", (lat.weighted_sum(st.phi)
                                            - lat.L_sum * st.phibulk)
        if key == "mol" and name in self.molecules:
            m = self.molecules[name]
            # per-state chemical potential mu-STATE = Mu + ln(alphabulk_s)
            # (Namics molecule.cpp:2079-2088, chainlength-1 molecules only;
            # computed FRESH here -- Namics accumulates across output
            # events, a live bug reported to the Namics authors)
            if prop.startswith("mu-") and m.N == 1:
                seg0 = m.seq[0]
                for st in seg0.states:
                    if prop == f"mu-{st.name}":
                        return "real", (self.chemical_potential(m)
                                        + np.log(st.alphabulk))
            theta = m.get_theta()

            def _gn():
                # the propagators keep GN = exp(min(lnGN, 700)) (a display
                # value; mu/theta use lnGN). Report the TRUE GN: exact up to
                # the double range, inf beyond it (with a note) rather than
                # a silently clamped exp(700) (review 7 Oct 2026, #77).
                if m.lnGN < 700.0:
                    return m.GN
                if m.lnGN < 709.78:
                    return float(np.exp(m.lnGN))
                print(f"  warning: kal mol:{name}:GN overflows a double "
                      f"(ln GN = {m.lnGN:.6g}); printing inf (mu and theta "
                      "use ln GN and are exact)")
                return float("inf")

            table = {"theta": m.get_theta, "theta_exc": m.get_theta_exc,
                     # Namics accepts both spellings (oracle-checked 21 Jul
                     # 2026: identical columns)
                     "thetaexc": m.get_theta_exc,
                     "phibulk": lambda: m.phibulk,
                     "Mu": lambda: self.chemical_potential(m),
                     "MU": lambda: self.chemical_potential(m),
                     "mu": lambda: self.chemical_potential(m),
                     "n": lambda: theta / m.N,
                     "N": lambda: m.N, "GN": _gn,
                     "chainlength": lambda: m.N,
                     "phiMax": lambda: float(m.phi[self.lat.interior].max()),
                     # Namics phiM = phitot[M-2*fjc]: phi at the last interior
                     # layer (the bulk/reservoir side), NOT the maximum.
                     "phiM": lambda: float(m.phi[self.lat.M - 2 * self.lat.fjc]
                                           if self.lat.gradients == 1
                                           else m.phi[self.lat.interior][-1]),
                     "phiMin": lambda: float(m.phi[self.lat.interior].min())}
            if prop in table:
                v = table[prop]()
                return ("int", v) if isinstance(v, int) else ("real", v)
        if key == "mon" and name in self.segments:
            s = self.segments[name]
            # per-state scalars, pushed on the parent mon exactly like
            # Namics (segment.cpp:1844-1885): alphabulk_S, valence_S,
            # phibulk_S, theta_S, theta_exc_S
            for st in s.states:
                if prop == f"alphabulk_{st.name}":
                    return "real", st.alphabulk
                if prop == f"valence_{st.name}":
                    return "real", st.valence
                if prop == f"phibulk_{st.name}":
                    return "real", st.phibulk
                if prop == f"theta_{st.name}":
                    return "real", lat.weighted_sum(st.phi)
                if prop == f"theta_exc_{st.name}":
                    return "real", (lat.weighted_sum(st.phi)
                                    - lat.L_sum * st.phibulk)
                # system-average state fraction alpha-STATE = theta_state/theta_mon
                # (Namics 'mon : X : alpha-S': the mean degree of that state
                # over the segment, e.g. the average degree of dissociation)
                if prop in (f"alpha-{st.name}", f"alpha_{st.name}"):
                    tot = lat.weighted_sum(s.phi)
                    return "real", (lat.weighted_sum(st.phi) / tot
                                    if tot > 0 else 0.0)
            pk = prop.replace(" ", "")           # chi_X / chi-X, like Segment
            if (pk.startswith("chi_") or pk.startswith("chi-")) \
                    and pk[4:] in self.segments:
                return "real", s.chi_with(self.segments[pk[4:]])
            phib = self.phibulk_seg.get(name, 0.0)
            theta = lat.weighted_sum(s.phi)
            # excess in the lattice's own measure (L_sum == volume except
            # curved fjc>1; see lattice.py) so uniform bulk => exactly 0
            theta_exc = theta - lat.L_sum * phib
            if prop == "theta":
                return "real", theta
            if prop == "theta_exc":
                return "real", theta_exc
            if prop == "phibulk":
                return "real", phib
            if prop in ("1st_M_phi_z", "2nd_M_phi_z", "fluctuations", "RMS"):
                # moments of the EXCESS profile, normalised by theta_exc.
                # A uniform state has theta_exc = round-off and the ratio
                # an O(box) number (Namics prints that garbage, dividing
                # whenever theta_exc != 0 exactly); report nan instead
                # when |theta_exc| is round-off relative to the amount.
                # (review 7 Oct 2026, #62)
                if abs(theta_exc) <= 1e-10 * max(theta, lat.L_sum * phib):
                    return "real", float("nan")
                m1 = lat.moment(s.phi, phib, 1) / theta_exc
                m2 = lat.moment(s.phi, phib, 2) / theta_exc
                if prop == "1st_M_phi_z":
                    return "real", m1
                if prop == "2nd_M_phi_z":
                    return "real", m2
                if prop == "RMS":
                    # M2 < 0 (an excess that changes sign): no real RMS;
                    # Namics' pow(M2, 0.5) prints nan here too
                    return "real", np.sqrt(m2) if m2 >= 0 else float("nan")
                fl = m2 - m1 * m1
                return "real", np.sqrt(fl) if fl > 0 else 0.0
        return None, None

    def _full_potential(self, i):
        """The potential iteration species i actually feels, in kT: the
        exponent of its Boltzmann weight in compute_phis, u + valence*psi
        - eps*EE. This is what Namics' Seg->u holds after PutU and what
        `pro mon : X : u` / `u-<state>` print; the bare iteration variable
        u omits the electrostatic and Born parts (review finding 50,
        7 Oct 2026: counter-ions at a charged wall printed a repulsive u
        where they accumulate). Uncharged species: unchanged."""
        sp = self.it_species[i]
        u_tot = self._last_u[i]
        if self.charged:
            if sp.valence != 0.0:
                u_tot = u_tot + sp.valence * self.psi
            u_tot = u_tot - sp.seg.epsilon * self.EE
        return u_tot

    def get_profile(self, key, name, prop):
        """Profile arrays for .pro output (interior incl. ghosts)."""
        if key == "mol" and name in self.molecules and prop == "phi":
            return self.molecules[name].phi
        if key == "mol" and name in self.molecules and prop.startswith("phi_"):
            # per-monomer contribution to a molecule's density
            # (Namics underscore notation, e.g. mol : poly : phi_A). Zero
            # when the molecule contains no such segment; NiN if the mon
            # name is unknown.
            mon = prop[len("phi_"):]
            per = self.molecules[name].phi_per_seg
            if mon in per:
                return per[mon]
            if mon in self.segments:
                return np.zeros(self.lat.M)
            return None
        if key == "mon" and name in self.segments and prop == "phi":
            return self.segments[name].phi
        if key == "mon" and name in self.segments:
            # per-state profiles on the parent mon: phi-S / alpha-S / u-S
            # (hyphenated, like Namics segment.cpp:1869-1885; Namics' own
            # phi-<first state> output is broken by a profile-number
            # collision -- PySFBox emits the correct profile)
            s = self.segments[name]
            for st in s.states:
                if prop == f"phi-{st.name}":
                    return st.phi
                if prop == f"alpha-{st.name}":
                    return st.alpha_prof
                if prop == f"u-{st.name}":
                    for i, sp in enumerate(self.it_species):
                        if sp.state is st:
                            return self._full_potential(i)
        if key == "mon" and name in self.segments and prop == "u":
            for i, sp in enumerate(self.it_species):
                if sp.state is None and sp.name == name:
                    return self._full_potential(i)
            return None
        if key == "sys" and prop == "alpha":
            return self.alpha
        if key == "sys" and prop == "beta" and self.constraintfields:
            # the delta-constraint Lagrange field (PySFBox extension --
            # Namics does not expose it; nonzero only on delta_range sites)
            return self.beta
        if key == "sys" and prop == "free_energy_density":
            return self.free_energy_density()
        if key == "sys" and prop == "grand_potential_density":
            return self.grand_potential_density()
        if key == "sys" and self.charged and prop == "psi":
            return self.psi
        if key == "sys" and self.charged and prop == "q":
            return self.q
        if key == "sys" and self.charged and prop == "eps":
            return self.eps_prof            # local relative permittivity
        return None

    def fill_profile_bounds(self, key, name, prop, arr):
        """Return a copy of a .pro profile with its ghost layers filled the
        Namics way, for `output : pro : write_bounds : true`. POTENTIALS /
        intensive fields (psi, u, alpha, and per-state u-/alpha-) MIRROR
        regardless of the wall (Namics set_M_bounds: the wall condition is
        zero-field, not zero-value). DENSITIES (phi, q) use set_bounds:
        surface-zero at a solid wall, mirror at a mirror bound -- and a frozen
        segment sitting on that surface overrides its ghost to the wall
        density (1.0), reproducing e.g. the frozen S wall in silica.in."""
        lat = self.lat
        if key == "sys" and prop in ("free_energy_density",
                                     "grand_potential_density"):
            # INTEGRAND densities keep ZERO ghosts, as Namics writes them:
            # they are summed over the interior only, and a mirror-filled
            # ghost row made the written column no longer sum to Omega/F
            # (review 7 Oct 2026, #76)
            out = np.array(arr, dtype=float)
            out[~lat.interior] = 0.0
            return out
        if (prop in ("psi", "u", "alpha")
                or prop.startswith("u-") or prop.startswith("alpha-")):
            return lat.set_mirror_bounds(arr)
        lower = upper = None
        if key == "mon" and name in self.segments:
            seg = self.segments[name]
            if getattr(seg, "on_lower_surface", False):
                lower = 1.0
            if getattr(seg, "on_upper_surface", False):
                upper = 1.0
        return lat.set_bounds(arr, lower_value=lower, upper_value=upper)

    # ---- analytic initial guesses (cf. Namics system.cpp:399-427) ------------
    def initial_guess(self):
        """Analytic starting potentials from `sys : ... : initial_guess`
        (Namics System::PrepareForCalculations + Segment::PutAdsorptionGuess
        / PutMembranePotential). Returns a full potential vector x0, or None
        for previous_result/none. The runner applies this to the FIRST
        calculation only, exactly like Namics (start == 1, then the type
        resets to previous_result)."""
        kind = None
        for _, params in get_blocks(self.settings, "sys"):
            kind = last(params, "initial_guess", kind)
        if kind in (None, "previous_result", "none"):
            return None
        if kind not in ("polymer_adsorption", "membrane", "micelle"):
            raise NotImplementedError(
                f"sys : initial_guess : {kind} is not supported in PySFBox "
                "(supported: polymer_adsorption, membrane, micelle, "
                "previous_result, none); guess files and membrane_torus "
                "need the C++ Namics")
        if self.lat.gradients > 1:
            # the analytic adsorption/membrane guesses are 1-gradient; N-D
            # calculations cold-start (the runner still warm-starts scans).
            # Say so: a declared guess must never vanish silently (review
            # 6 Oct 2026, #34 -- an unknown kind used to be dropped here too)
            print(f"  note: initial_guess : {kind} is 1-gradient only; this "
                  "N-D calculation cold-starts")
            return None
        lat = self.lat
        U = np.zeros((len(self.it_species), lat.M))
        interior = np.zeros(lat.M, dtype=bool)
        interior[lat.iv] = True
        if kind == "polymer_adsorption":
            # u = -lambda*chi at sites adjacent to each solid's mask
            # (segment.cpp:1259-1291; lambda = 0.25 hexagonal, 1/6 else --
            # which is exactly the lattice lambda_1). NOTE: Namics
            # system.cpp:400-404 indexes Seg[i] with the FrozenList
            # POSITION (a latent bug: correct only when all frozen mons
            # are declared first); PySFBox deliberately uses the actual
            # frozen segments, so multi-wall guesses can differ from the
            # C++ when frozen mons are declared late -- ours is the intent.
            for fs in self.frozen:
                m = fs.phi                      # range mask incl. wall ghosts
                adj = np.zeros(lat.M, dtype=bool)
                adj[1:-1] = (m[:-2] > 0.5) | (m[2:] > 0.5)
                adj &= interior
                wall = _Species(fs)
                for j, sp in enumerate(self.it_species):
                    chi = _species_chi(sp, wall)
                    if chi != 0.0:
                        U[j][adj] = -lat.lam * chi
        elif kind in ("membrane", "micelle"):
            # u = -log(1.8) on the first 4*fjc interior layers for segments
            # that repel the solvent (chi > 0.8), cf. segment.cpp:1344
            solv = self.solvent.seq[0]
            found = False
            for j, sp in enumerate(self.it_species):
                if sp.seg.chi_with(solv) > 0.8:
                    found = True
                    U[j][lat.fjc: lat.fjc + 4 * lat.fjc] = -np.log(1.8)
            if not found:
                print("  note: no 'solvo'phobic segment found; the "
                      f"{kind} initial guess may not help (as in Namics)")
        # pad with zeros for the trailing psi block (charged systems), else
        # solve() silently rejects the guess by size (latent bug, fixed
        # 21 Jul 2026 in lockstep with the dev tree)
        x0 = U.ravel()
        return np.concatenate([x0, np.zeros(self.n_var() - x0.size)])

    # ---- solver --------------------------------------------------------------
    # the dense pseudohessian stores n^2 Hessian factors and pays O(n^2) work
    # per secant update, so it is the primary only below this many variables
    # (n_it_species * M); above it we stay with Anderson only.
    NEWTON_MAX_VARS = 4000

    def solve(self, x0=None, tolerance=1e-7, iterationlimit=1000,
              deltamax=0.1, m_anderson=8, warmup=200, verbose=False,
              method="pseudohessian", engine=None):
        """Namics-style solver cascade. With the default method (pseudohessian,
        the Namics default) and a problem small enough for its dense Hessian
        (n_var <= NEWTON_MAX_VARS), the translated Namics pseudohessian
        quasi-Newton (sfnewton.py) runs as the PRIMARY solver, from the
        warm-start guess or from zeros -- exactly like Namics, and typically in
        Namics-like iteration counts. Anderson-accelerated Picard (with a
        damped-Picard warmup on cold starts) is the fallback, and remains the
        primary for large problems and for explicitly non-default methods.
        Two final rescue stages handle the stiffest cases: an extended-budget
        pseudohessian from the original guess, then a one-shot full-Hessian
        anchor."""
        # `engine` and any unrecognised `method` are accepted but ignored:
        # PySFBox runs on pure NumPy, and an unknown method simply uses the
        # default solver cascade below (the runner notes it). Both give the
        # same converged result, so this is never silently wrong.
        n = self.n_var()
        x0 = None if (x0 is None or x0.size != n) else x0
        # method:hessian = Namics' full-Hessian mode (recomputed every
        # iteration; solve_scf.cpp:195), using the finite-difference
        # numhessian -- expensive beyond a few hundred variables.
        full_h = (method == "hessian")
        ph_primary = (method in (None, "pseudohessian", "hessian")
                      and n <= self.NEWTON_MAX_VARS)
        it = 0
        x, err, ok = np.zeros(n), np.inf, False
        if ph_primary:
            # (0) pseudohessian primary, Namics budget semantics. A wild first
            # step can kill the propagator (full underflow raises
            # FloatingPointError in compute_phi); treat that as a failed stage
            # and fall through to Anderson instead of aborting.
            try:
                xp, itp, errp, okp = self._solve_pseudohessian(
                    x0, tolerance, iterationlimit, deltamax, verbose=verbose,
                    full_hessian=full_h)
                it += itp
                if okp or errp < err:
                    x, err, ok = xp, errp, okp
            except FloatingPointError:
                pass
        if not ok:
            # (1) Anderson(+warmup on cold starts): the fallback -- and the
            # primary for large n_var or non-default methods.
            if ph_primary and verbose:
                print(f"    pseudohessian stalled at max|g| = {err:.2e}; "
                      f"Anderson fallback")
            xa, ita, erra, oka = self._solve_anderson(
                x0, tolerance, iterationlimit, deltamax, m_anderson, warmup,
                verbose)
            it += ita
            if oka or erra < err:
                x, err, ok = xa, erra, oka
        if not ok and n <= self.NEWTON_MAX_VARS:
            # (2) extended-budget pseudohessian from the ORIGINAL guess (the
            # warm-start solution, or cold) -- it is a better basin than the
            # iterate a stalled stage drifted to. Skipped when stage (0) already
            # ran from the same guess with at least this budget.
            budget = max(int(iterationlimit), 3000)
            if not ph_primary or budget > int(iterationlimit):
                if verbose:
                    print(f"    stalled at max|g| = {err:.2e}; extended "
                          f"pseudohessian rescue")
                try:
                    xn, itn, errn, okn = self._solve_pseudohessian(
                        x0, tolerance, budget, deltamax, anchor_full=False,
                        verbose=verbose)
                    it += itn
                    if okn or errn < err:
                        x, err, ok = xn, errn, okn
                except FloatingPointError:
                    pass                      # keep the best iterate so far
            # (3) if the residual is small but not converged, anchor with the
            # full numerical Hessian (once) from there and continue with secant
            # updates. Gated on a small residual (cf. Namics'
            # minAccuracyForHessian: the full Newton step blows up when the
            # Jacobian is near-singular far from the solution) and on n_var
            # (numhessian costs n_var+1 residual evaluations).
            if not ok and err < 0.5 and n <= 1000:
                try:
                    xf, itf, errf, okf = self._solve_pseudohessian(
                        x, tolerance, budget, deltamax, anchor_full=True,
                        verbose=verbose)
                    it += itf
                    if okf or errf < err:
                        x, err, ok = xf, errf, okf
                except FloatingPointError:
                    pass                      # keep the best iterate so far
        # sync all observables (phi, alpha, GN, ...) to the returned x
        self.residual(x)
        self.iterations, self.residual_norm = it, err
        self._last_u = self.unpack(x)
        progress.clear()
        if ok:
            self._check_bulk_signs()
        if not ok:
            raise RuntimeError(
                f"no convergence in {it} iterations (max|g| = {err:.2e}); "
                f"try a smaller deltamax")
        return x, it, err

    def _check_bulk_signs(self):
        """Refuse a converged state whose bulk needs a NEGATIVE amount of
        the solvent or the neutralizer -- the two fractions _update_bulk
        sets by closure. A negative neutralizer gets zero density
        (lnC = -inf) while the electroneutrality bookkeeping still counts
        it, so the far field silently drifts to a different reservoir
        (review 6 Oct 2026, #29; Namics warns, system.cpp:2582)."""
        if self.solvent.phibulk < 0:
            # cf. Namics' refusal (system.cpp:2604): the constrained amounts
            # imply more material than an equilibrium bulk can hold
            raise RuntimeError(
                f"converged to a state with negative solvent bulk fraction "
                f"({self.solvent.phibulk:.3e}); the restricted theta values "
                f"exceed what the box can hold in equilibrium")
        nm = self.neutralizer
        if nm is not None and nm.phibulk < -1e-12:
            raise RuntimeError(
                f"the bulk needs a NEGATIVE amount of the neutralizer "
                f"{nm.name} (phibulk {nm.phibulk:.3e}): the other bulk "
                "charges already carry its sign. Choose a neutralizer of "
                "the opposite charge, or add salt")

    def _solve_anderson(self, x0, tolerance, iterationlimit, deltamax,
                        m_anderson, warmup, verbose):
        """Anderson mixing (type II) on a damped-Picard baseline. Returns
        (x_best, iterations, best_err, converged)."""
        residual = self.residual
        n = self.n_var()
        x = np.zeros(n) if x0 is None else x0.copy()
        warm = 0 if x0 is not None else int(warmup)
        X_hist, G_hist = [], []
        delta = float(deltamax)
        best_err = np.inf
        x_best = x.copy()
        since_improve = 0
        it = 0
        for it in range(1, int(iterationlimit) + 1):
            g = residual(x)
            err = np.abs(g).max()
            progress.update("anderson", it, float(err), float(delta))
            if verbose and it % 100 == 0:
                print(f"    it {it:5d}  max|g| = {err:.3e}  delta = {delta:.1e}")
            if err < tolerance:
                progress.clear()
                return x, it, err, True
            if not np.isfinite(err) or err > 1e4 * max(best_err, 1.0):
                # blow-up: roll back to best, shrink step, reset history
                # (cf. the Hessian-reset / deltamax-decay rescue in Namics)
                x = x_best.copy()
                delta = max(delta * 0.5, 1e-4)
                X_hist, G_hist = [], []
                since_improve = 0
                continue
            if err < best_err - 1e-15:
                best_err, x_best, since_improve = err, x.copy(), 0
            else:
                since_improve += 1
                if since_improve > 400:  # stagnation: gentle step decay
                    delta = max(delta * 0.7, 1e-4)
                    X_hist, G_hist = [], []
                    since_improve = 0
            gc = np.clip(g, -1.0, 1.0)
            if it <= warm:                          # damped-Picard warmup
                x = x - min(delta, 0.02) * gc
                continue
            X_hist.append(x.copy())
            G_hist.append(gc.copy())
            if len(X_hist) > m_anderson + 1:
                X_hist.pop(0)
                G_hist.pop(0)
            k = len(X_hist) - 1
            if k:
                dX = np.stack([X_hist[i + 1] - X_hist[i]
                               for i in range(k)], 1)
                dG = np.stack([G_hist[i + 1] - G_hist[i]
                               for i in range(k)], 1)
                A = dG.T @ dG
                # ADAPTIVE TIKHONOV regularisation (regularised Anderson, cf.
                # Saad): the normal-equations matrix dG^T dG goes singular near
                # the fixed point, and a fixed absolute floor (the old
                # 1e-12*I) is either too weak on stiff transients (noisy,
                # over-large steps) or wrong-scaled. Regularise RELATIVE to the
                # LS scale, STRONG on stiff transients (large residual ->
                # damped, stable) and VANISHING near convergence (small residual
                # -> weak, so local convergence is not slowed). Measured 2-5x
                # fewer Anderson iterations on stiff scans; identical fixed
                # point (regularisation only reshapes the step, not the root).
                scale = np.trace(A) / k + 1e-300
                lam = min(1e-2, max(1e-10, float(err)))
                try:
                    gam = np.linalg.solve(A + lam * scale * np.eye(k),
                                          dG.T @ gc)
                    dx = -delta * gc - (dX - delta * dG) @ gam
                    # trust region: cap the Anderson step (cf. Namics'
                    # trust-region safeguards in sfnewton)
                    step = np.abs(dx).max()
                    if step > 1.0:
                        dx *= 1.0 / step
                    x = x + dx
                except np.linalg.LinAlgError:
                    X_hist, G_hist = [], []      # singular LS -> restart history
                    x = x - delta * gc
            else:
                x = x - delta * gc
        progress.clear()
        return x_best, it, best_err, False

    def _solve_pseudohessian(self, x0, tolerance, iterationlimit, deltamax,
                             anchor_full=False, verbose=False,
                             full_hessian=False):
        """Rescue with the translated Namics quasi-Newton (sfnewton.py).
        anchor_full=False starts from the diagonal Hessian (fast, most cases);
        anchor_full=True computes the full numerical Hessian once as the anchor,
        then continues with secant updates (Namics' option for the stiffest
        problems). Iterates only the non-frozen, non-ghost variables (the `ksam`
        mask, Namics' variable filter). Returns (x_full, iterations, max|g|,
        converged)."""
        mask = self.var_mask()
        idx = np.where(mask)[0]
        xred = (x0[idx].copy() if x0 is not None and x0.size == self.n_var()
                else np.zeros(idx.size))
        stage = "full-hessian anchor" if anchor_full else "pseudohessian"
        cb = ((lambda it, gmax, alpha: progress.update(stage, it, gmax, alpha))
              if progress.enabled() else None)   # skip the per-it max|g| when off
        sn = SFNewton(self.residual, mask, progress=cb)
        converged, it, _ = sn.iterate(
            xred, tolerance, int(iterationlimit), float(deltamax),
            min(float(deltamax) * 1e-3, 1e-6), anchor_full=anchor_full,
            full_hessian=full_hessian)
        progress.clear()
        x = np.zeros(self.n_var())
        x[idx] = xred
        err = float(np.abs(self.residual(x)).max())
        if verbose:
            print(f"    {'full-hessian anchor' if anchor_full else 'pseudohessian'}"
                  f": {it} it, max|g| = {err:.2e}, converged={converged}")
        return x, it, err, converged
