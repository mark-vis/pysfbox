"""The SCF system: assembles lattice, segments, and molecules from parsed
input; provides the Namics-style residual and the observables used by the
kal/pro output machinery.

Residual (cf. Namics System::Classical_residual): the iteration variable is
the stack of segment potentials u_i(z) for all non-frozen segment types;

    g_i = u_i - u_int_i ;  g_i -= mean_i(g_i) ;  g_i += 1/phi_T - 1

masked to free lattice sites. The incompressibility field alpha is the mean
over components, eliminated analytically inside the residual.
"""

import numpy as np

from .inputreader import get_blocks, last, substitute_aliases
from .lattice import Lattice1D
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
        # asked for -- never silently wrong.
        for block in ("mesodyn", "cleng", "teng"):
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
        if gradients == 1:
            # FJC_choices = 3 + 2*i -> fjc = (FJC-1)/2 sub-layers/bond (Namics)
            FJC = int(float(last(lp, "FJC_choices", 3)))
            if FJC < 3 or (FJC - 3) % 2 != 0:
                raise ValueError(
                    "FJC_choices must be 3 + 2*i (i.e. 3, 5, 7, ...); "
                    f"got {FJC}")
            self.lat = Lattice1D(
                n_layers=int(float(substitute_aliases(
                    last(lp, "n_layers"), settings))),
                geometry=last(lp, "geometry", "planar"),
                lattice_type=last(lp, "lattice_type", "simple_cubic"),
                lowerbound=last(lp, "lowerbound", "mirror"),
                upperbound=last(lp, "upperbound", "mirror"),
                offset_first_layer=float(last(lp, "offset_first_layer", 0.0)),
                fjc=(FJC - 1) // 2)
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
            # face radii for the curved Poisson/field-energy (refined units,
            # Namics LGrad1: r_plus = offset*fjc + (x-fjc+1), r_minus = r-1)
            if self.geom != "planar":
                x = np.arange(lat.M, dtype=float)
                self.r_plus = lat.offset * lat.fjc + (x - lat.fjc + 1.0)
                self.r_minus = self.r_plus - 1.0
                if self.fixedPsi0:
                    raise NotImplementedError(
                        "fixed surface potential (e.psi0/kT) on a curved "
                        "lattice is not supported yet (the curved electrode "
                        "condition is not yet ported to PySFBox)")
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
            self.psi = np.zeros(lat.M)
            self.q = np.zeros(lat.M)
            self.EE = np.zeros(lat.M)
            self.eps_prof = np.full(lat.M, 80.0)
        else:
            if any(m.freedom == "neutralizer"
                   for m in self.molecules.values()):
                raise ValueError(
                    "freedom : neutralizer requires a charged system")
            self.neutralizer = None

        # per-segment-type bulk fractions (for u_int reference and moments);
        # refreshed every iteration by _update_bulk (restricted molecules
        # contribute their implied bulk density, which depends on GN)
        self.phibulk_seg = {name: 0.0 for name in self.segments}
        self._update_bulk()
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
            n += self.lat.M                 # the psi block (last)
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
        return m

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
        # non-solvent, non-constrained molecules first: their normalisation
        # (phibulk- or theta-based) does not depend on the solvent's bulk
        # fraction ...
        for m in self.molecules.values():
            if m is not self.solvent and m is not self.neutralizer:
                m.compute_phi(G1)
        # ... which is refreshed from their current GN before the solvent
        # AND the neutralizer are evaluated -- both have their bulk fraction
        # SET by _update_bulk (the neutralizer by electroneutrality), so like
        # the solvent they must be computed AFTER it, with the current value.
        # This MATCHES Namics, which normalises the neutralizer density with
        # the freshly-computed norm in the same ComputePhis call
        # (system.cpp:2633-2650) -- there is NO lag there.
        self._update_bulk()
        if self.neutralizer is not None:
            self.neutralizer.compute_phi(G1)
        self.solvent.compute_phi(G1)
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
        if self.charged:
            S = len(self.it_species)
            psi_raw = x[S * lat.M:]
            # FIDELITY NOTE (matches Namics PutU/DoElectrostatics order):
            # the segment weights and the field energy EE use the RAW
            # iterated psi -- fixed-potential sites are filtered out of the
            # solver and their x entries stay at the initial guess -- while
            # the Poisson update below sees the EFFECTIVE psi with the
            # fixed values (PSI0) substituted.
            EE = self._field_energy(lat.set_mirror_bounds(psi_raw))
            psi = psi_raw.copy()
            if self.fixedPsi0:
                psi[self.psiMask] = self.psi0_profile[self.psiMask]
            psib = lat.set_mirror_bounds(psi)   # psi ghosts ALWAYS mirror
            self.psi, self.EE = psi, EE
            self.compute_phis(u, psi=psi_raw, EE=EE)
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
        if not self.charged:
            return g.ravel()
        g_psi = self._psi_residual(psi_raw, psib, phitot)
        return np.concatenate([g.ravel(), g_psi])

    # ---- electrostatics (planar, fjc = 1; cf. LG1Planar.cpp) ---------------
    def _field_energy(self, psib):
        """Electric-field energy density EE(z) (Namics UpdateEE). Planar
        (LG1Planar): EE = pf_ee*[(dpsi_left)^2 + (dpsi_right)^2]. Curved
        (LGrad1): each squared field difference is weighted by its FACE
        radius (cyl: r; sph: r^2) and divided by the shell volume L, with
        pf = pf_base*pi (cyl) or pf_base*2pi/fjc (sph) -- so EE is an
        energy DENSITY per unit volume and L*eps*EE sums to the bond
        energies under the L-weighted weighted_sum."""
        lat = self.lat
        iv = slice(lat.fjc, lat.M - lat.fjc)
        EE = np.zeros(lat.M)
        if self.geom == "planar":
            d = np.diff(psib)                # d[x] = psi[x+1]-psi[x]
            EE[1:-1] = self.pf_ee * (d[:-1] ** 2 + d[1:] ** 2)
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
        + C_geo*q[x]*L[x] - (cm+cp)*psi[x] = 0. r_minus = 0 at the origin
        gives the inner zero-flux symmetry automatically. This is the
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
            # The face coefficients use the INTEGER refined radius r (Namics
            # r++ per site) = fjc x the lattice radius r_lat. The correct
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
        C2 = self.C_psi * 2.0 / fjc**2
        epsm = eps[:-2] + eps[1:-1]
        epsp = eps[1:-1] + eps[2:]
        X = ((epsm * psib[:-2] + C2 * q[1:-1] + epsp * psib[2:])
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
            free = ~self.psiMask[1:-1]
            g_psi[1:-1] = np.where(free, psi_raw[1:-1] - X,
                                   (psi_raw - x_target)[1:-1])
        else:
            g_psi[1:-1] = psi_raw[1:-1] - X
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

    def grand_potential_density(self):
        """Per-layer grand-potential density: the integrand whose lattice
        weighted_sum is grand_potential(). Exposed as the
        `sys : <name> : grand_potential_density` profile (the scalar split by
        site; the total is oracle-validated, the per-site split is not
        separately validated column-by-column)."""
        lat = self.lat
        omega = np.zeros(lat.M)
        for m in self.molecules.values():
            omega -= (m.phi - m.phibulk) / m.N
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
                    sa = self.side_phi(sp[b].seg, sp[b].phi)
                    pb_b = (self.phibulk_seg.get(sp[b].seg.name, 0.0)
                            * sp[b].alphabulk)
                    omega -= chi * (sp[a].phi * sa - pb_a * pb_b)
        if self.charged:
            # Namics GetGrandPotential charged tail: add EE*eps - q*psi/2
            # inside the KSAM mask, plus q*psi/2 at the masked (surface)
            # sites — the surface-charge work term
            omega += self.EE * self.eps_prof - 0.5 * self.q * self.psi
            omega *= self.ksam
            omega += (1.0 - self.ksam) * 0.5 * self.q * self.psi
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

    # ---- output property lookup (kal) -----------------------------------------
    def get_value(self, key, name, prop):
        """Returns ('int'|'real'|None, value). None -> NiN, like Namics."""
        lat = self.lat
        if prop.endswith("-value"):
            alias = prop[:-6]
            v = last(self.settings.get(("alias", alias), {}), "value")
            if v is not None:
                fv = float(v)
                return ("int", int(fv)) if fv == int(fv) else ("real", fv)
        if key == "lat":
            if prop == "n_layers":
                return "int", getattr(self.lat, "MX",
                                      int(np.prod(getattr(self.lat, "dims",
                                                          (0,)))))
            if prop == "volume":
                return "real", self.lat.volume
        if key == "sys":
            if prop == "grand_potential":
                return "real", self.grand_potential()
            # free_energy (po) is Namics' "GP + n*mu" route to the same F
            if prop == "free_energy" or prop.replace(" ", "") == "free_energy(po)":
                return "real", self.free_energy()
            if prop == "iterations":
                return "int", self.iterations
            if prop == "residual":
                return "real", self.residual_norm
        if key == "state":
            for seg in self.segments.values():
                for st in seg.states:
                    if st.name == name:
                        if prop == "alphabulk":
                            return "real", st.alphabulk
                        if prop == "valence":
                            return "real", st.valence
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
            table = {"theta": m.get_theta, "theta_exc": m.get_theta_exc,
                     "phibulk": lambda: m.phibulk,
                     "Mu": lambda: self.chemical_potential(m),
                     "MU": lambda: self.chemical_potential(m),
                     "mu": lambda: self.chemical_potential(m),
                     "n": lambda: theta / m.N,
                     "N": lambda: m.N, "GN": lambda: m.GN,
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
                                    - lat.volume * st.phibulk)
            pk = prop.replace(" ", "")           # chi_X / chi-X, like Segment
            if (pk.startswith("chi_") or pk.startswith("chi-")) \
                    and pk[4:] in self.segments:
                return "real", s.chi_with(self.segments[pk[4:]])
            phib = self.phibulk_seg.get(name, 0.0)
            theta = lat.weighted_sum(s.phi)
            theta_exc = theta - lat.volume * phib
            if prop == "theta":
                return "real", theta
            if prop == "theta_exc":
                return "real", theta_exc
            if prop == "phibulk":
                return "real", phib
            if prop in ("1st_M_phi_z", "2nd_M_phi_z", "fluctuations", "RMS"):
                m1 = lat.moment(s.phi, phib, 1) / theta_exc if theta_exc else 0
                m2 = lat.moment(s.phi, phib, 2) / theta_exc if theta_exc else 0
                if prop == "1st_M_phi_z":
                    return "real", m1
                if prop == "2nd_M_phi_z":
                    return "real", m2
                if prop == "RMS":
                    return "real", np.sqrt(m2) if m2 > 0 else 0.0
                fl = m2 - m1 * m1
                return "real", np.sqrt(fl) if fl > 0 else 0.0
        return None, None

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
                            return self._last_u[i]
        if key == "mon" and name in self.segments and prop == "u":
            for i, sp in enumerate(self.it_species):
                if sp.state is None and sp.name == name:
                    return self._last_u[i]
            return None
        if key == "sys" and prop == "alpha":
            return self.alpha
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
        if self.lat.gradients > 1:
            # the analytic adsorption/membrane guesses are 1-gradient; N-D
            # calculations cold-start (the runner still warm-starts scans)
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
        else:
            raise NotImplementedError(
                f"sys : initial_guess : {kind} is not supported in PySFBox "
                "(supported: polymer_adsorption, membrane, micelle, "
                "previous_result, none); guess files and membrane_torus "
                "need the C++ Namics")
        return U.ravel()

    # ---- solver --------------------------------------------------------------
    # dense finite-difference Newton is affordable below this many variables
    # (n_it_segs * M); above it the FD Jacobian (n residual evals per step)
    # gets too expensive and we stay with Anderson only.
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
        if ok and self.solvent.phibulk < 0:
            # cf. Namics' refusal (system.cpp:2604): the constrained amounts
            # imply more material than an equilibrium bulk can hold
            raise RuntimeError(
                f"converged to a state with negative solvent bulk fraction "
                f"({self.solvent.phibulk:.3e}); the restricted theta values "
                f"exceed what the box can hold in equilibrium")
        if not ok:
            raise RuntimeError(
                f"no convergence in {it} iterations (max|g| = {err:.2e}); "
                f"try a smaller deltamax")
        return x, it, err

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
            if verbose and it % 100 == 0:
                print(f"    it {it:5d}  max|g| = {err:.3e}  delta = {delta:.1e}")
            if err < tolerance:
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
        sn = SFNewton(self.residual, mask)
        converged, it, _ = sn.iterate(
            xred, tolerance, int(iterationlimit), float(deltamax),
            min(float(deltamax) * 1e-3, 1e-6), anchor_full=anchor_full,
            full_hessian=full_hessian)
        x = np.zeros(self.n_var())
        x[idx] = xred
        err = float(np.abs(self.residual(x)).max())
        if verbose:
            print(f"    {'full-hessian anchor' if anchor_full else 'pseudohessian'}"
                  f": {it} it, max|g| = {err:.2e}, converged={converged}")
        return x, it, err, converged
