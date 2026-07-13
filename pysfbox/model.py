"""Segments, molecules, and chain propagation: linear multiblock and branched
chains (brackets, @dend, @comb), charged segments, on the one- and N-gradient
lattices, with log-domain renormalised propagators. Mirrors the corresponding
Namics classes (segment.cpp, molecule.cpp, mol_linear.cpp, mol_branched.cpp,
mol_comb.cpp, mol_dendrimer.cpp).
"""

import re

import numpy as np

from .inputreader import last


# ---------------------------------------------------------------- segments
class Segment:
    def __init__(self, name, params, lat):
        self.name = name
        self.lat = lat
        self.freedom = last(params, "freedom", "free")
        self.chi = {}  # chi[other_name] = value
        for p, v in params.items():
            pk = p.replace(" ", "")          # accept chi_X / chi-X / "chi - X"
            if pk.startswith("chi_") or pk.startswith("chi-"):
                self.chi[pk[4:]] = float(v[-1])
        # electrostatics (cf. Namics segment.cpp:1078-1113): valence in units
        # of e, relative permittivity (default 80, water-like), and an
        # optional fixed dimensionless surface potential e*psi0/kT on FROZEN
        # segments (mutually exclusive with valence, as in Namics)
        self.valence = float(last(params, "valence", 0.0))
        self.epsilon = float(last(params, "epsilon", 80.0))
        psi0 = last(params, "e.psi0/kT")
        self.fixed_psi0 = psi0 is not None
        self.psi0 = float(psi0) if self.fixed_psi0 else 0.0
        if self.fixed_psi0:
            if self.freedom != "frozen":
                raise ValueError(
                    f"mon {name}: e.psi0/kT requires freedom : frozen "
                    "(as in Namics)")
            if self.valence != 0.0:
                raise ValueError(
                    f"mon {name}: valence and e.psi0/kT are mutually "
                    "exclusive (as in Namics)")
        self.range_mask = np.zeros(lat.M)   # 1 on pinned/frozen layers
        self.on_lower_surface = False
        self.on_upper_surface = False
        self.surface_face = None            # (axis, 'lo'|'hi') for ND walls
        rng = last(params, "frozen_range") or last(params, "pinned_range")
        if self.freedom in ("frozen", "pinned"):
            if rng is None:
                raise ValueError(f"mon {name}: freedom {self.freedom} "
                                 f"requires a {self.freedom}_range")
            self._set_range(rng)
        self.phi = np.zeros(lat.M)
        self.phibulk = 0.0
        # internal states (weak charges); attached by System from the
        # `state : NAME : mon : <this>` blocks. Empty = ordinary segment.
        self.states = []

    def mean_valence(self):
        """Bulk-average valence: sum_s alphabulk_s * v_s for a multistate
        segment (Namics vbar), the plain valence otherwise."""
        if self.states:
            return sum(st.alphabulk * st.valence for st in self.states)
        return self.valence

    def _set_range(self, rng):
        # ranges are given in physical layers; with fjc>1 each physical layer
        # spans fjc refined sites, so [lo;hi] -> refined [lo*fjc, (hi+1)*fjc-1]
        # (cf. Namics segment.cpp: r[0]*=fjc; r[3]=(r[3]+1)*fjc-1).
        lat = self.lat
        if getattr(lat, "gradients", 1) > 1:
            # 2D/3D box grammar 'xlo,ylo[,zlo];xhi,yhi[,zhi]' (Namics), or a
            # 'lowerbound'/'upperbound' wall on the first axis
            self.range_mask, self.surface_face = lat.parse_range(rng)
            if self.surface_face is not None:
                axis, end = self.surface_face
                self.on_lower_surface = (axis == 0 and end == "lo")
                self.on_upper_surface = (axis == 0 and end == "hi")
            return
        fjc = lat.fjc
        r = rng.strip().lower().rstrip(";")  # Namics tolerates 'lowerbound;'
        if r == "lowerbound":
            self.on_lower_surface = True       # lives in the ghost layers
        elif r == "upperbound":
            self.on_upper_surface = True
        elif r in ("firstlayer", "first_layer"):
            self.range_mask[fjc:2 * fjc] = 1.0
        elif r in ("lastlayer", "last_layer"):
            self.range_mask[lat.MX:lat.MX + fjc] = 1.0
        else:
            lo, hi = (int(x) for x in r.split(";"))
            self.range_mask[lo * fjc:(hi + 1) * fjc] = 1.0

    def chi_with(self, other):
        # chi table is symmetric; defined on either side, default 0
        if other.name in self.chi:
            return self.chi[other.name]
        return other.chi.get(self.name, 0.0)


# Potential clip applied in System.compute_phis (G1 = exp(-clip(u))), shared
# here because the composition-law overflow cap must account for 1/G1.
U_CLIP = 200.0
# Lazy-renormalisation drift budget for the propagators: a stored row is
# renormalised only when its maximum leaves [1/BIG, BIG] (see compute_phi).
BIG = 1e30


# ---------------------------------------------------------------- molecules
_BLOCK = re.compile(r"\(\s*([A-Za-z0-9_]+)\s*\)\s*(\d+)")


def parse_composition(comp):
    """Linear multi-block compositions like (X)1(A)100(G)1 -> [(mon, n)...].
    Branched/ring/dendrimer syntax is rejected with a clear message (the
    branched path lives in parse_architecture / Molecule._compute_phi_tree)."""
    if any(tok in comp for tok in ("[", "@")):
        raise NotImplementedError(
            f"composition '{comp}': branched syntax reached the linear parser "
            "(branched architectures — brackets, @dend, @comb — are handled by "
            "parse_architecture)")
    blocks = _BLOCK.findall(comp)
    if not blocks or _BLOCK.sub("", comp).strip():
        raise ValueError(f"cannot parse composition '{comp}'")
    return [(mon, int(n)) for mon, n in blocks]


# ------------------------------------------------------- branched (trees)
class TreeSeg:
    """One segment of a branched molecule: a node in the architecture tree.
    `children` holds both the backbone continuation and any side chains --
    the tree is rooted at the first segment of the composition string and
    makes no backbone/side distinction."""

    __slots__ = ("mon", "children")

    def __init__(self, mon):
        self.mon = mon
        self.children = []


def _parse_seq(s, pos, segments, mol_name):
    """Recursive-descent parse of a composition sequence starting at pos.
    Returns (first_seg, last_seg, next_pos); stops at ']' or end of string.

    Grammar (cf. Namics molecule.cpp Decomposition):
        seq     := item+
        item    := run | group | bracket
        run     := '(' NAME ')' INT              chain of INT segments
        group   := '(' seq ')' INT               the whole group INT times
        bracket := '[' seq ']'                   side chain on the PRECEDING
                                                 segment
    """
    first = tail = None

    def attach(seg):
        nonlocal first, tail
        if tail is None:
            first = seg
        else:
            tail.children.append(seg)
        tail = seg

    n = len(s)
    while pos < n:
        c = s[pos]
        if c == "]":
            break
        if c == "[":
            if tail is None:
                raise ValueError(
                    f"mol {mol_name}: side chain '[' with nothing to attach "
                    f"to in '{s}'")
            sub_first, _, pos = _parse_seq(s, pos + 1, segments, mol_name)
            if pos >= n or s[pos] != "]":
                raise ValueError(f"mol {mol_name}: unbalanced '[' in '{s}'")
            pos += 1
            tail.children.append(sub_first)      # side chain: tail unchanged
            continue
        if c != "(":
            raise ValueError(
                f"mol {mol_name}: cannot parse composition '{s}' at '{s[pos:]}'")
        m = re.match(r"\(\s*([A-Za-z0-9_#]+)\s*\)\s*(\d+)", s[pos:])
        if m:                                    # plain run (A)12
            mon, cnt = m.group(1), int(m.group(2))
            if mon not in segments:
                raise ValueError(f"mol {mol_name}: unknown mon '{mon}'")
            for _ in range(cnt):
                attach(TreeSeg(mon))
            pos += m.end()
            continue
        # group repeat ((O)1(C)2)5 -- parse the inner seq, then repeat
        depth, j = 1, pos + 1
        while j < n and depth:
            depth += {"(": 1, ")": -1}.get(s[j], 0)
            j += 1
        if depth:
            raise ValueError(f"mol {mol_name}: unbalanced '(' in '{s}'")
        inner = s[pos + 1:j - 1]
        m = re.match(r"\s*(\d+)", s[j:])
        if not m:
            raise ValueError(
                f"mol {mol_name}: group '({inner})' needs a repeat count")
        for _ in range(int(m.group(1))):
            sub_first, sub_tail, endp = _parse_seq(inner, 0, segments,
                                                   mol_name)
            if endp != len(inner) or sub_first is None:
                raise ValueError(
                    f"mol {mol_name}: cannot parse group '({inner})'")
            attach(sub_first)
            tail = sub_tail
        pos = j + m.end()
    return first, tail, pos


def _chain(mons, segments, mol_name):
    """A plain linear chain of TreeSegs from a composition substring."""
    first, tail, endp = _parse_seq(mons, 0, segments, mol_name)
    if first is None or endp != len(mons):
        raise ValueError(f"mol {mol_name}: cannot parse '{mons}'")
    return first, tail


def _expand_dend(body, segments, mol_name):
    """@dend(X,(A)24,2;A,(B)24,2;...) -> tree. Each ';'-separated generation
    is (junction mon, arm composition, functionality[, arm2, f2, ...]); the
    junctions of generation g+1 are appended to the END of every generation-g
    arm (grammar confirmed against the compiled Namics via total chain
    lengths and per-type fractions: N = 391 for the dendrimers-on-a-line
    example, N = 34/1006 for the solvation stars, N = 4 for @dend water)."""
    gens = []
    for gtxt in body.split(";"):
        parts = [p.strip() for p in gtxt.split(",")]
        if len(parts) < 3 or len(parts) % 2 == 0:
            raise ValueError(
                f"mol {mol_name}: @dend generation '{gtxt}' must be "
                "junction,arm,f[,arm2,f2,...]")
        junction = parts[0]
        if junction not in segments:
            raise ValueError(f"mol {mol_name}: unknown mon '{junction}'")
        arms = [(parts[i], int(parts[i + 1]))
                for i in range(1, len(parts), 2)]
        if len(arms) > 1:
            raise NotImplementedError(
                f"mol {mol_name}: asymmetric dendrimers (several arm types "
                "per generation) are not supported yet -- needs the full "
                "Namics")
        gens.append((junction, arms))

    def build(g):
        node = TreeSeg(gens[g][0])
        for arm_txt, f in gens[g][1]:
            for _ in range(f):
                first, tailseg = _chain(arm_txt, segments, mol_name)
                if g + 1 < len(gens):
                    tailseg.children.append(build(g + 1))
                node.children.append(first)
        return node

    return build(0)


def _expand_comb(body, segments, mol_name):
    """@comb((B)10;B,(A)25,(B)9,100;(B)1) -> tree: lead-in; n_rep repeats of
    [junction + tooth side chain + spacer]; lead-out (grammar confirmed
    against the compiled Namics: N = 3511, per-type fractions exact)."""
    parts = [p.strip() for p in body.split(";")]
    if len(parts) != 3:
        raise ValueError(
            f"mol {mol_name}: @comb needs lead-in;junction,tooth,spacer,n;"
            "lead-out")
    rep = [p.strip() for p in parts[1].split(",")]
    if len(rep) != 4:
        raise ValueError(
            f"mol {mol_name}: @comb repeat part '{parts[1]}' must be "
            "junction,tooth,spacer,n")
    junction, tooth_txt, spacer_txt, n_rep = rep[0], rep[1], rep[2], int(rep[3])
    if junction not in segments:
        raise ValueError(f"mol {mol_name}: unknown mon '{junction}'")
    root, tail = _chain(parts[0], segments, mol_name)
    for _ in range(n_rep):
        j = TreeSeg(junction)
        tail.children.append(j)
        tooth_first, _ = _chain(tooth_txt, segments, mol_name)
        j.children.append(tooth_first)
        sp_first, sp_tail = _chain(spacer_txt, segments, mol_name)
        j.children.append(sp_first)
        tail = sp_tail
    out_first, out_tail = _chain(parts[2], segments, mol_name)
    tail.children.append(out_first)
    return root


def _expand_group_sugar(s):
    """Expand parenthesised group repeats textually: ((O)1(C)2)5 ->
    (O)1(C)2(O)1(C)2... (linear sugar, cf. the Namics C12E5 example).
    Innermost groups first; the caller strips whitespace."""
    while True:
        i = s.find("((")
        if i < 0:
            return s
        depth, j = 1, i + 1
        while j < len(s) and depth:
            depth += {"(": 1, ")": -1}.get(s[j], 0)
            j += 1
        if depth:
            raise ValueError(f"unbalanced '(' in composition '{s}'")
        m = re.match(r"\d+", s[j:])
        if not m:
            raise ValueError(
                f"group '({s[i + 1:j - 1]})' needs a repeat count in '{s}'")
        s = s[:i] + s[i + 1:j - 1] * int(m.group(0)) + s[j + m.end():]


def parse_architecture(comp, segments, mol_name):
    """Branched composition -> rooted TreeSeg tree (or None for linear)."""
    comp = comp.strip()
    if comp.startswith("@"):
        m = re.match(r"@(dend|comb)\s*\((.*)\)\s*$", comp)
        if not m:
            raise NotImplementedError(
                f"mol {mol_name}: composition '{comp}' not supported "
                "(supported generators: @dend, @comb)")
        if m.group(1) == "dend":
            return _expand_dend(m.group(2), segments, mol_name)
        return _expand_comb(m.group(2), segments, mol_name)
    if "[" in comp:
        root, _, endp = _parse_seq(comp, 0, segments, mol_name)
        if root is None or endp != len(comp):
            raise ValueError(f"mol {mol_name}: cannot parse '{comp}'")
        return root
    return None                                   # linear (incl. group sugar)


class Molecule:
    def __init__(self, name, params, segments, lat, composition):
        self.name = name
        self.lat = lat
        self.freedom = last(params, "freedom", "free")
        comp = re.sub(r"\s+", "", composition)
        if not comp.startswith("@"):
            # group-repeat sugar ((O)1(C)2)5 is linear-syntax only; @dend/
            # @comb bodies contain '((' patterns of their own grammar
            comp = _expand_group_sugar(comp)
        self.tree = parse_architecture(comp, segments, name)
        if self.tree is not None:
            # branched: flatten once (pre-order; parents before children)
            nodes, kids = [], []
            stack = [self.tree]
            while stack:
                nd = stack.pop()
                idx = len(nodes)
                nodes.append(nd)
                kids.append([])
                stack.extend(reversed(nd.children))
            # rebuild child index lists from object identity, in one pass
            pos_of = {id(nd): i for i, nd in enumerate(nodes)}
            for i, nd in enumerate(nodes):
                kids[i] = [pos_of[id(c)] for c in nd.children]
            self._tree_nodes = nodes
            self._tree_kids = kids
            self.seq = [segments[nd.mon] for nd in nodes]
            counts = {}
            for nd in nodes:
                counts[nd.mon] = counts.get(nd.mon, 0) + 1
            self.blocks = list(counts.items())
        else:
            self.blocks = parse_composition(comp)
            self.seq = []  # segment object per chain position s = 0..N-1
            for mon, n in self.blocks:
                if mon not in segments:
                    raise ValueError(f"mol {name}: unknown mon '{mon}'")
                self.seq.extend([segments[mon]] * n)
        self.N = len(self.seq)
        self.phibulk = float(last(params, "phibulk", 0.0))
        self.n = float(last(params, "n", 0.0))
        self.theta = float(last(params, "theta", self.n * self.N))
        self.phi_per_seg = {}   # segment name -> profile
        self.phi = np.zeros(lat.M)
        self.GN = 0.0
        self.lnGN = np.inf      # unknown until the first propagation ->
        #                         implied bulk theta/GN starts at 0
        # consecutive same-type runs of the chain -- the composition law sums
        # each run with one vectorised operation instead of per-segment steps
        self._type_runs = []
        pos = 0
        for mon, cnt in self.blocks:
            self._type_runs.append((mon, slice(pos, pos + cnt)))
            pos += cnt
        # a palindromic sequence (every homopolymer, symmetric multiblocks)
        # makes the backward propagator the exact mirror of the forward one:
        # Gb[s] = Gf[N-1-s] bit-for-bit, so the backward pass can be skipped
        names = [s.name for s in self.seq]
        self._palindrome = names == names[::-1]
        # a molecule with a pinned segment is grafted: its implied bulk
        # density is zero (cf. Namics IsPinned in System::ComputePhis)
        self.has_pinned = any(s.freedom == "pinned" for s in self.seq)
        if self.has_pinned and self.freedom != "restricted":
            raise ValueError(
                f"mol {name}: contains pinned segment(s), which requires "
                "freedom : restricted (as in Namics)")
        self._Gf = None         # propagator buffers, reused between calls
        self._Gb = None
        # only monodisperse chains are supported; dispersity : 1.0 is the
        # monodisperse default and is accepted silently
        if float(last(params, "dispersity", 1.0)) > 1.0 + 1e-9:
            raise NotImplementedError(
                f"mol {name}: dispersity (polydisperse chains) is not "
                "supported in PySFBox; enumerate the chain lengths as "
                "separate mol blocks")
        # chain stiffness: only Markov : 1 (fully flexible chains) is
        # supported, at both lat and mol level
        markov = int(float(last(params, "Markov", getattr(lat, "markov", 1))))
        if markov != 1:
            raise NotImplementedError(
                f"mol {name}: Markov : {markov} semiflexibility is not "
                "supported; the compiled Namics binary also forces Markov 1")
        if last(params, "k_stiff") is not None:
            raise NotImplementedError(
                f"mol {name}: k_stiff (Markov : 2 semiflexibility) is not "
                "supported; the compiled Namics binary also forces Markov 1")
        # ring molecules change the propagator topology; refuse loudly rather
        # than silently treating the chain as linear
        if str(last(params, "ring", "false")).lower() in ("true", "1"):
            raise NotImplementedError(
                f"mol {name}: ring molecules (mol : {name} : ring : true) "
                "are not supported in PySFBox; use the C++ Namics")

    def _propagate_chain(self, Gseq, out, forward):
        """One propagation sweep (forward or backward) with LAZY log-domain
        renormalisation: the row maximum is checked every step (this is also
        the propagator-death check), but the row is only rescaled -- and its
        log booked -- when the magnitude drifts outside [1/BIG, BIG]. That
        keeps the invariant-4 guarantee (nothing can overflow: stored rows
        stay within the budget by construction) at a fraction of the
        every-step divide+log cost. Returns L, the per-row applied log
        prefactor: true row s = exp(L[s]) * stored row s."""
        lat, N = self.lat, self.N
        L = np.zeros(N)
        order = range(1, N) if forward else range(N - 2, -1, -1)
        first, step = (0, -1) if forward else (N - 1, +1)
        # the seed row gets the same budget check as every other row, so the
        # [1/BIG, BIG] bound below holds for ALL stored rows by construction
        # (bit-identical whenever max(G1) <= BIG, i.e. any remotely sane state)
        g = Gseq[first]
        m = g.max()
        ln_applied = 0.0
        if m > 0 and np.isfinite(m) and (m > BIG or m < 1.0 / BIG):
            ln_applied = np.log(m)
            g = g / m
        out[first] = g
        L[first] = ln_applied
        for s in order:
            g = lat.propagate(out[s + step], Gseq[s])
            m = g.max()
            if not np.isfinite(m) or m <= 0:
                raise FloatingPointError(
                    f"mol {self.name}: propagator died at segment {s}")
            if m > BIG or m < 1.0 / BIG:
                g /= m
                ln_applied += np.log(m)
            out[s] = g
            L[s] = ln_applied
        return L

    def compute_phi(self, G1):
        """Forward + backward propagators -> phi. G1: dict segment name ->
        single-segment weight (Boltzmann factor with site masks applied).

        Log-domain bookkeeping (invariant 4): stored propagator rows are kept
        within [1/BIG, BIG] by lazy renormalisation (_propagate_chain), with
        the applied log prefactors in Lf/Lb, so strongly adsorbing or long
        chains cannot overflow -- not even transiently. The composition law
        restores the prefactors per segment as a SCALAR,
        phi(z) = sum_s exp(lnC + Lf[s] + Lb[s]) * Gf~[s] Gb~[s] / G1,
        evaluated with one weighted matrix product per same-type run of the
        chain. A palindromic sequence skips the backward sweep entirely
        (Gb[s] = Gf[N-1-s] bit-for-bit)."""
        if self.tree is not None:
            return self._compute_phi_tree(G1)
        lat, N = self.lat, self.N
        Gseq = [G1[s.name] for s in self.seq]
        if self._Gf is None or self._Gf.shape != (N, lat.M):
            self._Gf = np.empty((N, lat.M))
            self._Gb = None if self._palindrome else np.empty((N, lat.M))
        Gf = self._Gf
        Lf = self._propagate_chain(Gseq, Gf, forward=True)
        if self._palindrome:
            Gb, Lb = Gf[::-1], Lf[::-1]        # exact mirror, zero cost
        else:
            Gb = self._Gb
            Lb = self._propagate_chain(Gseq, Gb, forward=False)

        ws = lat.weighted_sum(Gf[N - 1])           # GN = ws * exp(Lf[N-1])
        self.lnGN = np.log(ws) + Lf[N - 1] if ws > 0 else -np.inf
        self.GN = np.exp(min(self.lnGN, 700.0))
        if self.freedom in ("free", "solvent", "neutralizer"):
            # neutralizer: like free, but its phibulk is set each iteration
            # by System._update_bulk to keep the bulk electroneutral
            lnC = np.log(self.phibulk / N) if self.phibulk > 0 else -np.inf
        elif self.freedom == "restricted":
            if not np.isfinite(self.lnGN):
                raise FloatingPointError(f"mol {self.name}: GN <= 0")
            lnC = np.log(self.theta / N) - self.lnGN
            # phibulk (the implied bulk density theta/GN, or 0 when grafted)
            # is owned and updated per iteration by System.compute_phis
        else:
            raise NotImplementedError(
                f"mol {self.name}: freedom '{self.freedom}' not supported")
        if not np.isfinite(lnC):
            # phibulk = 0 (or theta = 0): exactly zero density, not the
            # subnormal exp(-745) the clipped prefactor would produce
            self.phi_per_seg = {mon: np.zeros(lat.M)
                                for mon, _ in self._type_runs}
            self.phi = np.zeros(lat.M)
            return self.phi

        # ---- composition law, one weighted matrix product per run ----------
        # Overflow safety (invariant 4): the scalar weights W are capped so
        # that even on the wildest transient every intermediate stays below
        # DBL_MAX ~ exp(709): stored rows are within [1/BIG, BIG] (so a
        # product Gf*Gb <= BIG^2), 1/G1 <= exp(U_CLIP) wherever G1 > 0, and a
        # run sums at most N such terms -- hence the cap
        #     709 - 2 ln BIG - U_CLIP - ln N - margin.
        # Legitimate converged states sit orders of magnitude below the cap
        # (their W is at most ~BIG^2 exp(U_CLIP) since phi is O(1)).
        cap = 709.0 - 2.0 * np.log(BIG) - U_CLIP - np.log(N) - 10.0
        W = np.exp(np.clip(lnC + Lf + Lb, -745.0, cap))
        per_seg = {}
        for mon, run in self._type_runs:
            contrib = W[run] @ (Gf[run] * Gb[run])
            per_seg[mon] = per_seg.get(mon, 0.0) + contrib
        with np.errstate(divide="ignore", invalid="ignore"):
            for mon in per_seg:
                g1 = G1[mon]
                per_seg[mon] = np.where(g1 > 0, per_seg[mon] / g1, 0.0)
        self.phi = sum(per_seg.values())
        self.phi_per_seg = per_seg
        return self.phi

    def _compute_phi_tree(self, G1):
        """Branched molecules: message passing on the architecture tree.
        Two sweeps over the pre-order node list: up (children before
        parents) builds
            up[i] = G1(i) * prod_c <up[c]>,
        down (parents before children) builds the parent-side message
            down[c] = G1(i) * <down[i]> * prod_{c' != c} <up[c']>
        with leave-one-out products (NO division by G1 or by child messages
        -- masked-zero weights stay exact). Then
            phi(i) = C * up[i] * <down[i]>,   GN = sum_z L * up[root].
        Every stored message is lazily renormalised into [1/BIG, BIG] with
        its log booked (invariant 4), exactly like the linear propagators."""
        lat, N = self.lat, self.N
        nodes, kids = self._tree_nodes, self._tree_kids
        n = len(nodes)
        lv = 0.0 if lat.lowerbound == "surface" else None

        def avg(x):
            return lat.site_average(lat.set_bounds(x, lower_value=lv))

        def renorm(x, lg):
            m = x.max()
            if not np.isfinite(m):
                raise FloatingPointError(
                    f"mol {self.name}: propagator overflow (tree)")
            if m > BIG or (0.0 < m < 1.0 / BIG):
                return x / m, lg + np.log(m)
            return x, lg

        up = [None] * n
        aup = [None] * n            # <up[i]>, cached for the down pass
        up_log = np.zeros(n)
        for i in range(n - 1, -1, -1):          # children before parents
            x, lg = G1[nodes[i].mon], 0.0
            for c in kids[i]:                   # progressive renorm: a
                x, lg = renorm(x * aup[c], lg + up_log[c])  # many-arm node
            up[i], up_log[i] = x, lg            # cannot overflow doubles
            aup[i] = avg(x)

        ws = lat.weighted_sum(up[0])            # GN = ws * exp(up_log[0])
        self.lnGN = np.log(ws) + up_log[0] if ws > 0 else -np.inf
        self.GN = np.exp(min(self.lnGN, 700.0))
        if self.freedom in ("free", "solvent", "neutralizer"):
            lnC = np.log(self.phibulk / N) if self.phibulk > 0 else -np.inf
        elif self.freedom == "restricted":
            if not np.isfinite(self.lnGN):
                raise FloatingPointError(f"mol {self.name}: GN <= 0")
            lnC = np.log(self.theta / N) - self.lnGN
        else:
            raise NotImplementedError(
                f"mol {self.name}: freedom '{self.freedom}' not supported")
        if not np.isfinite(lnC):
            self.phi_per_seg = {mon: np.zeros(lat.M) for mon, _ in self.blocks}
            self.phi = np.zeros(lat.M)
            return self.phi

        # down sweep + density accumulation (same overflow cap as the linear
        # composition law; up*<down> <= BIG^2 and no 1/G1 factor here)
        cap = 709.0 - 2.0 * np.log(BIG) - np.log(N) - 10.0
        adn = [None] * n                        # <down[i]>; ones at the root
        adn[0] = np.ones(lat.M)
        down_log = np.zeros(n)
        per_seg = {mon: 0.0 for mon, _ in self.blocks}
        for i in range(n):                      # parents before children
            W = np.exp(min(lnC + up_log[i] + down_log[i], cap))
            per_seg[nodes[i].mon] = (per_seg[nodes[i].mon]
                                     + W * (up[i] * adn[i]))
            ks = kids[i]
            if not ks:
                continue
            # leave-one-out products of <up[c]> via prefix/suffix arrays,
            # renormalised progressively (logs carried alongside)
            k = len(ks)
            pre, prelog = [None] * (k + 1), [0.0] * (k + 1)
            pre[0], prelog[0] = renorm(G1[nodes[i].mon] * adn[i],
                                       down_log[i])
            for j in range(k):
                pre[j + 1], prelog[j + 1] = renorm(
                    pre[j] * aup[ks[j]], prelog[j] + up_log[ks[j]])
            suf, suflog = np.ones(lat.M), 0.0
            for j in range(k - 1, -1, -1):
                d, dlg = renorm(pre[j] * suf, prelog[j] + suflog)
                adn[ks[j]] = avg(d)
                down_log[ks[j]] = dlg
                suf, suflog = renorm(suf * aup[ks[j]],
                                     suflog + up_log[ks[j]])
        self.phi_per_seg = {mon: (v if isinstance(v, np.ndarray)
                                  else np.zeros(lat.M))
                            for mon, v in per_seg.items()}
        self.phi = sum(self.phi_per_seg.values())
        return self.phi

    def charge_per_seg(self):
        """Average charge per segment (Namics Molecule::Charge): the
        molecule's total valence divided by its chain length. Segments with
        internal states contribute their bulk-average valence
        sum_s alphabulk_s * v_s (Namics molecule.cpp:1898-1911)."""
        return sum(s.mean_valence() for s in self.seq) / self.N

    # observables (Namics names)
    def get_theta(self):
        return self.lat.weighted_sum(self.phi)

    def get_theta_exc(self):
        return self.get_theta() - self.lat.volume * self.phibulk
