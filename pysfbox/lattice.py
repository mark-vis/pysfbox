"""One-gradient lattice: planar, cylindrical, or spherical.

Interior sites z = fjc .. MX+fjc-1 with `fjc` ghost layers on each side for the
boundary conditions, exactly as in Namics LGrad1.cpp. With fjc = 1 (the default)
this is the familiar z = 1..MX plus ghosts 0 and MX+1, first-order Markov chains,
and the three-point step weights lambda_1 (toward z-1), lambda0, lambda1 (toward
z+1).

`FJC_choices > 3` refines the lattice: fjc = (FJC_choices-1)/2 sub-layers per
bond length, so the internal MX = fjc * n_layers and the propagator becomes a
(2*fjc+1)-point stencil `LAMBDA[c, z]` (c = 0..FJC-1, neighbour offset c-fjc)
with curvature-weighted, position-dependent coefficients (Namics
ComputeLambdas). fjc = 1 keeps the original three-point code bit-for-bit.
"""

import numpy as np

LAMBDA = {"simple_cubic": 1.0 / 6.0, "hexagonal": 0.25}

# Lattice1D._check_stencil: a transition weight of the folded one-step
# operator below -STENCIL_TOL, or a detailed-balance or row-sum defect above
# STENCIL_TOL (relative), refuses the lattice
STENCIL_TOL = 1e-12

# once-per-process notes of the refined cylinder's axis rows (the offset-0
# cylinder at FJC_choices >= 11 has a negative stay weight on its axis row;
# Lattice1D._check_stencil exempts it, System audits phi >= 0)
_AXIS_NOTES = set()


def curved_face_radii(offset, fjc, M):
    """Face radii (refined units) of the curved 1-gradient Poisson operator
    and field energy: site x has the inner face r_minus[x] and the outer
    face r_plus[x] = r_minus[x] + 1, areas r (cylinder) / r^2 (sphere).

    fjc = 1 (the Namics shells, LGrad1 `r++` per site): the faces are the
    site's own shell [r - 1, r], r = offset + x.
    fjc > 1 (refined): the site sits at r_site = offset*fjc + (x - fjc + 1)
    (= fjc * r_k, `Lattice1D._setup_fjc`) and its faces are the
    propagator's own one-step channel surfaces r_site -+ 1/2, with plain
    GEOMETRIC areas -- the finite-volume operator of the cell centred on the
    site, O(h^2)-consistent (decision 9 Oct 2026). Namics keeps the
    fjc = 1 rule at fjc > 1 (faces [r_site - 1, r_site], half a refined
    cell inward): first-order inconsistent, a deliberate deviation from
    Namics (reported to the Namics authors). The source C_geo q L and the
    coefficients' prefactors are frame-independent."""
    x = np.arange(M, dtype=float)
    r_plus = offset * fjc + (x - fjc + 1.0)
    if fjc > 1:
        r_plus = r_plus + 0.5
    return r_plus, r_plus - 1.0


class Lattice1D:
    def __init__(self, n_layers, geometry="planar",
                 lattice_type="simple_cubic", lowerbound="mirror",
                 upperbound="mirror", offset_first_layer=0.0, fjc=1,
                 lam=None):
        if geometry == "flat":          # Namics accepts flat as planar synonym
            geometry = "planar"
        self.geometry = geometry
        self.lattice_type = lattice_type
        for side, b in (("lowerbound", lowerbound),
                        ("upperbound", upperbound)):
            # every value other than 'surface' used to run as mirror -- a
            # typo silently, and Namics' 1-D 'periodic' (the lamellar
            # repeat unit) as a reflecting wall (review 6 Oct 2026, #30)
            if b == "periodic":
                raise NotImplementedError(
                    f"lat : {side} : periodic is not yet ported to PySFBox "
                    "on the 1-gradient lattice; use the C++ Namics, or "
                    "gradients : 2 (flat) with lowerbound_x/upperbound_x : "
                    "periodic and n_layers_y : 1")
            if b not in ("mirror", "surface"):
                raise ValueError(
                    f"lat : {side} : '{b}' not recognized (mirror or "
                    "surface)")
        self.lowerbound, self.upperbound = lowerbound, upperbound
        # chain-stiffness defaults (Namics reads Markov/k_stiff at both lat
        # and mol level; System fills these from the lat block and each
        # Molecule may override with its own mol : X : Markov / k_stiff).
        self.markov = 1
        self.k_stiff = 0.0
        self.fjc = int(fjc)
        self.FJC = 2 * self.fjc + 1
        # the radial offset is a curved-lattice notion: Namics reads it only
        # for non-planar geometry (lattice.cpp:377-385), so a planar offset
        # is ignored rather than shifting the .pro x column (review #70)
        if geometry == "planar":
            offset_first_layer = 0.0
        self.offset = float(offset_first_layer)
        # a curved lattice's first shell starts at r = offset: a negative
        # value gives negative-volume (cylinder) or centre-folded (sphere)
        # shells and silently wrong restricted amounts. Namics prints a
        # message and resets it to 0 (lattice.cpp:380-384); PySFBox refuses
        # rather than change the geometry silently (review 27 Oct 2026, #26)
        if geometry != "planar" and not self.offset >= 0:
            raise ValueError(
                f"lat : offset_first_layer : {offset_first_layer} < 0 -- a "
                f"{geometry} lattice needs a non-negative radial offset (the "
                "first shell starts at r = offset_first_layer; Namics resets "
                "a negative value to 0, PySFBox refuses rather than change "
                "the geometry silently)")
        self.MX = self.fjc * int(n_layers)          # internal (refined) sites
        self.M = self.MX + 2 * self.fjc             # incl. fjc ghosts each side
        self.iv = slice(self.fjc, self.M - self.fjc)   # interior slice
        self.gradients = 1
        interior = np.zeros(self.M, dtype=bool)         # flat interior mask
        interior[self.iv] = True                        # (uniform ND-style API)
        self.interior = interior
        # base step weight lambda_1: from lattice_type by default, but an
        # explicit `lat : X : lambda : v` overrides it (sfbox/Namics allow
        # a custom a-priori step probability, e.g. lambda = 1/3)
        lam = LAMBDA[lattice_type] if lam is None else float(lam)
        self.lam = lam
        # physical coordinate of the interior sites (Namics .pro: (k+0.5)/fjc);
        # for fjc=1 this is k+0.5 = z-0.5, matching the original output/moments
        self.z_phys = (np.arange(self.MX) + 0.5) / self.fjc
        # the negative stay weight of the refined cylinder's axis row, the
        # one exemption of _check_stencil (None: every weight is >= 0)
        self.axis_stay = None

        if self.fjc == 1:
            self._setup_fjc1(geometry, lam, offset_first_layer)
        else:
            self._setup_fjc(geometry)
        # second line of defence: every interior site must have a positive
        # volume
        if not np.all(self.L[self.iv] > 0):
            raise ValueError(
                f"Lattice1D {geometry}: a site with non-positive volume (min "
                f"{float(np.min(self.L[self.iv])):.3g}) -- check "
                "offset_first_layer")

    # ---- fjc = 1: original three-point lattice (unchanged) ------------------
    def _setup_fjc1(self, geometry, lam, offset_first_layer):
        z = np.arange(self.M, dtype=float)   # ghost, 1..MX, ghost
        r = offset_first_layer + z
        if geometry == "planar":
            self.L = np.ones(self.M)
            self.l1 = np.full(self.M, lam)   # weight toward z+1
            self.l_1 = np.full(self.M, lam)  # weight toward z-1
        elif geometry == "cylindrical":
            self.L = np.pi * (r**2 - (r - 1) ** 2)
            self.l1 = 2 * np.pi * r * lam / self.L
            self.l_1 = 2 * np.pi * (r - 1) * lam / self.L
        elif geometry == "spherical":
            self.L = 4.0 / 3.0 * np.pi * (r**3 - (r - 1) ** 3)
            self.l1 = 4 * np.pi * r**2 * lam / self.L
            self.l_1 = 4 * np.pi * (r - 1) ** 2 * lam / self.L
        else:
            raise ValueError(f"unknown geometry '{geometry}'")
        self.l0 = 1.0 - self.l1 - self.l_1
        # a custom `lat : lambda` must leave a step PROBABILITY in every
        # layer (l1, l_1, l0 >= 0): 0 < lambda <= 1/2 planar/cylindrical,
        # <= 1/3 for a sphere at offset 0 (its first layer stays with
        # 1 - 3 lambda), between the two at offset > 0. Testing the built
        # weights covers the offset dependence; out of range the chain
        # statistics are no random walk and densities can leave [0, 1]
        # with a converged status (review 27 Oct 2026, #41)
        iv = self.iv
        lo = min(float(self.l0[iv].min()), float(self.l1[iv].min()),
                 float(self.l_1[iv].min()))
        if not lam > 0 or lo < -1e-12:
            raise ValueError(
                f"lat : lambda : {lam} gives a negative or zero step weight "
                f"(min {lo:.3g}) on the {geometry} lattice: need 0 < lambda "
                "<= 1/2 (planar, cylindrical), <= 1/3 for a sphere at "
                "offset_first_layer 0 (between 1/3 and 1/2 at offset > 0)")
        self.volume = (self.L[self.iv].sum() if geometry != "planar"
                       else float(self.MX))
        # at fjc = 1 the layers partition the domain exactly, so the
        # lattice's own measure sum(L) IS the geometric volume
        self.L_sum = self.volume

    # ---- fjc > 1: refined (2*fjc+1)-point lattice (Namics ComputeLambdas) ---
    def _setup_fjc(self, geometry):
        fjc, FJC, MX, M = self.fjc, self.FJC, self.MX, self.M
        if geometry == "planar":
            # Namics LG1Planar.cpp:14-22: position-independent uniform
            # FJC weights -- a bond of length b lands uniformly across
            # the 2*fjc+1 grit offsets (the continuum freely-jointed
            # cos-theta projection), with HALF weight at the two end
            # channels (half-cells): interior 1/(2fjc), ends 1/(4fjc).
            # L = 1/fjc per refined layer (volume in b^3 units). No
            # reflection folding: mirrors are carried by the fjc-deep
            # ghost fills in set_bounds, exactly like LG1Planar's
            # truncated AddTimes loops over ghosted arrays.
            L = np.zeros(M)
            LAM = np.zeros((FJC, M))
            L[fjc:M - fjc] = 1.0 / fjc
            iv = slice(fjc, M - fjc)
            LAM[0, iv] = LAM[FJC - 1, iv] = 1.0 / (2.0 * (FJC - 1))
            for j in range(1, FJC - 1):
                LAM[j, iv] = 1.0 / (FJC - 1)
            self.L, self.LAM = L, LAM
            self.volume = float(MX) / fjc     # Namics LGrad1.cpp:183
            self.L_sum = self.volume          # = sum(L[iv]) exactly
            return
        if geometry not in ("spherical", "cylindrical"):
            raise NotImplementedError(
                f"FJC_choices > 1 with geometry '{geometry}' is not "
                "supported")
        off = self.offset
        L = np.zeros(M)
        LAM = np.zeros((FJC, M))            # LAM[c, i], neighbour offset c-fjc
        sph = geometry == "spherical"
        if sph:                            # sphere: C_ext=1/2 C_mid, area=4r^2
            area = lambda rr: 4.0 * rr * rr
            c_ext, c_mid = 0.5 / (FJC - 1), 1.0 / (FJC - 1)
        else:                              # cylinder: middle area doubled, area=r
            area = lambda rr: rr
            c_ext, c_mid = 1.0 / (FJC - 1), 2.0 / (FJC - 1)
        edge = MX / fjc
        # Channel j (inward) and FJC-1-j (outward) of refined site k land on
        # site kt = k + j - fjc and k + fjc - j. EVERY keep/fold/absorb
        # decision is taken on that INTEGER index, never on a float radius:
        #   1 <= kt <= MX           interior: free-space area at the midpoint
        #   kt <= 0, offset 0       the axis: dropped, the centre closes the row
        #   kt <= 0, inner mirror   folded with area(min(rlow, 2 m - rlow)),
        #                           m = off + 1/(2 fjc) the mirror plane: the
        #                           smaller of the bond's free-space face and
        #                           its image's (review 27 Oct 2026, #8)
        #   kt <= 0, inner surface  absorbed by the solid core r < off with
        #                           its free-space area (review 27 Oct, #9)
        #   kt > MX, outer mirror   folded by _reflect (Namics' else-branch)
        #   kt > MX, outer surface  absorbed by the solid shell with its
        #                           free-space area (review 27 Oct, #2)
        # The radii rlow/rhigh, accumulated in steps of 0.5/fjc as before,
        # enter only the AREAS, so LAM is bit-identical wherever no decision
        # changed. Why: (#3) at offset 0 the old axis test 2*rlow - r > 0
        # evaluated the channel that ends ON the axis (r' = 0) to +5.6e-17
        # at FJC_choices 11, 19, 23, 25, 27, 35, 37, 41, kept it, and broke
        # detailed balance (restricted molecules lost mass, results
        # depended on the chain orientation; Namics LGrad1.cpp:80/97/126/143
        # carries the same test). (#2) a surface absorbs with the weight the
        # bond has in free space -- the same model as a frozen shell on a
        # larger mirror box, and how fjc = 1 and the inner surface already
        # worked; the mirror-fold area made a concave outer wall absorb LESS
        # than a planar one (Namics folds regardless of the bound).
        # (#9) a bond from a site just outside a core of radius 0 < off < 1
        # can land behind the axis (r' <= 0) while it enters the solid; the
        # axis test returned it to the stay channel, so wall observables
        # jumped at every off = 1 - k/fjc and did not converge in fjc.
        # (#8) the inner mirror plane lies half a refined cell below the
        # first site (set_bounds maps ghost k = 0 onto site 1, at r = off and
        # off + 1/fjc). A bond from site k that lands in the ghost band at
        # r_t arrives at the image site k' = 1 - kt, whose own bond into the
        # band has the free-space face 2 m - rlow where site k's has rlow.
        # Detailed balance needs ONE area for the pair; the smaller of the
        # two free-space areas (the Metropolis choice) keeps every folded
        # weight at or below its free-space value, so the stay weight never
        # drops below the free-space stay weight: T >= 0 wherever that is
        # (every cylinder; a sphere down to offset ~0.01 at FJC_choices 35+,
        # where _check_stencil refuses). The 7 Oct 2026 rule, the face
        # midway between site and image area(off + r - rlow), was
        # reciprocal too, but its folded faces exceed the window VL at small
        # offsets: negative stay weights (-0.19 at a sphere FJC 11, offset
        # 0.5) and phi < 0 under a ~2 kT field on one refined site. On a
        # cylinder below offset 1/2 - 1/fjc the smaller face can lie behind
        # the axis (a negative signed area, as in free space); the direct
        # bond to the same image outweighs it and T stays >= 0. The rule is
        # an O(b/R) convention that does not vanish under refinement
        # (Namics books the unreciprocated forward area): it moved
        # tests/cylindrical_fjc_offset_mirror (GN 8e-4) and nothing else.
        m_lo = off + 0.5 / fjc
        lo_core = off > 0 and self.lowerbound == "surface"
        lo_mirror = off > 0 and not lo_core
        hi_surface = self.upperbound == "surface"
        if lo_core and off < 0.5 - 1.0 / fjc - 1e-12:
            # the first site's outermost inward bond has its midpoint face
            # at r = off + 1/fjc - 1/2 < 0: behind the axis, where the area
            # weight of a bond entering the core is undefined (negative on
            # a cylinder)
            raise NotImplementedError(
                f"lat : lowerbound : surface with offset_first_layer {off} "
                f"at FJC_choices {FJC}: the solid core is thinner than "
                f"1/2 - 1/fjc = {0.5 - 1.0 / fjc:.6g} b, so refined bonds "
                "from the first sites reach behind the axis through the "
                "solid -- not yet ported to PySFBox. Use offset_first_layer "
                f">= {0.5 - 1.0 / fjc:.6g}, offset_first_layer : 0 (no "
                "core), or a lower FJC_choices (FJC_choices 5 takes any "
                "offset)")
        for i in range(fjc, M - fjc):
            k = i - fjc + 1                        # 1-based refined index
            r = off + (i - fjc + 1.0) / fjc
            rlow, rhigh = r - 0.5, r + 0.5
            if sph:
                L[i] = np.pi * 4.0 / 3.0 * (rhigh**3 - rlow**3) / fjc
                VL = 4.0 / 3.0 * (rhigh**3 - rlow**3)
            else:
                L[i] = np.pi * (2.0 * r) / fjc
                VL = 2.0 * r
            for j in range(fjc):       # j = 0: the bond endpoints (C_ext)
                if j:
                    rlow += 0.5 / fjc
                    rhigh -= 0.5 / fjc
                coef = c_mid if j else c_ext
                kt = k + j - fjc                       # inward channel j
                if kt >= 1:                            # interior
                    LAM[j, i] += coef * area(rlow) / VL
                elif lo_mirror:                        # inner mirror fold
                    LAM[j, i] += coef * area(min(rlow, 2.0 * m_lo - rlow)) / VL
                elif lo_core:                          # absorbed by the core
                    # max(): at off = 1/2 - 1/fjc the face sits on the axis
                    LAM[j, i] += coef * area(max(rlow, 0.0)) / VL
                # (offset 0: kt <= 0 ends on or behind the axis -- dropped)
                kt = k + fjc - j                       # outward FJC-1-j
                if kt <= MX or hi_surface:             # interior / absorbed
                    LAM[FJC - 1 - j, i] += coef * area(rhigh) / VL
                else:
                    # the outer mirror sits at r = off + edge: the offset
                    # enters the reflected distance (before 25 Sep 2026 it
                    # did not, and every offset > 0 folded the outermost
                    # channel onto the centre at EVERY site)
                    self._reflect(LAM, FJC - 1 - j, i, r, rhigh, off + edge,
                                  fjc, coef, VL, area)
            LAM[fjc, i] += 1.0 - LAM[:, i].sum()         # centre closes the row
        self.L, self.LAM = L, LAM
        # geometric volume in closed form. The offset enters in PHYSICAL
        # bond units, matching the shell placement r = off + k/fjc above
        # (Namics LGrad1.cpp:184 looks like ((MX+off)^3-off^3)/fjc^3 but
        # its variable is pre-multiplied by fjc at lattice.cpp:387; a
        # line-copy without that pre-multiply put the offset in refined
        # units here until 21 Aug 2026).
        off = self.offset * fjc
        if geometry == "spherical":
            self.volume = 4.0 / 3.0 * np.pi * ((MX + off) ** 3 - off ** 3) \
                / fjc ** 3
        else:                                            # cylindrical
            self.volume = np.pi * ((MX + off) ** 2 - off ** 2) / fjc ** 2
        # The lattice's own measure sum(L[iv]) EXCEEDS the geometric
        # volume: the refined "shells" above are bond-thick smears (each
        # a [r-1/2, r+1/2] window /fjc), whose sum carries a surface
        # overhang ~ 2*pi*R^2/fjc (sphere; relative 3/(2*R*fjc)). Excess
        # bookkeeping (theta_exc = theta - phibulk*measure) must use THIS
        # measure, so a uniform bulk phi = phibulk has exactly zero
        # excess site-wise; subtracting the geometric volume instead
        # pollutes every excess/moment column by phibulk*(L_sum - volume)
        # — which is what Namics does (its theta_exc mixes the measures
        # the same way; intentional deviation, reported upstream).
        self.L_sum = float(L[self.iv].sum())
        self._check_stencil()

    def _check_stencil(self):
        """Construction-time safety net for the refined CURVED stencil
        (review 27 Oct 2026, #8): the one-step site average, ghost folds
        included, must be a reversible (sub)probability average -- weights
        T >= 0, rows summing to 1 (at most 1 next to a surface), and
        L_n T_nm = L_m T_mn. T >= 0 is what keeps a positive Boltzmann
        weight's propagator, and so phi, positive.

        The folds only touch the first and last 2 fjc rows, so the folded
        operator is built there column by column from the production code
        path, site_average(set_bounds(e_m)), for the first and last
        min(MX, 3 fjc) sites (O(fjc) averages); the rows in between have no
        fold, their T row is the raw LAM column, checked directly. Note the
        criterion is T, not LAM: a channel whose image is the site itself
        folds onto the diagonal, so LAM[fjc] < 0 can be a valid T_kk >= 0.

        EXEMPT (decided 28 Oct 2026): the offset-0 CYLINDER's axis rows.
        Its windows straddle the axis with the signed measure VL = 2 r, and
        at FJC_choices >= 11 the stay weight of the axis row is negative
        (T00 = -0.075 at FJC 11, -0.725 at 21; Namics has the same rows,
        and production cylindrical micelles use them).
        The stencil is kept; the value is stored in self.axis_stay, a
        once-per-process note says so, and System._audit_solution refuses
        any phi < 0."""
        fjc, MX, M, FJC = self.fjc, self.MX, self.M, self.FJC
        L, LAM, iv = self.L, self.LAM, self.iv
        Li = L[iv]
        scale = float(np.max(Li[:, None] * np.abs(LAM[:, iv]).T))
        where = (f"Lattice1D {self.geometry}, FJC_choices {FJC}, "
                 f"offset_first_layer {self.offset:g}, lowerbound "
                 f"{self.lowerbound}, upperbound {self.upperbound}")
        # (1) direct channels between two interior sites: L-reciprocal
        worst = 0.0
        for c in range(FJC):
            d = c - fjc
            i = np.arange(fjc + max(0, -d), M - fjc - max(0, d))
            a = L[i] * LAM[c, i]
            b = L[i + d] * LAM[FJC - 1 - c, i + d]
            worst = max(worst, float(np.max(np.abs(a - b), initial=0.0)))
        # (2) the folded boundary blocks, column by column: B[a, b] =
        # T[cols[a], cols[b]] (every row within 2 fjc of an end has its
        # whole support inside cols)
        K = min(MX, 3 * fjc)
        cols = sorted(set(range(K)) | set(range(MX - K, MX)))
        B = np.zeros((len(cols), len(cols)))
        for jc, m in enumerate(cols):
            e = np.zeros(M)
            e[fjc + m] = 1.0
            B[:, jc] = self.site_average(self.set_bounds(e))[iv][cols]
        F = Li[cols][:, None] * B
        worst = max(worst, float(np.max(np.abs(F - F.T))))
        if worst > STENCIL_TOL * scale:
            raise ValueError(
                f"{where}: the refined stencil breaks detailed balance "
                f"(L_n T_nm - L_m T_mn up to {worst / scale:.2e} of the "
                "largest weight) -- a PySFBox defect, please report the input")
        ones = np.zeros(M)
        ones[iv] = 1.0
        rows = self.site_average(self.set_bounds(ones))[iv]
        # a surface absorbs: rows within 2 fjc of one sum to at most 1
        absorbs = (((np.arange(MX) < 2 * fjc) & (self.lowerbound == "surface"))
                   | ((np.arange(MX) >= MX - 2 * fjc)
                      & (self.upperbound == "surface")))
        need_one = ~absorbs
        bad_row = (rows > 1.0 + STENCIL_TOL) | (
            need_one & (np.abs(rows - 1.0) > STENCIL_TOL))
        if np.any(bad_row):
            n = int(np.argmax(bad_row))
            raise ValueError(
                f"{where}: the site average of a uniform profile is "
                f"{rows[n]:.15g} at refined site {n + 1}, not 1 -- a PySFBox "
                "defect, please report the input")
        # (3) T >= 0: boundary rows from the columns, the rest from LAM
        mid = np.ones(MX, dtype=bool)
        mid[:min(MX, 2 * fjc)] = False
        mid[max(0, MX - 2 * fjc):] = False
        tmin = np.full(MX, np.inf)
        tmin[cols] = B.min(axis=1)
        if np.any(mid):
            tmin[mid] = LAM[:, iv][:, mid].min(axis=0)
        axis = self.geometry == "cylindrical" and self.offset == 0
        neg = tmin < -STENCIL_TOL
        if axis:                     # the exempt axis rows (docstring)
            ax = neg & (np.arange(MX) < fjc)
            if np.any(ax):
                self.axis_stay = float(tmin[ax].min())
                note = (f"the refined cylinder at FJC_choices {FJC} "
                        "and offset_first_layer 0 has a negative stay "
                        f"weight T00 = {self.axis_stay:.4g} on its axis row "
                        "(Namics' lattice, kept); results are audited for "
                        "phi >= 0")
                if note not in _AXIS_NOTES:      # once per process
                    _AXIS_NOTES.add(note)
                    print(f"  note: {note}")
            neg = neg & ~ax
        if np.any(neg):
            n = int(np.argmax(neg))
            raise ValueError(
                f"{where}: negative transition weight {tmin[n]:.3g} at "
                f"refined site {n + 1} -- the site average is no longer a "
                "probability average, and a sharp field there drives "
                "densities negative. Increase offset_first_layer or lower "
                "FJC_choices")

    @staticmethod
    def _reflect(LAM, c, i, r, rhigh, edge, fjc, coef, VL, area):
        """Fold a bond that reaches past the outer MIRROR back into channel c
        (Namics' else-branch: reflected radius rhigh - k/fjc). Mirror only:
        an outer surface absorbs the bond with its free-space area instead
        (review 27 Oct 2026, #2; Namics folds for both bounds)."""
        d = 2 * rhigh - r - edge
        if -0.001 < d < 0.001:
            LAM[c, i] += coef * area(rhigh) / VL
        for k in range(1, fjc + 1):
            if 0.99 * k / fjc < d < 1.01 * k / fjc:
                LAM[c, i] += coef * area(rhigh - k / fjc) / VL

    # -- boundary handling -------------------------------------------------
    def set_bounds(self, f, lower_value=None, upper_value=None):
        """Fill the fjc ghost layers on each side. mirror: reflect the interior;
        surface: 0 (impenetrable solid). Explicit values override the whole
        ghost block (used for frozen surface densities)."""
        f = f.copy()
        fjc, MX = self.fjc, self.MX
        for k in range(fjc):
            lo, hi = k, MX + fjc + k          # ghost indices (lower, upper)
            if lower_value is not None:
                f[lo] = lower_value
            elif self.lowerbound == "surface":
                f[lo] = 0.0
            else:                             # mirror: reflect about the edge
                f[lo] = f[2 * fjc - k - 1]
            if upper_value is not None:
                f[hi] = upper_value
            elif self.upperbound == "surface":
                f[hi] = 0.0
            else:
                f[hi] = f[MX + fjc - k - 1]
        return f

    def set_mirror_bounds(self, f):
        """Fill ghost layers by MIRRORING regardless of the boundary type
        (Namics set_M_bounds) — used for the electrostatic potential, whose
        wall condition is zero-field, not zero-value."""
        f = f.copy()
        fjc, MX = self.fjc, self.MX
        for k in range(fjc):
            f[k] = f[2 * fjc - k - 1]
            f[MX + fjc + k] = f[MX + fjc - k - 1]
        return f

    def site_average(self, f_with_bounds):
        """<f>(z): three-point (fjc=1) or (2*fjc+1)-point (fjc>1) weighted
        average over neighbours, interior only."""
        out = np.zeros(self.M)
        if self.fjc == 1:
            out[1:-1] = (self.l_1[1:-1] * f_with_bounds[:-2]
                         + self.l0[1:-1] * f_with_bounds[1:-1]
                         + self.l1[1:-1] * f_with_bounds[2:])
            return out
        f, fjc, M = f_with_bounds, self.fjc, self.M
        for c in range(self.FJC):
            d = c - fjc                       # neighbour offset
            if d == 0:
                out += self.LAM[c] * f
            elif d > 0:
                out[:-d] += self.LAM[c][:-d] * f[d:]
            else:
                out[-d:] += self.LAM[c][-d:] * f[:d]
        return out

    def propagate(self, gs, G1, lower_value=0.0, upper_value=None):
        """One chain-propagation step: G1 * <gs>. Propagators see solids as
        0 (default lower ghost 0 when lowerbound is a surface; mirror
        otherwise handled by set_bounds default)."""
        lv = lower_value if self.lowerbound == "surface" else None
        gb = self.set_bounds(gs, lower_value=lv, upper_value=upper_value)
        return G1 * self.site_average(gb)

    # -- observables --------------------------------------------------------
    def weighted_sum(self, X):
        return float(np.dot(X[self.iv], self.L[self.iv]))

    def moment(self, X, Xb, n, weight=None):
        # z_phys is in bond units and L is the physical layer volume, so
        # this sum IS the physical moment, fjc-independent under
        # refinement. Do NOT divide by fjc again: Namics' LGrad1::Moment
        # does, but its 1st_M_phi_z caller multiplies back
        # (segment.cpp:1837) while its 2nd-moment caller does not, so the
        # binary's refined 2nd moments are fjc too small and no oracle
        # here. The 5 Jul 2026 physics review copied the inner /fjc
        # without the caller's *fjc; both dropped 20 Aug 2026
        # (collaborator bug report). Bit-identical at fjc=1.
        # `weight` (a full-length site mask, System.ksam) restricts the sum
        # to the free sites: a frozen interior site holds no mobile density
        # and must not enter as -Xb of "depleted" volume (review 27 Oct
        # 2026, #39; Namics sums every site). A weight of 1.0 everywhere
        # is bit-identical to no weight.
        w = (X[self.iv] - Xb) * self.L[self.iv]
        if weight is not None:
            w = w * weight[self.iv]
        return float(np.dot(self.z_phys**n, w))
