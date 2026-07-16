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


class Lattice1D:
    def __init__(self, n_layers, geometry="planar",
                 lattice_type="simple_cubic", lowerbound="mirror",
                 upperbound="mirror", offset_first_layer=0.0, fjc=1,
                 lam=None):
        if geometry == "flat":          # Namics accepts flat as planar synonym
            geometry = "planar"
        self.geometry = geometry
        self.lattice_type = lattice_type
        self.lowerbound, self.upperbound = lowerbound, upperbound
        # chain-stiffness defaults (Namics reads Markov/k_stiff at both lat
        # and mol level; System fills these from the lat block and each
        # Molecule may override with its own mol : X : Markov / k_stiff).
        self.markov = 1
        self.k_stiff = 0.0
        self.fjc = int(fjc)
        self.FJC = 2 * self.fjc + 1
        self.offset = float(offset_first_layer)
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

        if self.fjc == 1:
            self._setup_fjc1(geometry, lam, offset_first_layer)
        else:
            self._setup_fjc(geometry)

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
        self.volume = (self.L[self.iv].sum() if geometry != "planar"
                       else float(self.MX))

    # ---- fjc > 1: refined (2*fjc+1)-point lattice (Namics ComputeLambdas) ---
    def _setup_fjc(self, geometry):
        if geometry not in ("spherical", "cylindrical"):
            raise NotImplementedError(
                f"FJC_choices > 1 with geometry '{geometry}' is not supported "
                "yet (not yet ported to PySFBox); PySFBox has spherical/cylindrical")
        fjc, FJC, MX, M = self.fjc, self.FJC, self.MX, self.M
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
        for i in range(fjc, M - fjc):
            r = off + (i - fjc + 1.0) / fjc
            rlow, rhigh = r - 0.5, r + 0.5
            if sph:
                L[i] = np.pi * 4.0 / 3.0 * (rhigh**3 - rlow**3) / fjc
                VL = 4.0 / 3.0 * (rhigh**3 - rlow**3)
            else:
                L[i] = np.pi * (2.0 * r) / fjc
                VL = 2.0 * r
            edge = MX / fjc
            # outermost channels (bond endpoints)
            if 2 * rlow - r > 0:
                LAM[0, i] += c_ext * area(rlow) / VL
            if 2 * rhigh - r < edge:
                LAM[FJC - 1, i] += c_ext * area(rhigh) / VL
            else:                          # reflect off the outer mirror
                self._reflect(LAM, FJC - 1, i, r, rhigh, edge, fjc, c_ext, VL, area)
            # inner channels
            for j in range(1, fjc):
                rlow += 0.5 / fjc
                rhigh -= 0.5 / fjc
                if 2 * rlow - r > 0:
                    LAM[j, i] += c_mid * area(rlow) / VL
                if 2 * rhigh - r < off + edge:
                    LAM[FJC - 1 - j, i] += c_mid * area(rhigh) / VL
                else:
                    self._reflect(LAM, FJC - 1 - j, i, r, rhigh, edge, fjc,
                                  c_mid, VL, area)
            LAM[fjc, i] += 1.0 - LAM[:, i].sum()         # centre closes the row
        self.L, self.LAM = L, LAM
        # geometric volume in closed form (Namics LGrad1.cpp:184-185); this
        # equals sum(L[iv]) at fjc=1 but the interior L-sum over-counts the
        # true volume for fjc>1 (boundary half-cells). Found 5 Jul 2026.
        off = self.offset
        if geometry == "spherical":
            self.volume = 4.0 / 3.0 * np.pi * ((MX + off) ** 3 - off ** 3) \
                / fjc ** 3
        else:                                            # cylindrical
            self.volume = np.pi * ((MX + off) ** 2 - off ** 2) / fjc ** 2

    @staticmethod
    def _reflect(LAM, c, i, r, rhigh, edge, fjc, coef, VL, area):
        """Fold a bond that reaches past the outer mirror back into channel c
        (Namics' else-branch: reflected radius rhigh - k/fjc)."""
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

    def moment(self, X, Xb, n):
        # Namics PutM divides the summed moment by fjc so refined-lattice
        # (fjc>1) moments come out in physical-layer units (factor 1 at
        # fjc=1). Found in the 5 Jul 2026 physics review.
        return float(np.dot(self.z_phys**n, (X[self.iv] - Xb)
                            * self.L[self.iv])) / self.fjc
