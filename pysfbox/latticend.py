"""Unified N-gradient lattice (2 or 3 gradients) via a finite-volume metric.

One class, `LatticeND`, covers the Namics 2- and 3-gradient geometries
through a single principle. A cell `i` has volume `V_i`; between adjacent
cells `i,j` there is a shared face of area `A_ij` and their centres are a
distance `d_ij` apart. The chain-step transition weight is

    lambda(i->j) = C * A_ij / (V_i * d_ij),   C = 1/6 (simple cubic),
    lambda(i->i) = 1 - sum_j lambda(i->j)     (the "stay" weight),

so that (1) **detailed balance** `V_i lambda(i->j) = C A_ij/d_ij = V_j
lambda(j->i)` holds identically (A_ij, d_ij symmetric) -- the site average is
self-adjoint under the L-weighted inner product, which is what makes the
forward/backward propagators consistent; and (2) the continuum limit is
`G + (1/6) grad^2 G`, the Gaussian-chain diffusion. Here `d_ij = 1` for
every axis, and this reduces bit-for-bit to `Lattice1D` (verified) and to
the flat/cylindrical 2-gradient stencil.

Geometries (axis roles in brackets; `x/z` cartesian, `r` radial), exactly
the Namics set (LGrad2/LGrad3):

    2D  flat/planar   [x, x]          cylindrical [r, z]
    3D  flat/planar   [x, x, x]

simple_cubic, fjc = 1, first-order Markov.

Profiles are FLAT arrays of length `M = prod(n_a + 2)` in C order over the
padded grid (one ghost layer per side per axis), so `System`'s residual and
`compute_phis` -- which operate on flat `(n_species, M)` stacks and only call
`site_average` / `set_bounds` -- work unchanged. The N-D structure lives
entirely inside this class.
"""

import numpy as np

LAMBDA = {"simple_cubic": 1.0 / 6.0}

# geometry name -> axis roles, per gradient count. "flat" and "planar" alias.
# Exactly the geometries Namics has: 2D flat, 2D cylindrical (r, z), 3D flat.
_GEOM = {
    2: {"flat": ("x", "x"), "planar": ("x", "x"),
        "cylindrical": ("r", "z")},
    3: {"flat": ("x", "x", "x"), "planar": ("x", "x", "x")},
}


class LatticeND:
    def __init__(self, dims, geometry="flat", lattice_type="simple_cubic",
                 bounds=None, offset_first_layer=0.0):
        self.dims = tuple(int(d) for d in dims)
        self.gradients = len(self.dims)
        if self.gradients not in (2, 3):
            raise NotImplementedError(
                "LatticeND handles 2 or 3 gradients (1 -> Lattice1D)")
        if lattice_type != "simple_cubic":
            raise NotImplementedError(
                "LatticeND supports the simple_cubic face-neighbour stencil "
                "only (the hexagonal/FCC diagonal stencil is separate)")
        if geometry == "planar":
            geometry = "flat"
        if geometry not in _GEOM[self.gradients]:
            raise NotImplementedError(
                f"lat : geometry : {geometry} with gradients : "
                f"{self.gradients} is not supported; supported: flat, "
                "cylindrical (2 gradients) and flat (3 gradients), as in "
                "Namics")
        self.geometry = geometry
        self.lattice_type = lattice_type
        self.roles = _GEOM[self.gradients][geometry]
        self.fjc = 1
        self.markov = 1
        self.k_stiff = 0.0
        self.offset = float(offset_first_layer)
        self.C = LAMBDA[lattice_type]
        self.lam = self.C                                # adsorption-guess weight

        self.pdims = tuple(n + 2 for n in self.dims)     # padded (ghost each side)
        self.M = int(np.prod(self.pdims))
        # interior selector (tuple of slices) and flat interior mask
        self.iv = tuple(slice(1, n + 1) for n in self.dims)
        interior = np.zeros(self.pdims, dtype=bool)
        interior[self.iv] = True
        self.interior = interior.ravel()

        # default boundaries: mirror. `bounds` may be a full list OR have
        # `None` entries for axes the caller did not set explicitly -- those
        # fall back to the default.
        default = [("mirror", "mirror") for _ in range(self.gradients)]
        if bounds is None:
            self.bounds = default
        else:
            self.bounds = [default[a] if bounds[a] is None else bounds[a]
                           for a in range(self.gradients)]
        self._build_metric()

    # -- finite-volume metric ---------------------------------------------
    def _axis_geom(self, a):
        """Per-cell coordinate bounds along axis a on the padded grid:
        returns (lo, hi, centre, step) index arrays broadcast to pdims."""
        role = self.roles[a]
        idx = np.arange(self.pdims[a], dtype=float)     # 0..n+1
        if role in ("x", "z"):
            lo, hi = idx - 1.0, idx                      # cell [i-1, i]
            step = 1.0
        elif role == "r":
            lo = self.offset + idx - 1.0
            hi = self.offset + idx
            step = 1.0
        shape = [1] * self.gradients
        shape[a] = self.pdims[a]
        return (lo.reshape(shape), hi.reshape(shape),
                (0.5 * (lo + hi)).reshape(shape), step)

    def _build_metric(self):
        g = self.pdims
        roles = self.roles
        # gather per-axis coordinate arrays (broadcast to the full grid)
        lo = {}
        hi = {}
        ctr = {}
        step = {}
        for a in range(self.gradients):
            lo[a], hi[a], ctr[a], step[a] = self._axis_geom(a)
        one = np.ones(g)

        # radial axis (at most one), its volume/face polynomial factors
        r_ax = roles.index("r") if "r" in roles else None

        def rint(p):                       # integral of r^p over the radial cell
            if r_ax is None:
                return one
            a = r_ax
            return (hi[a] ** (p + 1) - lo[a] ** (p + 1)) / (p + 1) * one

        # ---- volume L and the +face area / +distance per axis --------------
        L = one.copy()
        Fp = [None] * self.gradients       # +face area (between cell and cell+1)
        Dp = [None] * self.gradients       # +centre distance
        geom = self.geometry

        if "x" in roles and "r" not in roles:            # pure cartesian (flat)
            for a in range(self.gradients):
                L = L * (hi[a] - lo[a])
            for a in range(self.gradients):
                area = one.copy()
                for b in range(self.gradients):
                    if b != a:
                        area = area * (hi[b] - lo[b])
                Fp[a] = area
                Dp[a] = one.copy()
            self._finish(L, Fp, Dp)
            return

        # geometries with a radial axis: assign scale-factor integrals
        # volume integrand = r^pr * const, per geometry
        if geom == "cylindrical":                                # (r, z)
            L = rint(1) * 2.0 * np.pi * (hi[1] - lo[1])          # pi(r2-r1) * dz
            Fp[0] = 2.0 * np.pi * hi[0] * (hi[1] - lo[1]); Dp[0] = one
            Fp[1] = rint(1) * 2.0 * np.pi;                Dp[1] = one   # =L/dz
        else:
            raise NotImplementedError(f"metric for '{geom}' not implemented")
        self._finish(L, Fp, Dp)

    def _finish(self, L, Fp, Dp):
        C = self.C
        self.L = L.ravel()
        self.volume = float(L[self.iv].sum())
        L_safe = np.where(L > 0, L, 1.0)
        self._lam_p = []
        self._lam_m = []
        for a in range(self.gradients):
            Dp_safe = np.where(Dp[a] > 0, Dp[a], 1.0)
            lam_p = np.where((L > 0) & (Dp[a] > 0),
                             C * Fp[a] / (L_safe * Dp_safe), 0.0)
            # -face of cell = +face of the cell below (shift +1 along axis a)
            Fp_m = np.roll(Fp[a], 1, axis=a)
            Dp_m = np.roll(Dp[a], 1, axis=a)
            Dp_m_safe = np.where(Dp_m > 0, Dp_m, 1.0)
            lam_m = np.where((L > 0) & (Dp_m > 0),
                             C * Fp_m / (L_safe * Dp_m_safe), 0.0)
            self._lam_p.append(lam_p)
            self._lam_m.append(lam_m)
        lam0 = np.ones(self.pdims)
        for a in range(self.gradients):
            lam0 = lam0 - self._lam_p[a] - self._lam_m[a]
        self._lam0 = lam0

    # -- boundary handling -------------------------------------------------
    def _fill_ghosts(self, g, values):
        """Fill the two ghost hyperplanes of every axis in-place on the grid
        array g. `values` optionally overrides a face with a constant."""
        for a in range(self.gradients):
            lo_kind, hi_kind = self.bounds[a]
            n = self.dims[a]
            sl = [slice(None)] * self.gradients
            # lower ghost (index 0)
            src = list(sl); dst = list(sl)
            dst[a] = 0
            v = values.get((a, "lo")) if values else None
            if v is not None:
                g[tuple(dst)] = v
            elif lo_kind == "surface":
                g[tuple(dst)] = 0.0
            elif lo_kind == "periodic":
                src[a] = n
                g[tuple(dst)] = g[tuple(src)]
            else:                                       # mirror
                src[a] = 1
                g[tuple(dst)] = g[tuple(src)]
            # upper ghost (index n+1)
            src = list(sl); dst = list(sl)
            dst[a] = n + 1
            v = values.get((a, "hi")) if values else None
            if v is not None:
                g[tuple(dst)] = v
            elif hi_kind == "surface":
                g[tuple(dst)] = 0.0
            elif hi_kind == "periodic":
                src[a] = 1
                g[tuple(dst)] = g[tuple(src)]
            else:
                src[a] = n
                g[tuple(dst)] = g[tuple(src)]
        return g

    def set_bounds(self, f, values=None):
        """Return a copy of the flat profile with ghosts filled. `values` is a
        dict {(axis, 'lo'|'hi'): const} overriding a face (frozen surfaces)."""
        g = f.reshape(self.pdims).copy()
        self._fill_ghosts(g, values or {})
        return g.ravel()

    def set_mirror_bounds(self, f):
        """Ghosts by mirroring on every non-periodic face (the electrostatic
        potential's zero-field wall); periodic axes still wrap."""
        g = f.reshape(self.pdims).copy()
        for a in range(self.gradients):
            n = self.dims[a]
            sl = [slice(None)] * self.gradients
            periodic = self.bounds[a][0] == "periodic"
            dst = list(sl); src = list(sl)
            dst[a] = 0; src[a] = n if periodic else 1
            g[tuple(dst)] = g[tuple(src)]
            dst = list(sl); src = list(sl)
            dst[a] = n + 1; src[a] = 1 if periodic else n
            g[tuple(dst)] = g[tuple(src)]
        return g.ravel()

    def site_average(self, fb):
        """<f>: the finite-volume weighted average (ghosts must be set).
        Returns a flat interior-filled array (ghosts 0), like Lattice1D."""
        g = fb.reshape(self.pdims)
        out = self._lam0 * g
        for a in range(self.gradients):
            out = out + self._lam_p[a] * np.roll(g, -1, axis=a)
            out = out + self._lam_m[a] * np.roll(g, 1, axis=a)
        out = out.ravel()
        out[~self.interior] = 0.0
        return out

    def propagate(self, gs, G1, values=None):
        """One chain-propagation step G1 * <gs>. Solid (surface) walls enter
        as 0 via set_bounds; `values` can pin a frozen surface density."""
        vals = values
        if vals is None:
            vals = {}
            for a in range(self.gradients):
                if self.bounds[a][0] == "surface":
                    vals[(a, "lo")] = 0.0
                if self.bounds[a][1] == "surface":
                    vals[(a, "hi")] = 0.0
        return G1 * self.site_average(self.set_bounds(gs, vals))

    # -- observables -------------------------------------------------------
    def weighted_sum(self, X):
        Xg = X.reshape(self.pdims)
        return float((Xg[self.iv] * self.L.reshape(self.pdims)[self.iv]).sum())

    # -- ranges / masks (Namics 2D/3D grammar "xlo,ylo[,zlo];xhi,yhi[,zhi]") -
    def parse_range(self, rng):
        """Return (flat interior mask, face) for a frozen/pinned range string.
        `face` is (axis, 'lo'|'hi') if the box is a single boundary-adjacent
        layer flush against a wall (so a frozen wall can fill the ghost face),
        else None. Coordinates are 1-based interior indices per axis; a
        Namics 'lowerbound'/'upperbound' along axis 0 is also accepted."""
        r = rng.strip().lower().rstrip(";")
        mask = np.zeros(self.pdims, dtype=bool)
        if r in ("lowerbound", "upperbound"):
            # a wall on the first axis' lower/upper face (Namics 1D idiom)
            face = (0, "lo" if r == "lowerbound" else "hi")
            return np.zeros(self.M, dtype=bool), face
        lo_txt, _, hi_txt = r.partition(";")
        los = [int(v) for v in lo_txt.split(",")]
        his = [int(v) for v in hi_txt.split(",")] if hi_txt else los
        if len(los) != self.gradients or len(his) != self.gradients:
            raise ValueError(
                f"range '{rng}' needs {self.gradients} coordinates per corner "
                f"(got {len(los)}); grammar 'xlo,ylo[,zlo];xhi,yhi[,zhi]'")
        sl = tuple(slice(lo, hi + 1) for lo, hi in zip(los, his))
        mask[sl] = True
        # detect a wall face: a box one layer thick against a boundary AND
        # spanning the FULL face on every other axis (a whole wall). A PARTIAL
        # patch must NOT be treated as a face -- filling the entire ghost
        # hyperplane to 1.0 would fake solid contact along the whole boundary;
        # it stays an interior box mask, handled correctly by the default
        # ghost fill.
        face = None
        for a in range(self.gradients):
            if los[a] == his[a] and los[a] in (1, self.dims[a]):
                full = all(los[b] == 1 and his[b] == self.dims[b]
                           for b in range(self.gradients) if b != a)
                if full:
                    face = (a, "lo" if los[a] == 1 else "hi")
        return mask.ravel(), face

    def moment(self, X, Xb, n, axis=0):
        """n-th moment of (X - Xb) along `axis` (the radial/first axis by
        default), volume weighted -- e.g. the radial extent of an adsorbed
        layer. The coordinate is measured from the FIRST LAYER (offset
        subtracted on a radial axis), matching Lattice1D.moment / the Namics
        LGrad1 convention so 1D and N-D report the same moment for the same
        physical density."""
        _, _, ctr, _ = self._axis_geom(axis)
        c = np.broadcast_to(ctr, self.pdims).astype(float)
        if self.roles[axis] == "r":
            c = c - self.offset           # distance from the first layer, as 1D
        Xg = X.reshape(self.pdims)
        Lg = self.L.reshape(self.pdims)
        w = (Xg[self.iv] - Xb) * Lg[self.iv]
        return float((c[self.iv] ** n * w).sum())
