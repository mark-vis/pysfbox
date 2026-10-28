"""Free-site moment columns and the absent-molecule mu (28 Oct 2026,
review 27 Oct #39 / #74 / #66).

The suite compares converged output against committed references
(tests/frozen_interior_moments.in, tests/mu_absent.in,
tests/spherical_fjc_excess.in); this harness checks what a reference
cannot say: that the kal columns ARE the free-site sums, that equivalent
layouts agree, that the old convention is what Namics prints, and that the
NiN / raise branches fire.

 1. 1-D layouts (frozen_interior_moments.in starts 1-3: ghost wall, wall
    frozen on layer 1, on layers 1-2): Omega and fluctuations agree, <z>
    moves by one layer per frozen layer, <z^2> = <(z + k)^2>, phiMin of
    the solvent agrees; every kal moment == a direct loop over the free
    sites of the solved profile (Gamma = sum (phi - phib) L over them).
 2. Positive control: the all-site convention (lat.moment without the
    weight, / theta_exc) reproduces the Namics 3e9eafd oracle numbers of
    the same input (1st_M 4.35778 / fluctuations 0.84321 at 1;1, a
    clamped negative variance at 1;2).
 3. Sign-changing excess (start 4): fluctuations nan, RMS finite; the
    positive-excess start 5 stays finite.
 4. spherical_fjc_excess.in (refined sphere, frozen core 1;3): kal == the
    free-site L-measure integrals; theta_exc == the all-site excess.
 5. N-D: a y-uniform 2-D flat box with the wall frozen on x-layer 1 == the
    1-D simple-cubic column (moments, phiMin/phiMax), and == its own
    direct free-site sums.
 6. mu (#66): absent molecule -> -inf (free phibulk 0, restricted theta 0),
    X skips it (X == F); a present molecule with a non-finite ln GN ->
    (None, None) + a note; the mu search target raises on an absent probe.

Not part of run_tests.py (a reference comparison cannot say "this must
raise"). Run from the repo root: python tests/observable_conventions.py
(~1 s). Run after touching Lattice1D.moment / LatticeND.moment, the
moment / phiMin / phiMax branches of System.get_value,
System.chemical_potential, System.characteristic_X or runner._target_error.
"""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("PYSFBOX_NO_PROGRESS", "1")
from pysfbox import runner                                # noqa: E402
from pysfbox.inputreader import read_input                # noqa: E402
from pysfbox.system import System                         # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAIL.append(name)


def kal(s, key, name, prop):
    return s.get_value(key, name, prop)[1]


def direct_moments(s, mon="A", frozen=("S",)):
    """Gamma, <z>, <z^2> of `mon` over the free sites, by an explicit loop.
    The free sites are the interior sites outside every frozen range; z is
    the .pro x column of the axis (refined index k -> (k + 1/2)/fjc,
    measured from the box edge, offset not added)."""
    lat = s.lat
    seg = s.segments[mon]
    phib = s.phibulk_seg[mon]
    solid = np.zeros(lat.M)
    for f in frozen:
        solid += s.segments[f].range_mask
    if lat.gradients == 1:
        sites = range(lat.fjc, lat.M - lat.fjc)
        coord = {i: (i - lat.fjc + 0.5) / lat.fjc for i in sites}
    else:
        # axis 0 of a flat LatticeND (fjc 1): padded index (i, j) -> x = i - 1/2
        idx = np.arange(lat.M).reshape(lat.pdims)
        sites = [int(v) for v in idx[lat.iv].ravel()]
        coord = {v: np.unravel_index(v, lat.pdims)[0] - 0.5 for v in sites}
    G = M1 = M2 = 0.0
    for i in sites:
        if solid[i] > 0:
            continue
        w = (seg.phi[i] - phib) * lat.L[i]
        G += w
        M1 += coord[i] * w
        M2 += coord[i] ** 2 * w
    return G, M1 / G, M2 / G


def solve_all(path):
    out = []
    for calc in read_input(path):
        s = System(calc)
        s.solve(tolerance=1e-11)
        out.append(s)
    return out


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


print("1. 1-D layouts: ghost wall / frozen 1;1 / frozen 1;2")
S = solve_all(os.path.join(ROOT, "tests", "frozen_interior_moments.in"))
a, b, c = S[0], S[1], S[2]
om = [s.grand_potential() for s in (a, b, c)]
check("Omega equal in all layouts", max(abs(o - om[0]) for o in om) < 1e-10,
      " ".join(f"{o:.12e}" for o in om))
fl = [kal(s, "mon", "A", "fluctuations") for s in (a, b, c)]
check("fluctuations equal (1.871135)",
      max(rel(f, fl[0]) for f in fl) < 1e-7 and abs(fl[0] - 1.871135) < 1e-6,
      " ".join(f"{f:.10f}" for f in fl))
m1 = [kal(s, "mon", "A", "1st_M_phi_z") for s in (a, b, c)]
m2 = [kal(s, "mon", "A", "2nd_M_phi_z") for s in (a, b, c)]
for k, s in ((1, b), (2, c)):
    check(f"<z> shifts by {k} (box-edge origin)",
          abs(m1[k] - (m1[0] + k)) < 1e-7, f"{m1[k]:.10f}")
    check(f"<z^2> = <(z+{k})^2>",
          abs(m2[k] - (m2[0] + 2 * k * m1[0] + k * k)) < 1e-6,
          f"{m2[k]:.10f}")
pm = [kal(s, "mol", "W", "phiMin") for s in (a, b, c)]
check("phiMin(W) = 0.979495, not the frozen site's 0",
      max(abs(p - 0.979495) for p in pm) < 1e-6,
      " ".join(f"{p:.8f}" for p in pm))
for lbl, s in (("ghost", a), ("1;1", b), ("1;2", c)):
    G, d1, d2 = direct_moments(s)
    ok = (rel(kal(s, "mon", "A", "1st_M_phi_z"), d1) < 1e-12
          and rel(kal(s, "mon", "A", "2nd_M_phi_z"), d2) < 1e-12)
    check(f"kal == direct free-site sums ({lbl})", ok,
          f"Gamma {G:.10f} <z> {d1:.10f}")

print("2. positive control: the all-site (Namics) convention")
s = b
phib = s.phibulk_seg["A"]
tex = kal(s, "mon", "A", "theta_exc")
old1 = s.lat.moment(s.segments["A"].phi, phib, 1) / tex
old2 = s.lat.moment(s.segments["A"].phi, phib, 2) / tex
check("1;1 all-site <z> == Namics 4.357776", abs(old1 - 4.3577757933) < 1e-6,
      f"{old1:.10f}")
check("1;1 all-site width == Namics 0.843210",
      abs(np.sqrt(old2 - old1 ** 2) - 0.8432100727) < 1e-6,
      f"{np.sqrt(old2 - old1 ** 2):.10f}")
s = c
tex = kal(s, "mon", "A", "theta_exc")
o1 = s.lat.moment(s.segments["A"].phi, phib, 1) / tex
o2 = s.lat.moment(s.segments["A"].phi, phib, 2) / tex
check("1;2 all-site variance < 0 (Namics clamps it to 0.0)",
      o2 - o1 ** 2 < 0, f"{o2 - o1 ** 2:.6f}")

print("3. sign-changing excess -> fluctuations nan (#74)")
s4, s5 = S[3], S[4]
check("start 4: theta_exc < 0, fluctuations nan, RMS finite",
      kal(s4, "mon", "A", "theta_exc") < 0
      and np.isnan(kal(s4, "mon", "A", "fluctuations"))
      and np.isfinite(kal(s4, "mon", "A", "RMS")),
      f"theta_exc {kal(s4, 'mon', 'A', 'theta_exc'):.6f} "
      f"RMS {kal(s4, 'mon', 'A', 'RMS'):.6f}")
check("start 5 (positive excess): fluctuations finite > 0",
      kal(s5, "mon", "A", "fluctuations") > 0,
      f"{kal(s5, 'mon', 'A', 'fluctuations'):.6f}")

print("4. spherical_fjc_excess (refined sphere, frozen core 1;3)")
(sp,) = solve_all(os.path.join(ROOT, "tests", "spherical_fjc_excess.in"))
G, d1, d2 = direct_moments(sp)
check("kal <z>, <z^2> == free-site integrals",
      rel(kal(sp, "mon", "A", "1st_M_phi_z"), d1) < 1e-12
      and rel(kal(sp, "mon", "A", "2nd_M_phi_z"), d2) < 1e-12,
      f"<z> {d1:.10f} width {np.sqrt(d2 - d1 * d1):.10f}")
lat = sp.lat
tex_all = float(np.sum((sp.segments["A"].phi[lat.iv]
                        - sp.phibulk_seg["A"]) * lat.L[lat.iv]))
check("theta_exc keeps the box convention",
      rel(kal(sp, "mon", "A", "theta_exc"), tex_all) < 1e-12,
      f"{tex_all:.10f}")

print("5. N-D: y-uniform 2-D wall frozen on x-layer 1 == 1-D column")
common = {
    ("mon", "S"): {"freedom": ["frozen"]},
    ("mon", "A"): {"freedom": ["free"], "chi_S": ["-1"], "chi_W": ["0.4"]},
    ("mon", "W"): {"freedom": ["free"]},
    ("mol", "pol"): {"composition": ["(A)50"], "freedom": ["free"],
                     "phibulk": ["0.01"]},
    ("mol", "W"): {"composition": ["(W)1"], "freedom": ["solvent"]},
    ("newton", "NN"): {"tolerance": ["1e-11"]},
}
one = dict(common)
one[("lat", "NN")] = {"n_layers": ["41"], "geometry": ["planar"],
                      "lattice_type": ["simple_cubic"]}
one[("mon", "S")] = {"freedom": ["frozen"], "frozen_range": ["1;1"]}
two = dict(common)
two[("lat", "NN")] = {"gradients": ["2"], "geometry": ["flat"],
                      "n_layers_x": ["41"], "n_layers_y": ["3"],
                      "lowerbound_x": ["mirror"], "upperbound_x": ["mirror"],
                      "lowerbound_y": ["mirror"], "upperbound_y": ["mirror"]}
two[("mon", "S")] = {"freedom": ["frozen"], "frozen_range": ["1,1;1,3"]}
s1, s2 = System(one), System(two)
s1.solve(tolerance=1e-11)
s2.solve(tolerance=1e-11)
for prop in ("1st_M_phi_z", "2nd_M_phi_z", "RMS", "fluctuations"):
    v1, v2 = kal(s1, "mon", "A", prop), kal(s2, "mon", "A", prop)
    check(f"2-D == 1-D {prop}", rel(v2, v1) < 1e-7, f"{v1:.10f} {v2:.10f}")
for name, prop in (("W", "phiMin"), ("W", "phiMax"), ("pol", "phiMin")):
    v1, v2 = kal(s1, "mol", name, prop), kal(s2, "mol", name, prop)
    check(f"2-D == 1-D mol {name} {prop}", rel(v2, v1) < 1e-7,
          f"{v1:.10f} {v2:.10f}")
G, d1, d2 = direct_moments(s2)
check("2-D kal == direct free-site sums",
      rel(kal(s2, "mon", "A", "1st_M_phi_z"), d1) < 1e-12
      and rel(kal(s2, "mon", "A", "2nd_M_phi_z"), d2) < 1e-12,
      f"<z> {d1:.10f}")

print("6. mu of an absent molecule (#66)")
M = solve_all(os.path.join(ROOT, "tests", "mu_absent.in"))
for k, lbl in ((1, "free phibulk 0"), (3, "restricted theta 0")):
    s = M[k]
    mu = kal(s, "mol", "pol", "mu")
    X = kal(s, "sys", "NN", "X")
    F = kal(s, "sys", "NN", "free_energy")
    check(f"{lbl}: mu == -inf, X == F (finite)",
          mu == -np.inf and np.isfinite(X) and X == F, f"mu {mu} X {X:.3e}")
    try:
        runner._target_error(s, ("mol", "pol", "mu", -30.0))
        check(f"{lbl}: mu search target raises", False, "no raise")
    except ValueError as e:
        check(f"{lbl}: mu search target raises", "absent" in str(e),
              str(e)[:60])
s = M[0]
ok_mu = np.isclose(kal(s, "mol", "pol", "mu"), -32.0258509300, atol=1e-8)
check("phibulk 1e-10: mu == the closed form -32.0258509300", ok_mu,
      f"{kal(s, 'mol', 'pol', 'mu'):.10f}")
m = s.molecules["pol"]
saved = m.lnGN
m.lnGN = np.inf
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = s.get_value("mol", "pol", "mu")
m.lnGN = saved
check("theta > 0 with a non-finite ln GN -> (None, None) + note",
      r == (None, None) and "ln GN is not finite" in buf.getvalue(),
      repr(r))

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL))
    sys.exit(1)
print("all frozen-interior moment / absent-mu checks pass")
