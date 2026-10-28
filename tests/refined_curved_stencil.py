#!/usr/bin/env python
"""Anchors for the REFINED CURVED stencil `Lattice1D._setup_fjc` (review
27 Oct 2026, findings #3, #2, #9, #26, #41 and #8).
Every keep/fold/absorb decision of a channel is taken on its INTEGER target
index kt; the radii enter only the areas. The `.in` regressions
(tests/spherical_fjc11_axis.in, tests/spherical_fjc_outer_surface.in)
compare converged output; these are the lattice-level identities behind
them, which no committed reference can express.

 S1 DETAILED BALANCE of the FOLDED one-step operator T (built from
    site_average(set_bounds(e_m)), so the ghost folds are included) at
    offset 0, mirror bounds, sph/cyl x every FJC_choices 5..41, n_layers 8
    and 9: rows of T sum to 1, |L_n T_nm - L_m T_mn| / max|L T| <
    1e-13, and every channel that ends on or behind the axis (kt <= 0) is
    exactly 0 (#3: the old float axis test kept the kt = 0 channel at
    FJC_choices 11, 19, 23, 25, 27, 35, 37, 41 -- asymmetry 4e-4..2e-2).
    The same DB at offsets 0.5 and 2 (inner mirror fold) and with surfaces.
 S2 CLOSED FORM: every LAM entry equals its exact-rational closed form
    (Fractions, no accumulation): interior c area((r + r')/2)/VL; axis 0;
    inner mirror c area(min(f, 2m - f))/VL with f = (r + r')/2 and m =
    off + 1/(2 fjc) the mirror plane (#8); inner/outer surface
    c area((r + r')/2)/VL; outer mirror c area(R_e - (r' - r)/2)/VL; the
    centre closes the row -- to 1e-14, sph/cyl x FJC 5..21 x offsets
    0, 1/2, 4/5, 2 x all bounds x n_layers 6, 7.
 S3 OUTER SURFACE == FROZEN SHELL (#2): ln Q of an ideal (A)40 with
    `upperbound : surface` at n_layers equals the mirror box at n_layers + 2
    whose extra layers carry G = 0, to 1e-12 (sph/cyl x FJC 5/11/21 x
    n 6/12 x offset 0/5); the absorbed fraction at the outermost site of a
    sphere (FJC 11, R 6) lies ABOVE planar (concave wall) and within 1e-3
    of the exact concave value (it was 0.4086 < planar 0.45 before).
 S4 INNER SURFACE CONTINUITY (#9): ln(Q/V) of an ideal (A)20 with -1 kT on
    the first bond layer and the first-site absorbed fraction are
    continuous (1e-6) across every off* = 1 - k/fjc >= 1/2 - 1/fjc (they
    jumped by up to a factor 2 in Q); the cylinder sequence at offset 0.5
    is monotone in FJC_choices 5..21 (it oscillated); T >= 0 and DB there.
 S5 GUARDS (API level; the .in level is tests/input_guards.py):
    a solid core thinner than 1/2 - 1/fjc raises NotImplementedError naming
    offset_first_layer, the threshold itself builds (zero-area face); a
    negative curved offset raises ValueError (#26), planar ignores it; an
    out-of-range lambda raises ValueError (#41), 1/3 on a sphere and 1/2
    planar build.
 S6 SCF ORIENTATION at FJC_choices 19, offset 0: a pinned restricted
    triblock written both ways converges (the mass audit passes) with
    theta exact and identical mu/F/Omega (the FJC 11 twin is the suite
    case tests/spherical_fjc11_axis.in).
 S7 NON-NEGATIVE OPERATOR (#8, decided 28 Oct 2026): over sph/cyl x fjc
    2,3,4,5,10,20 x offsets 0..10 x every bound pair the lattice builds (or
    refuses a tiny solid core, S5) with rows 1 (<= 1 next to a surface), DB
    1e-13 and T >= 0 -- the offset-0 cylinder's axis rows aside, which are
    exempt: lat.axis_stay = T00 (-3/40 at FJC 11), one note per process;
    at R = 1e3 / 1e5 the folded rows tend to the planar refined mirror as
    1/R.
    Pins: sph FJC 11 offset 0.5 T00 = 25/172 (the 7 Oct rule: -0.1895);
    sph FJC 35 offset 0.01 builds with a NEGATIVE LAM stay weight but T >= 0
    (the guard tests T, not LAM); sph FJC 41 offset 0.001, the remaining
    corner, is refused. Positive controls: the 7 Oct rule swapped in gives
    phi < 0 for an ideal chain under a 2-2.25 kT one-site field (phi > 0
    now) and is refused by _check_stencil; a 1 % detailed-balance break is
    refused; a one-site field on the exempt cylinder axis drives phi < 0
    (what System._audit_positivity refuses; tests/input_audit.py).

Run after touching Lattice1D._setup_fjc / _check_stencil / _reflect /
set_bounds / site_average or the curved branch of Lattice1D.__init__.

    python tests/refined_curved_stencil.py        (~15 s)
"""
import contextlib
import io
import os
import sys
import tempfile
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from pysfbox.lattice import _AXIS_NOTES, Lattice1D           # noqa: E402
from pysfbox.runner import run_file                          # noqa: E402

FAIL = []
TMP = tempfile.mkdtemp(prefix="refined_curved_stencil_")


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(label)


def transfer(lat):
    """The folded one-step operator T[n, m] = <e_m>(n) on the interior."""
    MX, fjc = lat.MX, lat.fjc
    T = np.zeros((MX, MX))
    for m in range(MX):
        e = np.zeros(lat.M)
        e[fjc + m] = 1.0
        T[:, m] = lat.site_average(lat.set_bounds(e))[fjc:fjc + MX]
    return T


def db_rel(lat, T):
    F = lat.L[lat.iv][:, None] * T
    return float(np.abs(F - F.T).max() / np.abs(F).max())


def lat(geo, FJC, off=0.0, lb="mirror", ub="mirror", n=8):
    fjc = (FJC - 1) // 2
    return Lattice1D(n, geometry=geo, fjc=fjc, offset_first_layer=off,
                     lowerbound=lb, upperbound=ub)


GEOS = ("spherical", "cylindrical")

# ---- S1 ------------------------------------------------------------------
print("[S1] detailed balance of the folded operator (offset 0, #3)")
worst_db = worst_row = worst_axis = 0.0
was_bad = []
for geo in GEOS:
    for FJC in range(5, 43, 2):
        fjc = (FJC - 1) // 2
        for n in (8, 9):
            with contextlib.redirect_stdout(io.StringIO()):  # axis notes
                lt = lat(geo, FJC, n=n)
            T = transfer(lt)
            d = db_rel(lt, T)
            worst_db = max(worst_db, d)
            worst_row = max(worst_row, float(np.abs(T.sum(1) - 1).max()))
            for i in range(fjc, lt.M - fjc):
                k = i - fjc + 1
                for c in range(fjc):
                    if k + c - fjc <= 0:
                        worst_axis = max(worst_axis, abs(lt.LAM[c, i]))
            if FJC in (11, 19, 23, 25, 27, 35, 37, 41) and n == 8:
                was_bad.append(f"{geo[:3]} {FJC}: {d:.1e}")
check("rows of T sum to 1 (mirror/mirror)", worst_row < 1e-13, f"{worst_row:.1e}")
check("L-detailed balance, FJC 5..41", worst_db < 1e-13, f"worst {worst_db:.1e}")
check("channels ending on/behind the axis are exactly 0", worst_axis == 0.0,
      f"{worst_axis:.1e}")
print("     formerly broken FJC values now: " + ", ".join(was_bad[:4]) + " ...")
worst = 0.0
for geo in GEOS:
    for FJC in (5, 7, 11, 19, 21):
        for off in (0.5, 2.0):
            for lb in ("mirror", "surface"):
                for ub in ("mirror", "surface"):
                    lt = lat(geo, FJC, off, lb, ub, n=7)
                    worst = max(worst, db_rel(lt, transfer(lt)))
check("L-detailed balance at offsets 0.5 / 2, every bound pair",
      worst < 1e-13, f"{worst:.1e}")

# ---- S2 ------------------------------------------------------------------
print("[S2] every LAM entry equals its exact-rational closed form")


def closed_form(geo, fjc, off, lb, ub, MX, fold="min"):
    """LAM from closed-form radii in exact arithmetic (no accumulation).
    fold: the inner-mirror rule -- "min" (since 28 Oct 2026, review #8)
    or "mid", the 7 Oct 2026 face midway between site and image (kept for
    the positive controls of S7)."""
    FJC = 2 * fjc + 1
    sph = geo == "spherical"
    area = (lambda x: 4 * x * x) if sph else (lambda x: x)
    cext = Fraction(1, 2 * (FJC - 1)) if sph else Fraction(1, FJC - 1)
    cmid = Fraction(1, FJC - 1) if sph else Fraction(2, FJC - 1)
    Re = off + Fraction(MX, fjc)
    LAM = np.zeros((FJC, MX + 2 * fjc))
    for k in range(1, MX + 1):
        r = off + Fraction(k, fjc)
        rl, rh = r - Fraction(1, 2), r + Fraction(1, 2)
        VL = Fraction(4, 3) * (rh ** 3 - rl ** 3) if sph else 2 * r
        row = [Fraction(0)] * FJC
        for c in range(FJC):
            if c == fjc:
                continue
            coef = cext if c in (0, FJC - 1) else cmid
            kt = k + c - fjc
            rp = off + Fraction(kt, fjc)
            if 1 <= kt <= MX:
                w = area((r + rp) / 2)
            elif kt <= 0:
                if off == 0:
                    w = Fraction(0)
                elif lb == "mirror":
                    m = off + Fraction(1, 2 * fjc)      # the mirror plane
                    face = (r + rp) / 2
                    w = (area(min(face, 2 * m - face)) if fold == "min"
                         else area(off + (r - rp) / 2))
                else:
                    w = area((r + rp) / 2)
            else:
                w = area((r + rp) / 2) if ub == "surface" \
                    else area(Re - (rp - r) / 2)
            row[c] = coef * w / VL
        row[fjc] = 1 - sum(row)
        LAM[:, fjc + k - 1] = [float(v) for v in row]
    return LAM


worst = 0.0
n_cases = 0
for geo in GEOS:
    for FJC in (5, 7, 9, 11, 13, 19, 21):
        fjc = (FJC - 1) // 2
        for off in (Fraction(0), Fraction(1, 2), Fraction(4, 5), Fraction(2)):
            for lb in ("mirror", "surface"):
                if lb == "surface" and 0 < off < Fraction(1, 2) - Fraction(1, fjc):
                    continue                      # refused (S5)
                for ub in ("mirror", "surface"):
                    for n in (6, 7):
                        lt = lat(geo, FJC, float(off), lb, ub, n=n)
                        ref = closed_form(geo, fjc, off, lb, ub, n * fjc)
                        worst = max(worst, float(np.abs(lt.LAM - ref).max()))
                        n_cases += 1
check(f"LAM == closed form ({n_cases} lattices)", worst < 1e-14, f"{worst:.1e}")

# ---- S3 ------------------------------------------------------------------
print("[S3] outer surface == frozen-shell twin (#2)")


def lnQ(lt, N, G1):
    G = G1.copy()
    for _ in range(N - 1):
        G = G1 * lt.site_average(lt.set_bounds(G))
    return float(np.log((lt.L * G)[lt.iv].sum()))


worst = worst_lam = 0.0
for geo in GEOS:
    for FJC in (5, 11, 21):
        for n, off in ((6, 0.0), (12, 0.0), (6, 5.0), (12, 5.0)):
            A = lat(geo, FJC, off, ub="surface", n=n)
            B = lat(geo, FJC, off, ub="mirror", n=n + 2)
            GA = np.zeros(A.M); GA[A.iv] = 1.0
            GB = np.zeros(B.M); GB[B.fjc:B.fjc + A.MX] = 1.0   # shell: G = 0
            worst = max(worst, abs(lnQ(A, 40, GA) - lnQ(B, 40, GB)))
            worst_lam = max(worst_lam, float(np.abs(
                A.LAM[:, A.iv] - B.LAM[:, B.fjc:B.fjc + A.MX]).max()))
check("ln Q(surface, n) == ln Q(mirror, n + 2, G = 0 shell)", worst < 1e-12,
      f"{worst:.1e}")
check("surface rows == the twin's rows", worst_lam < 1e-15, f"{worst_lam:.1e}")
A = lat("spherical", 11, ub="surface", n=6)
e = np.zeros(A.M); e[A.iv] = 1.0
absorbed = 1.0 - A.site_average(A.set_bounds(e))[A.fjc + A.MX - 1]
P = Lattice1D(6, geometry="planar", fjc=5, upperbound="surface")
ep = np.zeros(P.M); ep[P.iv] = 1.0
planar = 1.0 - P.site_average(P.set_bounds(ep))[P.fjc + P.MX - 1]
R, delta = 6.0, 0.5 / 5                 # outermost cell centre, 1/(2 fjc) deep
c = (delta * (2 * R - delta) - 1) / (2 * (R - delta))
exact = (1 - c) / 2
check("concave outer wall absorbs more than planar",
      absorbed > planar + 0.03, f"{absorbed:.4f} vs planar {planar:.4f}")
check("absorbed fraction == exact concave value to 1e-3",
      abs(absorbed - exact) < 1e-3, f"{absorbed:.4f} vs {exact:.4f}")

# ---- S4 ------------------------------------------------------------------
print("[S4] inner surface at 0 < offset < 1: continuity in the offset (#9)")


def obs(lt, N=20, du=1.0):
    fjc = lt.fjc
    G1 = np.zeros(lt.M); G1[lt.iv] = 1.0
    G1[fjc:2 * fjc] *= np.exp(du)           # first BOND layer attractive
    G = G1.copy()
    for _ in range(N - 1):
        G = G1 * lt.site_average(lt.set_bounds(G))
    e = np.zeros(lt.M); e[lt.iv] = 1.0
    absorbed = 1.0 - lt.site_average(lt.set_bounds(e))[fjc]
    return float(np.log((lt.L * G)[lt.iv].sum() / lt.L_sum)), float(absorbed)


worst_q = worst_a = 0.0
worst_db, min_T = 0.0, 1.0
for geo in GEOS:
    for FJC in (5, 11):
        fjc = (FJC - 1) // 2
        for k in range(1, fjc):
            ostar = 1 - k / fjc
            if ostar < 0.5 - 1 / fjc:
                continue
            lo = lat(geo, FJC, ostar - 1e-7, lb="surface", n=10)
            hi = lat(geo, FJC, ostar + 1e-7, lb="surface", n=10)
            a, b = obs(lo), obs(hi)
            worst_q = max(worst_q, abs(a[0] - b[0]))
            worst_a = max(worst_a, abs(a[1] - b[1]))
            for lt in (lo, hi):
                T = transfer(lt)
                worst_db = max(worst_db, db_rel(lt, T))
                min_T = min(min_T, float(T.min()))
check("ln(Q/V) continuous across off* = 1 - k/fjc", worst_q < 1e-6,
      f"max jump {worst_q:.1e} (was up to 0.75)")
check("first-site absorbed fraction continuous", worst_a < 1e-6,
      f"max jump {worst_a:.1e}")
check("DB and T >= 0 on those lattices", worst_db < 1e-13 and min_T > -1e-12,
      f"DB {worst_db:.1e}, min T {min_T:+.2e}")
q = obs(lat("cylindrical", 11, 0.8001, lb="surface", n=10))
check("pinned value: cyl FJC 11 offset 0.8001",
      abs(q[0] - 6.549789346) < 1e-6 and abs(q[1] - 0.3250125) < 1e-6,
      f"ln(Q/V) {q[0]:.9f}, absorbed {q[1]:.7f}")
seq = [obs(lat("cylindrical", F, 0.5, lb="surface", n=10))[0]
       for F in (5, 7, 9, 11, 15, 21)]
check("cyl offset 0.5: ln(Q/V) monotone in FJC_choices 5..21",
      all(np.diff(seq) > 0), " ".join(f"{v:.4f}" for v in seq))

# ---- S5 ------------------------------------------------------------------
print("[S5] guards")


def raises(label, fn, exc, words):
    try:
        fn()
    except exc as e:
        check(label, all(w in str(e) for w in words), str(e)[:90])
        return
    except Exception as e:                       # noqa: BLE001
        check(label, False, f"{type(e).__name__}: {e}")
        return
    check(label, False, "built without complaint")


for geo in GEOS:
    for FJC, off in ((7, 0.1), (11, 0.2), (21, 0.35)):
        raises(f"tiny core {geo[:3]} FJC {FJC} offset {off}",
               lambda: lat(geo, FJC, off, lb="surface", n=8),
               NotImplementedError, ["offset_first_layer", "1/2 - 1/fjc"])
    for FJC, off in ((9, 0.25), (11, 0.3), (21, 0.4)):    # face on the axis
        lt = lat(geo, FJC, off, lb="surface", n=8)
        T = transfer(lt)
        check(f"threshold {geo[:3]} FJC {FJC} offset {off} builds",
              db_rel(lt, T) < 1e-13 and T.min() > -1e-12,
              f"min T {T.min():+.2e}")
for geo, fjc, off in (("cylindrical", 1, -1.0), ("cylindrical", 1, -0.3),
                      ("spherical", 1, -0.5), ("cylindrical", 2, -1.0),
                      ("spherical", 2, -2.0)):
    raises(f"negative offset {geo[:3]} fjc {fjc} offset {off}",
           lambda: Lattice1D(20, geometry=geo, fjc=fjc, offset_first_layer=off),
           ValueError, ["offset_first_layer", "non-negative"])
check("planar ignores a negative offset",
      Lattice1D(20, geometry="planar", offset_first_layer=-1.0).offset == 0.0)
for geo, lam in (("spherical", 0.45), ("cylindrical", 0.55), ("planar", 0.6),
                 ("planar", -0.1), ("planar", 0.0)):
    raises(f"lambda {lam} on {geo}",
           lambda: Lattice1D(10, geometry=geo, lam=lam), ValueError, ["lambda"])
for geo, lam in (("spherical", 1 / 3), ("planar", 0.5), ("cylindrical", 0.5)):
    lt = Lattice1D(10, geometry=geo, lam=lam)
    check(f"lambda {lam:.4g} on {geo} builds", lt.l0[lt.iv].min() >= -1e-12)

# ---- S6 ------------------------------------------------------------------
print("[S6] SCF orientation identity at FJC_choices 19, offset 0")
BASE = """lat : NN : geometry : spherical
lat : NN : n_layers : 12
lat : NN : lattice_type : hexagonal
lat : NN : FJC_choices : 19
mon : W : freedom : free
mon : A : freedom : free
mon : P : freedom : pinned
mon : P : pinned_range : 1;2
mon : A : chi_W : 0.4
mon : P : chi_W : 1.0
mol : solvent : freedom : solvent
mol : solvent : composition : (W)1
mol : poly : freedom : restricted
mol : poly : composition : {comp}
mol : poly : theta : 30
newton : NN : tolerance : 1e-10
start
"""
res = []
for comp in ("(A)5(P)3(A)12", "(A)12(P)3(A)5"):
    path = os.path.join(TMP, f"s6_{len(res)}.in")
    with open(path, "w") as f:
        f.write(BASE.format(comp=comp))
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            s = run_file(path, verbose=False)
        res.append((float(s.molecules["poly"].get_theta()),
                    s.get_value("mol", "poly", "mu")[1],
                    s.get_value("sys", "NN", "free_energy")[1],
                    s.get_value("sys", "NN", "grand_potential")[1]))
    except Exception as e:                       # noqa: BLE001
        res.append(f"{type(e).__name__}: {str(e)[:120]}")
ok = all(not isinstance(r, str) for r in res)
check("both orientations converge (mass audit passes)", ok,
      "" if ok else str(res))
if ok:
    gap = max(abs(a - b) for a, b in zip(res[0][1:], res[1][1:]))
    check("theta == 30 in both", max(abs(r[0] - 30) for r in res) < 1e-10,
          f"{res[0][0]:.12f} / {res[1][0]:.12f}")
    check("mu, F, Omega orientation gaps < 1e-9", gap < 1e-9, f"{gap:.1e}")

# ---- S7 ------------------------------------------------------------------
print("[S7] non-negative folded operator: the 'min' inner fold and the "
      "_check_stencil guard (#8)")
worst_db = worst_row = 0.0
min_T = 1.0
built = refused = 0
other = []
for geo in GEOS:
    for fjc in (2, 3, 4, 5, 10, 20):
        FJC = 2 * fjc + 1
        for off in (0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.5, 3.0, 10.0):
            for lb in ("mirror", "surface"):
                for ub in ("mirror", "surface"):
                    try:
                        lt = lat(geo, FJC, off, lb, ub, n=7)
                    except NotImplementedError:     # tiny solid core (S5)
                        refused += 1
                        continue
                    except ValueError as e:         # the guard: not expected
                        other.append(f"{geo[:3]} {FJC} {off} {lb}/{ub}: {e}")
                        continue
                    built += 1
                    T = transfer(lt)
                    worst_db = max(worst_db, db_rel(lt, T))
                    rows = T.sum(1)
                    if lb == "mirror" and ub == "mirror":
                        worst_row = max(worst_row, float(np.abs(rows - 1).max()))
                    else:
                        worst_row = max(worst_row, float(rows.max() - 1))
                    skip = fjc if lt.axis_stay is not None else 0
                    min_T = min(min_T, float(T[skip:].min()))
check(f"grid: {built} lattices build, {refused} tiny cores refused, "
      "no stencil refusal", not other, "; ".join(other)[:200])
check("rows sum to 1 (mirror), <= 1 next to a surface", worst_row < 1e-13,
      f"{worst_row:.1e}")
check("L-detailed balance", worst_db < 1e-13, f"{worst_db:.1e}")
check("min T >= 0 (cylinder axis rows at offset 0 aside)", min_T > -1e-12,
      f"{min_T:+.2e}")
worst_lim = 0.0
for FJC in (5, 11):
    fjc = (FJC - 1) // 2
    P = transfer(Lattice1D(6, geometry="planar", fjc=fjc))[:2 * fjc]
    for geo in GEOS:
        d3, d5 = (float(np.abs(transfer(lat(geo, FJC, R, n=6))[:2 * fjc] - P).max())
                  for R in (1e3, 1e5))
        worst_lim = max(worst_lim, d5, abs(d3 / d5 - 100) / 100)
check("planar limit: the folded rows -> the planar refined mirror as 1/R",
      worst_lim < 0.01, f"worst {worst_lim:.1e} (|dT| <= 4e-6 at R 1e5)")
T = transfer(lat("spherical", 11, 0.5, n=6))
check("pin: sph FJC 11 offset 0.5 T00 = 25/172 (was -0.1895)",
      abs(T[0, 0] - 25 / 172) < 1e-13, f"{T[0, 0]:.15f}")
lt = lat("spherical", 35, 0.01, n=4)
T = transfer(lt)
check("pin: sph FJC 35 offset 0.01 builds with LAM stay < 0 but T >= 0 "
      "(the criterion is T, not LAM)",
      lt.LAM[lt.fjc, lt.fjc] < -1e-5 and T.min() > -1e-12,
      f"LAM {lt.LAM[lt.fjc, lt.fjc]:+.2e}, min T {T.min():+.2e}")
raises("remaining corner: sph FJC 41 offset 0.001 refused",
       lambda: lat("spherical", 41, 0.001, n=4), ValueError,
       ["FJC_choices 41", "offset_first_layer 0.001",
        "negative transition weight", "Increase offset_first_layer"])

# the offset-0 cylinder axis rows: exempt, stored, noted once per process
_AXIS_NOTES.clear()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    a = lat("cylindrical", 11, 0.0, n=6)
    b = lat("cylindrical", 11, 0.0, n=9)
notes = [ln for ln in buf.getvalue().splitlines() if "axis row" in ln]
T = transfer(a)
check("cyl FJC 11 offset 0: axis_stay = T00 = -3/40, the only negative "
      "entry", abs(a.axis_stay + 0.075) < 1e-13
      and abs(T[0, 0] - a.axis_stay) < 1e-15 and T[1:].min() > -1e-12
      and T[0, 1:].min() > -1e-12, f"{a.axis_stay!r}")
check("the axis note is printed once per process", len(notes) == 1
      and "T00 = -0.075" in notes[0] and b.axis_stay == a.axis_stay,
      f"{len(notes)} note line(s)")
check("cyl FJC 9 offset 0 and cyl FJC 11 offset 0.1: no exemption",
      lat("cylindrical", 9, 0.0, n=6).axis_stay is None
      and lat("cylindrical", 11, 0.1, n=6).axis_stay is None)


def ideal_phi(lt, du, N=10):
    """(A)N with G1 = exp(du) on the first refined site only, through the
    production operator: min phi over the interior (bulk value N)."""
    fjc = lt.fjc
    G1 = np.zeros(lt.M); G1[lt.iv] = 1.0
    G1[fjc] = np.exp(du)
    Gs = [G1.copy()]
    for _ in range(N - 1):
        Gs.append(G1 * lt.site_average(lt.set_bounds(Gs[-1])))
    phi = sum(Gs[s] * Gs[N - 1 - s] for s in range(N))[lt.iv] / G1[lt.iv]
    return float(phi.min())


# production path: a sharp one-site field keeps phi > 0 under the min fold;
# the 7 Oct 'mid' rule (LAM swapped in behind the guard) gave phi < 0
for geo, FJC, off, du in (("spherical", 9, 0.5, 2.25), ("spherical", 11, 0.5, 2.0),
                          ("cylindrical", 11, 0.3, 2.0)):
    lt = lat(geo, FJC, off, n=8)
    now = ideal_phi(lt, du)
    lt.LAM = closed_form(geo, lt.fjc, Fraction(off).limit_denominator(100),
                         "mirror", "mirror", lt.MX, fold="mid")
    old = ideal_phi(lt, du)
    check(f"{geo[:3]} FJC {FJC} offset {off}, {du} kT on one site: phi > 0 "
          "(the 7 Oct rule went negative)", now > 0 and old < 0,
          f"min phi {now:.4g}, 7 Oct rule {old:.4g}")
    raises(f"guard refuses the 7 Oct rule on {geo[:3]} FJC {FJC} offset {off}",
           lt._check_stencil, ValueError,
           [f"FJC_choices {FJC}", f"offset_first_layer {off:g}",
            "negative transition weight"])
lt = lat("spherical", 7, 2.0, n=8)
i = lt.fjc + 3 * lt.fjc
lt.LAM[lt.fjc + 1, i] *= 1.01                    # one direct weight, row kept
lt.LAM[lt.fjc, i] = 1.0 - (lt.LAM[:, i].sum() - lt.LAM[lt.fjc, i])
raises("guard refuses a broken detailed balance", lt._check_stencil,
       ValueError, ["detailed balance", "please report"])
# the exempt axis row is real: a one-site field on the axis of a refined
# cylinder at offset 0 drives an ideal chain negative -- what
# System._audit_positivity refuses
for FJC, du in ((11, 3.0), (21, 1.0)):
    v = ideal_phi(lat("cylindrical", FJC, 0.0, n=8), du)
    check(f"cyl FJC {FJC} offset 0, {du} kT on the axis site: phi < 0 "
          "(the audit's job)", v < 0, f"min phi {v:.3g}")

print(f"\n{'ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: ' + '; '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
