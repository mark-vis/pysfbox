"""The post-solve input audit (System._audit_solution, 26 Oct 2026): every
check must FIRE on a deliberately violated state and stay SILENT on real
converged ones. The .in suite cannot express "this state must be refused"
(it compares converged output), hence this harness. The amount part's
original positive control -- the pre-fix composition law, a REAL density
loss -- is tests/mass_accounting_guard.py.

    python tests/input_audit.py        (~3 s)

1. NEGATIVE CONTROLS: representative suite inputs (two grafted linear
   brushes, a branched restricted star, a 2-D grafted patch, a weak-charge
   sphere with reactions + neutralizer, a fixed-psi0 electrode, a charged
   cylinder, a refined charged sphere) solve and the audit is silent --
   also re-run by hand with the stored residual. (Every suite input runs
   the audit at its solve exit, so a green tests/run_tests.py is the wider
   negative control.)
2. POSITIVE CONTROLS, one per check, on a converged state:
   restricted amount (m.theta off by 1e-6), segment amount (density moved
   between two segment types of a chain, theta intact), bulk closure (one
   segment's bulk split +1e-6), bulk electroneutrality (bulk charge moved
   between Na and Cl at fixed total volume), bulk reaction equilibrium
   (alphabulk moved between two states at fixed sum; pK changed after the
   bulk solve), incompressibility (one molecule's printed phi +1e-6 at one
   site, its segment split untouched), electroneutrality (a
   NON-conservative Poisson stencil: the outer face radii of a charged
   cylinder scaled by 1.001 -- the solve converges, the box is not neutral
   -- and an electrode charge booked off by 1e-6). Each must raise the
   right InputAuditError subclass naming the check and the culprit.
3. THRESHOLDS: amounts at rtol 1e-8 (1e-6 raises, 1e-10 passes); the
   incompressibility bound passes at 0.5 max|g|.
4. RUNNER: an InputAuditError is re-raised, not retried.
5. COST: one audit vs one residual evaluation on the largest input.
Run after touching _audit_solution or any _audit_* helper, its call site
(the cascade exit), the composition law, _update_bulk, or the Poisson
stencil.
"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = HERE
sys.path.insert(0, ROOT)
os.environ.setdefault("PYSFBOX_NO_PROGRESS", "1")

import pysfbox.runner as runner_mod               # noqa: E402
from pysfbox.inputreader import read_input        # noqa: E402
from pysfbox.runner import _newton_options        # noqa: E402
from pysfbox.system import (InputAuditError,      # noqa: E402
                            MassAccountingError, System)

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"  {'ok   ' if ok else 'FAIL '} {label}"
          + (f" ({detail})" if detail else ""))
    if not ok:
        failures += 1


def build(name, start=0):
    settings = read_input(os.path.join(TESTS, name))[start]
    return System(settings), _newton_options(settings)


def solved(name, start=0):
    s, opts = build(name, start)
    x, it, err = s.solve(**opts)
    return s, x


def g_at(s, x):
    """The residual at x (re-syncs every density, like the cascade exit)."""
    return s.residual(x)


def expect(label, fn, cls, *needles):
    try:
        fn()
    except InputAuditError as exc:
        msg = str(exc)
        miss = [n for n in needles if n not in msg]
        check(label, isinstance(exc, cls) and not miss,
              f"{type(exc).__name__}: {msg[:110]}"
              + (f" -- missing {miss}" if miss else ""))
        return
    except Exception as exc:                      # wrong exception type
        check(label, False, f"{type(exc).__name__}: {str(exc)[:110]}")
        return
    check(label, False, "no error raised")


def silent(label, fn):
    try:
        fn()
        check(label, True)
    except Exception as exc:
        check(label, False, f"{type(exc).__name__}: {str(exc)[:110]}")


NEG = ["two_brushes_quick.in", "branched_star.in", "brush2d_flat.in",
       "charged_weak_sphere.in", "edl_fixed_psi.in", "charged_cylinder.in",
       "charged_sphere_fjc.in"]

print("1. negative controls (solve + audit silent)")
STATES = {}
for name in NEG:
    try:
        t0 = time.time()
        s, x = solved(name)
        STATES[name] = (s, x)
        g = g_at(s, x)
        s._audit_solution(g)
        check(f"{name}", True, f"{time.time() - t0:.1f} s")
    except Exception as exc:
        check(f"{name}", False, f"{type(exc).__name__}: {str(exc)[:110]}")

print("2. positive controls (one per check)")
s, x = STATES["two_brushes_quick.in"]
g = g_at(s, x)
m = s.molecules["pol"]

# restricted amount
th0 = m.theta
m.theta = th0 * (1 + 1e-6)
expect("restricted amount: theta off by 1e-6", lambda: s._audit_solution(g),
       MassAccountingError, "[restricted amount]", "mol pol")
m.theta = th0
for rel, raise_ in ((1e-6, True), (1e-10, False)):
    m.theta = th0 * (1 + rel)
    try:
        s._audit_amounts()
        r = False
    except MassAccountingError:
        r = True
    check(f"amount threshold {rel:g} {'raises' if raise_ else 'passes'}",
          r == raise_)
m.theta = th0

# segment amount: move density from A to the pinned graft segment X
# (theta intact)
pA, pX = m.phi_per_seg["A"].copy(), m.phi_per_seg["X"].copy()
d = 1e-5 * float(s.lat.weighted_sum(pA))
k = int(np.argmax(pX))
m.phi_per_seg["A"] = pA * (1 - 1e-5)
m.phi_per_seg["X"] = pX.copy()
m.phi_per_seg["X"][k] += d / float(np.asarray(s.lat.L).ravel()[k])
expect("segment amount: density moved A -> X, theta intact",
       lambda: s._audit_amounts(), MassAccountingError, "[segment amount]",
       "mol pol, mon ")
m.phi_per_seg["A"], m.phi_per_seg["X"] = pA, pX
silent("brush state restored passes", lambda: s._audit_solution(g))

# bulk closure
seg = "W"
b0 = s.phibulk_seg[seg]
s.phibulk_seg[seg] = b0 + 1e-6
expect("bulk closure: one split entry +1e-6", lambda: s._audit_bulk(),
       InputAuditError, "[bulk closure]")
s.phibulk_seg[seg] = b0

# incompressibility: printed molecule phi off at one free site (the
# solvent: a restricted molecule's theta check would fire first)
m = s.molecules["water"]
phi0 = m.phi.copy()
free = np.where((s.ksam > 0) & s.lat.interior)[0]
m.phi = phi0.copy()
m.phi[free[5]] += 1e-6
expect("incompressibility: mol phi +1e-6 at one free site",
       lambda: s._audit_solution(g), InputAuditError, "[incompressibility]",
       f"site {free[5]}")
m.phi = phi0
# at the bound: 0.5 max|g| passes
gmax = float(np.max(np.abs(g[:len(s.it_species) * s.lat.M])))
m.phi = phi0.copy()
m.phi[free[5]] -= 0.5 * gmax
silent("incompressibility: deviation 0.5 max|g| passes",
       lambda: s._audit_solution(g))
m.phi = phi0

# segment amount on a branched molecule (the tree's per-type shares)
s, x = STATES["branched_star.in"]
g = g_at(s, x)
m = s.molecules["star"]
pA, pG = m.phi_per_seg["A"].copy(), m.phi_per_seg["G"].copy()
m.phi_per_seg["A"], m.phi_per_seg["G"] = pA * (1 - 1e-5), pG + 1e-5 * pA
expect("segment amount (tree): density moved A -> G, theta intact",
       lambda: s._audit_amounts(), MassAccountingError, "[segment amount]",
       "mol star, mon ")
m.phi_per_seg["A"], m.phi_per_seg["G"] = pA, pG
silent("star state restored passes", lambda: s._audit_solution(g))

# bulk electroneutrality + reactions (weak sphere: neutralizer Na, Cl free)
s, x = STATES["charged_weak_sphere.in"]
g = g_at(s, x)
na0, cl0 = s.phibulk_seg["Na"], s.phibulk_seg["Cl"]
s.phibulk_seg["Na"] = na0 + 1e-6
s.phibulk_seg["Cl"] = cl0 - 1e-6          # closure intact, charge not
expect("bulk electroneutrality: Na/Cl split moved by 1e-6",
       lambda: s._audit_bulk(), InputAuditError, "[bulk electroneutrality]",
       "neutralizer Na")
s.phibulk_seg["Na"], s.phibulk_seg["Cl"] = na0, cl0
st = {t.name: t for sg in s.segments.values() for t in sg.states}
a0, b0 = st["AH"].alphabulk, st["AM"].alphabulk
st["AH"].alphabulk, st["AM"].alphabulk = a0 * (1 + 1e-6), b0 - a0 * 1e-6
expect("bulk reaction equilibrium: alphabulk moved AH -> AM at fixed sum",
       lambda: s._audit_bulk(), InputAuditError,
       "[bulk reaction equilibrium]", "reaction weak")
st["AH"].alphabulk, st["AM"].alphabulk = a0, b0
r = s.reactions[0]
pk0 = r.pK
r.pK = pk0 + 0.01
expect("bulk reaction equilibrium: pK changed after the bulk solve",
       lambda: s._audit_bulk(), InputAuditError,
       "[bulk reaction equilibrium]")
r.pK = pk0
silent("weak sphere restored passes", lambda: s._audit_solution(g))

# electroneutrality, electrode booking (fixed psi0)
s, x = STATES["edl_fixed_psi.in"]
g = g_at(s, x)
qe0 = s.q_electrode.copy()
el = int(np.where(s.psiMask)[0][0])
s.q_electrode[el] *= 1 + 1e-6
expect("electroneutrality: electrode charge off by 1e-6",
       lambda: s._audit_solution(g), InputAuditError, "[electroneutrality]",
       "electrode charge")
s.q_electrode = qe0
silent("electrode restored passes", lambda: s._audit_solution(g))

# electroneutrality, a NON-conservative stencil (real converged state)
s, opts = build("charged_cylinder.in")
s.r_plus = s.r_plus * 1.001        # faces no longer shared by neighbours
expect("electroneutrality: non-conservative curved stencil (solve)",
       lambda: s.solve(**opts), InputAuditError, "[electroneutrality]")

print("3. runner")
n_solve = []
orig_solve = System.solve


def failing_solve(self, *a, **k):
    n_solve.append(1)
    raise InputAuditError("input audit [test]: refused")


System.solve = failing_solve
try:
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(TESTS, "charged_nlayers_scan.in")) as f:
            src = f.read()
        p = os.path.join(d, "scan.in")
        with open(p, "w") as f:
            f.write(src)
        try:
            runner_mod.run_file(p, verbose=False)
            check("runner re-raises InputAuditError", False, "no raise")
        except InputAuditError:
            check("runner re-raises InputAuditError (one solve, no retry)",
                  len(n_solve) == 1, f"{len(n_solve)} solve call(s)")
finally:
    System.solve = orig_solve

print("4. cost")
big = max(STATES.items(), key=lambda kv: kv[1][0].n_var())
s, x = big[1]
g = g_at(s, x)
reps = 20
t0 = time.perf_counter()
for _ in range(reps):
    s.residual(x)
t_res = (time.perf_counter() - t0) / reps
t0 = time.perf_counter()
for _ in range(reps):
    s._audit_solution(g)
t_aud = (time.perf_counter() - t0) / reps
print(f"  {big[0]} ({s.n_var()} vars): residual {t_res * 1e3:.2f} ms, "
      f"audit {t_aud * 1e3:.3f} ms ({t_aud / t_res:.1%} of ONE residual)")
check("audit cheaper than one residual evaluation", t_aud < t_res)

print("all checks passed" if not failures else f"{failures} check(s) FAILED")
sys.exit(1 if failures else 0)
