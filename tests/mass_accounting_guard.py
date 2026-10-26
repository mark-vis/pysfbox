"""The restricted-theta accounting guard (System._audit_solution, amount part,
26 Oct 2026) that the .in suite cannot express: it compares converged
output, so it cannot say "this state must be REFUSED".

    python tests/mass_accounting_guard.py        (~3 s)

1. POSITIVE CONTROL: with the pre-fix composition law (the GLOBAL cap
   709 - 2 ln BIG - U_CLIP - ln N - 10 clipping every row) monkeypatched
   back into model._composition_law, the small crowded-slab input below
   converges with mol:pol:theta 13.75 % low -- and the guard must REFUSE
   it with a RuntimeError naming the molecule and the lost density.
2. NEGATIVE CONTROL: the same input on the shipped law converges, theta =
   0.01 to <= 1e-12, and the guard stays silent.
3. THRESHOLD: on that converged state, a 1e-6 relative mismatch between
   the profile and m.theta raises, a 1e-10 one does not (rtol 1e-8,
   system.THETA_ACCOUNTING_RTOL); non-restricted molecules are never
   checked.
Run after touching _audit_solution/_audit_amounts, its call site (the
cascade exit) or the composition law.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("PYSFBOX_NO_PROGRESS", "1")

import pysfbox.model as model                     # noqa: E402
from pysfbox import system as system_mod          # noqa: E402
from pysfbox.inputreader import read_input        # noqa: E402
from pysfbox.system import System                 # noqa: E402

# a sparse grafted 80-mer inside a 50-layer crowded slab (composition_cap
# mechanism with one linear chain; lnGN -381, max e 372.5 > old cap 356.5)
INPUT = """
lat : flat : geometry : flat
lat : flat : gradients : 1
lat : flat : n_layers : 70
lat : flat : lattice_type : simple_cubic
lat : flat : lowerbound : mirror
lat : flat : upperbound : mirror
mon : B : freedom : pinned
mon : B : pinned_range : 1;50
mon : W : freedom : free
mon : G : freedom : pinned
mon : G : pinned_range : 1;1
mon : A : freedom : free
mon : A : chi_B : 3
mol : crowd : composition : (B)1
mol : crowd : freedom : restricted
mol : crowd : theta : 47.5
mol : water : composition : (W)1
mol : water : freedom : solvent
mol : pol : composition : (G)1(A)79
mol : pol : freedom : restricted
mol : pol : theta : 0.01
newton : isaac : tolerance : 1e-9
newton : isaac : iterationlimit : 10000
"""

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"  {'ok   ' if ok else 'FAIL '} {label}"
          + (f" ({detail})" if detail else ""))
    if not ok:
        failures += 1


def old_composition_law(runs, e, Gf, Gb, N, u_clip=model.U_CLIP):
    """The composition law before 26 Oct 2026: one global clip for every
    row (bit-identical to the shipped classic path, minus the exact
    beyond-cap branch)."""
    cap = 709.0 - 2.0 * np.log(model.BIG) - model.U_CLIP - np.log(N) - 10.0
    W = np.exp(np.clip(e, -745.0, cap))
    per_seg = {}
    for mon, run in runs:
        per_seg[mon] = per_seg.get(mon, 0.0) + W[run] @ (Gf[run] * Gb[run])
    return per_seg


def build():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "guard.in")
        with open(path, "w") as f:
            f.write(INPUT)
        return System(read_input(path)[0])


def solve(s):
    return s.solve(tolerance=1e-9, iterationlimit=10000)


print("1. positive control: the pre-fix law must be refused")
shipped = model._composition_law
model._composition_law = old_composition_law
try:
    s = build()
    try:
        solve(s)
        check("pre-fix law refused", False, "solve returned "
              f"theta {s.molecules['pol'].get_theta():.6e}")
    except RuntimeError as exc:
        msg = str(exc)
        check("pre-fix law refused", "density was lost" in msg
              and "mol pol" in msg, msg[:120])
        check("message quotes the deficit", "-1.37e-01" in msg, msg[:120])
        check("raised as MassAccountingError (runner handlers re-raise it, "
              "no retry)",
              isinstance(exc, system_mod.MassAccountingError),
              type(exc).__name__)
finally:
    model._composition_law = shipped

print("2. negative control: the shipped law converges, guard silent")
s = build()
x, it, err = solve(s)
th = s.molecules["pol"].get_theta()
check("theta exact", abs(th - 0.01) <= 1e-12,
      f"theta {th!r}, {it} it")

print("3. threshold")
m = s.molecules["pol"]
theta0 = m.theta
for rel, should_raise in ((1e-6, True), (1e-10, False)):
    m.theta = theta0 * (1.0 + rel)
    try:
        s._audit_solution()
        raised = False
    except RuntimeError:
        raised = True
    check(f"mismatch {rel:g} {'raises' if should_raise else 'passes'}",
          raised == should_raise,
          f"rtol {system_mod.THETA_ACCOUNTING_RTOL:g}")
m.theta = theta0
w = s.molecules["water"]
w_theta = getattr(w, "theta", None)
w.theta = 1e9                     # nonsense on a solvent: must be ignored
try:
    s._audit_solution()
    check("solvent never checked", True)
except RuntimeError as exc:
    check("solvent never checked", False, str(exc)[:80])
w.theta = w_theta
s._audit_solution()
check("restored state passes", True)

print("all checks passed" if not failures else f"{failures} check(s) FAILED")
sys.exit(1 if failures else 0)
