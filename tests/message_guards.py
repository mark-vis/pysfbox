"""Console-message guards (28 Oct 2026): what the .in suite cannot see,
because it compares the .kal/.pro files and not stdout. Also run by
tests/run_tests.py as the 'message_guards' case (a few seconds).

    python tests/message_guards.py

[NiN] a RECOGNISED kal column that a gate turns into NiN (kJ0; X without
      its definition) prints its own note and the neutral line 'has no
      value for this state', never 'kal property ... unknown'; a misspelt
      column prints 'unknown' once per kal row.
[file] the once-per-file notes are reset at the start of every input file:
      two files run in one process both print the kJ0 note and the
      two-sided Laplace_pressure note; a var scan still prints each once
      per file, not once per step.
[rf]  regula-falsi bracket collapse above super_tolerance: the error is
      either the inner solves' NOISE (loosening super_tolerance is the
      honest remedy) or a JUMP in the target (a branch switch: loosening it
      accepts a state on one side of the discontinuity). The diagnosis
      reads the readouts clustered at the collapse point
      (runner._classify_readouts), never the distance from the start: a
      jump reached from a distant start used to be called noise.
[rf]  an inner failure at the search's START value names the scan step,
      not super_deltamax (the first probe takes no search step).
"""
import contextlib
import hashlib
import io
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("PYSFBOX_NO_PROGRESS", "1")

from pysfbox import runner  # noqa: E402
from pysfbox.runner import _classify_readouts, _regula_falsi  # noqa: E402

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"  {'ok   ' if ok else 'FAIL '} {label}"
          + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        failures += 1


# an athermal A10/B10 mixture against a solid wall (chi 0: one phase,
# converges in tens of iterations on every platform -- the chi 1.0 blend is
# two-phase, its interface position a soft mode, ~4000 iterations here and
# over the 8000 limit on the Linux CI): kJ0 is gated (Namics' compute_kJ0,
# not ported), X is not declared, kJ0x is a typo
SLAB = """
lat : NN : n_layers : 40
lat : NN : geometry : flat
lat : NN : lowerbound : surface
lat : NN : upperbound : mirror
mon : A : freedom : free
mon : B : freedom : free
mon : A : chi_B : 0.0
mol : A : composition : (A)10
mol : A : freedom : restricted
mol : A : theta : {theta}
mol : B : composition : (B)10
mol : B : freedom : solvent
newton : n : tolerance : 1e-9
{scan}
output : kal : append : false
kal : sys : NN : grand_potential
kal : sys : NN : kJ0
kal : sys : NN : X
kal : sys : NN : kJ0x
"""
# a spherical droplet on a frozen core whose far side is NOT at omega = 0
# (restricted (A)20 in a 14-layer box): the two-sided Laplace_pressure note
DROPLET = """
lat : NN : geometry : spherical
lat : NN : n_layers : 14
lat : NN : lowerbound : mirror
lat : NN : upperbound : mirror
mon : C : freedom : frozen
mon : C : frozen_range : 1;2
mon : A : freedom : free
mon : W : freedom : free
mon : A : chi_C : -3
mon : A : chi_W : 1.2
mol : A : composition : (A)20
mol : A : freedom : restricted
mol : A : theta : 1500
mol : W : composition : (W)1
mol : W : freedom : solvent
newton : n : tolerance : 1e-9
output : kal : append : false
kal : sys : NN : Laplace_pressure
"""
KJ0_NOTE = "note: sys : kJ0 is Namics' compute_kJ0 column"
X_NOTE = "note: kal sys : X needs its definition"
LP_NOTE = "note: sys : Laplace_pressure: the far side"


def run(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, "w") as fp:
        fp.write(text)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runner.run_file(path)
    with open(os.path.splitext(path)[0] + ".kal") as fp:
        rows = [ln.rstrip("\n").split("\t") for ln in fp]
    return out.getvalue(), rows


def count(text, s):
    return sum(s in ln for ln in text.splitlines())


with tempfile.TemporaryDirectory() as tmp:
    print("[NiN] gated column vs misspelt column (one kal row)")
    out, rows = run(tmp, "slab.in", SLAB.format(theta=20, scan=""))
    hdr, row = rows[0], rows[1]
    for col in ("kJ0", "X", "kJ0x"):
        check(f"kal sys:NN:{col} is NiN",
              row[hdr.index(f"sys:NN:{col}")] == "NiN")
    check("grand_potential is a number",
          row[hdr.index("sys:NN:grand_potential")] != "NiN")
    for col in ("kJ0", "X"):
        check(f"no 'unknown' warning for the recognised {col}",
              count(out, f"kal property sys:NN:{col} unknown") == 0)
        check(f"one neutral 'has no value' line for {col}",
              count(out, f"kal property sys:NN:{col} has no value for "
                    "this state -> NiN") == 1)
    check("the misspelt kJ0x prints 'unknown' once",
          count(out, "kal property sys:NN:kJ0x unknown -> NiN") == 1)
    check("no 'has no value' line for the misspelt kJ0x",
          count(out, "sys:NN:kJ0x has no value") == 0)
    check("kJ0 gate note present", count(out, KJ0_NOTE) == 1)
    check("X gate note present", count(out, X_NOTE) == 1)

    print("[NiN/file] a three-step var scan in one file")
    scan = ("var : mol-A : scan : theta\nvar : mol-A : step : 1\n"
            "var : mol-A : end_value : 22")
    out, rows = run(tmp, "scan.in", SLAB.format(theta=20, scan=scan))
    n_rows = len(rows) - 1
    check("three kal rows", n_rows == 3, f"{n_rows}")
    check("'unknown' for kJ0x once per kal row",
          count(out, "kal property sys:NN:kJ0x unknown") == n_rows)
    check("'has no value' for kJ0 once per kal row",
          count(out, "sys:NN:kJ0 has no value") == n_rows)
    check("the deduplicated kJ0 note once per FILE, not per step",
          count(out, KJ0_NOTE) == 1, f"{count(out, KJ0_NOTE)}")

    print("[file] two input files in one process")
    out1, _ = run(tmp, "a.in", SLAB.format(theta=20, scan=""))
    out2, _ = run(tmp, "b.in", SLAB.format(theta=20, scan=""))
    for note, what in ((KJ0_NOTE, "kJ0"), (X_NOTE, "X")):
        check(f"{what} note in file 1", count(out1, note) == 1)
        check(f"{what} note in file 2 too (reset per file)",
              count(out2, note) == 1)
    check("file 2 never calls kJ0 'unknown'",
          count(out2, "sys:NN:kJ0 unknown") == 0)
    lp1, r1 = run(tmp, "drop1.in", DROPLET)
    lp2, r2 = run(tmp, "drop2.in", DROPLET)
    check("two-sided Laplace_pressure note in file 1",
          count(lp1, LP_NOTE) == 1, lp1.strip()[-120:])
    check("two-sided Laplace_pressure note in the identical file 2",
          count(lp2, LP_NOTE) == 1)
    check("the two files print the same Laplace_pressure",
          r1[1] == r2[1], f"{r1[1]} / {r2[1]}")


# ---- regula-falsi diagnostics on mock targets (pure Python, ms each) ------
def jump(s, j_lo, j_hi, xs=37.3):
    """A target with slope s and a step from -j_lo to +j_hi at xs: no
    root, no noise."""
    return lambda X: s * (X - xs) + (-j_lo if X < xs else j_hi)


def hash_noise(X):
    """Deterministic pseudo-noise in [-1, 1): a function of the bits of X,
    so repeated probes at the same X agree and neighbouring X do not."""
    h = hashlib.sha256(struct.pack("<d", X)).digest()
    return int.from_bytes(h[:8], "little") / 2.0**64 * 2.0 - 1.0


def noisy(s, sigma, xs=37.3):
    return lambda X: s * (X - xs) + sigma * hash_noise(X)


def falsi(f, x0, tol=1e-6, itlimit=400):
    """Run the falsi on a mock target -> message, or None on success."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            _regula_falsi(f, x0, tol, 0.1, itlimit, False, "t")
        except RuntimeError as e:
            return str(e)
    return None


def suggested(text):
    """The tolerance value a message suggests, or None."""
    key = "super_tolerance : "
    if key not in text:
        return None
    return float(text.split(key)[1].split("'")[0])


print("[rf] bracket-collapse diagnosis")
for label, f, x0 in (
        ("jump slope 5, start 10", jump(5, .15, .20), 10.0),
        ("jump slope 1, start 22", jump(1, .15, .20), 22.0),
        ("jump slope 50, start 30", jump(50, .15, .20), 30.0),
        ("jump +-1e-3, start 36.8", jump(1, 1e-3, 1e-3), 36.8)):
    msg = falsi(f, x0)
    check(label + ": JUMP, no tolerance suggested",
          msg is not None and "JUMPS" in msg and suggested(msg) is None,
          (msg or "returned")[:160])

for label, f, x0, tol, sigma in (
        ("noise 1e-3, start 60, tol 1e-5", noisy(1, 1e-3), 60.0, 1e-5,
         1e-3),
        ("noise 1e-4, start 30", noisy(1, 1e-4), 30.0, 1e-6, 1e-4)):
    msg = falsi(f, x0, tol=tol)
    sug = suggested(msg) if msg else None
    check(label + ": NOISE, suggested tolerance >= sigma",
          msg is not None and "noise-limited" in msg and sug is not None
          and sug >= sigma, f"suggested {sug}: {(msg or 'returned')[:120]}")

check("a clean continuous target converges",
      falsi(lambda X: 0.3 * (X - 37.3) ** 3 + (X - 37.3), 10.0) is None)

# the classifier on hand-made windows around c = 10
c = 10.0
check("reproducible opposite-sign sides -> jump",
      _classify_readouts([(c - 1e-12, -0.1), (c - 2e-12, -0.1),
                          (c, 0.2), (c + 1e-12, 0.2)], c)[0] == "jump")
kind, floor = _classify_readouts([(c - 1e-12, -1e-3), (c - 2e-12, -1e-5),
                                  (c, 4e-4), (c + 1e-12, 2e-3)], c)
check("sides scattering by their own size -> noise",
      kind == "noise" and 1e-4 < floor < 2e-3, f"{kind}, floor {floor}")
check("one scattering side is enough for noise",
      _classify_readouts([(c - 1e-12, -1e-3), (c - 2e-12, -1e-5),
                          (c, 0.2)], c)[0] == "noise")
check("one readout per side -> unclear",
      _classify_readouts([(c - 1e-12, -0.1), (c, 0.2)], c)[0] == "unclear")
check("readouts outside the window do not count",
      _classify_readouts([(c - 1e-3, -0.1), (c - 2e-3, -0.3),
                          (c, 0.2), (c + 1e-12, 0.2)], c)[0] == "unclear")

print("[rf] inner-failure messages")


def fail_at(n):
    calls = [0]

    def resid(X):
        calls[0] += 1
        if calls[0] == n:
            raise RuntimeError("no convergence in 6300 iterations")
        return X - 37.3
    return resid


msg = falsi(fail_at(1), 30.0)
check("a failure at the start value names the scan step",
      msg is not None and "start value" in msg and "scan step" in msg
      and "super_deltamax'" not in msg, (msg or "returned")[:160])
msg = falsi(fail_at(2), 30.0)
check("a failure at a later probe names super_deltamax",
      msg is not None and "super_deltamax" in msg
      and "start value" not in msg, (msg or "returned")[:160])

print(f"all done ({failures} failure(s))")
sys.exit(1 if failures else 0)
