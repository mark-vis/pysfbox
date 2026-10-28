"""Guards on the regression comparator itself (tests/run_tests.py
`compare`): the suite can only be trusted if a wrong table FAILS. No
solver; well under a second; also run by tests/run_tests.py as the
'compare_guards' case.

    python tests/compare_guards.py

[non-finite] an inf in a reference column used to make the deviation NaN,
     and the whole column then passed as "max rel dev 0.0"; a lone NiN
     against a number passed the same way. A non-finite entry must now be
     matched by the same class (NaN-NaN or the same-signed inf) in the
     other file, and the finite rows of the column are still checked.
[trace] the absolute floor RTOL*ATOL = 1e-6 made every |ref| < 1e-2 entry
     an absolute check, so a dendrimer GN of 1e-35 (x1e25, or -> 0) and a
     5e-8 bulk fraction (x100) passed. Labels in LOG_SUFFIXES are now also
     compared in log space; sign-changing and exact-zero columns (psi) keep
     the absolute floor.
The verdicts are checked on synthetic one-column tables written with the
production `output.write_kal`, and on mutated copies of committed
references.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pysfbox.output import write_kal  # noqa: E402
from run_tests import compare  # noqa: E402

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"  {'ok   ' if ok else 'FAIL '} {label}"
          + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        failures += 1


class _Rows:
    """Stand-in for a System: get_value returns the next row's value."""

    def __init__(self, value):
        self.value = value

    def get_value(self, key, name, prop):
        return "real", self.value


def write_column(path, label, values):
    """A one-column kal file written by the production writer."""
    key, name, prop = label.split(":")
    for i, v in enumerate(values):
        write_kal(path, [(key, name, prop)], _Rows(v), new_file=(i == 0))


def verdict(tmp, label, ref, new):
    write_column(os.path.join(tmp, "r.kal"), label, ref)
    write_column(os.path.join(tmp, "n.kal"), label, new)
    return compare(os.path.join(tmp, "r.kal"), os.path.join(tmp, "n.kal"))


def synthetic(tmp, label, col, ref, new, expect):
    ok, msg, _ = verdict(tmp, col, ref, new)
    check(f"{label}: {'PASS' if expect else 'FAIL'} expected", ok == expect,
          f"compare returned ({ok}, {msg!r})")


def mutated(tmp, label, ref_name, col, f, expect):
    """Copy a committed reference, apply f to one column, compare."""
    ref = os.path.join(HERE, ref_name)
    with open(ref) as fp:
        lines = fp.read().splitlines()
    head = lines[0].split("\t")
    j = head.index(col)
    out = [lines[0]]
    for ln in lines[1:]:
        row = ln.split("\t")
        row[j] = f"{f(float(row[j])):.16e}"
        out.append("\t".join(row))
    new = os.path.join(tmp, ref_name)
    with open(new, "w") as fp:
        fp.write("\n".join(out) + "\n")
    ok, msg, _ = compare(ref, new)
    check(f"{label}: {'PASS' if expect else 'FAIL'} expected", ok == expect,
          f"compare returned ({ok}, {msg!r})")


inf, nan = float("inf"), float("nan")
with tempfile.TemporaryDirectory() as tmp:
    mu = "mol:A:mu"
    print("[non-finite] entries")
    synthetic(tmp, "-inf vs +inf, finite row off", mu,
              [-inf, -5.0], [inf, 7.0], False)
    synthetic(tmp, "-inf becomes finite", mu,
              [-inf, -5.0], [-3.0, -5.0], False)
    synthetic(tmp, "matching -inf hides a 200x finite row", mu,
              [-inf, -5.0], [-inf, 1000.0], False)
    synthetic(tmp, "matching -inf, finite rows equal (control)", mu,
              [-inf, -5.0], [-inf, -5.0], True)
    synthetic(tmp, "finite deviation (control)", mu,
              [-1.0, -5.0], [-1.0, 7.0], False)
    synthetic(tmp, "+inf only in the new table (control)", mu,
              [-1.0, -5.0], [inf, -5.0], False)
    synthetic(tmp, "matching NiN hides a 200x finite row", mu,
              [nan, -5.0], [nan, 1000.0], False)
    synthetic(tmp, "+inf vs -inf (sign)", mu,
              [inf, -5.0], [-inf, -5.0], False)
    synthetic(tmp, "NiN vs a number", mu, [nan, -5.0], [-1.0, -5.0], False)
    synthetic(tmp, "matching NiN, finite rows equal (control)", mu,
              [nan, -5.0], [nan, -5.0], True)

    print("[trace] columns in log space")
    synthetic(tmp, "trace GN x2", "mol:pol:GN", [9.5e-35], [1.9e-34], False)
    synthetic(tmp, "trace GN within RTOL in ln", "mol:pol:GN",
              [9.5e-35], [9.5e-35 * (1 + 1e-6)], True)
    synthetic(tmp, "plateau 4.1e-9 x2", "mol:C:phi_lower",
              [4.1e-9], [8.2e-9], False)
    synthetic(tmp, "trace phibulk -> 0", "mol:C:phibulk", [4.1e-9], [0.0],
              False)
    synthetic(tmp, "phibulk 0 (pinned) stays on the absolute rule",
              "mol:C:phibulk", [0.0], [1e-9], True)
    synthetic(tmp, "psi-like column keeps the absolute floor",
              "sys:NN:psi", [1e-12], [2e-12], True)
    mutated(tmp, "branched_dendrimer GN x1e25", "branched_dendrimer.kal",
            "mol:pol:GN", lambda v: v * 1e25, False)
    mutated(tmp, "branched_dendrimer GN x2", "branched_dendrimer.kal",
            "mol:pol:GN", lambda v: v * 2, False)
    mutated(tmp, "branched_dendrimer GN -> 0", "branched_dendrimer.kal",
            "mol:pol:GN", lambda v: 0.0, False)
    mutated(tmp, "planar_fjc_free_lamella phibulk (5e-8) x100",
            "planar_fjc_free_lamella.kal", "mol:poly:phibulk",
            lambda v: v * 100, False)
    mutated(tmp, "delta_micelle phibulk (3e-5) x2", "delta_micelle.kal",
            "mol:pol:phibulk", lambda v: v * 2, False)
    mutated(tmp, "depletion_mu poly n (4e-4) x1.001",
            "depletion_mu.kal", "mol:poly:n", lambda v: v * 1.001, False)
    mutated(tmp, "unchanged reference (control)", "branched_dendrimer.kal",
            "mol:pol:GN", lambda v: v, True)

print(f"all done ({failures} failure(s))")

if __name__ == "__main__":
    sys.exit(1 if failures else 0)
