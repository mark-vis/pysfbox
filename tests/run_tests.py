#!/usr/bin/env python
"""Regression runner for PySFBox.

For every `<case>.in` in this folder: run PySFBox in a temp dir and compare
the produced `.kal` / `.pro` files against the committed reference files with
the same names. Solver-dependent kal columns (`iterations`, `residual`) are
ignored; everything else must agree to RTOL (with an absolute floor for
near-zero entries, a log-space check for strictly positive trace columns, and
NiN/inf entries matched class for class -- see `compare`). The guard scripts
`input_guards.py` and `compare_guards.py` run as two further cases.

Usage:
    python tests/run_tests.py            # run all cases
    python tests/run_tests.py brush      # only cases whose name contains 'brush'

Add a new regression case by dropping `<case>.in` here, checking its output
by hand (against Namics where possible), and committing the resulting
`<case>.kal` / `<case>*.pro` as the reference.
"""

import glob
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pysfbox.runner import run_file  # noqa: E402

# Column comparison: |new - ref| is judged against RTOL * max(|ref|, ATOL),
# i.e. a relative tolerance RTOL = 1e-4 for |ref| >= 1e-2 and an ABSOLUTE
# tolerance RTOL*ATOL = 1e-6 below. Two independently converged solutions at
# tolerance 1e-7 differ by ~5e-6 in derived observables SAME-PLATFORM; ACROSS
# platforms the sensitive iterated fields (notably the electrostatic potential
# psi, which is ~0 over neutral regions where relative comparison is
# meaningless) diverge by ~5e-5 through BLAS/libm rounding. RTOL=1e-4 sits
# above that; the ATOL floor keeps a rounding wiggle in a near-zero psi from
# blowing up to a large relative deviation.
# The floor would leave strictly positive TRACE columns (a dendrimer GN of
# 1e-35, a dilute bulk fraction of 5e-8) unchecked, so the kal labels in
# LOG_SUFFIXES are ALSO compared in log space: every entry with
# ref > LOG_TINY must stay positive and satisfy |ln new - ln ref| <= RTOL
# (entries with ref = 0, e.g. a pinned molecule's phibulk, stay on the
# absolute rule).
# What a green suite does NOT mean: the stored numbers bound neither the
# size of a physics error nor its absence -- a small accounting error can
# move a reference by less than RTOL and pass. Accounting fixes therefore
# get dedicated guard inputs (e.g. delta_critical_ratio.in) whose reference
# flips visibly when the fix is reverted.
RTOL, ATOL = 1e-4, 1e-2
LOG_SUFFIXES = (":GN", ":phibulk", ":phi_lower", ":phi_upper", ":n", ":theta")
LOG_TINY = 1e-300

# stiff strong-adsorption corner (N=10000, chi_Si=-6): now converges via the
# extended-budget pseudohessian rescue and matches the compiled Namics exactly.
# Its input's iterationlimit is capped so the primary stages bow out quickly
# (the extended rescue then converges); this is the slowest test (~3 min).
EXPECT_FAIL = set()
CAP_ITERLIMIT = {"homopolymer_adsorption.in"}

# Namics scaling demo: start 1 (N=1000) converges in ~2 min, but the later
# starts scale N up to 5e7 -- far outside PySFBox' scope (and memory).
# Two_brushes: real Namics example with a 991-step n_layers scan (1000 -> 10)
# and N=2000 chains -- hours of work, and its committed .kal holds only the
# first scan row. Kept for manual/stress runs; the regression case is the
# Namics-validated two_brushes_quick.in instead.
SKIP = {"polad.in", "Two_brushes_interactingSurfaceBC.in"}

# guard scripts in this folder run as suite cases (exit 0 = PASS), with the
# message printed on success: (script stem, message)
GUARD_SCRIPTS = (("input_guards", "all guards raise"),
                 ("compare_guards", "all comparator verdicts as expected"))


def _field(v, strings):
    """One table field -> float; NiN -> nan; a STRING property (e.g. kal
    lat : X : geometry, printed bare like Namics) -> nan, with the text
    collected in `strings` for an exact comparison."""
    if v == "NiN":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        strings.append(v)
        return float("nan")


def read_table(path):
    """Read a .kal or .pro file -> (header list, 2D float array, NiN->nan,
    list of string fields in reading order)."""
    strings = []
    with open(path) as fp:
        header = fp.readline().rstrip("\n").split("\t")
        rows = [[_field(v, strings) for v in ln.rstrip("\n").split("\t")]
                for ln in fp if ln.strip()]
    return header, np.array(rows, dtype=float), strings


def compare(ref_path, new_path):
    """Column-wise compare; returns (ok, message, worst) with `worst` the
    largest deviation found (a float, so callers can rank files)."""
    href, ref, sref = read_table(ref_path)
    hnew, new, snew = read_table(new_path)
    if href != hnew:
        return False, "headers differ", float("inf")
    if sref != snew:
        return False, "string fields differ", float("inf")
    if ref.shape != new.shape:
        return False, f"shape {new.shape} != reference {ref.shape}", \
            float("inf")
    worst = 0.0
    for j, label in enumerate(href):
        if label.endswith((":iterations", ":residual")):
            continue
        r, n = ref[:, j], new[:, j]
        # a non-finite entry must be matched by the SAME class in the other
        # file (NiN <-> NiN, +inf <-> +inf, -inf <-> -inf), else FAIL. A lone
        # NaN or an inf in the reference makes `rel` NaN below, and both
        # `nan > RTOL` and max(0, nan) are False/0.0, so such a column used
        # to pass as "max rel dev 0.0" -- even with a 200x deviation in a
        # finite row of the same column.
        nonfin = ~np.isfinite(r) | ~np.isfinite(n)
        match = (np.isnan(r) & np.isnan(n)) | (r == n)
        if np.any(nonfin & ~match):
            return False, f"column '{label}' non-finite mismatch vs " \
                "reference", float("inf")
        r, n = np.where(nonfin, 0.0, r), np.where(nonfin, 0.0, n)
        denom = np.maximum(np.abs(r), ATOL)
        rel = float(np.max(np.abs(n - r) / denom))
        if not np.isfinite(rel) or rel > RTOL:
            return False, f"column '{label}' deviates (max rel {rel:.2e})", \
                rel
        worst = max(worst, rel)
        # strictly positive trace columns: the log rule (see LOG_SUFFIXES)
        if label.endswith(LOG_SUFFIXES):
            pos = r > LOG_TINY
            if np.any(pos & (n <= 0)):
                return False, f"column '{label}' lost a positive trace " \
                    "value", float("inf")
            if np.any(pos):
                dln = float(np.max(np.abs(np.log(n[pos]) - np.log(r[pos]))))
                if dln > RTOL:
                    return False, f"column '{label}' deviates in log space " \
                        f"(max |d ln| {dln:.2e})", dln
                worst = max(worst, dln)
    return True, f"max rel dev {worst:.1e}", worst


def run_case(in_name):
    """Run one input in a temp dir; compare all reference outputs."""
    stem = os.path.splitext(in_name)[0]
    # A reference belongs to this input iff its basename is `<stem>.<ext>` or a
    # Namics-numbered `.pro` (`<stem>_<j>.pro` / `<stem>_<start>_<j>.pro`). The
    # `_<digit>` guard stops a sibling case whose stem merely PREFIXES this one
    # (e.g. charged_sphere_fjc.pro vs charged_sphere.in) from being claimed.
    refs = sorted(p for p in glob.glob(os.path.join(HERE, stem + "*"))
                  if os.path.basename(p) != in_name
                  and (os.path.basename(p).startswith(stem + ".")
                       or re.match(re.escape(stem) + r"_\d", os.path.basename(p))))
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(HERE, in_name), tmp)
        if in_name in EXPECT_FAIL or in_name in CAP_ITERLIMIT:
            # cap the input's iterationlimit so Anderson bows out quickly (the
            # pseudohessian rescue then does the work / graceful failure shows)
            p = os.path.join(tmp, in_name)
            with open(p) as fp:
                text = fp.read()
            with open(p, "w") as fp:
                fp.write(re.sub(r"(iterationlimit\s*:\s*)\d+",
                                r"\g<1>80", text))
        t0 = time.time()
        try:
            run_file(os.path.join(tmp, in_name), verbose=False)
        except RuntimeError as e:
            if in_name in EXPECT_FAIL:
                return True, f"expected non-convergence ({time.time()-t0:.1f} s)"
            return False, f"solver failed: {e}"
        dt = time.time() - t0
        if in_name in EXPECT_FAIL:
            return False, "expected non-convergence, but it converged"
        if not refs:
            return True, f"smoke only, no reference ({dt:.1f} s)"
        worst = 0.0
        for ref in refs:
            new = os.path.join(tmp, os.path.basename(ref))
            if not os.path.exists(new):
                return False, f"did not produce {os.path.basename(ref)}"
            ok, msg, dev = compare(ref, new)
            if not ok:
                return False, f"{os.path.basename(ref)}: {msg}"
            # a NUMERIC max: max() over the message strings used to rank
            # "9.5e-08" above "1.2e-05"
            worst = max(worst, dev)
        return True, (f"{len(refs)} file(s), max rel dev {worst:.1e} "
                      f"({dt:.1f} s)")


def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    inputs = sorted(os.path.basename(p)
                    for p in glob.glob(os.path.join(HERE, "*.in")))
    cases = [n for n in inputs if pattern in n and n not in SKIP]
    stems = {os.path.splitext(n)[0] for n in inputs}
    orphans = [os.path.basename(p)
               for p in sorted(glob.glob(os.path.join(HERE, "*.kal")))
               if os.path.splitext(os.path.basename(p))[0] not in stems]

    failures = 0
    for name in cases:
        ok, msg = run_case(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:40s} {msg}")
        failures += not ok
    # guard scripts whose claims cannot be expressed as a reference
    # comparison: the input guards (tests/input_guards.py, "this input must
    # RAISE") and the comparator's own verdicts on mutated tables
    # (tests/compare_guards.py)
    for guard, ok_msg in GUARD_SCRIPTS:
        if pattern not in guard:
            continue
        import subprocess
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(
            HERE, guard + ".py")], capture_output=True, text=True)
        ok = p.returncode == 0
        bad = [ln for ln in p.stdout.splitlines() if "FAIL" in ln]
        msg = (f"{ok_msg} ({time.time() - t0:.1f} s)" if ok
               else (bad[0].strip() if bad else p.stderr.strip()[-200:]))
        print(f"  {'PASS' if ok else 'FAIL'}  {guard:40s} {msg}")
        failures += not ok
        cases = cases + [guard]
    for name in sorted(SKIP):
        if pattern in name:
            print(f"  SKIP  {name:40s} see SKIP note in this script")
    for o in orphans:
        print(f"  note: {o} is a reference without a .in (input never kept)")
    print(f"{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
