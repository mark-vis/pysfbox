"""Guards on the runner's warm seed (`runner._warm_seed`, `_remap_layers`,
`_seed_by_species`; synced from pysfbox-dev PR #170, 28 Oct 2026) that the
`.in` suite cannot express: it compares converged output, and a bad seed
mostly costs iterations (or, on a multi-solution problem, picks a branch --
that case is the suite input tests/nd_nlayers_scan_2d.in).

    python tests/warm_seed_guards.py        (~2 s)

Checks:
  1. `_remap_layers` on labelled cells: every kept cell keeps its lattice
     coordinate under the per-axis two-wall rule -- inner-axis (14,14) ->
     (14,12), outer (14,14) -> (12,14), odd outer (13,14) -> (15,14), 3-D
     innermost (10,10,10) -> (10,10,8), a same-M axis swap (12,14) ->
     (14,12) -- every block alike, middle cells zero; the 1-D column is
     BIT-identical to the historical flat split (the suite's n_layers
     scans depend on it). The flat split scored 12/168 and 8/800 cells.
  2. Charge switched ON under a delta constraint (Na/Cl valence 0 -> +-1,
     symmetric salt, so psi = 0 and the previous state is exact): the seed
     has psi block 0 and the previous beta in the beta block (the old tail
     append put beta into psi and restarted beta at 0), the charged start
     converges in <= 2 iterations, Omega equal to 1e-9.
  3. Charge switched OFF (asymmetric chi_O_Cl, so psi != 0): the seed's
     beta block is the previous beta exactly (truncation put psi there).

Run after touching runner._warm_seed / _remap_layers / _seed_by_species,
the main-loop seed wiring or System.n_var's block layout.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pysfbox.system as S                                  # noqa: E402
from pysfbox.runner import _remap_layers, run_file          # noqa: E402

# ---------------------------------------------------------------- 1. remap
print("[1] _remap_layers keeps lattice coordinates per axis")


def expected_old_coord(nc, po, pn):
    """Old coordinate the per-axis two-wall rule copies into new cell nc,
    or None for a zero-filled middle cell."""
    oc = []
    for a in range(len(po)):
        k = min(po[a], pn[a])
        lo, hi = k // 2, k - k // 2
        if nc[a] < lo:
            oc.append(nc[a])
        elif nc[a] >= pn[a] - hi:
            oc.append(po[a] - (pn[a] - nc[a]))
        else:
            return None
    return tuple(oc)


for po, pn in [((14, 14), (14, 12)), ((14, 14), (12, 14)),
               ((13, 14), (15, 14)), ((10, 10, 10), (10, 10, 8)),
               ((12, 14), (14, 12)), ((22,), (26,)), ((22,), (17,))]:
    Mo, Mn = int(np.prod(po)), int(np.prod(pn))
    lab = np.arange(Mo, dtype=float) + 1.0          # label = flat index + 1
    seed = _remap_layers(np.concatenate([lab, lab + Mo]), po, pn)
    assert seed.size == 2 * Mn, "block count changed"
    old_coords = np.indices(po).reshape(len(po), -1).T
    new_coords = np.indices(pn).reshape(len(pn), -1).T
    kept = 0
    for j in range(Mn):
        exp = expected_old_coord(new_coords[j], po, pn)
        if exp is None:
            assert seed[j] == 0.0 and seed[Mn + j] == 0.0, (po, pn, j)
            continue
        got = tuple(old_coords[int(seed[j]) - 1])
        assert got == exp, f"{po}->{pn}: cell {j} from {got}, want {exp}"
        assert seed[Mn + j] == seed[j] + Mo, "second block remapped apart"
        kept += 1
    print(f"    {po} -> {pn}: {kept} kept cells on their coordinates")


def flat_split(x_old, M_old, M_new):
    """The pre-v1.4.4 1-D split, verbatim, as the bit-identity anchor."""
    nb = x_old.size // M_old
    Xo = x_old.reshape(nb, M_old)
    Xn = np.zeros((nb, M_new))
    k = min(M_old, M_new)
    lo, hi = k // 2, k - k // 2
    Xn[:, :lo] = Xo[:, :lo]
    Xn[:, M_new - hi:] = Xo[:, M_old - hi:]
    return Xn.ravel()


rng = np.random.default_rng(1)
for Mo, Mn in [(22, 26), (26, 22), (33, 34), (34, 33), (41, 40), (5, 5)]:
    x = rng.standard_normal(3 * Mo)
    assert np.array_equal(flat_split(x, Mo, Mn),
                          _remap_layers(x, (Mo,), (Mn,)))
print("    1-D column bit-identical to the historical split")

# --------------------------------------------------- 2-3. layout toggles
BASE = """
lat : NN : geometry : spherical
lat : NN : n_layers : 40
lat : NN : lowerbound : mirror
mon : O : freedom : pinned
mon : O : pinned_range : 1;6
mon : W : freedom : free
mon : O : chi_W : 1.2
mon : Na : freedom : free
mon : Cl : freedom : free
mon : Na : valence : {v0}
mon : Cl : valence : -{v0}
{chi}
mol : oil : composition : (O)20
mol : oil : freedom : restricted
mol : oil : theta : 700
mol : W : composition : (W)1
mol : W : freedom : solvent
mol : Na : composition : (Na)1
mol : Na : freedom : free
mol : Na : phibulk : 0.01
mol : Cl : composition : (Cl)1
mol : Cl : freedom : free
mol : Cl : phibulk : 0.01
newton : n : tolerance : 1e-9
output : kal : append : false
kal : sys : NN : grand_potential
start
mon : O : freedom : free
start
{constraint}
start
mon : Na : valence : {v1}
mon : Cl : valence : -{v1}
start
"""
DELTA = """sys : NN : constraint : delta
sys : NN : delta_molecules : oil;W
sys : NN : delta_range : (5)
sys : NN : phi_ratio : 1.0"""
def run_record(tag, **kw):
    """Run the input (nucleate, release, constrain, toggle the charge);
    return per-solve records and the kal."""
    rec = []
    orig = S.System.solve

    def solve(self, x0=None, **k):
        out = orig(self, x0=x0, **k)
        rec.append(dict(sys=self, x0=None if x0 is None else x0.copy(),
                        x=out[0].copy(), it=out[1]))
        return out
    S.System.solve = solve
    d = tempfile.mkdtemp(prefix=f"warm_seed_{tag}_")
    path = os.path.join(d, f"{tag}.in")
    with open(path, "w") as f:
        f.write(BASE.format(**kw))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_file(path)
    finally:
        S.System.solve = orig
    kal = np.loadtxt(os.path.join(d, f"{tag}.kal"), skiprows=1, ndmin=2)
    return rec, kal


def blocks(system, x):
    """Named slices of an iteration vector, in System.n_var order."""
    M = system.lat.M
    out, off = {}, len(system.it_species) * M
    out["u"] = x[:off]
    for flag, name, size in (("charged", "psi", M),
                             ("constraintfields", "beta", M)):
        if getattr(system, flag, False):
            out[name] = x[off:off + size]
            off += size
    assert off == x.size
    return out


print("[2] charge ON under a delta constraint")
rec, kal = run_record("on", v0=0, v1=1, chi="", constraint=DELTA)
prev, cur = rec[-2], rec[-1]
bp, bs = blocks(prev["sys"], prev["x"]), blocks(cur["sys"], cur["x0"])
assert not prev["sys"].charged and cur["sys"].charged
assert np.all(bs["psi"] == 0.0), "seed psi block is not zero"
assert np.array_equal(bs["beta"], bp["beta"]), "seed beta != previous beta"
assert np.abs(bp["beta"]).max() > 1e-3, "test needs a nonzero beta"
assert cur["it"] <= 2, f"start 3 took {cur['it']} iterations (want <= 2)"
assert abs(kal[-1, 0] - kal[-2, 0]) < 1e-9, kal[:, 0]
print(f"    seed psi 0, beta carried (max {np.abs(bp['beta']).max():.3e}); "
      f"{cur['it']} iterations; dOmega {abs(kal[-1, 0] - kal[-2, 0]):.1e}")

print("[3] charge OFF under a delta constraint (psi != 0 before)")
rec, kal = run_record("off", v0=1, v1=0, chi="mon : O : chi_Cl : 2",
                      constraint=DELTA)
prev, cur = rec[-2], rec[-1]
bp, bs = blocks(prev["sys"], prev["x"]), blocks(cur["sys"], cur["x0"])
assert prev["sys"].charged and not cur["sys"].charged
assert np.abs(bp["psi"]).max() > 1e-3, "test needs a nonzero psi"
assert np.array_equal(bs["beta"], bp["beta"]), "seed beta != previous beta"
assert np.array_equal(bs["u"], bp["u"])
print(f"    seed beta == previous beta (psi max {np.abs(bp['psi']).max():.3e}"
      " dropped)")

print("all warm-seed guards PASS")
