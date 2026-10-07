"""Input guards (the 6 Oct 2026 review fixes):
every case below used to run SILENTLY with wrong physics (or an empty
molecule, a dropped wall, a mirror instead of a wall, ...). Each must now
raise the named exception with an actionable message, and the legal
neighbour of each case must still build. Fast (System builds only, plus one
tiny scan); also run by tests/run_tests.py as the 'input_guards' case.

    python tests/input_guards.py

Review finding numbers in brackets: [4] charged ghost wall, [6] missing
neutralizer, [7] chi symmetry + partner-side scan, [10] ghost wall needs a
surface bound / one wall per face, [25] mon freedom values, [29] negative
neutralizer bulk, [30] bound values + wrong-dimension keys, [31] lambda at
fjc>1/N-D, [36] 1-D range bounds, [46] micro/bate blocks, [59] mol
freedom/amount, [61] unused state chi, [71] UTF-8 BOM, [84] mu : 0,
[45] var scan validation, [53] one target per var block.
"""
import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pysfbox.inputreader import read_input  # noqa: E402
from pysfbox.latticend import LatticeND  # noqa: E402
from pysfbox import runner  # noqa: E402
from pysfbox.system import System  # noqa: E402

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"  {'ok   ' if ok else 'FAIL '} {label}"
          + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        failures += 1


WALL = """
lat : flat : n_layers : 20
lat : flat : geometry : planar
lat : flat : lowerbound : surface
mon : S : freedom : frozen
mon : S : frozen_range : lowerbound
mon : A : freedom : free
mon : A : chi_S : -1.5
mon : W : freedom : free
mon : W : chi_A : 0.4
mol : pol : composition : (A)25
mol : pol : freedom : free
mol : pol : phibulk : 0.02
mol : water : composition : (W)1
mol : water : freedom : solvent
"""
SALT = """
mon : Na : freedom : free
mon : Na : valence : 1
mon : Cl : freedom : free
mon : Cl : valence : -1
mol : cl : composition : (Cl)1
mol : cl : freedom : free
mol : cl : phibulk : 0.01
"""
NEUT = """
mol : na : composition : (Na)1
mol : na : freedom : neutralizer
"""


def write(text, name="g.in"):
    d = tempfile.mkdtemp(prefix="pysfbox_guards_")
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(text.lstrip())
    return p


def build(text):
    with contextlib.redirect_stdout(io.StringIO()):
        return System(read_input(write(text))[-1])


def raises(label, text, exc, words):
    try:
        build(text)
    except exc as e:
        msg = str(e)
        check(label, all(w in msg for w in words), msg)
        return
    except Exception as e:                          # noqa: BLE001
        check(label, False, f"{type(e).__name__}: {e}")
        return
    check(label, False, "built without complaint")


def builds(label, text):
    try:
        build(text)
        check(label, True)
    except Exception as e:                          # noqa: BLE001
        check(label, False, f"{type(e).__name__}: {e}")


def sub(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)


print("[4] charged wall in the ghost layer")
raises("valence on a lowerbound wall", WALL + "mon : S : valence : -0.2\n"
       + SALT + NEUT, ValueError, ["ghost", "1;1"])
builds("the same wall on layer 1 (frozen_range : 1;1)",
       sub(WALL, "lat : flat : lowerbound : surface\n", "").replace(
           "frozen_range : lowerbound", "frozen_range : 1;1")
       + "mon : S : valence : -0.2\n" + SALT + NEUT)

print("[6] neutralizer needed")
raises("non-neutral free bulk, no neutralizer",
       WALL + SALT + "mol : na : composition : (Na)1\nmol : na : freedom : "
       "free\nmol : na : phibulk : 0.005\n", ValueError, ["neutralizer"])
raises("charged floating restricted molecule, no neutralizer",
       WALL + SALT + "mon : P : freedom : free\nmon : P : valence : -0.5\n"
       "mol : pe : composition : (P)5\nmol : pe : freedom : restricted\n"
       "mol : pe : theta : 1.5\nmol : na : composition : (Na)1\n"
       "mol : na : freedom : free\nmol : na : phibulk : 0.01\n",
       ValueError, ["neutralizer", "pe"])
builds("neutral free salt without a neutralizer",
       WALL + SALT + "mol : na : composition : (Na)1\nmol : na : freedom : "
       "free\nmol : na : phibulk : 0.01\n")

print("[7] chi table")
raises("conflicting chi on both partners",
       WALL + "mon : A : chi_W : 1.0\n", ValueError, ["disagree"])
builds("equal chi on both partners", WALL + "mon : A : chi_W : 0.4\n")
raises("nonzero self-chi", WALL + "mon : A : chi_A : 0.3\n", ValueError,
       ["self-interaction"])
builds("zero self-chi", WALL + "mon : A : chi_A : 0\n")

print("[7] chi scan from the partner side keeps the table symmetric")
p = write(WALL + "output : kal : append : false\nkal : sys : NN : "
          "grand_potential\nstart\nvar : mon-A : scan : chi_W\n"
          "var : mon-A : step : 0.1\nvar : mon-A : end_value : 0.6\nstart\n",
          "scan.in")
q = write(sub(WALL, "chi_A : 0.4", "chi_A : 0.6") + "output : kal : append "
          ": false\nkal : sys : NN : grand_potential\nstart\n", "direct.in")
with contextlib.redirect_stdout(io.StringIO()):
    runner.run_file(p, verbose=False)
    runner.run_file(q, verbose=False)
rows = open(p[:-3] + ".kal").read().split()[1:]
direct = open(q[:-3] + ".kal").read().split()[1:]
check("scan starts at the partner's 0.4 (start row + 0.4/0.5/0.6) and its "
      "0.6 row equals a direct chi 0.6", len(rows) == 4
      and abs(float(rows[-1]) / float(direct[0]) - 1) < 1e-4,
      f"{rows} vs {direct}")

print("[10] ghost wall needs a surface bound; one wall per face")
raises("lowerbound wall with the default mirror bound",
       sub(WALL, "lat : flat : lowerbound : surface\n", ""), ValueError,
       ["surface"])
raises("two walls in the same ghost layer",
       WALL + "mon : T : freedom : frozen\nmon : T : frozen_range : "
       "lowerbound\n", ValueError, ["Overpopulated"])
ND = """
lat : flat : gradients : 2
lat : flat : n_layers_x : 10
lat : flat : n_layers_y : 3
lat : flat : geometry : flat
""" + WALL.split("lat : flat : lowerbound : surface\n")[1]
raises("2-D lowerbound wall with a mirror x bound", ND, ValueError,
       ["surface"])
builds("2-D lowerbound wall with lowerbound_x : surface",
       ND + "lat : flat : lowerbound_x : surface\n")

print("[25] mon freedom values")
raises("'Frozen'", sub(WALL, "S : freedom : frozen", "S : freedom : Frozen"),
       ValueError, ["did you mean 'frozen'"])
raises("'pined'", sub(WALL, "A : freedom : free", "A : freedom : pined"),
       ValueError, ["did you mean 'pinned'"])
raises("'tagged'", sub(WALL, "A : freedom : free", "A : freedom : tagged"),
       NotImplementedError, ["tagged"])
try:
    runner._check_first_start(read_input(write(
        WALL + "mon : A : pinned_range : 1;1\n"))[0])
    check("free mon with a range in the first start", False, "accepted")
except ValueError as e:
    check("free mon with a range in the first start", "free" in str(e))

print("[29] negative neutralizer bulk")
raises("declared bulk needs a negative neutralizer",
       WALL + SALT.replace("0.01", "0.005") + "mon : P : freedom : free\n"
       "mon : P : valence : 1\nmol : pe : composition : (P)10\n"
       "mol : pe : freedom : free\nmol : pe : phibulk : 0.01\n" + NEUT,
       ValueError, ["NEGATIVE", "neutralizer"])

print("[30] bound values and keys")
raises("1-D typo 'surfce'", sub(WALL, "lowerbound : surface",
                                "lowerbound : surfce"), ValueError, ["surfce"])
raises("1-D periodic", WALL + "lat : flat : upperbound : periodic\n",
       NotImplementedError, ["periodic", "gradients : 2"])
raises("lowerbound_x with gradients 1",
       WALL + "lat : flat : lowerbound_x : surface\n", ValueError,
       ["lowerbound_x"])
raises("lowerbound with gradients 2",
       ND + "lat : flat : lowerbound : surface\n", ValueError, ["lowerbound"])
for label, kw, words in [
        ("N-D one-sided periodic",
         dict(bounds=[("periodic", "mirror"), None]), ["BOTH"]),
        ("N-D periodic on a radial axis",
         dict(geometry="cylindrical", bounds=[("periodic", "periodic"), None]),
         ["r axis"]),
        ("N-D bound typo", dict(bounds=[("surfce", "mirror"), None]),
         ["surfce"])]:
    try:
        LatticeND((6, 5), **kw)
        check(label, False, "accepted")
    except ValueError as e:
        check(label, all(w in str(e) for w in words), str(e))

print("[31] lambda where there is no lambda")
raises("lambda at FJC_choices 5 (spherical)", WALL
       + "lat : flat : geometry : spherical\nlat : flat : lambda : 0.3333\n"
       "lat : flat : FJC_choices : 5\n", NotImplementedError, ["lambda"])
raises("lambda on a 2-D lattice", ND + "lat : flat : lowerbound_x : surface\n"
       "lat : flat : lambda : 0.3333\n", NotImplementedError, ["lambda"])
builds("lambda at FJC_choices 3", WALL + "lat : flat : lambda : 0.3333\n")

print("[36] 1-D ranges")
raises("range past the upper ghost", sub(WALL, "frozen_range : lowerbound",
                                         "frozen_range : 8;24"),
       ValueError, ["outside"])
s0 = build(sub(WALL, "frozen_range : lowerbound", "frozen_range : 0;0"))
check("'0;0' is the lower wall (as lowerbound)",
      s0.segments["S"].on_lower_surface and s0.segments["S"].in_ghost)

print("[46] other Namics engines")
raises("micro block", WALL + "micro : m : x : 1\n", NotImplementedError,
       ["micro"])

print("[59] mol freedom and amount")
raises("no freedom line", sub(WALL, "mol : pol : freedom : free\n", ""),
       ValueError, ["freedom"])
raises("free without phibulk", sub(WALL, "mol : pol : phibulk : 0.02\n", ""),
       ValueError, ["phibulk"])
raises("restricted without theta/n",
       sub(sub(WALL, "mol : pol : phibulk : 0.02\n", ""),
           "pol : freedom : free", "pol : freedom : restricted"),
       ValueError, ["theta"])
builds("explicit phibulk : 0 stays legal",
       sub(WALL, "phibulk : 0.02", "phibulk : 0"))

print("[61] a chi that is never used gets a warning")
WEAK = WALL + """
state : H3O : mon : W
state : H3O : valence : 1
state : H3O : alphabulk : 1e-7
state : H2O : mon : W
state : H2O : valence : 0
state : OH : mon : W
state : OH : valence : -1
reaction : auto : equation : 2(H2O) = 1(OH) + 1(H3O)
reaction : auto : pK : 14
state : AH : mon : A
state : AH : valence : 0
state : AM : mon : A
state : AM : valence : -1
reaction : weak : equation : 1(AH) + 1(H2O) = 1(AM) + 1(H3O)
reaction : weak : pK : 5
state : AH : chi_W : 0.9
""" + SALT + NEUT
s1 = build(WEAK)
check("state AH : chi_W (W multistate) warns",
      any("AH : chi_W" in w and "never used" in w for w in s1.warnings),
      str(s1.warnings))

print("[71] UTF-8 byte-order mark")
pb = write("")
with open(pb, "wb") as f:
    f.write(b"\xef\xbb\xbf" + WALL.lstrip().encode())
check("BOM does not eat the first line",
      "n_layers" in read_input(pb)[-1].get(("lat", "flat"), {}))

print("[84] mu : 0 is a number, not a molecule name")
roles = runner._var_roles(read_input(write(
    WALL + "var : mol-pol : search : phibulk\nvar : mol-pol : mu : 0\n"))[-1])
check("mu : 0 parses as a target value",
      roles[2] and roles[2][2:] == ("mu", 0.0), str(roles))

print("[45] var scan blocks are validated; [53] one target per block")
for label, extra, words in [
        ("scan of an undeclared mol",
         "var : mol-poll : scan : theta\nvar : mol-poll : step : 1\n"
         "var : mol-poll : end_value : 3\n", ["no 'mol : poll'"]),
        ("scan of a parameter nobody reads",
         "var : mol-pol : scan : phibluk\nvar : mol-pol : step : 0.01\n"
         "var : mol-pol : end_value : 0.05\n", ["phibluk"]),
        ("scan of a chi with an undeclared partner",
         "var : mon-A : scan : chi_Q\nvar : mon-A : step : 0.1\n"
         "var : mon-A : end_value : 0.5\n", ["'Q'"]),
        ("scale typo",
         "var : mol-pol : scan : phibulk\nvar : mol-pol : step : 0.01\n"
         "var : mol-pol : end_value : 0.05\nvar : mol-pol : scale : "
         "expnential\n", ["scale"]),
        ("zero step",
         "var : mol-pol : scan : phibulk\nvar : mol-pol : step : 0\n"
         "var : mol-pol : end_value : 0.05\n", ["step : 0"]),
        ("two targets in one block",
         "var : mol-pol : search : theta\nvar : sys-NN : grand_potential : "
         "0\nvar : sys-NN : free_energy : 0\n", ["targets"])]:
    try:
        runner._var_plan(read_input(write(WALL + extra))[-1])
        check(label, False, "accepted")
    except ValueError as e:
        check(label, all(w in str(e) for w in words), str(e))

print("[lattice] FJC_choices > 3 ignores lattice_type")
SPH = WALL + "lat : flat : geometry : spherical\nlat : flat : FJC_choices : 5\n"
s_sc = build(SPH + "lat : flat : lattice_type : simple_cubic\n")
check("explicit simple_cubic + FJC 5 warns that lattice_type is ignored",
      any("ignores lattice_type" in w for w in s_sc.warnings),
      str(s_sc.warnings))
s_om = build(SPH)
check("omitted lattice_type + FJC 5 stays silent",
      not any("ignores lattice_type" in w for w in s_om.warnings))
s_hx = build(SPH + "lat : flat : lattice_type : hexagonal\n")
check("hexagonal + FJC 5 does not warn",
      not any("ignores lattice_type" in w for w in s_hx.warnings))

print("[34] a declared initial_guess never vanishes silently on N-D")
NDW = ND + "lat : flat : lowerbound_x : surface\n"
try:
    build(NDW + "sys : NN : initial_guess : membrane_tours\n").initial_guess()
    check("typo kind on N-D raises", False, "no raise")
except NotImplementedError as e:
    check("typo kind on N-D raises", "membrane_tours" in str(e), str(e))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    x0 = build(NDW + "sys : NN : initial_guess : polymer_adsorption\n"
               ).initial_guess()
check("1-gradient kind on N-D cold-starts with a note",
      x0 is None and "cold-starts" in buf.getvalue(), buf.getvalue())

print(f"all done ({failures} failure(s))")


def main():
    return failures


if __name__ == "__main__":
    sys.exit(1 if failures else 0)
