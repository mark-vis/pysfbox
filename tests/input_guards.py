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
[45] var scan validation, [53] one target per var block; [X] the
characteristic-function parse errors and compute_kJ0 (9 Oct 2026);
[27 Oct #9/#26/#41] curved offsets (a solid core thinner than 1/2 - 1/fjc,
negative offsets) and the lambda range (28 Oct 2026); [L]
Laplace_pressure around a frozen core with no bulk plateau (28 Oct 2026); [R27]
the 27 Oct 2026 review: wrong-dimension layer keys and their scans (#7),
the N-D box-size pre-flight (#42) and Namics' per-axis ghost coordinates
in N-D ranges (#28); from
the 27 Oct 2026 review: [F32] amounts outside their domain + the scan end
snap, [F44] alphabulk anchors, [F25] the range keyword follows the freedom,
[F56] missing n_layers/composition, [F57] numbers and alias scope
(newton/sys substitution, the unparseable-newton-value warning); [E] a
fixed-potential electrode on a refined lattice is refused at build, not
as an 'interior electrode' at the first residual (28 Oct 2026).
"""
import contextlib
import io
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pysfbox.inputreader import read_input, substitute_aliases  # noqa: E402
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

print("[A7] layer counts and n_layers steps are whole layers")
raises("non-integral n_layers", sub(WALL, "n_layers : 20", "n_layers : 20.5"),
       ValueError, ["positive integer", "whole layers"])
try:
    runner._var_plan(read_input(write(WALL + """
var : lat-flat : scan : n_layers
var : lat-flat : step : -0.5
var : lat-flat : end_value : 19
"""))[-1])
    check("fractional n_layers scan step raises", False, "planned")
except ValueError as e:
    check("fractional n_layers scan step raises",
          "whole number of layers" in str(e), str(e))

print("[27 Oct #9/#26/#41] curved offsets and lambda on the 1-gradient lattice")
CYL = sub(WALL, "geometry : planar", "geometry : cylindrical")
SPH = sub(WALL, "geometry : planar", "geometry : spherical")
for geo, text in (("cylindrical", CYL), ("spherical", SPH)):
    raises(f"{geo} FJC 11 solid core at offset 0.05 (thinner than "
           "1/2 - 1/fjc)", text + "lat : flat : FJC_choices : 11\n"
           "lat : flat : offset_first_layer : 0.05\n", NotImplementedError,
           ["offset_first_layer", "1/2 - 1/fjc"])
    builds(f"{geo} FJC 11 solid core at offset 0.3 (= 1/2 - 1/fjc)",
           text + "lat : flat : FJC_choices : 11\n"
           "lat : flat : offset_first_layer : 0.3\n")
for geo, text, extra in (("cylindrical", CYL, ""), ("spherical", SPH, ""),
                         ("cylindrical FJC 5", CYL,
                          "lat : flat : FJC_choices : 5\n")):
    for off in ("-1", "-0.5", "-0.3"):
        raises(f"negative offset_first_layer {off} on {geo}",
               text + extra + f"lat : flat : offset_first_layer : {off}\n",
               ValueError, ["offset_first_layer", "non-negative"])
builds("planar ignores a negative offset_first_layer",
       WALL + "lat : flat : offset_first_layer : -1\n")
for text, lam in ((WALL, "0.55"), (WALL, "-0.1"), (WALL, "0"),
                  (SPH, "0.45"), (CYL, "0.55")):
    raises(f"lambda {lam} ({text.split('geometry : ')[1].split()[0]})",
           text + f"lat : flat : lambda : {lam}\n", ValueError,
           ["lat : lambda", "1/3"])
builds("lambda 1/3 on a sphere at offset 0 (stay weight exactly 0)",
       SPH + "lat : flat : lambda : 0.3333333333333333\n")
builds("lambda 0.5 planar", WALL + "lat : flat : lambda : 0.5\n")

print("[X] the Namics characteristic function (sys : X) and compute_kJ0")
WEAK = (WALL.replace("mon : A : chi_S : -1.5\n", "")
        + "state : A1 : mon : A\nstate : A1 : valence : 0\n"
        "state : A2 : mon : A\nstate : A2 : valence : 0\n"
        "state : A2 : alphabulk : 0.3\n")
raises("X : ? prints the help", WALL + "sys : NN : X : ?\n", ValueError,
       ["characteristic function", "molname"])
raises("X not starting with F", WALL + "sys : NN : X : G-water\n",
       ValueError, ["expected 'F'"])
raises("X with an unknown molecule", WALL + "sys : NN : X : F-wat\n",
       ValueError, ["not a molecule name", "did you mean 'water'"])
raises("X with a trailing '-'", WALL + "sys : NN : X : F-water-\n",
       ValueError, ["empty entry"])
raises("X state entry with two arguments",
       WEAK + "sys : NN : X : F-(A1,A2)\n", ValueError, ["not recognised"])
raises("X state entry with an unknown state",
       WEAK + "sys : NN : X : F-(A1,B7,1)\n", ValueError, ["B7"])
raises("X state entry whose mu_j has no monomeric host (Namics: mu = 0)",
       WEAK + "sys : NN : X : F-(A1,A2,1)\n", ValueError,
       ["mu-<state>", "mu = 0"])
builds("X : F - water - pol (spaces are free, as in Namics)",
       WALL + "sys : NN : X : F - water - pol\n")
raises("compute_kJ0 is refused with the pointer",
       WALL + "sys : NN : compute_kJ0 : true\n", NotImplementedError,
       ["compute_kJ0", "bare first and second moments", "curved ladders"])

print("[L] Laplace_pressure around a frozen core with no bulk plateau")
# a thin oil shell (30-layer sphere, theta 1500) on a frozen core: no
# interior plateau past the core, so the kal column is NiN and a
# Laplace_pressure search raises instead of reading the frozen layer's
# omega = 0 and accepting its first probe (28 Oct 2026; the plateau case
# itself is the regression tests/laplace_core.in)
THIN = """
lat : NN : geometry : spherical
lat : NN : n_layers : 30
lat : NN : lowerbound : mirror
mon : X : freedom : frozen
mon : X : frozen_range : 1;4
mon : O : freedom : pinned
mon : O : pinned_range : 5;20
mon : W : freedom : free
mon : O : chi_W : 1.2
mol : oil : composition : (O)20
mol : oil : freedom : restricted
mol : oil : theta : 1500
mol : W : composition : (W)1
mol : W : freedom : solvent
newton : n : tolerance : 1e-9
output : kal : append : false
kal : mol : oil : theta
kal : sys : NN : Laplace_pressure
start
mon : O : freedom : free
{last}
start
"""
for label, last in (("thin-shell kal Laplace_pressure is NiN", ""),
                    ("thin-shell Laplace_pressure search raises",
                     "var : mol-oil : search : theta\n"
                     "var : sys-NN : Laplace_pressure : 0")):
    p = write(THIN.format(last=last), "thin.in")
    err = None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            runner.run_file(p, verbose=False)
        except Exception as e:                      # noqa: BLE001
            err = e
    rows = [ln.split("\t") for ln in
            open(p[:-3] + ".kal").read().split("\n")[1:] if ln]
    if last:
        check(label, isinstance(err, ValueError)
              and "Laplace_pressure undefined" in str(err)
              and len(rows) == 1, repr(err))
    else:
        check(label, err is None and len(rows) == 2
              and rows[1][1] == "NiN", f"{err!r} {rows}")

print("[R27] review 27 Oct 2026: N-D guards and grammar")
import numpy as np  # noqa: E402


def run(text, name="r.in"):
    """run_file on `text`; returns (kal rows as floats, .pro array|None).
    Raises what run_file raises."""
    p = write(text, name)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run_file(p, verbose=False)
    rows = [[float(v) for v in ln.split()] for ln in
            open(p[:-3] + ".kal").read().strip().split("\n")[1:]]
    pro = (np.loadtxt(p[:-3] + ".pro", skiprows=1)
           if os.path.exists(p[:-3] + ".pro") else None)
    return rows, pro


def run_raises(label, text, exc, words, name="r.in"):
    """run_file must raise BEFORE any .kal data row is written."""
    p = write(text, name)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run_file(p, verbose=False)
    except exc as e:
        kal = p[:-3] + ".kal"
        rows = (open(kal).read().strip().split("\n")[1:]
                if os.path.exists(kal) else [])
        check(label, all(w in str(e) for w in words) and not rows,
              f"{e} / {len(rows)} kal row(s)")
        return
    except Exception as e:                          # noqa: BLE001
        check(label, False, f"{type(e).__name__}: {e}")
        return
    check(label, False, "ran without complaint")


# a free (A)10 at an attractive wall, the x axis resolved, y columns
POL = """
mon : S : freedom : frozen
mon : S : frozen_range : {rng}
mon : A : freedom : free
mon : A : chi_S : -1
mon : W : freedom : free
mol : pol : composition : (A)10
mol : pol : freedom : free
mol : pol : phibulk : 0.01
mol : W : composition : (W)1
mol : W : freedom : solvent
newton : NN : tolerance : 1e-11
output : kal : append : false
kal : mol : pol : theta
kal : sys : NN : grand_potential
"""
PRO = "output : pro : append : false\npro : mon : A : phi\n"
L2 = """
lat : L : gradients : 2
lat : L : geometry : flat
lat : L : n_layers_x : {nx}
lat : L : n_layers_y : {ny}
"""


def nd(nx=12, ny=3, rng="lowerbound", extra="lat : L : lowerbound_x : "
       "surface\n"):
    return L2.format(nx=nx, ny=ny) + extra + POL.format(rng=rng)


print("[R27 #7] layer counts of the wrong dimensionality")
raises("gradients 2 + a stray n_layers", nd() + "lat : L : n_layers : 12\n",
       ValueError, ["n_layers", "n_layers_x / n_layers_y"])
raises("gradients 2 + n_layers_z", nd() + "lat : L : n_layers_z : 7\n",
       ValueError, ["n_layers_z"])
raises("gradients 1 + n_layers_x", WALL + "lat : flat : n_layers_x : 12\n",
       ValueError, ["n_layers_x", "read from n_layers only"])
raises("gradients 1 + phi_range", WALL + "lat : flat : phi_range : 0;1\n",
       ValueError, ["phi_range", "gradients : 1"])
SCAN = ("var : lat-L : scan : n_layers\nvar : lat-L : step : -1\n"
        "var : lat-L : end_value : 10\nstart\n")
for lbl, extra in (("with", "lat : L : n_layers : 12\n"), ("without", "")):
    run_raises(f"gradients 2 scan of n_layers ({lbl} a stray line) is "
               "refused before row 0", nd() + extra + SCAN,
               ValueError, ["n_layers_x / n_layers_y", "not read there"])
rows, _ = run(nd() + "var : lat-L : scan : n_layers_x\nvar : lat-L : "
              "step : -1\nvar : lat-L : end_value : 10\nstart\n")
th = [r[0] for r in rows]
check("2-D n_layers_x scan 12 -> 10: three distinct rows, "
      "theta(12) - theta(10) = 2 layers x 3 columns x phi_b",
      len(th) == 3 and len(set(th)) == 3
      and abs(th[0] - th[2] - 0.06) < 1e-9, str(th))

print("[R27 #42] N-D box sizes from an alias or an exponential scan: "
      "pre-flight")
run_raises("2-D n_layers_x : #H#, alias step -0.5",
           nd().replace("n_layers_x : 12", "n_layers_x : #H#")
           + "alias : H : value : 12\nvar : alias-H : scan : value\n"
           "var : alias-H : step : -0.5\nvar : alias-H : end_value : "
           "11\nstart\n", ValueError,
           ["step 1", "H = 11.5", "n_layers_x", "whole number"])
run_raises("2-D exponential n_layers_x 12 -> 10, steps 30",
           nd() + "var : lat-L : scan : n_layers_x\nvar : lat-L : "
           "scale : exponential\nvar : lat-L : steps : 30\n"
           "var : lat-L : end_value : 10\nstart\n", ValueError,
           ["10.954451150103322", "whole number"])

print("[R27 #28] N-D per-axis ghost coordinates (Namics grammar)")
YW = L2.format(nx=10, ny=6) + "lat : L : {b} : surface\n" + POL + PRO
one_d = run("lat : L : n_layers : 6\nlat : L : lowerbound : surface\n"
            + POL.format(rng="lowerbound") + PRO + "start\n")
res = {}
for rng, b in (("1,lowerbound;10,lowerbound", "lowerbound_y"),
               ("1,0;10,0", "lowerbound_y"), ("lowerbound_y", "lowerbound_y"),
               ("1,upperbound;10,upperbound", "upperbound_y"),
               ("1,7;10,7", "upperbound_y"), ("upperbound_y", "upperbound_y"),
               ("lowerbound", "lowerbound_x"),
               ("lowerbound,1;lowerbound,6", "lowerbound_x"),
               ("0,1;0,6", "lowerbound_x")):
    res[rng] = run(YW.format(b=b, rng=rng) + "start\n")
t1 = one_d[0][0][0]
ylo = res["1,lowerbound;10,lowerbound"]
A = ylo[1][:, 2].reshape(10, 6)
check("y-lo wall == 10 x the 1-gradient wall (theta, Omega)",
      abs(ylo[0][0][0] - 10 * t1) < 1e-9
      and abs(ylo[0][0][1] - 10 * one_d[0][0][1]) < 1e-9,
      f"{ylo[0][0]} vs 10 x {one_d[0][0]}")
check("y-lo wall: phi independent of x", np.ptp(A, axis=0).max() < 1e-12,
      str(np.ptp(A, axis=0).max()))
check("y-lo wall: phi(y) == the 1-gradient profile",
      np.abs(A[0] - one_d[1][:, 1]).max() < 1e-9)
for a, b in (("1,0;10,0", "1,lowerbound;10,lowerbound"),
             ("lowerbound_y", "1,lowerbound;10,lowerbound"),
             ("1,7;10,7", "1,upperbound;10,upperbound"),
             ("upperbound_y", "1,upperbound;10,upperbound"),
             ("lowerbound,1;lowerbound,6", "lowerbound"),
             ("0,1;0,6", "lowerbound")):
    check(f"'{a}' == '{b}' bit for bit",
          res[a][0] == res[b][0] and np.array_equal(res[a][1], res[b][1]))
check("y-hi wall == the y-lo wall mirrored",
      abs(res["1,7;10,7"][0][0][0] - ylo[0][0][0]) < 1e-11)
cyl = run(L2.format(nx=5, ny=6).replace("geometry : flat",
                                        "geometry : cylindrical")
          + "lat : L : lowerbound_y : surface\n"
          + POL.format(rng="1,lowerbound;5,lowerbound") + PRO + "start\n")
C = cyl[1][:, 2].reshape(5, 6)
check("2-D cylindrical z-lo wall: phi(r, z) == the 1-gradient profile "
      "for every r", np.abs(C - one_d[1][:, 1][None, :]).max() < 1e-9,
      str(np.abs(C - one_d[1][:, 1][None, :]).max()))
d3 = run("lat : L : gradients : 3\nlat : L : geometry : flat\n"
         "lat : L : n_layers_x : 3\nlat : L : n_layers_y : 3\n"
         "lat : L : n_layers_z : 6\nlat : L : upperbound_z : surface\n"
         + POL.format(rng="1,1,upperbound;3,3,upperbound") + "start\n")
check("3-D z-hi wall == 9 x the 1-gradient wall",
      abs(d3[0][0][0] - 9 * t1) < 1e-9, f"{d3[0][0][0]} vs {9 * t1}")
for lbl, b, rng, words in (
        ("ghost coordinate on a mirror face", "upperbound_y", "1,0;10,0",
         ["lowerbound_y : surface", "frozen_range : 1,1;10,1"]),
        ("partial ghost patch", "lowerbound_y", "1,0;5,0",
         ["partial patch", "whole face"]),
        ("ghost + interior span", "lowerbound_y", "1,0;10,2",
         ["AND interior"]),
        ("ghost coordinates on two axes", "lowerbound_y", "0,0;0,0",
         ["more than one axis"]),
        ("a coordinate beyond the ghost plane", "lowerbound_y", "1,8;10,8",
         ["not within"]),
        ("a non-numeric token", "lowerbound_y", "foo,1;foo,6",
         ["'foo'", "per-axis tokens"]),
        ("a z face on a 2-D lattice", "lowerbound_y", "lowerbound_z",
         ["no z axis"])):
    raises(lbl, YW.format(b=b, rng=rng), ValueError, words)
try:
    build(nd(rng="lowerbound", extra=""))
    check("N-D ghost-wall advice names lowerbound_x", False, "built")
except ValueError as e:
    check("N-D ghost-wall advice names lowerbound_x and an N-D box",
          "lowerbound_x : surface" in str(e)
          and "frozen_range : 1;1" not in str(e)
          and "frozen_range : 1,1;1,3" in str(e), str(e))

# ---- review 27 Oct 2026: input and amount guards ------------------------


def run(text, name="r.in"):
    """run_file on `text`; returns (kal rows as floats, stdout). Raises
    what run_file raises."""
    p = write(text, name)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runner.run_file(p, verbose=False)
    rows = [[float(v) for v in ln.split()] for ln in
            open(p[:-3] + ".kal").read().strip().split("\n")[1:]]
    return rows, out.getvalue()


def run_raises(label, text, exc, words, name="r.in"):
    """run_file must raise BEFORE any .kal data row is written."""
    p = write(text, name)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run_file(p, verbose=False)
    except exc as e:
        kal = p[:-3] + ".kal"
        rows = (open(kal).read().strip().split("\n")[1:]
                if os.path.exists(kal) else [])
        check(label, all(w in str(e) for w in words) and not rows,
              f"{e} / {len(rows)} kal row(s)")
        return
    except Exception as e:                          # noqa: BLE001
        check(label, False, f"{type(e).__name__}: {e}")
        return
    check(label, False, "ran without complaint")


# a 30-layer planar box, frozen adsorbing wall S, (A)20 in monomeric W
AMT = """
lat : flat : n_layers : 30
lat : flat : geometry : planar
mon : S : freedom : frozen
mon : S : frozen_range : 1;1
mon : A : freedom : free
mon : W : freedom : free
mon : A : chi_S : -1
mon : A : chi_W : 0.5
mol : W : composition : (W)1
mol : W : freedom : solvent
mol : pol : composition : (A)20
newton : NN : tolerance : 1e-10
output : kal : append : false
kal : mol : pol : phibulk
kal : mol : pol : theta
kal : mol : pol : theta_exc
kal : mol : W : phibulk
kal : sys : NN : grand_potential
"""
FREE = "mol : pol : freedom : free\nmol : pol : phibulk : {}\n"
RESTR = "mol : pol : freedom : restricted\nmol : pol : {} : {}\n"
SCAN = ("var : mol-pol : scan : phibulk\nvar : mol-pol : step : {}\n"
        "var : mol-pol : end_value : 0\nstart\n")

print("[F32] amounts outside their domain")
for v in ("-5e-4", "1.5", "nan"):
    raises(f"free phibulk {v} refused at build", AMT + FREE.format(v),
           ValueError, ["mol pol", "phibulk", "out of range 0 .. 1"])
for k, v in (("theta", "-1"), ("n", "-0.05"), ("theta", "nan")):
    raises(f"restricted {k} {v} refused at build", AMT + RESTR.format(k, v),
           ValueError, ["mol pol", k, "out of range"])
run_raises("descending scan 0.01 by -0.0035 to 0: refused before row 0, "
           "naming the overshoot row", AMT + FREE.format("0.01")
           + SCAN.format("-0.0035"), ValueError,
           ["step 3", "value 4 of 4", "phibulk to -0.0005", "[0, 1]",
            "end_value 0.003"])
rows, _ = run(AMT + FREE.format("0.15") + SCAN.format("-0.05"))
check("scan 0.15 by -0.05 to 0 runs 4 rows, the last at exactly 0.0 "
      "(end snap), absent molecule (theta_exc 0, W phibulk 1, |Omega| "
      "< 1e-9)", len(rows) == 4 and rows[-1][0] == 0.0
      and rows[-1][2] == 0.0 and rows[-1][3] == 1.0
      and abs(rows[-1][4]) < 1e-9, str(rows[-1:]))
for start, step in ((0.3, -0.1), (0.7, -0.1), (0.15, -0.05)):
    st = read_input(write(AMT + FREE.format(start)
                          + SCAN.format(step)))[0]
    vals = runner._var_plan(st)[3]
    check(f"_var_plan {start} by {step} ends at exactly 0.0 (was "
          "round-off below zero)", vals[-1] == 0.0, repr(vals[-1]))
st = read_input(write(AMT + FREE.format("0.1") + "var : mol-pol : scan : "
                      "phibulk\nvar : mol-pol : step : 0.1\nvar : mol-pol :"
                      " end_value : 0.3\nstart\n"))[0]
check("_var_plan 0.1 by 0.1 ends at exactly 0.3",
      runner._var_plan(st)[3][-1] == 0.3, repr(runner._var_plan(st)[3]))
run_raises("restricted theta scan 2 by -0.3 to 0 (last row -0.1): refused",
           AMT + RESTR.format("theta", "2") + "var : mol-pol : scan : theta"
           "\nvar : mol-pol : step : -0.3\nvar : mol-pol : end_value : 0\n"
           "start\n", ValueError, ["theta to -0.1", ">= 0"])
rows, _ = run(AMT + FREE.format("0") + "start\n")
check("phibulk 0 still runs: theta_exc 0, W phibulk 1, |Omega| < 1e-9",
      rows[0][2] == 0.0 and rows[0][3] == 1.0 and abs(rows[0][4]) < 1e-9,
      str(rows))
rows, _ = run(AMT + RESTR.format("theta", "0") + "start\n")
check("restricted theta 0 still runs (theta 0, |Omega| < 1e-9)",
      rows[0][1] == 0.0 and abs(rows[0][4]) < 1e-9, str(rows))

# 3-state water + a weak-acid 10-mer at a wall
WEAKA = """
lat : flat : n_layers : 20
lat : flat : geometry : planar
mon : S : freedom : frozen
mon : S : frozen_range : 1;1
mon : W : freedom : free
mon : A : freedom : free
mon : Na : freedom : free
mon : Na : valence : 1
state : H3O : mon : W
state : H3O : valence : 1
state : H3O : alphabulk : {h}
state : H2O : mon : W
state : H2O : valence : 0
state : OH : mon : W
state : OH : valence : -1
state : AH : mon : A
state : AH : valence : 0
state : AM : mon : A
state : AM : valence : -1
reaction : w : equation : 2(H2O)=1(H3O)+1(OH)
reaction : w : pK : 17.5
reaction : a : equation : 1(AH)+1(H2O)=1(AM)+1(H3O)
reaction : a : pK : 6
mol : water : composition : (W)1
mol : water : freedom : solvent
mol : pol : composition : (A)10
mol : pol : freedom : restricted
mol : pol : theta : 1
mol : na : composition : (Na)1
mol : na : freedom : neutralizer
output : kal : append : false
kal : mon : W : alphabulk_H3O
kal : mon : A : alphabulk_AM
"""
print("[F44] alphabulk anchors outside (0, 1)")
for h in ("0", "-1e-6", "1.5", "1e6", "nan", "1"):
    raises(f"H3O alphabulk {h}: named refusal, no math error",
           WEAKA.format(h=h), ValueError, ["state H3O", "mon W",
                                           "alphabulk", "(0, 1)"])
raises("anchors H3O 0.6 + OH 0.6 (no autoprotolysis): the sum is refused",
       WEAKA.format(h="0.6").replace(
           "reaction : w : equation : 2(H2O)=1(H3O)+1(OH)\n"
           "reaction : w : pK : 17.5\n", "")
       + "state : OH : alphabulk : 0.6\n", ValueError,
       ["mon W", "H3O 0.6", "OH 0.6", "sum to 1.2"])
s44 = build(WEAKA.format(h="1e-6"))
al = {st.name: st.alphabulk for seg in s44.segments.values()
      for st in seg.states}
check("control H3O 1e-6: AM/AH = alpha_H2O (Ka 1e-6 at alpha_H3O 1e-6) "
      "to 1e-12", abs(al["AM"] / al["AH"] / al["H2O"] - 1) < 1e-12,
      str(al))
run_raises("exponential alphabulk scan 1e-3 -> 10: refused before row 0",
           WEAKA.format(h="1e-3") + "var : state-H3O : scan : alphabulk\n"
           "var : state-H3O : scale : exponential\nvar : state-H3O : "
           "steps : 1\nvar : state-H3O : end_value : 10\nstart\n",
           ValueError, ["state : H3O : alphabulk", "(0, 1)", "step 3"])

# a restricted brush (X)1(A)20 grafted by X
BRUSH = """
lat : flat : n_layers : 30
lat : flat : geometry : planar
lat : flat : lowerbound : surface
mon : S : freedom : frozen
mon : S : frozen_range : lowerbound
mon : A : freedom : free
mon : W : freedom : free
mol : W : composition : (W)1
mol : W : freedom : solvent
mol : pol : composition : (X)1(A)20
mol : pol : freedom : restricted
mol : pol : n : 0.05
output : kal : append : false
kal : mon : X : 1st_M_phi_z
kal : mol : pol : theta
"""
print("[F25] the range keyword follows the freedom")
run_raises("pinned + pinned_range 10;10 + a leftover frozen_range 1;1 "
           "(first start): refused", BRUSH + "mon : X : freedom : pinned\n"
           "mon : X : pinned_range : 10;10\nmon : X : frozen_range : 1;1\n"
           "start\n", ValueError, ["mon X", "frozen_range", "pinned"])
raises("pinned with only a frozen_range: requires a pinned_range",
       BRUSH + "mon : X : freedom : pinned\nmon : X : frozen_range : 10;10"
       "\n", ValueError, ["requires a pinned_range", "found frozen_range"])
raises("frozen with only a pinned_range: requires a frozen_range",
       BRUSH.replace("(X)1(A)20", "(A)20") + "mon : X : freedom : frozen\n"
       "mon : X : pinned_range : 1;1\n", ValueError,
       ["requires a frozen_range", "found pinned_range"])
two = (BRUSH.replace("(X)1(A)20", "(A)20") + "mon : X : freedom : frozen\n"
       "mon : X : frozen_range : 1;1\nstart\nmol : pol : composition : "
       "(X)1(A)20\nmon : X : freedom : pinned\nmon : X : pinned_range : "
       "10;10\nstart\n")
rows, _ = run(two)
check("frozen X at 1;1, then pinned X at 10;10: start 2 grafts at layer "
      "10 (1st_M_phi_z 9.5, was 0.5)", abs(rows[1][0] - 9.5) < 1e-12,
      str(rows))
s25 = build(two)
check("... and its X range_mask is [10] (layer 10, one ghost below)",
      np.nonzero(s25.segments["X"].range_mask)[0].tolist() == [10],
      str(np.nonzero(s25.segments["X"].range_mask)[0].tolist()))
back = (BRUSH + "mon : X : freedom : pinned\nmon : X : pinned_range : "
        "10;10\nstart\nmol : pol : composition : (A)20\nmon : X : "
        "freedom : frozen\nmon : X : frozen_range : 1;1\nstart\n")
s25b = build(back)
check("pinned at 10;10, then frozen at 1;1: the frozen site is layer 1",
      np.nonzero(s25b.segments["X"].range_mask)[0].tolist() == [1],
      str(np.nonzero(s25b.segments["X"].range_mask)[0].tolist()))

print("[F56] missing n_layers / composition")
check("substitute_aliases(None, {}) is None",
      substitute_aliases(None, {}) is None)
raises("1-D lat without n_layers", sub(WALL, "lat : flat : n_layers : 20\n",
                                       ""), ValueError,
       ["lat : n_layers is required"])
raises("1-D lat without n_layers, an alias present",
       sub(WALL, "lat : flat : n_layers : 20\n", "alias : R : value : 3\n"),
       ValueError, ["lat : n_layers is required"])
raises("mol without composition", sub(WALL, "mol : pol : composition : "
                                      "(A)25\n", ""), ValueError,
       ["mol pol", "composition is required"])
raises("gradients 2 without n_layers_y (N-D guard held)",
       sub(NDW, "lat : flat : n_layers_y : 3\n", ""), ValueError,
       ["n_layers_y"])

print("[F57] numbers and alias scope")


def newton_opts(lines):
    out = io.StringIO()
    st = read_input(write(WALL + lines))[-1]
    with contextlib.redirect_stdout(out):
        nopt = runner._newton_options(st)
        sopt = runner._super_options(st)
    return nopt, sopt, out.getvalue()


runner._NEWTON_NOTES.clear()
o, _, txt = newton_opts("newton : n : tolerance : .5\n")
check("newton tolerance '.5' falls back to the default, with a warning",
      "tolerance" not in o and "tolerance : .5 is not a number" in txt,
      f"{o} / {txt!r}")
o, _, txt = newton_opts("newton : n : tolerance : e-10\n")
check("newton tolerance 'e-10' falls back, with a warning",
      "tolerance" not in o and "is not a number" in txt, f"{o} / {txt!r}")
o, so, txt = newton_opts("alias : R : value : 1e-9\nalias : S : value : "
                         "1e-8\nnewton : n : tolerance : #R#\nnewton : n : "
                         "super_tolerance : #S#\n")
check("#R# in newton tolerance / super_tolerance is substituted",
      o.get("tolerance") == 1e-9 and so["tolerance"] == 1e-8
      and "warning" not in txt, f"{o} / {so} / {txt!r}")
o, _, txt = newton_opts("newton : n : tolerance : #TOL#\n")
check("undefined #TOL# in newton: default + a warning naming it",
      "tolerance" not in o and "no alias defines #TOL#" in txt,
      f"{o} / {txt!r}")
o, _, txt = newton_opts("newton : n : tolerance : 1e-9\n")
check("a plain number stays silent", o.get("tolerance") == 1e-9
      and txt == "", txt)
check("mol phibulk '.02' reads 0.02 (Python float, not the Namics rule)",
      build(sub(WALL, "phibulk : 0.02", "phibulk : .02"))
      .molecules["pol"].phibulk == 0.02)
raises("mon chi 'e-1' raises", sub(WALL, "chi_A : 0.4", "chi_A : e-1"),
       ValueError, ["e-1"])
s57 = build("alias : G : value : polymer_adsorption\n" + WALL
            + "sys : NN : initial_guess : #G#\n")
check("#G# in a sys value is substituted (initial_guess builds a seed)",
      s57.initial_guess() is not None)
raises("#R# in a mon chi is not substituted: raises as unparseable",
       "alias : R : value : 0.3\n" + sub(WALL, "chi_A : 0.4", "chi_A : #R#"),
       ValueError, ["#R#"])
raises("#R# in lat offset_first_layer still raises",
       "alias : R : value : 2\n" + WALL + "lat : flat : offset_first_layer :"
       " #R#\n", ValueError, ["#R#"])

print("[E] fixed surface potential (e.psi0/kT) on a refined planar lattice")
EL = """
lat : flat : n_layers : 30
lat : flat : geometry : planar
lat : flat : lowerbound : surface
lat : flat : FJC_choices : {fjc}
mon : S : freedom : frozen
mon : S : frozen_range : 1;1
mon : S : {wall}
mon : Na : freedom : free
mon : Na : valence : 1
mon : Cl : freedom : free
mon : Cl : valence : -1
mon : W : freedom : free
mol : cl : composition : (Cl)1
mol : cl : freedom : free
mol : cl : phibulk : 0.01
mol : na : composition : (Na)1
mol : na : freedom : neutralizer
mol : water : composition : (W)1
mol : water : freedom : solvent
newton : NN : tolerance : 1e-11
"""
for fjc in (5, 7):
    raises(f"e.psi0/kT at FJC_choices {fjc}: refused at System build",
           EL.format(fjc=fjc, wall="e.psi0/kT : -1"), NotImplementedError,
           ["FJC_choices", "refined", "valence"])
    try:
        build(EL.format(fjc=fjc, wall="e.psi0/kT : -1"))
    except NotImplementedError as e:
        check(f"FJC_choices {fjc}: the message no longer says 'interior'",
              "interior" not in str(e), str(e))
    builds(f"a fixed-charge wall (valence) at FJC_choices {fjc} still builds",
           EL.format(fjc=fjc, wall="valence : -0.2"))
sel = build(EL.format(fjc=3, wall="e.psi0/kT : -1"))
with contextlib.redirect_stdout(io.StringIO()):
    sel.solve()
check("the FJC_choices 3 electrode still converges with psi[1] = psi0",
      abs(sel.psi[1] + 1.0) < 1e-12
      and math.isfinite(sel.grand_potential()), f"psi[1] = {sel.psi[1]}")
mlw = EL.format(fjc=3, wall="e.psi0/kT : -1").replace(
    "frozen_range : 1;1", "frozen_range : 1;2")
try:
    with contextlib.redirect_stdout(io.StringIO()):
        build(mlw).solve()
    check("a two-layer electrode (1;2) is refused", False,
          "solved without complaint")
except NotImplementedError as e:
    check("a two-layer electrode (1;2) is refused as a multi-layer block",
          "multi-layer" in str(e), str(e))

print(f"all done ({failures} failure(s))")


def main():
    return failures


if __name__ == "__main__":
    sys.exit(1 if failures else 0)
