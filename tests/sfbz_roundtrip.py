"""Guard for the experimental .sfbz container (pysfbox/sfbz.py): for a few
suite inputs (1-gradient scan, multi-start, N-D, string/int/NiN kal
cells) run with `output : sfbz : write : true` and check that

1. the .kal/.pro text files are still written, and the container's
   `export_text` regenerates every one of them BYTE-identically;
2. `output : sfbz : text : false` writes the container only, with the
   same content;
3. plain `numpy.load` reads the file without PySFBox;
4. a writer process KILLED between two syncs leaves a file without zip
   directory: the Reader recovers every complete calculation, `repair`
   makes it a proper archive again;
5. the plotting helpers: `stack` equals the individual profiles and
   carries the scan values, `load_many` is the row-wise union of its
   files, a box-size scan refuses to stack.

Run: python tests/sfbz_roundtrip.py
"""
import filecmp
import glob
import os
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pysfbox import sfbz                                    # noqa: E402
from pysfbox.runner import run_file                         # noqa: E402

CASES = ["brush_in_good_solvent", "search_laplace", "brush2d_flat",
         "weak_titration"]


def run(case, tmp, extra):
    d = os.path.join(tmp, case + extra.replace(" ", "").replace(":", "_"))
    os.makedirs(d)
    with open(os.path.join(HERE, case + ".in")) as fp:
        text = fp.read()
    path = os.path.join(d, case + ".in")
    with open(path, "w") as fp:
        fp.write("output : sfbz : write : true\n" + extra + "\n" + text)
    run_file(path, verbose=False)
    return d


KILLED = """
import sys, time, types
import numpy as np
sys.path.insert(0, {root!r})
from pysfbox import sfbz
sfbz.SYNC_SECONDS = 1e9
lat = types.SimpleNamespace(gradients=1, geometry="planar", fjc=1, offset=0.)
s = types.SimpleNamespace(lat=lat, iterations=1, residual_norm=1e-9)
w = sfbz.Writer({base!r}, None, ["a:b:c"])
for i in range(100000):
    w.add(1, i, s, kal_row=[("real", float(i))],
          pro=(["a"], [np.arange(5000.)], np.full((1, 5000), float(i))),
          pro_name="x.pro")
    time.sleep(0.01)
"""


def killed_writer(tmp):
    base = os.path.join(tmp, "killed")
    root = os.path.join(HERE, "..")
    p = subprocess.Popen([sys.executable, "-c",
                          KILLED.format(root=root, base=base)])
    time.sleep(3.0)
    p.kill()
    p.wait()
    try:
        zipfile.ZipFile(base + ".sfbz")
        broken = False
    except zipfile.BadZipFile:
        broken = True
    with sfbz.load(base + ".sfbz") as r:
        n = len(r.steps)
        good = n > 5 and np.array_equal(r.kal["a:b:c"], np.arange(n)) \
            and all(np.all(r.profile(1, i).values == i) for i in range(n))
    fixed = sfbz.repair(base + ".sfbz") == n \
        and sfbz.repair(base + ".sfbz") is None \
        and len(np.load(base + ".sfbz").files) == 1 + 4 * n
    ok = broken and good and fixed
    print(f"{'ok  ' if ok else 'FAIL'} killed writer: no directory={broken}, "
          f"{n} calculations recovered intact={good}, repair={fixed}")
    return ok


def helpers(tmp, made):
    # a same-grid scan with profiles: the titration plus one pro line
    d = run("weak_titration", tmp, "pro : mol : * : phi")
    with sfbz.load(os.path.join(d, "weak_titration.sfbz")) as r:
        lab = r.steps[0]["pro_labels"][0]
        st = r.stack(lab)
        n = len(r.steps)
        ok = (n > 1 and st.values.shape == (n, len(st.x))
              and st.scan_label == "state:H3O:alphabulk"
              and np.all(np.isfinite(st.scan))
              and np.array_equal(st.scan, r.scan)
              and all(np.array_equal(st.values[i], r.profile(1, i)[lab])
                      for i in range(n)))
    with sfbz.load_many(made) as c:
        rows = sum(len(r.start) for r in c)
        ok = ok and len(c.file) == rows and all(
            len(v) == rows for v in c.kal.values())
        for r, f in zip(c, c.files):
            mine = c.file == f
            cols = r.columns()
            for lab in r.kal_labels:
                if cols[lab].dtype != object:       # numeric columns
                    ok = ok and np.array_equal(
                        np.asarray(c.kal[lab][mine], dtype=float),
                        r.kal[lab], equal_nan=True)
    d = run("charged_nlayers_scan", tmp, "")
    with sfbz.load(os.path.join(d, "charged_nlayers_scan.sfbz")) as r:
        try:
            r.stack(r.steps[0]["pro_labels"][0])
            refused = False
        except ValueError:
            refused = True
    ok = bool(ok and refused)
    print(f"{'ok  ' if ok else 'FAIL'} helpers: stack/load_many consistent, "
          f"box-size scan refused={refused}")
    return ok


def main():
    fails = 0
    made = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            d = run(case, tmp, "")
            made.append(os.path.join(d, case + ".sfbz"))
            texts = sorted(glob.glob(os.path.join(d, "*.kal"))
                           + glob.glob(os.path.join(d, "*.pro")))
            exp = os.path.join(d, "export")
            os.makedirs(exp)
            with sfbz.load(os.path.join(d, case + ".sfbz")) as r:
                written = r.export_text(exp)
            same = (len(written) == len(texts) and texts and all(
                filecmp.cmp(t, os.path.join(exp, os.path.basename(t)),
                            shallow=False) for t in texts))
            d2 = run(case, tmp, "output : sfbz : text : false")
            only = not (glob.glob(os.path.join(d2, "*.kal"))
                        + glob.glob(os.path.join(d2, "*.pro")))
            a = np.load(os.path.join(d, case + ".sfbz"))
            b = np.load(os.path.join(d2, case + ".sfbz"))
            keys = [k for k in a.files if not k.endswith(".json")]
            equal = (a.files == b.files and all(
                np.array_equal(a[k], b[k], equal_nan=True) for k in keys))
            ok = same and only and equal
            fails += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {case}: {len(texts)} text "
                  f"files byte-identical={same}, text:false container-only="
                  f"{only}, numpy.load arrays equal={equal}")
        fails += not killed_writer(tmp)
        fails += not helpers(tmp, made)
    print("sfbz_roundtrip:", "PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
