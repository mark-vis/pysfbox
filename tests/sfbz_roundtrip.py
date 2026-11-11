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
5. a kill INSIDE a member (zipfile's provisional local header, CRC 0 and
   compressed size 0, not yet rewritten; stored and deflated): that
   calculation is dropped, the earlier ones are intact, `repair` keeps
   them and each member's stored/deflated mode;
6. unit cases of the 11 Nov 2026 review: an int kal column with a NiN row
   (columns/load_many), strict JSON for a NaN residual, export_text next
   to the container refuses to replace a differing text file, a renamed
   container exports .kal and .pro under its own name, `output : sfbz`
   flag typos raise (before any solve), the sync after the first and
   after slow calculations (and not after every fast one),
   `load_many([])` and the CLI on a missing file;
7. a run that ends in an exception closes its container while the
   caller holds the traceback;
8. the plotting helpers: `stack` equals the individual profiles and
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
sfbz.SYNC_COST_FACTOR = 1e9
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


PROVISIONAL = """
import sys, os, types, zipfile
import numpy as np
sys.path.insert(0, {root!r})
from pysfbox import sfbz
sfbz.SYNC_SECONDS = 1e9
sfbz.SYNC_COST_FACTOR = 1e9
orig = zipfile._ZipWriteFile.close
def close(self):
    # the member's data is on disk, zipfile has not yet rewritten its
    # provisional local header (CRC 0, compressed size 0): kill here
    if self._zinfo.filename == {member!r}:
        self._fileobj.flush()
        os._exit(9)
    return orig(self)
zipfile._ZipWriteFile.close = close
lat = types.SimpleNamespace(gradients=1, geometry="planar", fjc=1, offset=0.)
s = types.SimpleNamespace(lat=lat, iterations=1, residual_norm=1e-9)
w = sfbz.Writer({base!r}, None, ["a:b:c"], compress={compress})
w.add(1, 0, s, kal_row=[("real", 0.)],
      pro=(["a"], [np.arange(50.)], np.zeros((1, 50))), pro_name="x0.pro")
w.sync()          # a directory exists; the next adds append over it
for i in range(1, 5):
    w.add(1, i, s, kal_row=[("real", float(i))],
          pro=(["a"], [np.arange(50.)], np.full((1, 50), float(i))),
          pro_name="x%d.pro" % i)
"""


def provisional_header(tmp):
    """S1: a kill between a member's data and zipfile's header rewrite.
    Stored members: the provisional header must not count as a complete
    empty member (it did: crc32(b"") == 0), so step 3 is dropped, steps
    0-2 are intact, --repair keeps 3 and the stored mode."""
    root = os.path.join(HERE, "..")
    oks = []
    for compress in (False, True):
        for member in ("1/3/step.json", "1/3/kal.npy", "1/3/pro.npy"):
            base = os.path.join(tmp, f"prov_{int(compress)}_"
                                + member.replace("/", "_"))
            subprocess.run([sys.executable, "-c", PROVISIONAL.format(
                root=root, base=base, member=member, compress=compress)])
            f = base + ".sfbz"
            try:
                with sfbz.load(f) as r:
                    n = len(r.steps)
                    good = n == 3 and all(
                        np.all(r.profile(1, i).values == i)
                        for i in range(n))
                kept = sfbz.repair(f)
                with zipfile.ZipFile(f) as z:
                    modes = {i.compress_type for i in z.infolist()}
                    names = z.namelist()
                good = (good and kept == 3 and len(names) == 1 + 4 * 3
                        and modes == ({zipfile.ZIP_DEFLATED} if compress
                                      else {zipfile.ZIP_STORED}))
            except Exception as e:                   # noqa: BLE001
                good = False
                print(f"     {member} compress={compress}: "
                      f"{type(e).__name__}: {e}")
            oks.append(good)
    ok = all(oks)
    print(f"{'ok  ' if ok else 'FAIL'} provisional header (kill inside a "
          f"member, stored + deflated): {sum(oks)}/{len(oks)} recovered "
          "3 intact calculations, repair keeps the mode")
    return ok


class _Sys:
    def __init__(self, res=1e-9):
        import types
        self.lat = types.SimpleNamespace(gradients=1, geometry="planar",
                                         fjc=1, offset=0.)
        self.iterations, self.residual_norm = 1, res


def unit_cases(tmp):
    """S2 int column with NiN, S3 export refuses to clobber, S4 flag
    typos, S5 sync timing, strict JSON, renamed export, load_many([]),
    CLI on a missing file."""
    import io
    import json
    from contextlib import redirect_stderr
    from pysfbox.inputreader import read_input
    res = {}
    # S2: an int kal column with a NiN row
    try:
        base = os.path.join(tmp, "ints")
        w = sfbz.Writer(base, None, ["s:n:i", "s:n:big", "s:n:r"])
        rows = [[("int", 3), ("int", 2**60), ("real", 1.)],
                [("nin", None), ("nin", None), ("real", 2.)],
                [("int", 5), ("int", 7), ("real", float("nan"))]]
        for i, row in enumerate(rows):
            w.add(1, i, _Sys(float("nan") if i == 2 else 1e-9), kal_row=row)
        w.close()
        with sfbz.load(base + ".sfbz") as r:
            try:
                c = r.columns()
                res["S2 int+NiN column"] = (
                    c["s:n:i"].dtype == float
                    and np.array_equal(c["s:n:i"], [3, np.nan, 5],
                                       equal_nan=True)
                    and c["s:n:big"].dtype == object
                    and list(c["s:n:big"]) == [2**60, None, 7])
            except Exception:                            # noqa: BLE001
                res["S2 int+NiN column"] = False
        with sfbz.load_many([base + ".sfbz"]) as cc:
            res["S2 load_many int+NiN"] = len(cc.kal["s:n:i"]) == 3
    except Exception as e:                      # noqa: BLE001
        res['S2' + f' raised {type(e).__name__}: {e}'] = False
    # strict JSON: no bare NaN in step.json, read back as nan
    try:
        with zipfile.ZipFile(base + ".sfbz") as z:
            text = z.read("1/2/step.json").decode()
        try:
            json.loads(text, parse_constant=lambda t: 1 / 0)
            strict = True
        except ZeroDivisionError:
            strict = False
        with sfbz.load(base + ".sfbz") as r:
            res["strict JSON residual"] = strict and np.isnan(
                r.steps[2]["residual"]) and "nan" in r.summary()
    except Exception as e:                      # noqa: BLE001
        res['strict JSON' + f' raised {type(e).__name__}: {e}'] = False
    # S3: export next to the container keeps a differing text file
    try:
        base = os.path.join(tmp, "part")
        w = sfbz.Writer(base, None, ["a:b:c"])
        w.add(2, 0, _Sys(), kal_row=[("real", 1.)])
        w.close()
        with open(base + ".kal", "w") as fp:
            fp.write("a:b:c\nthe complete table of the run\n")
        with sfbz.load(base + ".sfbz") as r:
            try:
                r.export_text()
                kept = False
            except FileExistsError:
                kept = True
            with open(base + ".kal") as fp:
                kept = kept and "complete table" in fp.read()
            os.makedirs(os.path.join(tmp, "partdir"))
            into = r.export_text(os.path.join(tmp, "partdir"))
            forced = r.export_text(overwrite=True)
            with open(base + ".kal") as fp:
                same = fp.read() == open(into[0]).read()
            r.export_text()                      # identical content: fine
        res["S3 export keeps a differing file"] = kept and same and forced
    except Exception as e:                      # noqa: BLE001
        res['S3' + f' raised {type(e).__name__}: {e}'] = False
    # renamed container: .kal and .pro both follow the container's name
    try:
        base = os.path.join(tmp, "orig")
        w = sfbz.Writer(base, None, ["a:b:c"])
        w.add(1, 0, _Sys(), kal_row=[("real", 1.)],
              pro=(["a"], [np.arange(3.)], np.zeros((1, 3))),
              pro_name="orig_1_0.pro")
        w.close()
        os.rename(base + ".sfbz", os.path.join(tmp, "renamed.sfbz"))
        os.makedirs(os.path.join(tmp, "ren"))
        with sfbz.load(os.path.join(tmp, "renamed.sfbz")) as r:
            names = sorted(os.path.basename(p) for p in
                           r.export_text(os.path.join(tmp, "ren")))
        res["renamed export names"] = names == ["renamed.kal",
                                                "renamed_1_0.pro"]
    except Exception as e:                      # noqa: BLE001
        res['renamed container' + f' raised {type(e).__name__}: {e}'] = False
    # S4: flag typos raise, the accepted spellings do not
    try:
        def opts(line):
            path = os.path.join(tmp, "flag.in")
            with open(path, "w") as fp:
                fp.write(line + "\n")
            return sfbz.options(read_input(path)[0])
        bad = 0
        for line in ("output : sfbz : text : flase",
                     "output : sfbz : write : ture",
                     "output : sfbz : compress : of"):
            try:
                opts(line)
            except ValueError as e:
                bad += line.split(" : ")[-1] in str(e)
        res["S4 flag typos raise"] = bad == 3 and opts(
            "output : sfbz : text : False") == (True, False, True) and opts(
            "output : sfbz : write : no") == (False, True, True)
        d = os.path.join(tmp, "typo")
        os.makedirs(d)
        with open(os.path.join(HERE, "brush_in_good_solvent.in")) as fp:
            text = fp.read()
        with open(os.path.join(d, "typo.in"), "w") as fp:
            fp.write("output : sfbz : text : flase\n" + text)
        from pysfbox.system import System
        solves = [0]
        solve = System.solve

        def counted(self, *a, **k):
            solves[0] += 1
            return solve(self, *a, **k)
        System.solve = counted
        try:
            run_file(os.path.join(d, "typo.in"), verbose=False)
            early = False
        except ValueError:
            early = solves[0] == 0 and not glob.glob(
                os.path.join(d, "*.kal"))
        finally:
            System.solve = solve
        res["S4 run_file refuses before solving"] = early
    except Exception as e:                      # noqa: BLE001
        res['S4' + f' raised {type(e).__name__}: {e}'] = False
    # S5: the first add syncs (the writer is created after a solve), and
    # an add after a calculation slower than SYNC_COST_FACTOR x the sync
    # cost syncs too, with the 2 s rule out of the way
    try:
        old = sfbz.SYNC_SECONDS
        sfbz.SYNC_SECONDS = 1e9
        try:
            base = os.path.join(tmp, "sync")
            w = sfbz.Writer(base, None, ["a:b:c"])
            valid = []
            for i in range(3):
                if i:
                    time.sleep(0.3)              # a "solve" >> 20 x sync cost
                w.add(1, i, _Sys(), kal_row=[("real", float(i))],
                      pro=(["a"], [np.arange(5.)], np.zeros((1, 5))),
                      pro_name="s.pro")
                try:
                    with zipfile.ZipFile(base + ".sfbz") as z:
                        valid.append(len(z.namelist()) == 1 + 4 * (i + 1))
                except zipfile.BadZipFile:
                    valid.append(False)
            # a burst of fast adds does not sync each time
            n = [0]
            orig = sfbz.Writer.sync

            def counting(self):
                n[0] += 1
                orig(self)
            sfbz.Writer.sync = counting
            try:
                for i in range(3, 203):
                    w.add(1, i, _Sys(), kal_row=[("real", float(i))])
            finally:
                sfbz.Writer.sync = orig
            w.close()
            res["S5 sync after slow calculations"] = all(valid) and n[0] < 20
        finally:
            sfbz.SYNC_SECONDS = old
    except Exception as e:                      # noqa: BLE001
        res['S5' + f' raised {type(e).__name__}: {e}'] = False
    # load_many([]) and the CLI on a missing file: clear messages
    try:
        sfbz.load_many([])
        res["load_many([]) message"] = False
    except ValueError as e:
        res["load_many([]) message"] = "no files" in str(e)
    except Exception:                           # noqa: BLE001
        res["load_many([]) message"] = False
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = sfbz.main([os.path.join(tmp, "missing.sfbz")])
        res["CLI missing file"] = (code == 2
                                   and "no such file" in err.getvalue())
    except Exception:                           # noqa: BLE001
        res["CLI missing file"] = False
    for k, v in res.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    return all(res.values())


EXC_INPUT = """lat : flat : n_layers : 10
lat : flat : lowerbound : mirror
lat : flat : upperbound : mirror
mon : W : freedom : free
mol : water : composition : (W)1
mol : water : freedom : solvent
output : sfbz : write : true
kal : mol : water : phibulk
output : pro : append : false
pro : mon : W : phi
"""


def exception_closes(tmp):
    """S6: a run that ends in an exception still closes its container
    (the caller may keep the traceback alive)."""
    import pysfbox.runner as runner
    d = os.path.join(tmp, "exc")
    os.makedirs(d)
    path = os.path.join(d, "exc.in")
    with open(path, "w") as fp:
        fp.write(EXC_INPUT + "start\nlat : flat : n_layers : 12\nstart\n"
                 "lat : flat : n_layers : 14\nstart\n")
    calls = [0]
    orig = runner._Outputs.write

    def write_then_fail(self, *a, **k):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("emulated failure in start 3")
        return orig(self, *a, **k)
    runner._Outputs.write = write_then_fail
    old = sfbz.SYNC_SECONDS, getattr(sfbz, "SYNC_COST_FACTOR", 20.0)
    sfbz.SYNC_SECONDS = sfbz.SYNC_COST_FACTOR = 1e9
    held = None
    try:
        try:
            run_file(path, verbose=False)
        except RuntimeError as e:
            held = e                        # keep the traceback alive
    finally:
        runner._Outputs.write = orig
        sfbz.SYNC_SECONDS, sfbz.SYNC_COST_FACTOR = old
    try:
        with zipfile.ZipFile(os.path.join(d, "exc.sfbz")) as z:
            # start 1 synced (the first add does), start 2 did not
            ok = held is not None and "2/0/step.json" in z.namelist()
    except zipfile.BadZipFile:
        ok = False
    print(f"{'ok  ' if ok else 'FAIL'} exception: container closed while "
          "the traceback is held")
    del held
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
                # next to the run's own (identical) text files: no refusal
                beside = len(r.export_text()) == len(written)
            same = (beside and len(written) == len(texts) and texts and all(
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
        fails += not provisional_header(tmp)
        fails += not unit_cases(tmp)
        fails += not exception_closes(tmp)
        fails += not helpers(tmp, made)
    print("sfbz_roundtrip:", "PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
