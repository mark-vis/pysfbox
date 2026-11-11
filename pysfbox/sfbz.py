"""The `.sfbz` container (10 Nov 2026, format version 1) -- one binary,
self-describing output file per input, next to (or instead of) the Namics
`.kal` / `.pro` text files.

Why. A var scan of J steps writes one `.kal` and J `.pro` text files; every
number is formatted digit by digit in a Python loop (the `.pro` at %.20g,
~24 bytes per value) and parsed back the same way. The container holds the
same numbers as raw float64 in ONE file:

    base.sfbz                 a plain zip archive (append-only)
      header.json             format version, PySFBox version, the input
                              file's text, the kal column labels
      <start>/<step>/step.json    what this calculation was: lattice,
                              iterations, residual, kal value kinds, pro
                              labels, the name its .pro file has / would have
      <start>/<step>/kal.npy  the kal row, float64 (NaN where not a real)
      <start>/<step>/pro.npy  the profiles, (ncols, rows) on one gradient,
                              (ncols, nx, ny[, nz]) on two or three
      <start>/<step>/x.npy [y.npy, z.npy]   the coordinate axes

Design rules.
- Pure NumPy + stdlib (Invariant 3). Because every array member is an
  ordinary `.npy`, the file is an `.npz`: `numpy.load("base.sfbz")` reads
  it WITHOUT PySFBox (`d["1/0/pro"]`), and so does any zip tool plus any
  npy reader (Julia NPZ.jl, MATLAB readNPY, R RcppCNPy).
- Append-only and crash-safe: each finished calculation is appended;
  the archive's directory is written out (`sync`) at the end of an
  `add` when the previous sync is more than SYNC_SECONDS old OR the
  calculation was slow compared with a sync (more than SYNC_COST_FACTOR
  x the measured sync cost since the previous `add`; the first one
  counts as slow), at the end of the run (also when it ends in an
  exception) and when the writer is garbage-collected. Nothing runs
  during a solve (no thread, no timer), so the window in which the file
  on disk has no directory is: from an `add` that did not sync -- a
  calculation shorter than SYNC_COST_FACTOR x the sync cost, within
  SYNC_SECONDS of the last sync -- until the next `add` or the end of the
  run, i.e. at most one solve; a run of slow calculations syncs after
  every one, with an overhead below ~1/SYNC_COST_FACTOR of the run time.
  A process KILLED in that window leaves a file without directory (zip
  appends over the old one); the Reader then reads the members back from
  their local headers (`_recover`: every complete calculation, typically
  all but the one being written; a member whose header is still
  zipfile's provisional one is incomplete) and `--repair` rewrites the
  file for other tools. (Closing after EVERY step was measured first: the
  directory rewrite is O(members), 23 s for a 2000-step scan against
  1.9 s for the text files.) Nothing is ever rewritten.
- Lossless and Namics-compatible: `export_text` regenerates the `.kal`
  and every `.pro` BYTE-identically to the text writers (same
  output.format_kal_row / write_pro_text), so the text formats stay the
  product (Invariant 1) and the container is checked against them.
- N-D profiles keep their shape; the axes are stored once per axis, not
  as coordinate columns repeated on every row.

Input surface (opt-in):

    output : sfbz : write : true      write base.sfbz
    output : sfbz : text : false      ... and skip the .kal/.pro text files
    output : sfbz : compress : false  store instead of deflate

Reading:

    from pysfbox import sfbz
    r = sfbz.load("base.sfbz")
    r.kal["sys:NN:grand_potential"]          # one value per calculation
    p = r.profile(start=1, step=3)           # p.x, p["mol:A:phi"]
    r.export_text()                          # base.kal, base_*.pro (named
                                             #   after the container; keeps
                                             #   differing files, see doc)
    s = r.stack("mol:A:phi")                 # a whole scan as one array:
    plt.plot(s.x, s.values.T)                #   s.values (steps, sites),
                                             #   s.scan the scanned values
    r.frame()                                # the kal table as a pandas
                                             #   DataFrame (if installed)
    c = sfbz.load_many("ladder_*.sfbz")      # several files as ONE table:
    c.kal["sys:NN:grand_potential"], c.file  #   c.stack(...), c.frame()

    python -m pysfbox.sfbz base.sfbz              summary
    python -m pysfbox.sfbz base.sfbz --export [DIR]

Not in the container yet: `.dist` and `.sk` (still text).
"""

import glob
import io
import json
import os
import re
import struct
import sys
import time
import weakref
import zipfile
import zlib

import numpy as np

FORMAT = "pysfbox-sfbz"
VERSION = 1
SYNC_SECONDS = 2.0
# a sync is also written after any calculation that took longer than
# SYNC_COST_FACTOR x the measured cost of a sync (directory write + the
# reopen it implies): the sync overhead stays below ~1/SYNC_COST_FACTOR of
# the run time, so fast scans do not go quadratic
SYNC_COST_FACTOR = 20.0

_KIND = {"real": "r", "int": "i", "string": "s", "nin": "n"}
_KIND_BACK = {v: k for k, v in _KIND.items()}


def sfbz_path(base):
    return base + ".sfbz"


_TRUE = ("true", "1", "yes")
_FALSE = ("false", "0", "no")


def options(settings):
    """(write, text, compress) from the `output : sfbz : ...` block. A
    value that is neither true nor false raises (a typo such as
    `text : flase` must not silently drop the text output)."""
    from .inputreader import get_blocks, last

    def flag(params, key, default):
        v = last(params, key)
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(
            f"output : sfbz : {key} : {v} -- not a true/false value "
            f"(accepted: {' / '.join(_TRUE)} or {' / '.join(_FALSE)})")

    for name, params in get_blocks(settings, "output"):
        if name == "sfbz":
            return (flag(params, "write", True), flag(params, "text", True),
                    flag(params, "compress", True))
    return False, True, True


def _npy_bytes(a):
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(a, dtype=np.float64),
                              allow_pickle=False)
    return buf.getvalue()


def _close(holder):
    if holder[0] is not None:
        holder[0].close()
        holder[0] = None


class Writer:
    """Appends one calculation at a time; the file is a valid archive
    after every `sync` (automatic, see the module text) and `close`."""

    def __init__(self, base, input_path=None, kal_labels=(), compress=True):
        from . import __version__
        self.path = sfbz_path(base)
        self.compress = (zipfile.ZIP_DEFLATED if compress
                         else zipfile.ZIP_STORED)
        text = None
        if input_path is not None and os.path.exists(input_path):
            with open(input_path, errors="replace") as fp:
                text = fp.read()
        header = {"format": FORMAT, "version": VERSION,
                  "pysfbox": __version__,
                  "input_file": (os.path.basename(input_path)
                                 if input_path else None),
                  "input_text": text,
                  "kal_labels": list(kal_labels),
                  # the stem the text files are named after (export_text
                  # renames them after a renamed container)
                  "base": os.path.basename(base)}
        with zipfile.ZipFile(self.path, "w", self.compress) as z:
            z.writestr("header.json", json.dumps(header, indent=1))
        self._holder = [None]
        self._synced = time.monotonic()
        self._last_add = None       # end of the previous add
        self._sync_cost = 0.0       # duration of the last close
        self._reopen_cost = 0.0     # ... and of the reopen after it
        weakref.finalize(self, _close, self._holder)

    def sync(self):
        """Write the archive directory: everything added so far is on
        disk and readable."""
        t = time.monotonic()
        was_open = self._holder[0] is not None
        _close(self._holder)
        self._synced = time.monotonic()
        if was_open:
            self._sync_cost = self._synced - t

    close = sync

    def add(self, start, step, system, kal_row=None, pro=None,
            pro_name=None, scan=None):
        """kal_row: output.collect_kal result (or None: no kal row);
        pro: output.collect_pro result (or None); pro_name: basename of
        the .pro text file this dump has / would have; scan: (label,
        value) of the var step."""
        lat = system.lat
        res = float(getattr(system, "residual_norm", float("nan")))
        meta = {"start": int(start), "step": int(step),
                "iterations": int(getattr(system, "iterations", 0)),
                # strict JSON: a non-finite residual is stored as null
                "residual": res if np.isfinite(res) else None,
                "lattice": {
                    "gradients": int(getattr(lat, "gradients", 1)),
                    "geometry": str(getattr(lat, "geometry", "")),
                    "fjc": int(getattr(lat, "fjc", 1)),
                    "offset": float(getattr(lat, "offset", 0.0))}}
        if scan is not None:
            meta["scan"] = {"label": scan[0], "value": str(scan[1])}
        members = []
        pre = f"{int(start)}/{int(step)}/"
        if kal_row is not None:
            vals = np.full(len(kal_row), np.nan)
            strings = {}
            for j, (typ, val) in enumerate(kal_row):
                if typ in ("real", "int"):
                    vals[j] = val
                    if typ == "int" and int(vals[j]) != val:
                        strings[str(j)] = str(val)   # beyond 2**53: exact
                elif typ == "string":
                    strings[str(j)] = str(val)
            meta["kal_kinds"] = "".join(_KIND[t] for t, _ in kal_row)
            if strings:
                meta["kal_strings"] = strings
            members.append((pre + "kal.npy", _npy_bytes(vals)))
        if pro is not None:
            labels, axes, values = pro
            meta["pro_labels"] = list(labels)
            meta["pro_name"] = pro_name
            meta["axes"] = ["x", "y", "z"][:len(axes)]
            for nm, ax in zip(meta["axes"], axes):
                members.append((pre + nm + ".npy", _npy_bytes(ax)))
            members.append((pre + "pro.npy", _npy_bytes(values)))
        if self._holder[0] is None:
            t = time.monotonic()
            self._holder[0] = zipfile.ZipFile(self.path, "a", self.compress)
            self._reopen_cost = time.monotonic() - t
        z = self._holder[0]
        z.writestr(pre + "step.json", json.dumps(meta))
        for name, data in members:
            z.writestr(name, data)
        # Sync when the last sync is more than SYNC_SECONDS old, and after
        # every calculation that was slow compared with a sync (more than
        # SYNC_COST_FACTOR x its cost since the previous add): the next
        # one is probably slow too, and nothing runs during its solve to
        # write the directory. The first calculation counts as slow (the
        # writer is created after it was solved). Syncing after EVERY add
        # costs O(members) each, quadratic over a fast scan.
        now = time.monotonic()
        cost = self._sync_cost + self._reopen_cost
        slow = (self._last_add is None
                or now - self._last_add > SYNC_COST_FACTOR * cost)
        self._last_add = now
        if slow or now - self._synced > SYNC_SECONDS:
            self.sync()


def _complete(name, raw):
    """A recovered member's content is what its name says: a .npy array
    header, a JSON document."""
    if name.endswith(".npy"):
        return raw[:6] == b"\x93NUMPY"
    if name.endswith(".json"):
        try:
            json.loads(raw)
        except ValueError:
            return False
    return True


def _recover_members(path):
    """[(name, bytes, method)] of an archive whose directory is missing
    (the writing process was killed between two syncs: zip appends new
    members over the old directory). Every member starts with a local
    header carrying its name, sizes and CRC, so the complete ones are
    read back in file order, stopping at the first incomplete member. A
    member counts as complete only when its data has the uncompressed
    size its header announces, the CRC matches and the content parses:
    zipfile writes a PROVISIONAL header (CRC 0, compressed size 0) before
    the data and rewrites it after, so a kill in between leaves a header
    that would otherwise read as a complete empty member."""
    out = []
    with open(path, "rb") as fp:
        blob = fp.read()
    pos = 0
    while blob[pos:pos + 4] == b"PK\x03\x04" and pos + 30 <= len(blob):
        (_, flags, method, _, _, crc, csize, usize, nlen,
         xlen) = struct.unpack("<HHHHHIIIHH", blob[pos + 4:pos + 30])
        name = blob[pos + 30:pos + 30 + nlen].decode()
        extra = blob[pos + 30 + nlen:pos + 30 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize):        # zip64: sizes in the extra
            k = 0
            while k + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[k:k + 4])
                if hid == 0x0001:
                    vals = list(struct.unpack(
                        "<" + "Q" * (hlen // 8), extra[k + 4:k + 4 + hlen]))
                    if usize == 0xFFFFFFFF and vals:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF and vals:
                        csize = vals.pop(0)
                    break
                k += 4 + hlen
        start = pos + 30 + nlen + xlen
        data = blob[start:start + csize]
        if flags & 0x08 or len(data) < csize:
            break                         # streamed or truncated member
        try:
            raw = zlib.decompress(data, -15) if method == 8 else data
        except zlib.error:
            break
        if (len(raw) != usize or zlib.crc32(raw) != crc
                or not _complete(name, raw)):
            break                         # provisional header / garbage
        out.append((name, raw, method))
        pos = start + csize
    return out


def _recover(path):
    """{name: bytes} of the complete members of a directory-less file
    (see _recover_members)."""
    return {n: raw for n, raw, _ in _recover_members(path)}


def _complete_steps(members):
    """The step prefixes ('1/3/') whose step.json and every array it
    announces were recovered, in file order, up to the first incomplete
    step."""
    steps = []
    for n in members:
        if n.endswith("/step.json"):
            pre = n[:-len("step.json")]
            want = _step_members(json.loads(members[n]))
            if not all(pre + w in members for w in want):
                break
            steps.append(pre)
    return steps


def repair(path):
    """Rewrite a killed run's container with a proper directory (so
    `numpy.load` and zip tools read it again). Returns the number of
    calculations kept; a healthy file is left alone. Each member keeps
    its stored / deflated mode."""
    try:
        with zipfile.ZipFile(path):
            return None
    except zipfile.BadZipFile:
        pass
    found = _recover_members(path)
    members = {n: raw for n, raw, _ in found}
    method = {n: m for n, _, m in found}
    if "header.json" not in members:
        raise ValueError(f"{path}: nothing recoverable")
    steps = _complete_steps(members)

    def put(z, name):
        z.writestr(name, members[name],
                   compress_type=(zipfile.ZIP_DEFLATED if method[name] == 8
                                  else zipfile.ZIP_STORED))

    tmp = path + ".repair"
    with zipfile.ZipFile(tmp, "w") as z:
        put(z, "header.json")
        for pre in steps:
            for w in ["step.json"] + _step_members(
                    json.loads(members[pre + "step.json"])):
                put(z, pre + w)
    os.replace(tmp, path)
    return len(steps)


def _step_members(meta):
    """The array members a step.json announces."""
    want = ["kal.npy"] if "kal_kinds" in meta else []
    if "pro_labels" in meta:
        want += [a + ".npy" for a in meta["axes"]] + ["pro.npy"]
    return want


class Profile:
    """One profile dump: `axes` (list of 1-D coordinate arrays), `x`
    (= axes[0]), `labels`, `values` (ncols, *shape); p["mol:A:phi"]."""

    def __init__(self, labels, axes, values, meta):
        self.labels, self.axes, self.values, self.meta = (labels, axes,
                                                          values, meta)
        self.x = axes[0] if axes else None

    def __getitem__(self, label):
        return self.values[self.labels.index(label)]

    def __contains__(self, label):
        return label in self.labels


class Stack:
    """One profile over many calculations: `values` (ncalc, *shape),
    `axes` / `x` (shared), and one entry per calculation in `start`,
    `step`, `scan` (the scanned value as a float, NaN outside a scan or
    when it is not a number), `scan_label`, and `file` (load_many)."""

    def __init__(self, label, profs, files=None):
        shapes = {p.values.shape[1:] for p in profs}
        if len(shapes) > 1 or any(
                not np.array_equal(a, b) for p in profs[1:]
                for a, b in zip(p.axes, profs[0].axes)):
            raise ValueError(
                f"stack('{label}'): the calculations do not share one grid "
                f"(shapes {sorted(shapes)}) -- a box-size scan cannot be "
                "one array; pass start=... to select one start block, or "
                "loop over profiles()")
        self.label = label
        self.axes = profs[0].axes
        self.x = self.axes[0]
        self.values = np.array([p[label] for p in profs])
        self.start = np.array([p.meta["start"] for p in profs])
        self.step = np.array([p.meta["step"] for p in profs])
        self.scan = np.array([_scan_value(p.meta) for p in profs])
        labels = {p.meta["scan"]["label"] for p in profs if "scan" in p.meta}
        self.scan_label = labels.pop() if len(labels) == 1 else None
        self.file = None if files is None else np.array(files)

    def __len__(self):
        return len(self.values)


def _scan_value(meta):
    """The scanned value of a step.json as a float ('71/3' included)."""
    text = meta.get("scan", {}).get("value")
    try:
        if text is not None and "/" in text:
            a, b = text.split("/")
            return float(a) / float(b)
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _frame(columns):
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "frame() needs pandas (pip install pandas); without it the "
            "same columns are NumPy arrays in .kal / .start / .step / "
            ".scan") from None
    return pandas.DataFrame(columns)


class Reader:
    """A loaded container. `header`, `steps` (list of step.json dicts in
    file order), `kal_labels`, `kal` ({label: float64 array, one entry
    per calculation with a kal row; NaN where the cell is a string or
    NiN}), `kal_cell(row, label)` for the typed value."""

    def __init__(self, path):
        self.path = path
        self.recovered = False
        try:
            self._z = zipfile.ZipFile(path)
            self._names = self._z.namelist()
            self._read = self._z.read
        except zipfile.BadZipFile:
            # killed between two syncs: read the members directly
            members = _recover(path)
            if "header.json" not in members:
                raise
            self._z, self.recovered = None, True
            self._names, self._read = list(members), members.__getitem__
        self.header = json.loads(self._read("header.json"))
        if self.header.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        if self.header.get("version", 0) > VERSION:
            raise ValueError(
                f"{path}: container version {self.header['version']} is "
                f"newer than this PySFBox reads ({VERSION})")
        if self.recovered:
            # a step whose arrays did not all reach the disk is dropped
            names = [p + "step.json" for p in _complete_steps(members)]
        else:
            names = [n for n in self._names if n.endswith("/step.json")]
        self.steps = [json.loads(self._read(n)) for n in names]
        for s in self.steps:
            if s.get("residual") is None:       # stored as null
                s["residual"] = float("nan")
        self._by_step = {}
        for s in self.steps:
            self._by_step.setdefault((s["start"], s["step"]), s)
        if self.recovered:
            print(f"  note: {path} has no archive directory (the run was "
                  f"killed); {len(self.steps)} complete calculation(s) "
                  "recovered -- `python -m pysfbox.sfbz FILE --repair` "
                  "rewrites it")
        self.kal_labels = self.header["kal_labels"]
        self._kal_steps = [s for s in self.steps if "kal_kinds" in s]
        rows = [self._array(s, "kal") for s in self._kal_steps]
        table = (np.vstack(rows) if rows
                 else np.empty((0, len(self.kal_labels))))
        self.kal_table = table
        self.kal = {lab: table[:, j]
                    for j, lab in enumerate(self.kal_labels)}

    def close(self):
        if self._z is not None:
            self._z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _array(self, step, name):
        data = self._read(f"{step['start']}/{step['step']}/{name}.npy")
        return np.lib.format.read_array(io.BytesIO(data),
                                        allow_pickle=False)

    def kal_row(self, i):
        """Row i as output.collect_kal's list of (kind, value)."""
        s = self._kal_steps[i]
        strings = s.get("kal_strings", {})
        row = []
        for j, c in enumerate(s["kal_kinds"]):
            kind = _KIND_BACK[c]
            if kind == "real":
                row.append((kind, float(self.kal_table[i, j])))
            elif kind == "int":
                row.append((kind, int(strings[str(j)]) if str(j) in strings
                            else int(self.kal_table[i, j])))
            elif kind == "string":
                row.append((kind, strings[str(j)]))
            else:
                row.append((kind, None))
        return row

    def kal_cell(self, i, label):
        return self.kal_row(i)[self.kal_labels.index(label)][1]

    def _profile_of(self, s):
        axes = [self._array(s, nm) for nm in s["axes"]]
        return Profile(s["pro_labels"], axes, self._array(s, "pro"), s)

    def profile(self, start=1, step=0):
        s = self._by_step.get((start, step))
        if s is None or "pro_labels" not in s:
            raise KeyError(f"{self.path}: no profile for start {start}, "
                           f"step {step}")
        return self._profile_of(s)

    def profiles(self, start=None):
        """Every stored profile dump, in file order (of one start block
        when `start` is given)."""
        return [self._profile_of(s) for s in self.steps
                if "pro_labels" in s and start in (None, s["start"])]

    def stack(self, label, start=None):
        """The profile `label` of every calculation that stored it (of
        one start block when `start` is given) as one Stack."""
        profs = [p for p in self.profiles(start) if label in p]
        if not profs:
            have = sorted({lab for s in self.steps
                           for lab in s.get("pro_labels", [])})
            raise KeyError(f"{self.path}: no profile '{label}'"
                           + (f" in start {start}" if start else "")
                           + f"; stored: {', '.join(have) or 'none'}")
        return Stack(label, profs)

    @property
    def start(self):
        """Start block of every kal row."""
        return np.array([s["start"] for s in self._kal_steps], dtype=int)

    @property
    def step(self):
        """Scan step of every kal row."""
        return np.array([s["step"] for s in self._kal_steps], dtype=int)

    @property
    def scan(self):
        """Scanned value of every kal row (NaN outside a scan)."""
        return np.array([_scan_value(s) for s in self._kal_steps])

    def columns(self):
        """The kal table as {name: column}: start, step, scan, then the
        kal labels. A column of strings is object (None for NiN); a column
        of ints is int64, or float64 with NaN when some row is NiN, or
        object (None for NiN) when a value does not fit exactly; every
        other column is float64 with NaN for NiN."""
        cols = {"start": self.start, "step": self.step, "scan": self.scan}
        rows = [self.kal_row(i) for i in range(len(self._kal_steps))]
        for j, lab in enumerate(self.kal_labels):
            all_kinds = {r[j][0] for r in rows}
            kinds = all_kinds - {"nin"}
            if kinds == {"int"}:
                ints = [r[j][1] for r in rows if r[j][0] == "int"]
                if "nin" not in all_kinds and all(
                        -2**63 <= v < 2**63 for v in ints):
                    cols[lab] = np.array([r[j][1] for r in rows],
                                         dtype=np.int64)
                elif "nin" in all_kinds and all(
                        abs(v) <= 2**53 for v in ints):
                    cols[lab] = np.array([np.nan if r[j][0] == "nin"
                                          else float(r[j][1])
                                          for r in rows])
                else:
                    cols[lab] = np.array([r[j][1] for r in rows],
                                         dtype=object)
            elif "string" in kinds:
                cols[lab] = np.array([r[j][1] for r in rows], dtype=object)
            else:
                cols[lab] = self.kal_table[:, j]
        return cols

    def frame(self):
        """The kal table as a pandas DataFrame (needs pandas)."""
        return _frame(self.columns())

    def _run_stem(self):
        """The stem the run named its text files after: stored since the
        header carries `base`, else the input file's stem."""
        b = self.header.get("base")
        if b is None and self.header.get("input_file"):
            b = os.path.splitext(self.header["input_file"])[0]
        return b

    def text_names(self):
        """[(file name, kind, payload)] of the text files export_text
        writes. All are named after the CONTAINER: `stem.kal` and the
        run's .pro names with the run's stem replaced by the container's
        (identical to the run's names unless the file was renamed)."""
        stem = os.path.splitext(os.path.basename(self.path))[0]
        run = self._run_stem()
        out = []
        if self._kal_steps:
            out.append((stem + ".kal", "kal", None))
        for s in self.steps:
            if "pro_labels" not in s:
                continue
            name = s["pro_name"]
            if run and name.startswith(run) and run != stem:
                name = stem + name[len(run):]
            out.append((name, "pro", s))
        return out

    def export_text(self, directory=None, overwrite=False):
        """Regenerate the Namics text files (byte-identical to what the
        run writes without the container). Returns the paths written.

        Without `directory` the files go next to the container, and an
        existing text file there whose content differs is NOT replaced
        (FileExistsError, nothing written): the container may hold only
        part of the run (declared in a later start), and the run's own
        complete `.kal` sits there. Pass a directory, or overwrite=True."""
        from .output import format_kal_row, write_pro_text
        explicit = directory is not None
        if directory is None:
            directory = os.path.dirname(os.path.abspath(self.path))
        texts = []
        for name, kind, s in self.text_names():
            buf = io.StringIO()
            if kind == "kal":
                buf.write("\t".join(self.kal_labels) + "\n")
                for i in range(len(self._kal_steps)):
                    buf.write(format_kal_row(self.kal_row(i)))
            else:
                prof = self._profile_of(s)
                write_pro_text(buf, prof.labels, prof.axes, prof.values)
            texts.append((os.path.join(directory, name), buf.getvalue()))
        if not (explicit or overwrite):
            differ = []
            for p, text in texts:
                if os.path.exists(p):
                    with open(p) as fp:
                        if fp.read() != text:
                            differ.append(os.path.basename(p))
            if differ:
                raise FileExistsError(
                    f"{self.path}: export_text would replace "
                    f"{len(differ)} existing text file(s) with different "
                    f"content ({', '.join(differ[:3])}"
                    + (", ..." if len(differ) > 3 else "") + ") -- the "
                    "container may hold only part of the run. Export into "
                    "a directory (export_text(DIR), `--export DIR`) or "
                    "pass overwrite=True (`--overwrite`)")
        for p, text in texts:
            with open(p, "w") as fp:
                fp.write(text)
        return [p for p, _ in texts]

    def summary(self):
        h = self.header
        lines = [f"{self.path}: {FORMAT} v{h['version']}, PySFBox "
                 f"{h['pysfbox']}, input {h['input_file']}",
                 f"  {len(self.steps)} calculation(s), "
                 f"{len(self._kal_steps)} kal row(s) x "
                 f"{len(self.kal_labels)} column(s), "
                 f"{sum('pro_labels' in s for s in self.steps)} profile "
                 "dump(s)"]
        for s in self.steps:
            bits = [f"  start {s['start']} step {s['step']}"]
            if "scan" in s:
                bits.append(f"{s['scan']['label']} = {s['scan']['value']}")
            lt = s["lattice"]
            bits.append(f"{lt['geometry']} {lt['gradients']}-gradient")
            bits.append(f"{s['iterations']} its, max|g| {s['residual']:.1e}")
            if "pro_labels" in s:
                bits.append(f"{len(s['pro_labels'])} profiles")
            lines.append(", ".join(bits))
        return "\n".join(lines)


def load(path):
    return Reader(path)


class Collection:
    """Several containers read as one table (load_many). `readers` and
    `files` in order; `kal` = {label: column over ALL rows} (a label a
    file lacks is NaN / None there), with `file`, `start`, `step`, `scan`
    naming each row's origin; `stack(label)` and `frame()` as on a
    Reader. Indexing gives the Reader: c[0], c["ladder_3.sfbz"]."""

    def __init__(self, paths):
        if not paths:
            raise ValueError("load_many: no files given")
        self.readers = [Reader(p) for p in paths]
        self.files = [os.path.basename(p) for p in paths]
        per = [r.columns() for r in self.readers]
        n = [len(r.start) for r in self.readers]
        names = []
        for c in per:
            names += [k for k in c if k not in names]
        self._columns = {"file": np.repeat(
            np.array(self.files, dtype=object), n)}
        for k in names:
            parts = [np.asarray(c[k]) if k in c
                     else np.full(m, np.nan) for c, m in zip(per, n)]
            if any(p.dtype == object for p in parts):
                parts = [p if p.dtype == object else
                         np.where(np.isnan(p.astype(float)), None,
                                  p.astype(object)) for p in parts]
            self._columns[k] = (np.concatenate(parts) if parts
                                else np.empty(0))
        self.file, self.start, self.step, self.scan = (
            self._columns[k] for k in ("file", "start", "step", "scan"))
        self.kal = {k: v for k, v in self._columns.items()
                    if k not in ("file", "start", "step", "scan")}

    def __len__(self):
        return len(self.readers)

    def __iter__(self):
        return iter(self.readers)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self.files.index(os.path.basename(key))
        return self.readers[key]

    def columns(self):
        return dict(self._columns)

    def frame(self):
        """One pandas DataFrame over every file (needs pandas)."""
        return _frame(self._columns)

    def stack(self, label, start=None):
        """The profile `label` of every calculation of every file."""
        profs, files = [], []
        for r, f in zip(self.readers, self.files):
            mine = [p for p in r.profiles(start) if label in p]
            profs += mine
            files += [f] * len(mine)
        if not profs:
            raise KeyError(f"no profile '{label}' in {len(self)} file(s)")
        return Stack(label, profs, files)

    def close(self):
        for r in self.readers:
            r.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _natural(path):
    """Sort key with numbers in numeric order (ladder_2 < ladder_10)."""
    return [int(t) if t.isdigit() else t
            for t in re.split(r"(\d+)", os.path.basename(path))]


def load_many(paths):
    """Read several containers as one Collection. `paths`: a glob
    pattern ("ladder_*.sfbz") or a list of paths (kept in the given
    order; a pattern is sorted with numbers in numeric order)."""
    if isinstance(paths, str):
        found = sorted(glob.glob(paths), key=_natural)
        if not found:
            raise FileNotFoundError(f"no file matches '{paths}'")
        paths = found
    return Collection(list(paths))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m pysfbox.sfbz FILE.sfbz [--export [DIR] "
              "[--overwrite] | --repair]\n"
              "  no option: one line per stored calculation\n"
              "  --export:  write the Namics .kal/.pro text files (next to "
              "the container, or in DIR);\n"
              "             next to the container an existing text file "
              "with other content\n"
              "             is kept unless --overwrite is given\n"
              "  --repair:  give a killed run's file its archive directory "
              "back")
        return 0
    path = argv[0]
    if not os.path.exists(path):
        print(f"python -m pysfbox.sfbz: no such file: {path}",
              file=sys.stderr)
        return 2
    try:
        if "--repair" in argv[1:]:
            n = repair(path)
            print(f"{path}: " + ("healthy, left alone" if n is None
                                 else f"repaired, {n} calculation(s)"))
            return 0
        with load(path) as r:
            if "--export" in argv[1:]:
                k = argv.index("--export")
                d = (argv[k + 1] if len(argv) > k + 1
                     and not argv[k + 1].startswith("--") else None)
                for p in r.export_text(d, overwrite="--overwrite" in argv):
                    print(p)
            else:
                print(r.summary())
    except (ValueError, zipfile.BadZipFile, FileExistsError) as e:
        print(f"python -m pysfbox.sfbz: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
