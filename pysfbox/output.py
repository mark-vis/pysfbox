"""Namics-format output writers (.kal and .pro).

Formats follow Namics output.cpp / LGrad1::PutProfiles:
- .kal: one file per input (basename.kal), tab-separated; header line of
  `key:name:prop` labels written when the file is created; one row per
  calculation / var step; reals as %.16e, ints as %i, strings bare, unknown as NiN.
- .pro: header `x` + `key:name:prop` labels; rows: physical coordinate
  (offset_first_layer + z - 0.5, like Namics) as %e, profile values as
  %.20g; one file per profile dump, numbered basename.pro /
  basename_<j>.pro / basename_<start>_<j>.pro.
"""

import os

import numpy as np


def kal_path(base):
    return base + ".kal"


def pro_path(base, n_starts, start, subl, has_var):
    if n_starts == 1 and not has_var:
        return f"{base}.pro"
    if n_starts == 1 and has_var:
        return f"{base}_{subl}.pro"
    if n_starts > 1 and not has_var:
        return f"{base}_{start}.pro"
    return f"{base}_{start}_{subl}.pro"


def collect_kal(specs, system):
    """One kal row as data: a list of (kind, value), kind in 'int' / 'real'
    / 'string' / 'nin' (value None). Prints the NiN notes; the text writer
    and the .sfbz container (sfbz.py) both format this one list, so a
    property is evaluated once per row."""
    row = []
    for k, n, p in specs:
        typ, val = system.get_value(k, n, p)
        if typ in ("int", "real", "string"):
            row.append((typ, val))
        elif typ == "unknown":
            # no such property (System.get_value's fall-through)
            row.append(("nin", None))
            print(f"  warning: kal property {k}:{n}:{p} unknown -> NiN")
        else:
            # a RECOGNISED property with no value for this state (a
            # gate; most print their reason first): a neutral line, so
            # a deliberate NiN never reads like a typo and a silent
            # gate still leaves a trace
            row.append(("nin", None))
            print(f"  note: kal property {k}:{n}:{p} has no value for "
                  "this state -> NiN")
    return row


def format_kal_row(row):
    """The Namics text of a collect_kal row: reals %.16e, ints %i, strings
    bare (like Namics' %s), no value NiN."""
    fields = []
    for typ, val in row:
        if typ == "int":
            fields.append(f"{val:d}")
        elif typ == "real":
            fields.append(f"{val:.16e}")
        elif typ == "string":
            fields.append(str(val))
        else:
            fields.append("NiN")
    return "\t".join(fields) + "\n"


def write_kal(path, specs, system, new_file, row=None):
    """specs: list of (key, name, prop); row: a collect_kal result to
    reuse (else collected here)."""
    if row is None:
        row = collect_kal(specs, system)
    mode = "w" if new_file or not os.path.exists(path) else "a"
    with open(path, mode) as fp:
        if mode == "w":
            fp.write("\t".join(f"{k}:{n}:{p}" for k, n, p in specs) + "\n")
        fp.write(format_kal_row(row))


def collect_pro(specs, system, write_bounds=False):
    """The profile dump as data: (labels, axes, values). 1-gradient: axes
    = [x], the physical coordinate of every written row (interior, or with
    write_bounds the fjc ghost layers too), values of shape (ncols, rows).
    N-D: axes = the per-axis cell-centre coordinates of the interior,
    values of shape (ncols, nx, ny[, nz]). Unknown properties are skipped
    with a warning."""
    lat = system.lat
    nd = getattr(lat, "gradients", 1) > 1
    cols, labels = [], []
    for k, n, p in specs:
        prof = system.get_profile(k, n, p)
        if prof is None:
            print(f"  warning: pro property {k}:{n}:{p} unknown; skipped")
            continue
        if write_bounds and not nd:
            prof = system.fill_profile_bounds(k, n, p, prof)
        cols.append(prof)
        labels.append(f"{k}:{n}:{p}")
    if nd:
        interior = lat.iv                 # fjc-deep ghost bands excluded
        axes = []
        for a in range(lat.gradients):
            _, _, ctr, _ = lat._axis_geom(a)
            axes.append(np.ravel(ctr)[interior[a]].copy())
        shape = tuple(len(ax) for ax in axes)
        values = np.empty((len(cols),) + shape)
        for j, c in enumerate(cols):
            values[j] = np.asarray(c).reshape(lat.pdims)[interior]
        return labels, axes, values
    a = 0 if write_bounds else lat.fjc        # first site written (Namics)
    x = np.arange(a, lat.M - a)               # refined site index
    # physical coordinate incl. the radial domain offset (Namics
    # LGrad1.cpp:243 prints offset + (k-0.5)/fjc; 0 for planar and
    # unset offsets, so this only shows on offset curved lattices
    # -- and matches the N-D writer's offset-inclusive radial axis)
    coord = lat.offset + (x - lat.fjc + 0.5) / lat.fjc
    values = np.empty((len(cols), len(x)))
    for j, c in enumerate(cols):
        values[j] = np.asarray(c)[x]
    return labels, [coord], values


def write_pro(path, specs, system, write_bounds=False, data=None):
    """specs: list of (key, name, prop). Writes the interior layers; with
    write_bounds (Namics `output : pro : write_bounds : true`) also writes the
    fjc ghost layers on each side (Namics LGrad1::PutProfiles: `a = 0` if
    writebounds else `fjc`, rows x = a .. M-a, coord = offset +
    (x-fjc+0.5)/fjc). Ghost
    values are filled per the boundary condition (System.fill_profile_bounds),
    so they match Namics rather than the zeroed stored ghosts. data: a
    collect_pro result to reuse (else collected here)."""
    if data is None:
        data = collect_pro(specs, system, write_bounds)
    with open(path, "w") as fp:
        write_pro_text(fp, *data)


def write_pro_text(fp, labels, axes, values):
    """The Namics .pro text of a collect_pro result. 1-gradient: `x` + one
    row per site. N-D: leading cell-centre coordinate columns (x, y[, z]),
    one row per interior cell in C order; Cartesian/radial coordinates in
    bond lengths, radial ones offset-inclusive (like the 1-gradient offset
    + z-0.5); angular axes (theta, phi) in radians."""
    names = ["x", "y", "z"][:len(axes)]
    fp.write("\t".join(names) + "\t" + "\t".join(labels) + "\n")
    shape = tuple(len(ax) for ax in axes)
    grids = np.meshgrid(*axes, indexing="ij") if len(axes) > 1 else axes
    coord_cols = [np.ravel(g) for g in grids]
    val_cols = [np.ravel(v) for v in values]
    for i in range(int(np.prod(shape))):
        row = [f"{cc[i]:e}" for cc in coord_cols]
        row += [f"{vc[i]:.20g}" for vc in val_cols]
        fp.write("\t".join(row) + "\n")
