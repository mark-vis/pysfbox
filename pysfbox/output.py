"""Namics-format output writers (.kal and .pro).

Formats follow Namics output.cpp / LGrad1::PutProfiles:
- .kal: one file per input (basename.kal), tab-separated; header line of
  `key:name:prop` labels written when the file is created; one row per
  calculation / var step; reals as %.16e, ints as %i, unknown as NiN.
- .pro: header `x` + `key:name:prop` labels; rows: coordinate (z - 0.5) as
  %e, profile values as %.20g; one file per profile dump, numbered
  basename.pro / basename_<j>.pro / basename_<start>_<j>.pro.
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


def write_kal(path, specs, system, new_file):
    """specs: list of (key, name, prop)."""
    mode = "w" if new_file or not os.path.exists(path) else "a"
    with open(path, mode) as fp:
        if mode == "w":
            fp.write("\t".join(f"{k}:{n}:{p}" for k, n, p in specs) + "\n")
        fields = []
        for k, n, p in specs:
            typ, val = system.get_value(k, n, p)
            if typ == "int":
                fields.append(f"{val:d}")
            elif typ == "real":
                fields.append(f"{val:.16e}")
            else:
                fields.append("NiN")
                print(f"  warning: kal property {k}:{n}:{p} unknown -> NiN")
        fp.write("\t".join(fields) + "\n")


def write_pro(path, specs, system, write_bounds=False):
    """specs: list of (key, name, prop). Writes the interior layers; with
    write_bounds (Namics `output : pro : write_bounds : true`) also writes the
    fjc ghost layers on each side (Namics LGrad1::PutProfiles: `a = 0` if
    writebounds else `fjc`, rows x = a .. M-a, coord = (x-fjc+0.5)/fjc). Ghost
    values are filled per the boundary condition (System.fill_profile_bounds),
    so they match Namics rather than the zeroed stored ghosts."""
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
        _write_pro_nd(path, lat, cols, labels)
        return
    a = 0 if write_bounds else lat.fjc        # first site written (Namics)
    with open(path, "w") as fp:
        fp.write("x\t" + "\t".join(labels) + "\n")
        for x in range(a, lat.M - a):         # refined site index
            coord = (x - lat.fjc + 0.5) / lat.fjc   # == z_phys[k] on interior
            row = [f"{coord:e}"] + [f"{c[x]:.20g}" for c in cols]
            fp.write("\t".join(row) + "\n")


def _write_pro_nd(path, lat, cols, labels):
    """2D/3D .pro: leading cell-centre coordinate columns (x, y[, z]) then the
    profile values, one row per interior cell in C order. Cartesian/radial
    coordinates are in bond lengths (like the 1-gradient z-0.5); angular axes
    (theta, phi) are in radians."""
    names = ["x", "y", "z"][:lat.gradients]
    interior = tuple(slice(1, n + 1) for n in lat.dims)
    coord_cols = []
    for a in range(lat.gradients):
        _, _, ctr, _ = lat._axis_geom(a)
        coord_cols.append(np.broadcast_to(ctr, lat.pdims)[interior].ravel())
    val_cols = [c.reshape(lat.pdims)[interior].ravel() for c in cols]
    with open(path, "w") as fp:
        fp.write("\t".join(names) + "\t" + "\t".join(labels) + "\n")
        for i in range(coord_cols[0].size):
            row = [f"{cc[i]:e}" for cc in coord_cols]
            row += [f"{vc[i]:.20g}" for vc in val_cols]
            fp.write("\t".join(row) + "\n")
