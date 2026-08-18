# PySFBox logo generator.
#
# The icon is a converged PySFBox calculation: each lattice row of the tile is
# shaded by the mean-field volume fraction phi(z) of the grafted brush in
# logo_brush.in (binned to 9 rows -- one value per row is exactly what
# "1-gradient mean field" means), with a grafted chain walking the lattice in
# orange (free end slightly larger). Regenerate after changing logo_brush.in:
#
#     python -m pysfbox logo_brush.in
#     python make_logo.py                    (writes the two SVGs here)
#
# Rasterize (macOS, no extra deps -- any SVG renderer works):
#     chrome --headless --screenshot=... --default-background-color=00000000
#
# The wordmark text is converted to outlines (Helvetica Neue Bold via
# matplotlib's TextPath), so the SVG renders identically without the font.
# matplotlib is needed only for the wordmark; the icon is numpy-only.
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

P_INK, INK, ORANGE = "#934CD2", "#333333", "#EB6834"
TOP, BOT = "#F1E8FA", "#6B2FB0"       # gradient endpoints (mixed by phi)
WALL = "#3A3A3A"

# ---- the physics: phi(z) of the logo brush, binned to 9 lattice rows --------
with open(os.path.join(HERE, "logo_brush.pro")) as f:
    cols = f.readline().rstrip("\n").split("\t")
d = np.loadtxt(os.path.join(HERE, "logo_brush.pro"), skiprows=1)
z, phi = d[:, 0], d[:, cols.index("mol:brush:phi")]
keep = z >= 1
z, phi = z[keep], phi[keep]
edges = np.linspace(1, 43, 10)                 # the brush extends to ~42 layers
PHI = [phi[(z >= a) & (z < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
PHI = [p / max(PHI) for p in PHI]              # bottom row -> 1.0

# ---- shared geometry ---------------------------------------------------------
PAD, NC = 16.0, 10
PITCH = (512 - 2 * PAD) / NC

def site(col, row):                            # lattice-site centre, row 0 = wall
    return PAD + (col + 0.5) * PITCH, 512 - PAD - (row + 0.5) * PITCH

def hexmix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))

# the grafted chain: a self-avoiding upward walk on lattice sites
CHAIN = [(3, 1), (4, 1), (4, 2), (5, 2), (5, 3), (4, 3), (4, 4), (5, 4),
         (5, 5), (6, 5), (6, 6), (6, 7)]

def chain_svg():
    pts = [site(c, r) for c, r in CHAIN]
    gx, gy = site(CHAIN[0][0], 0)              # graft stub into the wall
    d = f"M {gx:.0f} {gy:.0f} " + " ".join(f"L {x:.0f} {y:.0f}" for x, y in pts)
    out = [f'<path d="{d}" fill="none" stroke="#FFFFFF" stroke-width="22" '
           f'stroke-linejoin="round" stroke-linecap="round"/>',
           f'<path d="{d}" fill="none" stroke="{ORANGE}" stroke-width="11" '
           f'stroke-linejoin="round" stroke-linecap="round"/>']
    for i, (x, y) in enumerate(pts):
        r = 17 if i == len(pts) - 1 else 13.5  # the free end, slightly larger
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r + 4}" fill="#FFFFFF"/>')
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{ORANGE}"/>')
    return "\n".join(out)

def icon_body():
    stops = []
    profile = [0.05] + PHI[::-1]               # top -> bottom, faint sky floor
    for i, o in enumerate(profile):
        stops.append(f'<stop offset="{i / len(PHI) * 100:.0f}%" '
                     f'stop-color="{hexmix(TOP, BOT, o)}"/>')
    grid = []
    for k in range(1, NC):
        p = PAD + k * PITCH
        grid.append(f'<line x1="{p:.0f}" y1="16" x2="{p:.0f}" y2="496" '
                    f'stroke="#FFFFFF" stroke-opacity="0.16" stroke-width="3"/>')
        grid.append(f'<line x1="16" y1="{p:.0f}" x2="496" y2="{p:.0f}" '
                    f'stroke="#FFFFFF" stroke-opacity="0.16" stroke-width="3"/>')
    wall_y = 512 - PAD - PITCH
    return f'''<defs>
<linearGradient id="mf" x1="0" y1="0" x2="0" y2="1">
{chr(10).join(stops)}
</linearGradient>
<clipPath id="tile"><rect x="16" y="16" width="480" height="480" rx="72"/></clipPath>
</defs>
<g clip-path="url(#tile)">
<rect x="16" y="16" width="480" height="480" fill="url(#mf)"/>
{chr(10).join(grid)}
<rect x="16" y="{wall_y:.0f}" width="480" height="{PITCH + PAD:.0f}" fill="{WALL}"/>
{chain_svg()}
</g>'''

def write_svg(name, body, w, h):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
                f'height="{h}" viewBox="0 0 {w} {h}">\n{body}\n</svg>\n')
    print("wrote", name)

write_svg("pysfbox_icon.svg", icon_body(), 512, 512)

# ---- wordmark: icon + "PySFBox" as outlines ---------------------------------
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path

# a .ttc collection hides its bold faces from matplotlib, so name concrete
# bold files (Arial Bold is metrically Helvetica-compatible and present on
# macOS); DejaVu Sans Bold ships with matplotlib as the portable fallback
for _cand in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/Library/Fonts/Arial Bold.ttf"):
    if os.path.exists(_cand):
        FONT = FontProperties(fname=_cand)
        break
else:
    FONT = FontProperties(family="DejaVu Sans", weight="bold")
SIZE = 132.0

def outline(s):
    return TextPath((0, 0), s, size=SIZE, prop=FONT)

def path_to_svg(tp, dx, dy):
    """matplotlib Path -> SVG d string, y flipped (SVG is y-down)."""
    out, verts, codes = [], tp.vertices, tp.codes
    i = 0
    def pt(j):
        return f"{dx + verts[j][0]:.1f} {dy - verts[j][1]:.1f}"
    while i < len(codes):
        c = codes[i]
        if c == Path.MOVETO:
            out.append("M " + pt(i)); i += 1
        elif c == Path.LINETO:
            out.append("L " + pt(i)); i += 1
        elif c == Path.CURVE3:
            out.append("Q " + pt(i) + " " + pt(i + 1)); i += 2
        elif c == Path.CURVE4:
            out.append("C " + pt(i) + " " + pt(i + 1) + " " + pt(i + 2)); i += 3
        else:                                  # CLOSEPOLY
            out.append("Z"); i += 1
    return " ".join(out)

# pen offsets: bbox("PySF").xmax - bbox("SF").xmax is exactly the pen x where
# "SF" starts (kerning included) -- same trick for "Box"
off_SF = outline("PySF").get_extents().xmax - outline("SF").get_extents().xmax
off_Box = outline("PySFBox").get_extents().xmax - outline("Box").get_extents().xmax
total_w = outline("PySFBox").get_extents().xmax

IH = 200                                       # icon height in the wordmark
X0, BASE = IH + 42, 158                        # text pen origin, baseline
W = int(X0 + total_w + 14)
runs = [("Py", 0, INK), ("SF", off_SF, P_INK), ("Box", off_Box, INK)]
texts = "\n".join(f'<path d="{path_to_svg(outline(s), X0 + dx, BASE)}" fill="{c}"/>'
                  for s, dx, c in runs)
body = (f'<g transform="translate(8,10) scale({IH / 512:.4f})">\n'
        f"{icon_body()}\n</g>\n{texts}")
write_svg("pysfbox_wordmark.svg", body, W, 220)
print("wordmark", W, "x 220; text width", round(total_w))
