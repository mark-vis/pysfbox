# The PySFBox logo

The icon is not an illustration of a mean-field lattice calculation — it *is*
one. Each lattice row of the tile is shaded by the volume fraction φ(z) of a
converged PySFBox brush (`logo_brush.in`: N = 100 grafted chain, θ = 10, good
solvent, 60 planar layers, binned to the 9 rows of the tile). One value per
row is exactly what "1-gradient mean field" means. The dark bar is the
grafting surface, and the orange walk is a chain on the lattice — grafted at
the wall, free end drawn slightly larger.

Colors: purple `#934CD2` (the field; also the SF in the wordmark), orange
`#EB6834` (the chain), ink `#333333`. Purple = good-solvent/corona and
orange = chain/hydrophobic matches the poster/figure convention. The
purple–orange pair is color-vision-safe.

## Files

- `pysfbox_icon.svg` — square icon (512 viewBox, transparent corners)
- `pysfbox_wordmark.svg` — icon + "PySFBox"; the type is outlined (no font
  needed to render it faithfully)
- `png/` — rasterized exports: icon at 512/256/128/64/32, wordmark @2x,
  and a 1280×640 banner (GitHub social preview)

## Regenerating

```
python -m pysfbox logo_brush.in     # writes logo_brush.pro (committed)
python make_logo.py                 # writes the two SVGs
```

`make_logo.py` needs numpy for the icon and matplotlib only for the wordmark
outlines (Helvetica-compatible bold: Arial Bold on macOS, DejaVu Sans Bold as
the portable fallback — regenerating the wordmark on a machine without Arial
changes the glyphs slightly; the committed SVG is the reference). Rasterize
with any SVG renderer, e.g. headless Chrome:

```
chrome --headless --screenshot=png/pysfbox_icon_512.png \
  --window-size=512,512 --default-background-color=00000000 pysfbox_icon.svg
```
