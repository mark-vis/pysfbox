"""Namics input-file parser.

Namics input files consist of lines

    keyword : name : parameter : value

with `//` comments, blank lines, and the word `start` on its own line closing
a calculation. Settings accumulate across `start` blocks (later blocks add to
or override earlier ones), exactly as in Namics. `alias : X : value : v`
defines `#X#` substitutions inside composition strings, and a `var` block
defines a parameter scan within one calculation.

Because output-specification lines may repeat a parameter with different
values (e.g. `kal : mon : G : 1st_M_phi_z` and `kal : mon : G : 2nd_M_phi_z`),
every parameter stores a LIST of values; use `last(...)` for normal settings
(override semantics) and the full list for output specs.
"""

from collections import OrderedDict


def _parse_line(line):
    line = line.split("//")[0].strip()
    if not line:
        return None
    if line.lower() == "start":
        return "START"
    parts = [p.strip() for p in line.split(":")]
    if len(parts) < 4:
        raise ValueError(f"cannot parse input line: '{line}'")
    key, name, param = parts[0], parts[1], parts[2]
    value = ":".join(parts[3:]).strip()
    return key, name, param, value


def read_input(path):
    """Parse a Namics .in file -> list of calculations.

    Each calculation is settings[(key, name)][param] = [values...], with
    settings accumulated over consecutive `start` blocks.
    """
    with open(path) as f:
        lines = f.readlines()

    calculations = []
    settings = OrderedDict()
    block_has_content = False
    for ln in lines:
        parsed = _parse_line(ln)
        if parsed is None:
            continue
        if parsed == "START":
            calculations.append(_copy(settings))
            block_has_content = False
            continue
        key, name, param, value = parsed
        params = settings.setdefault((key, name), OrderedDict())
        params.setdefault(param, []).append(value)
        block_has_content = True
    if block_has_content or not calculations:
        calculations.append(_copy(settings))
    return calculations


def _copy(settings):
    return OrderedDict((k, OrderedDict((p, list(v)) for p, v in d.items()))
                       for k, d in settings.items())


def last(params, key, default=None):
    """Override semantics: the last value given for a parameter."""
    v = params.get(key)
    return v[-1] if v else default


def get_blocks(settings, key):
    """All (name, params) blocks for a keyword, in input order."""
    return [(name, params) for (k, name), params in settings.items()
            if k == key]


def set_value(settings, key, name, param, value):
    settings.setdefault((key, name), OrderedDict())[param] = [str(value)]


def substitute_aliases(text, settings):
    """Replace #X# by the value of `alias : X : value`."""
    for (key, name), params in settings.items():
        if key == "alias" and "value" in params:
            text = text.replace(f"#{name}#", str(params["value"][-1]))
    return text
