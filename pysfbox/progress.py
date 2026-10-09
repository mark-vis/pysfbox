"""Live one-line solver progress for interactive terminals.

A single carriage-return-rewritten stderr line -- stage, iteration, max|g|,
step alpha -- so a user watching a long solve can tell "converging slowly"
from "going haywire" without waiting for the final summary (added Aug
2026). Three deliberate properties:

- ONLY when stderr is an interactive terminal (isatty): CI logs, redirected
  output, and pipelines never see a byte of this. PYSFBOX_NO_PROGRESS=1
  force-disables it (e.g. for terminals that mishandle \\r).
- stderr, not stdout: the parsable result stream (notes, per-start
  summaries, .kal content on any future pipe) stays clean.
- throttled to ~20 updates/s so the line costs nothing even in solver
  loops that run thousands of iterations per second.

Output files and stdout are never touched, so results are byte-identical
with the line on or off.
"""
import os
import sys
import time

_INTERVAL = 0.05          # min seconds between redraws
_last_draw = 0.0
_dirty = False            # a progress line is currently on screen
_enabled = None


def enabled():
    global _enabled
    if _enabled is None:
        _enabled = (not os.environ.get("PYSFBOX_NO_PROGRESS")
                    and hasattr(sys.stderr, "isatty")
                    and sys.stderr.isatty())
    return _enabled


def update(stage, it, err, alpha=None, extra=""):
    """Redraw the status line (throttled). err is max|g|; alpha is the
    accepted step/line-search factor of the stage, when it has one."""
    global _last_draw, _dirty
    if not enabled():
        return
    now = time.monotonic()
    if _dirty and now - _last_draw < _INTERVAL:
        return
    _last_draw = now
    msg = f"  {stage} it {it}  max|g| = {err:.3e}"
    if alpha is not None:
        msg += f"  alpha = {alpha:.2g}"
    if extra:
        msg += f"  {extra}"
    sys.stderr.write("\r\x1b[2K" + msg)
    sys.stderr.flush()
    _dirty = True


def status(msg):
    """A free-form status line (same single-line contract as update)."""
    global _last_draw, _dirty
    if not enabled():
        return
    _last_draw = time.monotonic()
    sys.stderr.write("\r\x1b[2K  " + msg)
    sys.stderr.flush()
    _dirty = True


def clear():
    """Erase the line. Call before any normal print that should not land
    next to a stale progress line, and at every solver exit."""
    global _dirty
    if _dirty:
        sys.stderr.write("\r\x1b[2K")
        sys.stderr.flush()
        _dirty = False
