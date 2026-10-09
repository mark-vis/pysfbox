"""Guards for the live solver progress line (pysfbox/progress.py).

The regression suite runs non-interactively, so it can never see the line
at all -- these checks need a pty. Three guards:

1. INTERACTIVE: under a pty, a solve redraws a single carriage-return
   status line (stage, iteration, max|g|, alpha) and CLEARS it before the
   normal per-start summary prints -- the "is it taking long or going
   haywire" display.
2. NON-INTERACTIVE: with stderr redirected (the CI / file-redirect case),
   NOT ONE BYTE of progress output appears -- the parsable output contract
   is unchanged.
3. KILL SWITCH: PYSFBOX_NO_PROGRESS=1 silences the line even on a pty.

Run after touching pysfbox/progress.py or the progress wiring in
sfnewton.iterate / system._solve_anderson (POSIX only: needs a pty):

    python tests/progress_line.py
"""
import os
import pty
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# run on a COPY: pysfbox writes .kal/.pro next to the input, and running the
# committed tests/two_brushes_quick.in in place would clobber its references
TMP = tempfile.mkdtemp(prefix="pysfbox_progress_")
INPUT = os.path.join(TMP, "two_brushes_quick.in")
shutil.copy(os.path.join(ROOT, "tests", "two_brushes_quick.in"), INPUT)

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"  {'ok   ' if ok else 'FAIL '} {label}" + (f" ({detail})" if detail else ""))
    if not ok:
        failures += 1


def run_pty(env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    m, s = pty.openpty()
    p = subprocess.Popen([sys.executable, "-m", "pysfbox", INPUT],
                         stderr=s, stdout=subprocess.DEVNULL,
                         cwd=ROOT, env=env)
    os.close(s)
    chunks = []
    while True:
        try:
            b = os.read(m, 65536)
        except OSError:          # linux pty EOF
            break
        if not b:                # macOS pty EOF
            break
        chunks.append(b)
    os.close(m)
    p.wait()
    return b"".join(chunks).decode(errors="replace")


# 1: interactive -> live line, cleared at the end
err = run_pty()
redraws = [seg for seg in err.split("\r") if "max|g|" in seg]
check("pty shows live progress redraws", len(redraws) >= 3,
      f"{len(redraws)} redraws, e.g. {redraws[len(redraws) // 2].strip()[:60] if redraws else '-'}")
check("stage + iteration + alpha on the line",
      any("pseudohessian it" in seg and "alpha" in seg for seg in redraws))
last = err.split("\r")[-1]
check("line cleared at exit (no stale text after final clear)",
      last.replace("\x1b[2K", "").strip() == "")

# 2: non-interactive -> byte-silent
p = subprocess.run([sys.executable, "-m", "pysfbox", INPUT],
                   capture_output=True, text=True, cwd=ROOT)
# progress bytes are \r / escape sequences
check("redirected stderr carries no progress bytes",
      "\r" not in p.stderr and "\x1b" not in p.stderr, repr(p.stderr[:60]))

# 3: kill switch on a pty
err = run_pty({"PYSFBOX_NO_PROGRESS": "1"})
check("PYSFBOX_NO_PROGRESS=1 silences the line on a pty",
      "max|g|" not in err)

print(f"all done ({failures} failure(s))")
sys.exit(1 if failures else 0)
