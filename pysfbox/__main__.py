"""Command-line entry point:
  pysfbox input.in [input2.in ...]         run each file (one process, in order)
  pysfbox -j[N] f1.in f2.in ...            run files in parallel (N workers;
                                           default N = cpu_count - 1)

`python -m pysfbox ...` is the same runner from a source checkout.
"""

import os
import sys

from .runner import run_file


def _run_one(path):
    """Worker: run one file, return (path, error-or-None). Used by -j."""
    try:
        run_file(path)
        return path, None
    except Exception as e:                          # keep the pool going
        return path, f"{type(e).__name__}: {e}"


def main(argv=None):
    args = (argv or sys.argv)[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Runs Scheutjens-Fleer SCF on Namics-format input files;")
        print("writes .kal/.pro output files next to each input file.")
        return 1

    # -j / -jN : run independent input files in parallel across processes.
    # Each file is a self-contained calculation writing its own .kal/.pro, so
    # they parallelise cleanly; default to cpu_count-1 workers (leave one core
    # free). Only worth it for several files.
    if args and (args[0] == "-j" or args[0].startswith("-j")):
        spec, files = args[0][2:], args[1:]
        workers = int(spec) if spec else max(1, (os.cpu_count() or 2) - 1)
        workers = min(workers, len(files)) or 1
        if workers <= 1 or len(files) <= 1:
            for path in files:
                print(f"== {path}")
                run_file(path)
            return 0
        from multiprocessing import Pool
        print(f"== running {len(files)} files on {workers} workers")
        fail = 0
        with Pool(workers) as pool:
            for path, err in pool.imap_unordered(_run_one, files):
                if err:
                    fail += 1
                    print(f"== FAIL {path}: {err}")
                else:
                    print(f"== done {path}")
        return 1 if fail else 0

    for path in args:
        print(f"== {path}")
        run_file(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
