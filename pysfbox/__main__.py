"""Command-line entry point:
  python -m pysfbox input.in [input2.in ...]        run each file (one process)
"""

import sys

from .runner import run_file


def main(argv=None):
    args = (argv or sys.argv)[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Runs Scheutjens-Fleer SCF on Namics-format input files;")
        print("writes .kal/.pro output files next to each input file.")
        return 1
    for path in args:
        print(f"== {path}")
        run_file(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
