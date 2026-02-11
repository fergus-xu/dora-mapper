from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mapper.tools.irgen import IRGenError, c_to_llvm_ir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mapper", description="Generate LLVM IR (.ll) from a C file.")
    p.add_argument("c_file", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("-I", dest="include_dirs", action="append", default=[])
    p.add_argument("-D", dest="defines", action="append", default=[])
    p.add_argument("--std", default="c11")
    p.add_argument("--O", dest="opt_level", default="0")
    p.add_argument("--flag", action="append", default=[], help="Extra clang flag (repeatable)")

    args = p.parse_args(argv)

    try:
        res = c_to_llvm_ir(
            args.c_file,
            out_ll=args.out,
            include_dirs=args.include_dirs,
            defines=args.defines,
            std=args.std,
            opt_level=args.opt_level,
            extra_flags=args.flag,
        )
    except (IRGenError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Wrote: {res.output_ll}")
    print(f"Cmd:   {' '.join(res.cmd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
