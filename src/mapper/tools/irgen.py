from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class IRGenResult:
    input_c: Path
    output_ll: Path
    llvm_version: str
    clang: str
    cmd: tuple[str, ...]


class IRGenError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise IRGenError(
            f"Missing environment variable {name}. "
            "Did you run ./scripts/activate to enter the dev shell?"
        )
    return v


def _run(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise IRGenError(
            "LLVM IR generation failed.\n"
            f"Command:\n  {shlex.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"stderr:\n{proc.stderr.strip()}\n"
        )


def c_to_llvm_ir(
    c_file: str | Path,
    *,
    out_ll: str | Path | None = None,
    include_dirs: Iterable[str | Path] = (),
    defines: Iterable[str] = (),
    std: str = "c11",
    opt_level: str = "0",
    extra_flags: Iterable[str] = (),
) -> IRGenResult:
    """
    Compile a single C file to textual LLVM IR (.ll) using clang-$LLVM_VERSION.

    Requires env var LLVM_VERSION to be set (your dev shell does this).
    """
    llvm_version = _require_env("LLVM_VERSION")

    c_path = Path(c_file).expanduser().resolve()
    if not c_path.exists():
        raise FileNotFoundError(f"Input C file not found: {c_path}")
    if c_path.suffix.lower() != ".c":
        raise ValueError(f"Expected a .c file, got: {c_path.name}")

    if out_ll is None:
        ll_path = c_path.with_suffix(".ll")
    else:
        ll_path = Path(out_ll).expanduser().resolve()
        if ll_path.is_dir():
            ll_path = ll_path / (c_path.stem + ".ll")

    clang = f"clang-{llvm_version}"

    cmd: list[str] = [
        clang,
        "-S",
        "-emit-llvm",
        f"-std={std}",
        f"-O{opt_level}",
        str(c_path),
        "-o",
        str(ll_path),
    ]

    for d in include_dirs:
        cmd.extend(["-I", str(Path(d).expanduser().resolve())])

    for sym in defines:
        cmd.append(f"-D{sym}")

    cmd.extend(list(extra_flags))

    _run(cmd)

    return IRGenResult(
        input_c=c_path,
        output_ll=ll_path,
        llvm_version=llvm_version,
        clang=clang,
        cmd=tuple(cmd),
    )