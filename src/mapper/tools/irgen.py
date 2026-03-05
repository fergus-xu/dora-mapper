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
    output_dfg: Path | None
    llvm_version: str
    clang: str
    cmd: tuple[str, ...]
    dfg_cmd: tuple[str, ...] | None


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
    run_passes: bool = True,
    include_dirs: Iterable[str | Path] = (),
    defines: Iterable[str] = (),
    std: str = "c11",
    opt_level: str = "0",
    extra_flags: Iterable[str] = (),
) -> IRGenResult:
    """
    Compile a single C file to textual LLVM IR (.ll) using clang-$LLVM_VERSION.
    Optionally run LLVM passes to extract DFG.

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
        ll_path = Path(out_ll).expanduser()
        # If out_ll is just a filename (no directory), put it in the same dir as the C file
        if not ll_path.is_absolute() and len(ll_path.parts) == 1:
            ll_path = c_path.parent / ll_path
        else:
            ll_path = ll_path.resolve()
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

    # Run LLVM passes if requested
    dfg_path = None
    dfg_cmd = None
    if run_passes:
        opt = f"opt-{llvm_version}"
        
        # Find the pass plugin (look in repository root)
        repo_root = Path(__file__).parent.parent.parent.parent
        pass_plugin = repo_root / "llvm" / "llvm_passes.so"
        
        if pass_plugin.exists():
            # First lower GEPs, then extract DFG
            dfg_cmd_list = [
                opt,
                f"-load-pass-plugin={pass_plugin}",
                "-passes=lower-gep,extract-dfg",
                "-disable-output",
                str(ll_path),
            ]
            
            # Run from the directory of the .ll file so JSON is created there
            original_dir = Path.cwd()
            try:
                os.chdir(ll_path.parent)
                _run(dfg_cmd_list)
                
                # Look for generated JSON file
                # The pass creates <function_name>_dfg.json
                json_files = sorted(Path(".").glob("*_dfg.json"))
                if json_files:
                    # Use the most recently modified one
                    dfg_path = ll_path.parent / json_files[-1].name
            finally:
                os.chdir(original_dir)
            
            dfg_cmd = tuple(dfg_cmd_list)

    return IRGenResult(
        input_c=c_path,
        output_ll=ll_path,
        output_dfg=dfg_path,
        llvm_version=llvm_version,
        clang=clang,
        cmd=tuple(cmd),
        dfg_cmd=dfg_cmd,
    )