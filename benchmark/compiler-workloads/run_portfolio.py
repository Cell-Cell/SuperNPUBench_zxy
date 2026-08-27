#!/usr/bin/env python3
"""Run the locked LinxISA compiler workload portfolio and consolidate results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_COMPILER = Path("/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin")


@dataclass
class Result:
    workload: str
    state: str
    command: list[str]
    returncode: int | None
    log: str
    report: str | None = None


def run(name: str, command: list[str], log: Path, timeout: float) -> Result:
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout)
        log.write_text(proc.stdout)
        return Result(name, "PASS" if proc.returncode == 0 else "FAIL",
                      command, proc.returncode, str(log))
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        log.write_text(output)
        return Result(name, "TIMEOUT", command, None, str(log))


def compatible_base_runner(root: Path, out: Path, target: str) -> Path:
    """Bridge locked upstream's stale target verifier without changing sources.

    The compiler uses the canonical ``linx64v5`` triple while the locked
    workload runner only recognizes ``linx64`` for ELF64 Linx verification and
    musl-static auto-selection.  Patch a generated mirror, leaving the locked
    input tree byte-for-byte intact and auditable.
    """
    runner = root / "run_benchmarks.py"
    if not target.startswith("linx64v5-"):
        return runner
    mirror = out / ".compat-workloads"
    if mirror.exists():
        shutil.rmtree(mirror)
    shutil.copytree(root, mirror, symlinks=True)
    patched = mirror / "run_benchmarks.py"
    text = patched.read_text()
    text = text.replace('"linx64": ("Linx", "ELF64", little_endian),',
                        '"linx64": ("Linx", "ELF64", little_endian),\n'
                        '        "linx64v5": ("LinxV5", "ELF64", little_endian),')
    text = text.replace('target.startswith(("linx64-", "linx32-"))',
                        'target.startswith(("linx64-", "linx64v5-", "linx32-"))')
    text = text.replace('lib_dir = sysroot_path / "lib"',
                        'lib_dir = sysroot_path / "lib"\n'
                        '    if not lib_dir.is_dir():\n'
                        '        lib_dir = sysroot_path / "usr/lib"')
    text = text.replace('sysroot_path / "lib" / "liblinx_builtin_rt.a"',
                        '(sysroot_path / "usr/lib" if (sysroot_path / "usr/lib").is_dir() else sysroot_path / "lib") / "liblinx_builtin_rt.a"')
    text = text.replace('sysroot_path / "lib" / "libclang_rt.builtins-linx64.a"',
                        '(sysroot_path / "usr/lib" if (sysroot_path / "usr/lib").is_dir() else sysroot_path / "lib") / "libclang_rt.builtins-linx64.a"')
    patched.write_text(text)
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads-root", type=Path, required=True,
                        help="locked LinxISA workloads directory from fetch_sources.py")
    parser.add_argument("--compiler-dir", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--target", default="linx64v5-unknown-linux-musl")
    parser.add_argument("--sysroot", type=Path)
    parser.add_argument("--run-command", help="runtime wrapper containing {exe}")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--polybench", action="store_true")
    parser.add_argument("--tsvc", action="store_true")
    parser.add_argument("--qemu", type=Path)
    parser.add_argument("--ctuning-root", type=Path)
    parser.add_argument("--ctuning-limit", type=int, default=0)
    parser.add_argument("--opt", default="-O2")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=REPO / "output/compiler-workloads")
    args = parser.parse_args()
    root = args.workloads_root.resolve()
    if not (root / "run_benchmarks.py").is_file():
        parser.error(f"invalid workloads root: {root}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    clang = args.compiler_dir / "clang"
    if args.sysroot is None:
        candidate = args.compiler_dir.parent / "sysroot"
        if candidate.is_dir():
            args.sysroot = candidate
    base_runner = compatible_base_runner(root, out, args.target)
    common = ["--cc", str(clang), "--target", args.target, f"--opt={args.opt}",
              "--out-dir", str(out / "artifacts"), "--link-mode", "default"]
    if args.sysroot:
        common += ["--sysroot", str(args.sysroot)]
    if args.run_command and not args.compile_only:
        common += ["--run-command", args.run_command]
    results = []
    base_json = out / "coremark_dhrystone.json"
    results.append(run("coremark+dhrystone",
                       [os.environ.get("PYTHON", "python3"), str(base_runner),
                        *common, "--json-out", str(base_json)],
                       out / "logs/coremark_dhrystone.log", args.timeout * 3))
    results[-1].report = str(base_json) if base_json.exists() else None
    if args.compile_only and results[-1].state == "PASS":
        results[-1].state = "COMPILE_PASS"
    if args.polybench:
        poly_json = out / "polybench.json"
        poly = [os.environ.get("PYTHON", "python3"), str(root / "run_polybench.py"),
                "--cc", str(clang), "--target", args.target, f"--opt={args.opt}",
                "--kernels", "gemm,jacobi-2d", "--out-dir", str(out / "polybench"),
                "--json-out", str(poly_json)]
        if args.sysroot:
            poly += ["--sysroot", str(args.sysroot)]
        if args.run_command and not args.compile_only:
            poly += ["--run-command", args.run_command]
        results.append(run("polybench", poly, out / "logs/polybench.log", args.timeout * 3))
        results[-1].report = str(poly_json) if poly_json.exists() else None
        if args.compile_only and results[-1].state == "PASS":
            results[-1].state = "COMPILE_PASS"
    if args.tsvc:
        if not args.qemu:
            results.append(Result("tsvc", "NOT_RUN", [], None,
                                  str(out / "logs/tsvc.log"), None))
        else:
            for mode in ("off", "auto"):
                command = [os.environ.get("PYTHON", "python3"), str(root / "tsvc/run_tsvc.py"),
                           "--clang", str(clang), "--qemu", str(args.qemu),
                           "--source-policy", "linx-v058", "--vector-mode", mode]
                if mode == "auto":
                    baseline = root / "generated/qemu/tsvc/tsvc.off.stdout.txt"
                    command += ["--compare-baseline-log", str(baseline),
                                "--fail-on-checksum-mismatch"]
                results.append(run(f"tsvc-{mode}", command,
                                   out / f"logs/tsvc_{mode}.log", args.timeout * 5))
    if args.ctuning_limit:
        if not args.ctuning_root:
            results.append(Result("ctuning", "NOT_RUN", [], None,
                                  str(out / "logs/ctuning.log"), None))
        else:
            command = [os.environ.get("PYTHON", "python3"),
                       str(root / "ctuning/run_milepost_codelets.py"),
                       "--target", args.target, "--ctuning-root", str(args.ctuning_root),
                       "--clang", str(clang), "--limit", str(args.ctuning_limit)]
            if not args.compile_only:
                command.append("--run")
            results.append(run("ctuning", command, out / "logs/ctuning.log",
                               args.timeout * max(2, args.ctuning_limit)))
    payload = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "compiler": str(clang), "target": args.target,
               "workloads_root": str(root), "results": [asdict(r) for r in results]}
    report = out / "portfolio.json"
    report.write_text(json.dumps(payload, indent=2) + "\n")
    markdown = ["# Compiler workload report", "", f"- Compiler: `{clang}`",
                f"- Target: `{args.target}`", "", "| Workload | State | Return code | Log |",
                "| --- | --- | ---: | --- |"]
    for item in results:
        markdown.append(f"| {item.workload} | {item.state} | {item.returncode if item.returncode is not None else '-'} | `{item.log}` |")
    (out / "portfolio.md").write_text("\n".join(markdown) + "\n")
    print(report)
    return 0 if all(r.state in {"PASS", "COMPILE_PASS", "NOT_RUN"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
