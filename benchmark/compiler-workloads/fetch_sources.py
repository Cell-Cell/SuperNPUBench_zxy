#!/usr/bin/env python3
"""Fetch the locked LinxISA compiler-workload bundle into a local cache."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "sources.lock.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=HERE / ".cache")
    parser.add_argument("--with-third-party", action="store_true",
                        help="also run the locked LinxISA PolyBench fetch helper")
    args = parser.parse_args()
    commit = LOCK["linx_isa"]["commit"]
    destination = args.cache.resolve() / f"linx-isa-{commit}" / "workloads"
    if not destination.exists():
        url = f"https://github.com/LinxISA/linx-isa/archive/{commit}.tar.gz"
        args.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="supernpu-workloads-") as tmp:
            archive = Path(tmp) / "linx-isa.tar.gz"
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive, "r:gz") as bundle:
                root = Path(tmp).resolve()
                for member in bundle.getmembers():
                    target = (root / member.name).resolve()
                    if root not in target.parents and target != root:
                        raise RuntimeError(f"unsafe archive member: {member.name}")
                bundle.extractall(tmp)
            source = next(Path(tmp).glob("linx-isa-*/workloads"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
    if args.with_third_party:
        subprocess.run(["bash", str(destination / "fetch_third_party.sh")],
                       cwd=destination, check=True)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
