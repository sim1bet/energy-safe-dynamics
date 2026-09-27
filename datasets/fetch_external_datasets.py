#!/usr/bin/env python3
# Author: Simone Betteti
# Paper: Safe-by-design learning via energy-based neural network
"""Explicitly acquire and checksum external benchmark data.

No download occurs during package import or model inference. This command-line
tool fails closed if publisher bytes differ from the audited manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "EXTERNAL_DATA_MANIFEST.json"


def manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_file(record: dict) -> tuple[bool, str]:
    path = ROOT / record["path"]
    if not path.is_file():
        return False, f"MISSING  {record['path']}"
    actual_size = path.stat().st_size
    actual_hash = digest(path)
    if actual_size != record["bytes"] or actual_hash != record["sha256"]:
        return False, f"MISMATCH {record['path']}"
    return True, f"OK       {record['path']}"


def verify(names: list[str]) -> bool:
    data = manifest()["datasets"]
    valid = True
    for name in names:
        for record in data[name]["files"]:
            ok, message = check_file(record)
            print(message)
            valid &= ok
    return valid


def _find_verified_file(search_root: Path, record: dict) -> Path:
    matches = []
    for candidate in search_root.rglob(Path(record["path"]).name):
        if candidate.is_file() and candidate.stat().st_size == record["bytes"]:
            if digest(candidate) == record["sha256"]:
                matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one verified source for {record['path']}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _materialize(records: list[dict], search_root: Path) -> None:
    for record in records:
        target = ROOT / record["path"]
        if target.is_file() and check_file(record)[0]:
            continue
        source = _find_verified_file(search_root, record)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def fetch_nonlinear_benchmark(name: str) -> None:
    try:
        import nonlinear_benchmarks as nlb
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional loader first: python -m pip install "
            "nonlinear-benchmarks"
        ) from exc

    loader_name = {"ced": "CED", "silverbox": "Silverbox"}[name]
    loader = getattr(nlb, loader_name)
    records = manifest()["datasets"][name]["files"]
    with tempfile.TemporaryDirectory(prefix=f"ebm-{name}-") as tmp:
        work = Path(tmp)
        loader(dir_placement=str(work), force_download=True)
        _materialize(records, work)


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe archive path: {info.filename}")
            mode = info.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise RuntimeError(f"Archive symlink rejected: {info.filename}")
        bundle.extractall(destination)


def fetch_nanodrone() -> None:
    entry = manifest()["datasets"]["nanodrone"]
    with tempfile.TemporaryDirectory(prefix="ebm-nanodrone-") as tmp:
        work = Path(tmp)
        archive = work / "nanodrone.zip"
        request = urllib.request.Request(
            entry["archive"], headers={"User-Agent": "EBM-dataset-fetcher/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive.write_bytes(response.read())
        extracted = work / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        _materialize(entry["files"], extracted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("ced", "silverbox", "nanodrone", "all", "verify")
    )
    args = parser.parse_args()
    names = ["ced", "silverbox", "nanodrone"]

    if args.action == "verify":
        return 0 if verify(names) else 1
    selected = names if args.action == "all" else [args.action]
    for name in selected:
        print(f"Fetching {name} from its canonical publisher ...")
        if name == "nanodrone":
            fetch_nanodrone()
        else:
            fetch_nonlinear_benchmark(name)
    return 0 if verify(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
