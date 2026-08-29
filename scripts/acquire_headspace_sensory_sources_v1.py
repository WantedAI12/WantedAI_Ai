#!/usr/bin/env python
"""Acquire the hash-pinned public inputs for the headspace/sensory hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_headspace_sensory_hub_v1 as hub  # noqa: E402


PYRFUME_REPOSITORY = "https://github.com/pyrfume/pyrfume-data.git"
PUBCHEM_CID = 1_549_778
PUBCHEM_CASRN = "3796-70-1"
PUBCHEM_URI = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/1549778/"
    "property/CanonicalSMILES,IsomericSMILES,MolecularWeight,InChIKey,IUPACName/JSON"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_git(arguments: list[str], *, timeout: int = 300) -> None:
    subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def acquire_pyrfume(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Pyrfume path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=path.parent, prefix=f".{path.name}.acquire."
    ) as temporary_root:
        checkout = Path(temporary_root) / "repository"
        run_git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                PYRFUME_REPOSITORY,
                str(checkout),
            ]
        )
        run_git(["-C", str(checkout), "sparse-checkout", "init", "--cone"])
        run_git(
            [
                "-C",
                str(checkout),
                "sparse-checkout",
                "set",
                *hub.ARCHIVES,
            ]
        )
        run_git(
            ["-C", str(checkout), "checkout", "--detach", hub.PYRFUME_COMMIT]
        )
        os.replace(checkout, path)


def request_bytes(uri: str) -> bytes:
    request = urllib.request.Request(
        uri,
        headers={"User-Agent": "perfumery-ai-headspace-source-acquisition/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def acquire_opera(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite EPA OPERA archive: {path}")
    payload = request_bytes(hub.OPERA_URI)
    if len(payload) != hub.OPERA_BYTES or sha256_bytes(payload) != hub.OPERA_SHA256:
        raise RuntimeError("downloaded EPA OPERA archive failed byte contract")
    atomic_bytes(path, payload)


def pubchem_supplement(properties: Mapping[str, Any]) -> bytes:
    record = {
        "casrn": PUBCHEM_CASRN,
        "cid": int(properties["CID"]),
        "connectivity_smiles": str(properties["ConnectivitySMILES"]),
        "inchi_key": str(properties["InChIKey"]),
        "iupac_name": str(properties["IUPACName"]),
        "molecular_weight": float(properties["MolecularWeight"]),
        "smiles": str(properties["SMILES"]),
        "source_uri": PUBCHEM_URI,
    }
    if record["cid"] != PUBCHEM_CID:
        raise RuntimeError("PubChem returned a different compound")
    document = {
        "schema": "pubchem-headspace-molecule-supplement/v1",
        "source": "PubChem PUG REST",
        "records": [record],
    }
    return (
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def acquire_pubchem_supplement(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PubChem supplement: {path}")
    response = json.loads(request_bytes(PUBCHEM_URI))
    rows = response.get("PropertyTable", {}).get("Properties", [])
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("unexpected PubChem property response")
    payload = pubchem_supplement(rows[0])
    if sha256_bytes(payload) != hub.PUBCHEM_SUPPLEMENT_SHA256:
        raise RuntimeError("PubChem supplement bytes changed from reviewed contract")
    atomic_bytes(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyrfume-root",
        type=Path,
        default=ROOT / "tmp" / "pyrfume_data_source_20260828",
    )
    parser.add_argument(
        "--opera-zip",
        type=Path,
        default=ROOT / ".cache" / "epa_opera" / "S1.zip",
    )
    parser.add_argument(
        "--pubchem-supplement",
        type=Path,
        default=ROOT
        / "benchmarks"
        / "source_data"
        / "pubchem_headspace_molecules_v1.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform no network or writes; verify all existing inputs",
    )
    args = parser.parse_args()
    pyrfume = args.pyrfume_root.expanduser().resolve()
    opera = args.opera_zip.expanduser().resolve()
    supplement = args.pubchem_supplement.expanduser().resolve()
    if not args.verify_only:
        if not pyrfume.exists():
            acquire_pyrfume(pyrfume)
        if not opera.exists():
            acquire_opera(opera)
        if not supplement.exists():
            acquire_pubchem_supplement(supplement)
    audit = hub.verify_sources(
        pyrfume.resolve(strict=True),
        opera.resolve(strict=True),
        supplement.resolve(strict=True),
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
