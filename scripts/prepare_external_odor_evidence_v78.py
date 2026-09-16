"""Exact identity/label joins against all newly acquired public archives."""

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def norm(value):
    return "".join(c for c in value.casefold() if c.isalnum())


def labels(row):
    if "Descriptors" in row:
        return [v.strip() for v in row["Descriptors"].split(";") if v.strip()]
    if "Labels" in row:
        return [v.strip() for v in row["Labels"].split(";") if v.strip()]
    if "Filtered Descriptors" in row:
        return [v.strip() for v in row["Filtered Descriptors"].split(";") if v.strip()]
    if "Descriptor 1" in row:
        return [
            row[k].strip()
            for k in ("Descriptor 1", "Descriptor 2", "Descriptor 3")
            if row.get(k, "").strip()
        ]
    if "descriptors" in row:
        value = ast.literal_eval(row["descriptors"])
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ValueError("invalid descriptor list")
        return value
    return [k for k, v in row.items() if k != "Stimulus" and v in ("1", "1.0", "True")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--space", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from rdkit import Chem

    manifest = json.loads((a.sources / "manifest.json").read_text(encoding="utf8"))
    space = json.loads(a.space.read_text(encoding="utf8"))
    nodes = {r["id"]: r for r in space["concepts"]}
    remaining = {
        k: n
        for k, n in nodes.items()
        if n["resolution"]["status"] == "specific_odor_reference_still_missing"
    }
    aliases = defaultdict(set)
    for k, n in remaining.items():
        for term in [n["label_en"], *n["aliases"], *n["source_terms"]]:
            aliases[norm(term)].add(k)
    for name, meta in manifest["files"].items():
        if (
            hashlib.sha256((a.sources / name).read_bytes()).hexdigest()
            != meta["sha256"]
        ):
            raise ValueError("acquired source changed")
    annotations = defaultdict(dict)
    identities = defaultdict(dict)
    statistics = {}
    for dataset in (
        "flavornet",
        "goodscents",
        "arctander_1960",
        "leffingwell",
        "sigma_2014",
        "ifra_2019",
        "aromadb",
    ):
        folder = a.sources / dataset
        by_cid = {}
        stats = {
            "molecules": 0,
            "behavior_rows": 0,
            "missing_identity_rows": 0,
            "ambiguous_labels": 0,
        }
        with (folder / "molecules.csv").open(encoding="utf8", newline="") as f:
            for row in csv.DictReader(f):
                smiles = row.get("IsomericSMILES", "")
                if not smiles or "." in smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                graph = Chem.MolToSmiles(mol, isomericSmiles=True)
                cid = row["CID"]
                by_cid[cid] = graph
                stats["molecules"] += 1
                for name in (row.get("name", ""), row.get("IUPACName", "")):
                    matches = aliases.get(norm(name), set())
                    if len(matches) == 1:
                        key = next(iter(matches))
                        identities[key].setdefault(graph, []).append(
                            {"dataset": dataset, "CID": cid, "name": name}
                        )
        by_stimulus = {}
        with (folder / "stimuli.csv").open(encoding="utf8", newline="") as f:
            for row in csv.DictReader(f):
                cid = row.get(
                    "CID",
                    row.get(
                        "new_CID", row["Stimulus"] if dataset == "sigma_2014" else ""
                    ),
                )
                by_stimulus[row["Stimulus"]] = by_cid.get(cid)
        behavior = folder / (
            "behavior_1_sparse.csv" if dataset == "arctander_1960" else "behavior.csv"
        )
        with behavior.open(encoding="utf8", newline="") as f:
            for row in csv.DictReader(f):
                stats["behavior_rows"] += 1
                graph = by_stimulus.get(row["Stimulus"])
                if graph is None:
                    stats["missing_identity_rows"] += 1
                    continue
                for term in labels(row):
                    matches = aliases.get(norm(term), set())
                    if len(matches) > 1:
                        stats["ambiguous_labels"] += 1
                    if len(matches) != 1:
                        continue
                    key = next(iter(matches))
                    annotations[key].setdefault(graph, []).append(
                        {
                            "dataset": dataset,
                            "stimulus": row["Stimulus"],
                            "descriptor": term,
                        }
                    )
        statistics[dataset] = stats
    records = {}
    for key in remaining:
        groups = annotations.get(key, {})
        chemical = identities.get(key, {})
        records[key] = {
            "annotated_identities": groups,
            "named_identities": chemical,
            "source_annotation_reference_possible": len(groups) >= 3,
            "unique_named_molecule_reference_possible": len(chemical) == 1,
            "measured_146_axis_profile_obtained": False,
        }
    report = {
        "schema": "external-odor-evidence-audit/v78",
        "source_commit": manifest["commit"],
        "source_manifest_sha256": hashlib.sha256(
            (a.sources / "manifest.json").read_bytes()
        ).hexdigest(),
        "remaining_input_count": len(remaining),
        "dataset_statistics": statistics,
        "three_identity_annotation_cohorts": sum(
            r["source_annotation_reference_possible"] for r in records.values()
        ),
        "unique_named_molecules": sum(
            r["unique_named_molecule_reference_possible"] for r in records.values()
        ),
        "no_qualified_reference_route": sum(
            not r["source_annotation_reference_possible"]
            and not r["unique_named_molecule_reference_possible"]
            for r in records.values()
        ),
        "measured_146_axis_profiles_added": 0,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (a.output / "evidence.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
