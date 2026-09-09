"""Build a local-research catalog from fingerprinted odor fields, not taste.

Original registry and source snapshots are read-only. The produced catalog
does not grant redistribution, supplier, toxicology or manufacturing approval.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import zipfile
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.catalog import IngredientCatalog  # noqa: E402
from fragrance_ai.recommender.odor_integrity import ODOR_INTEGRITY_VERSION, normalize_term  # noqa: E402
from fragrance_ai.recommender.odor_integrity import LEGACY_ODOR_PROJECTION, EXPANDED_ODOR_PROJECTION, CONCEPT_ODOR_PROJECTION
from fragrance_ai.recommender.registry_activation import activate_registry_conditionals, write_runtime_catalog, load_runtime_catalog  # noqa: E402
from fragrance_ai.recommender.promotion_activation import _valid_cas_number  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_wheel_sources(wheel):
    """A claimed wheel hash must represent the code actually building the data."""
    with zipfile.ZipFile(wheel) as archive:
        sources = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                   if name.startswith("fragrance_ai/") and name.endswith(".py")}
    actual = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in (ROOT / "fragrance_ai").rglob("*.py")}
    if sources != actual:
        raise ValueError("wheel source does not match the catalog builder runtime")
    return sources


def tokens(value):
    return [part.strip() for part in re.split(r"[;,|/]", value or "") if normalize_term(part) not in {"", "nan", "none", "null", "0"}]


def preserve_existing_positive_profiles(candidate, previous):
    """A request-independent extension policy, never best-score model selection.

    Keep already established positive projections; extend formerly unsupported
    rows. New negative/conflicting/structural findings always take precedence.
    """
    old = {i.ingredient_id:i for i in previous.ingredients}
    if set(old) != {i.ingredient_id for i in candidate.ingredients}:
        raise ValueError('profile preservation requires the same identity inventory')
    items, preserved = [], 0
    for new in candidate.ingredients:
        prior = old[new.ingredient_id]
        if (new.odor_assertions != prior.odor_assertions or new.odor_evidence_refs != prior.odor_evidence_refs):
            raise ValueError('profile preservation cannot conceal changed source assertions')
        if (new.ingredient_id.startswith('registry_') and prior.formulation_ready and not prior.blocked
                and new.formulation_ready and not new.blocked):
            new = replace(new, profile=prior.profile, pyramid=prior.pyramid,
                odor_projection_version=prior.odor_projection_version)
            preserved += 1
        items.append(new)
    return IngredientCatalog(items, metadata=candidate.metadata), preserved


def resolve_cross_source_identity(source, stimulus, cas_map, identities, cid_identities):
    """Fallback only through an explicit valid CAS -> CID -> unique identity."""
    cid = str(cas_map.get(stimulus, stimulus)) if source == 'goodscents' else stimulus
    direct = identities.get((source, cid))
    if direct is not None:
        return direct, cid, False
    if (source != 'goodscents' or not _valid_cas_number(stimulus) or stimulus not in cas_map
            or not cid.isdecimal() or int(cid) <= 0):
        return None, cid, False
    linked = cid_identities.get(cid, set())
    return (next(iter(linked)), cid, True) if len(linked) == 1 else (None, cid, False)


def read_assertions(registry_path, source_root):
    connection = sqlite3.connect(Path(registry_path).resolve().as_uri() + "?mode=ro", uri=True)
    source_rows = connection.execute("SELECT source_id,file_kind,path,sha256,redistribution_allowed FROM source_files").fetchall()
    files = {(source, kind): (Path(path), sha, permission) for source, kind, path, sha, permission in source_rows}
    identity_rows = connection.execute("SELECT registry_id,source_id,source_cid FROM ingredient_sources").fetchall()
    identities = {(source, str(cid)): identifier for identifier, source, cid in identity_rows}
    cid_identities, cid_sources = defaultdict(set), defaultdict(set)
    for identifier, source, cid in identity_rows:
        if (source, 'molecules') in files and str(cid).isdecimal() and int(cid) > 0:
            cid_identities[str(cid)].add(identifier)
            cid_sources[str(cid)].add(source)
    connection.close()
    root = Path(source_root).resolve(strict=True)
    for path, expected, _ in files.values():
        path = path.resolve(strict=True)
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("source snapshot path or hash mismatch")
    mapping_file = files.get(("goodscents", "cas_to_cid")) or files.get(("goodscents", "cas_to_cid.json"))
    if mapping_file is None:
        matches = [value for (source, _), value in files.items() if source == "goodscents" and value[0].name == "cas_to_cid.json"]
        if len(matches) != 1:
            raise ValueError("fingerprinted GoodScents identity mapping required")
        mapping_file = matches[0]
    cas_map = json.loads(mapping_file[0].read_text(encoding="utf-8"))
    assertions, references, source_counts, unmatched = {}, {}, {}, {}
    fallback_records, fallback_identities = 0, set()
    registry_hash = digest(registry_path)
    for (source, kind), (path, source_hash, _) in files.items():
        if kind != "behavior":
            continue
        used, missed = 0, 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for record_number, row in enumerate(csv.DictReader(handle), 1):
                stimulus = str(row.get("Stimulus", "")).strip()
                identifier, cid, cross_source = resolve_cross_source_identity(source, stimulus, cas_map, identities, cid_identities)
                if identifier is None:
                    missed += 1
                    continue
                identity_link = {}
                if cross_source:
                    fallback_records += 1
                    fallback_identities.add(identifier)
                    identity_link = {'identity_resolution': {
                        'method':'fingerprinted_CAS_CID_unique_cross_source_registry_identity',
                        'cas':stimulus, 'cid':cid, 'registry_id':identifier,
                        'registry_sha256':registry_hash, 'cas_mapping_sha256':mapping_file[1],
                        'identity_source_sha256':{s:files[(s,'molecules')][1] for s in sorted(cid_sources[cid])}}}
                if source == "leffingwell":
                    fields = [(key, [key]) for key, value in row.items() if key != "Stimulus" and str(value).strip() == "1"]
                elif source in {"goodscents", "flavornet"}:
                    fields = [("Descriptors", tokens(row.get("Descriptors")))]
                elif source == "aromadb":
                    # A cleaned column cannot erase odorless/conflicting/raw
                    # assertions from the same observation.
                    fields = [(field, tokens(row.get(field))) for field in ("Raw Descriptors", "Filtered Descriptors")]
                elif source == "ifra_2019":
                    fields = [(key, tokens(row.get(key))) for key in ("Descriptor 1", "Descriptor 2", "Descriptor 3")]
                elif source == "flavordb":
                    # Flavor Percepts/Modifiers are never odor assertions.
                    fields = [("Odor Percepts", tokens(row.get("Odor Percepts")))]
                else:
                    continue
                tag = "flavordb_odor" if source == "flavordb" else source
                conditions = {key: row[key] for key in ("Concentration", "Dilution", "Temperature", "Solvent", "Odor Modifiers", "Modifiers") if row.get(key)}
                for field, descriptors in fields:
                    for descriptor in descriptors:
                        term = normalize_term(descriptor)
                        assertions.setdefault(identifier, set()).add(tag + ":" + term)
                        references.setdefault(identifier, []).append(json.dumps({
                            "source_tag": tag, "source_id": source, "source_file_sha256": source_hash,
                            "source_record_id": stimulus, "source_record_number": record_number, "source_field": field,
                            "semantic_role": "odor", "descriptor": descriptor,
                            "conditions_status": "source_modifiers_retained_not_quantitative_measurement" if conditions else "not_reported_in_source",
                            "source_conditions_or_modifiers": conditions, "normalization_version": ODOR_INTEGRITY_VERSION,
                            **identity_link,
                        }, ensure_ascii=False, sort_keys=True))
                        used += 1
        source_counts[source] = used
        unmatched[source] = missed
    return ({key: tuple(sorted(value)) for key, value in assertions.items()},
            {key: tuple(value) for key, value in references.items()},
            {"source_assertions": source_counts, "unmatched_source_records": unmatched,
             "flavor_fields_used_as_odor": False, "source_specific_redistribution_cleared": False,
             "unique_cross_source_records_recovered":fallback_records,
             "unique_cross_source_identities_recovered":len(fallback_identities),
             "source_sha256": {source + "/" + kind: value[1] for (source, kind), value in files.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument('--odor-projection-version', choices=[LEGACY_ODOR_PROJECTION, EXPANDED_ODOR_PROJECTION, CONCEPT_ODOR_PROJECTION],
                        default=LEGACY_ODOR_PROJECTION)
    parser.add_argument('--preserve-positive-runtime', type=Path)
    parser.add_argument('--preserve-runtime-sha256')
    parser.add_argument('--preserve-wheel-sha256')
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    preservation = (args.preserve_positive_runtime, args.preserve_runtime_sha256, args.preserve_wheel_sha256)
    if any(preservation) and not all(preservation):
        parser.error('profile preservation needs the source runtime and both artifact hashes')
    if digest(args.registry) != args.registry_sha256:
        parser.error("registry hash mismatch")
    sources = verify_wheel_sources(args.wheel)
    with zipfile.ZipFile(args.wheel) as archive:
        runtime_data = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                        if name.startswith("fragrance_ai/data/") and Path(name).suffix in {".db", ".json", ".npz"}}
    assertions, refs, provenance = read_assertions(args.registry, args.source_root)
    catalog, report = activate_registry_conditionals(IngredientCatalog.load_builtin(), args.registry,
        expected_sha256=args.registry_sha256, verified_odor_assertions=assertions, verified_odor_refs=refs,
        odor_projection_version=args.odor_projection_version)
    if args.preserve_positive_runtime:
        old, _, _ = load_runtime_catalog(args.preserve_positive_runtime,
            expected_sha256=args.preserve_runtime_sha256, expected_wheel_sha256=args.preserve_wheel_sha256,
            expected_registry_sha256=args.registry_sha256)
        catalog, kept = preserve_existing_positive_profiles(catalog, old)
        provenance['profile_extension_policy'] = {
            'mode':'preserve_existing_positive_profiles_extend_unrepresented_materials',
            'preserved_registry_profiles':kept, 'source_runtime_sha256':args.preserve_runtime_sha256,
            'source_wheel_sha256':args.preserve_wheel_sha256,
            'new_negative_and_structural_findings_override_preservation':True,
            'request_or_benchmark_scores_used_for_selection':False}
    args.output.mkdir(parents=True)
    wheel_sha = digest(args.wheel)
    catalog_path = args.output / "runtime_catalog_v3.json.gz"
    catalog_sha = write_runtime_catalog(catalog_path, catalog, report, {"reference_molecules": report.reference_molecules_connected}, wheel_sha256=wheel_sha)
    manifest = {"schema": "perfumery-odor-integrity-catalog/v1", "status": "local_research_only_not_deployed",
                "odor_integrity_version": ODOR_INTEGRITY_VERSION,
                "odor_projection_version": args.odor_projection_version,
                "material_projection_versions": sorted({i.odor_projection_version or LEGACY_ODOR_PROJECTION for i in catalog.ingredients}),
                "runtime_catalog": {"path": catalog_path.name, "sha256": catalog_sha,
                                    "registry_sha256": args.registry_sha256, "wheel_sha256": wheel_sha,
                                    "connected_rows": len(catalog.ingredients), "active_odorants": report.active_odorant_candidates},
                "activation_report": report.to_dict(), "provenance": provenance,
                "builder_script_sha256": digest(__file__), "runtime_source_sha256": sources,
                "runtime_data_sha256": runtime_data,
                "old_profile_scores_comparable": False, "actual_human_accuracy_claim": False,
                "source_specific_redistribution_cleared": False}
    (args.output / "catalog_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"catalog_sha256": catalog_sha, "activation_report": report.to_dict(), "source_assertions": provenance["source_assertions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
