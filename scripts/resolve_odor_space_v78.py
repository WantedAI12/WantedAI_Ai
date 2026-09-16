"""Resolve the entire V77 inventory with typed, source-bound evidence.

Model-inferred references are NEVER relabelled measured observations. Historical
words and taxonomy headers do not become fabricated quantitative odors.
"""

import argparse
import ast
import csv
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )


def label(word):
    return (
        re.sub(r"_(?:adj|noun|verb|adv)$", "", word)
        .replace("_", " ")
        .casefold()
        .strip()
    )


def synsets(record):
    raw = record.get("synset", "")
    if not raw:
        return ()
    if re.fullmatch(r"[a-z0-9_-]+\.[nvasr]\.\d{2}", raw):
        return (raw,)
    if re.fullmatch(r"s\d+", raw):
        # Opaque language-specific identifiers are not English WordNet names;
        # do not join unrelated namespaces merely because integers coincide.
        return ()
    values = ast.literal_eval(raw)
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError("invalid source synset list")
    return tuple(v for v in values if re.fullmatch(r"[a-z0-9_-]+\.[nvasr]\.\d{2}", v))


def identity(word):
    return "".join(c for c in word.casefold() if c.isalnum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-output", type=Path, required=True)
    p.add_argument("--external-evidence", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.profile_output.exists():
        raise ValueError("new profile filename required")
    os.environ["PERFUMERY_AI_LOCAL_PROFILE"] = "disabled"
    import numpy as np
    from fragrance_ai.recommender.formulation_core import FormulationCore

    parent = json.loads(a.parent_profile.read_text(encoding="utf8"))

    def read(role):
        item = parent[role]
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("parent binding mismatch: " + role)
        return json.loads(path.read_text(encoding="utf8"))

    old = read("odor_space")
    value = deepcopy(old)
    ref = read("lotion_target_reference")
    original_profiles = deepcopy(ref["profiles"])
    original_metadata = deepcopy(ref["concept_metadata"])
    core = FormulationCore(
        ROOT / parent["formulation_core"]["path"], parent["formulation_core"]["sha256"]
    )
    nodes = {r["id"]: r for r in value["concepts"]}
    bindings = value["reference_bindings"]
    old_missing = {
        k for k, n in nodes.items() if n["quantitative_status"] == "reference_missing"
    }
    assert len(old_missing) == 3595, "do not silently change the audited denominator"
    # Named chemical references are not population-level odor categories. One
    # identified molecule's actual observed profile can anchor its own name;
    # this does not relax the three-identity rule for general descriptor means.
    from rdkit import Chem
    from fragrance_ai.research.atlas_profiles import load_atlas

    source_path = ROOT / ".benchmarks/atlas_profiles_v50/source"
    observations, endpoints, atlas_source = load_atlas(source_path)
    if endpoints != ref["endpoints"]:
        raise ValueError("named molecule endpoint mismatch")
    observed_by_graph = defaultdict(list)
    for row in observations:
        if row["level"] == "high" and row["graph"]:
            q = np.array([row[h] for h in ("applicability", "use")], float)
            q /= q.sum(-1, keepdims=True)
            observed_by_graph[row["graph"]].append(q)
    names = defaultdict(set)
    with (source_path / "molecules.csv").open(encoding="utf8", newline="") as f:
        for record in csv.DictReader(f):
            mol = Chem.MolFromSmiles(record["IsomericSMILES"])
            if mol is None:
                continue
            graph = Chem.MolToSmiles(mol, isomericSmiles=True)
            if graph not in observed_by_graph:
                continue
            for name in (record["name"], record["IUPACName"]):
                names[name.casefold().strip()].add(graph)
    named = {}
    for key, node in nodes.items():
        if (
            key not in old_missing
            or node.get("contextual_only")
            or node["kind"] != "odor"
        ):
            continue
        identities = {
            g
            for name in [node["label_en"], *node["aliases"]]
            for g in names[name.casefold().strip()]
        }
        if len(identities) != 1:
            continue
        graph = next(iter(identities))
        refkey = "molecule_" + key
        mean = np.mean(observed_by_graph[graph], axis=0)
        mean /= mean.sum(-1, keepdims=True)
        metadata = {
            "reference_kind": "named_molecule_observed_profile",
            "measured_profile": True,
            "source_identity_smiles": graph,
            "distinct_identity_groups": 1,
            "source_observations": len(observed_by_graph[graph]),
            "atlas_source": atlas_source,
            "scope": "that_named_chemical_not_all_members_of_an_odor_category",
            "candidate_catalog_used": False,
            "recipe_outcomes_used": False,
        }
        ref["profiles"][refkey] = mean.tolist()
        ref["concept_metadata"][refkey] = metadata
        bindings[key] = refkey
        node.update(
            reference_key=refkey,
            quantitative_status="named_observed_reference_connected",
        )
        named[refkey] = metadata
    endpoint_index = {k: i for i, k in enumerate(core.fine_endpoints)}
    aliases = defaultdict(set)
    for key, node in nodes.items():
        for name in [
            key,
            node["label_en"],
            *node["aliases"],
            *node.get("source_terms", []),
        ]:
            aliases[identity(name)].add(key)
    annotation_routes = {}
    for key, node in nodes.items():
        if (
            key not in old_missing
            or bindings.get(key)
            or node.get("contextual_only")
            or node["kind"] != "odor"
        ):
            continue
        routes = {
            endpoint_index[name]
            for name in [key, *node.get("source_terms", [])]
            if name in endpoint_index
        }
        if routes:
            annotation_routes[key] = routes
    cohorts = {
        key: sorted(
            g
            for g, indices in core.manifest["source_annotations"].items()
            if "." not in g and set(indices) & route
        )
        for key, route in annotation_routes.items()
    }
    cohorts = {key: graphs for key, graphs in cohorts.items() if len(graphs) >= 3}
    graphs = sorted({g for values in cohorts.values() for g in values})
    prediction = {}
    for start in range(0, len(graphs), 128):
        batch = graphs[start : start + 128]
        result = core.molecular(batch)
        x = np.stack([result[h] for h in ("applicability", "use")], axis=1).astype(
            float
        )
        if not np.isfinite(x).all() or np.any(x < 0) or np.any(x.sum(-1) <= 0):
            raise ValueError("invalid inferred source profile")
        x /= x.sum(-1, keepdims=True)
        prediction.update(zip(batch, x))
        print(
            json.dumps(
                {
                    "source_molecules_predicted": min(start + 128, len(graphs)),
                    "total": len(graphs),
                }
            ),
            flush=True,
        )
    inferred = {}
    for key, group in cohorts.items():
        x = np.stack([prediction[g] for g in group])
        mean = x.mean(0)
        mean /= mean.sum(-1, keepdims=True)
        refkey = "inferred_" + key
        source_digest = hashlib.sha256(
            json.dumps(group, sort_keys=True).encode()
        ).hexdigest()
        metadata = {
            "reference_kind": "source_annotation_conditioned_model_inference",
            "measured_profile": False,
            "distinct_identity_groups": len(group),
            "cohort_sha256": source_digest,
            "source_model_sha256": core.sha256,
            "annotation_indices": sorted(annotation_routes[key]),
            "source_graphs_used": group,
            "candidate_catalog_used": False,
            "recipe_outcomes_used": False,
            "mean_absolute_profile_dispersion": float(np.abs(x - mean).mean()),
            "dispersion_is_not_human_error_interval": True,
            "reference_is_model_estimate": True,
        }
        ref["profiles"][refkey] = mean.tolist()
        ref["concept_metadata"][refkey] = metadata
        bindings[key] = refkey
        nodes[key]["reference_key"] = refkey
        nodes[key]["quantitative_status"] = "model_inferred_reference_connected"
        inferred[refkey] = metadata

    external_required = None
    external_review = {}
    if a.external_evidence:
        evidence_path = a.external_evidence / "evidence.json"
        additional = json.loads(evidence_path.read_text(encoding="utf8"))
        external_required = sorted(additional)
        for key, record in additional.items():
            if bindings.get(key):
                continue
            if key in {"benzoin", "odorless"}:
                external_review[key] = (
                    "chemical/natural-resin homonym"
                    if key == "benzoin"
                    else "absence/intensity is not a normalized positive odor profile"
                )
                continue
            if record["source_annotation_reference_possible"]:
                group = record["annotated_identities"]
                kind = "source_annotation_conditioned_model_inference"
            elif (
                key in {"acetaldehyde", "pyrazine", "styrene"}
                and record["unique_named_molecule_reference_possible"]
            ):
                group = record["named_identities"]
                kind = "named_molecule_model_inference"
            else:
                continue
            selected = sorted(group)
            output = core.molecular(selected)
            x = np.stack([output[h] for h in ("applicability", "use")], axis=1).astype(
                float
            )
            if not np.isfinite(x).all() or np.any(x.sum(-1) <= 0):
                raise ValueError("invalid externally anchored prediction")
            x /= x.sum(-1, keepdims=True)
            mean = x.mean(0)
            mean /= mean.sum(-1, keepdims=True)
            refkey = "external_" + key
            metadata = {
                "reference_kind": kind,
                "reference_is_model_estimate": True,
                "measured_profile": False,
                "source_model_sha256": core.sha256,
                "distinct_identity_groups": len(selected),
                "source_graphs_used": selected,
                "external_source_records": group,
                "external_evidence_sha256": sha(evidence_path),
                "source_annotation_is_not_intensity": True,
                "candidate_catalog_used": False,
                "recipe_outcomes_used": False,
            }
            ref["profiles"][refkey] = mean.tolist()
            ref["concept_metadata"][refkey] = metadata
            bindings[key] = refkey
            nodes[key].update(
                reference_key=refkey,
                quantitative_status="model_inferred_reference_connected",
            )
            inferred[refkey] = metadata

    # Exact source synsets link lexical aliases, not nearest-word similarities.
    synset_refs = defaultdict(set)
    for row in value["external_vocabulary"]:
        word = label(row["record"].get("word", ""))
        owners = aliases.get(identity(word), set())
        refs = {bindings[k] for k in owners if bindings.get(k)}
        if len(refs) == 1:
            for syn in synsets(row["record"]):
                synset_refs[syn].update(refs)
    alias_evidence = {}
    for row in value["external_vocabulary"]:
        word = label(row["record"].get("word", ""))
        concept = (
            "lex_"
            + row["language"]
            + "_"
            + hashlib.sha256(word.encode()).hexdigest()[:16]
        )
        if concept not in nodes or bindings.get(concept):
            continue
        senses = synsets(row["record"])
        refs = {refkey for syn in senses for refkey in synset_refs[syn]}
        if len(refs) != 1:
            continue
        refkey = next(iter(refs))
        bindings[concept] = refkey
        nodes[concept].update(
            reference_key=refkey,
            quantitative_status="source_synset_reference_connected",
            contextual_only=False,
        )
        alias_evidence[concept] = {
            "method": "unique_source_synset_reference",
            "synsets": list(senses),
            "reference": refkey,
            "source_repository": value["source"]["repository"],
            "source_commit": value["source"]["commit"],
        }
        # Existing ambiguous lexical ownership is never replaced.
        value["aliases"].setdefault(word, concept)

    family_compositions = {}
    for key, node in nodes.items():
        if node["kind"] != "family" or bindings.get(key):
            continue
        children = {
            bindings[c]
            for c, n in nodes.items()
            if bindings.get(c)
            and any(
                identity(parent_name) == identity(node["label_en"])
                or identity(parent_name) == identity(key)
                for path in n["hierarchy_paths"]
                for parent_name in path
            )
        }
        if len(children) < 2:
            continue
        refkey = "taxonomy_" + key
        x = np.array([ref["profiles"][c] for c in sorted(children)])
        mean = x.mean(0)
        mean /= mean.sum(-1, keepdims=True)
        ref["profiles"][refkey] = mean.tolist()
        metadata = {
            "reference_kind": "source_taxonomy_child_composition",
            "source_child_references": sorted(children),
            "reference_is_model_estimate": any(c in inferred for c in children),
            "measured_profile": False,
            "candidate_catalog_used": False,
            "recipe_outcomes_used": False,
        }
        ref["concept_metadata"][refkey] = metadata
        bindings[key] = refkey
        node.update(
            reference_key=refkey, quantitative_status="taxonomy_reference_connected"
        )
        family_compositions[refkey] = metadata

    ledger = []
    from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS

    for key, node in nodes.items():
        refkey = bindings.get(key)
        if refkey and not node["coarse_projection"]:
            q = np.asarray(ref["profiles"][refkey]).mean(0)
            coarse = {
                axis: sum(
                    float(q[ref["endpoints"].index(name)])
                    for name in ATLAS_PROJECTION.get(axis, ())
                )
                for axis in SCENT_DIMENSIONS
            }
            total = sum(coarse.values())
            if total > 0:
                node["coarse_projection"] = {
                    k: v / total for k, v in coarse.items() if v > 0
                }
                node["projection_basis"] = (
                    "fixed_reference_warm_start_only_final_full_146_axis_evaluation"
                )
        if refkey:
            status = (
                "synset_alias_connected"
                if key in alias_evidence
                else "model_estimate_connected"
                if refkey in inferred
                else "taxonomy_composition_connected"
                if refkey in family_compositions
                else "source_reference_connected"
            )
            role = "odor_target"
        elif key in value.get("compositional_bindings", {}):
            status, role = "lexical_composition_connected", "odor_target"
        elif node.get("contextual_only"):
            status, role = "lexical_context_requires_specific_odor", "context"
        elif node["kind"] == "quality":
            status, role = "modifier_requires_odor_anchor", "modifier"
        elif node["kind"] == "family":
            status, role = "taxonomy_requires_subtype", "taxonomy"
        else:
            status, role = "specific_odor_reference_still_missing", "odor_target"
        decision = {
            "status": status,
            "semantic_role": role,
            "quantitative_reference_available": refkey is not None,
            "originally_missing_v77": key in old_missing,
            "reference": refkey,
            "reference_evidence_kind": ref["concept_metadata"]
            .get(refkey, {})
            .get("reference_kind", "observed_source")
            if refkey
            else None,
            "measured_data_fabricated": False,
        }
        if key in alias_evidence:
            decision["alias_evidence"] = alias_evidence[key]
        node["resolution"] = decision
        if key in old_missing:
            ledger.append({"concept_id": key, "label": node["label_en"], **decision})
    counts = dict(Counter(r["status"] for r in ledger))
    extension = {
        "schema": "odor-source-resolution/v78",
        "original_missing_count": len(old_missing),
        "every_original_entry_audited": len(ledger) == len(old_missing),
        "counts": counts,
        "required_reference_ids_before_deployment": external_required
        or sorted(
            r["concept_id"]
            for r in ledger
            if r["status"] == "specific_odor_reference_still_missing"
        ),
        "external_source_review": external_review,
        "model_inferred_reference_count": len(inferred),
        "taxonomy_reference_count": len(family_compositions),
        "measured_profiles_added": len(named),
        "source_profiles_unchanged": all(
            ref["profiles"][k] == v for k, v in original_profiles.items()
        ),
        "source_model_sha256": core.sha256,
        "source_profile_count": len(original_profiles),
        "source_profiles_sha256": hashlib.sha256(
            json.dumps(original_profiles, sort_keys=True).encode()
        ).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(
            json.dumps(
                {k: original_metadata[k] for k in original_profiles}, sort_keys=True
            ).encode()
        ).hexdigest(),
        "validation_revision": 2,
        "source_profile_keys": sorted(original_profiles),
        "new_reference_metadata": {**inferred, **family_compositions, **named},
        "model_inferred_references_are_not_measured": True,
        "candidate_catalog_used": False,
        "recipe_outcomes_used": False,
    }
    value.update(
        version="v78",
        resolution_extension={
            k: v for k, v in extension.items() if k != "new_reference_metadata"
        },
    )
    write(a.output / "space.json", value)
    ref["hierarchical_extension"].update(
        space_path="space.json",
        space_sha256=sha(a.output / "space.json"),
        reference_bindings=bindings,
        predicted_odor_profiles_used=bool(inferred),
    )
    ref["resolution_extension"] = extension
    write(a.output / "reference.json", ref)
    for role, filename in (
        ("odor_space", "space.json"),
        ("lotion_target_reference", "reference.json"),
    ):
        path = a.output / filename
        parent[role] = {
            "path": path.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha(path),
        }
    write(a.profile_output, parent)
    write(a.output / "resolution-ledger.json", ledger)
    report = {
        k: v
        for k, v in extension.items()
        if k not in ("new_reference_metadata", "source_profile_keys")
    }
    report.update(
        total_concepts=len(nodes),
        total_reference_profiles=len(ref["profiles"]),
        newly_connected=sum(r["reference"] is not None for r in ledger),
        still_missing_specific_odors=sum(
            r["status"] == "specific_odor_reference_still_missing" for r in ledger
        ),
        all_3595_measured_profiles_completed=False,
        code_sha256=sha(__file__),
        profile=str(a.profile_output.resolve()),
        space_sha256=sha(a.output / "space.json"),
        reference_sha256=sha(a.output / "reference.json"),
    )
    write(a.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
