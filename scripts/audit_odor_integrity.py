"""Read-only material/physics audit for the corrected local R&D catalog."""

import argparse
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--installed-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh audit output")
    sys.path.insert(0, str(args.installed_package.resolve(strict=True)))
    import fragrance_ai
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.catalog import HistoricalReferenceCorpus
    from fragrance_ai.recommender.global_profile_search import profile_upper_bound
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    from fragrance_ai.recommender.odor_integrity import is_registry_material, registry_odor_rejection
    from fragrance_ai.recommender.registry_activation import load_runtime_catalog
    from fragrance_ai.recommender.science import ScientificPropertyStore
    if not Path(fragrance_ai.__file__).resolve().is_relative_to(args.installed_package.resolve()):
        raise RuntimeError("incorrect runtime package")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    binding = manifest["runtime_catalog"]
    catalog, activation, _ = load_runtime_catalog(args.manifest.parent / binding["path"],
        expected_sha256=binding["sha256"], expected_wheel_sha256=binding["wheel_sha256"],
        expected_registry_sha256=binding["registry_sha256"])
    with NaturalLanguagePerfumeryAI(catalog=catalog, corpus=HistoricalReferenceCorpus(args.output.parent / "absent-reference.db")) as ai:
        limits = RecipeConstraints(max_risk_tier=2, enable_registry_trace_candidates=True)
        brief = ai.parser.parse("balanced fresh floral woody fragrance", limits)
        pool, rejected = ai.screen.screen(ai.catalog, brief, ai.supplier_registry, date(2026, 9, 5))
        unrestricted, _ = ai.screen.screen(ai.catalog, replace(brief, constraints=replace(limits, experimental_disable_safety=True)), ai.supplier_registry, date(2026, 9, 5))
        records = ai.scientific_store.get_many([item.ingredient_id for item in pool])
        queries = []
        ai.scientific_store.connection.set_trace_callback(queries.append)
        connected = ScientificPropertyStore.with_catalog_structures(pool, records)
        ai.scientific_store.connection.set_trace_callback(None)
        assert not queries, "calculated descriptor merge must not query the database"
        assert all(registry_odor_rejection(item) is None for item in pool + unrestricted)
        probes = {}
        for short_id in ("7674cf9544f28b16fcecae9a", "c8a33c10b1bbd5e612dafeba", "34f83716011a251044cf41ee"):
            item = next(row for row in catalog.ingredients if row.ingredient_id == "registry_" + short_id)
            assert not item.formulation_ready and item.blocked and not item.profile
            assert item.ingredient_id not in {row.ingredient_id for row in unrestricted}
            probes[item.name] = {"ingredient_id": item.ingredient_id, "status": item.odor_evidence_status,
                                "odor_assertions": item.odor_assertions, "present_in_unrestricted_candidate_pool": False}
        report = {
            "schema": "perfumery-odor-integrity-audit/v1", "status": "local_research_only_not_deployed",
            "catalog_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "wheel_sha256": binding["wheel_sha256"], "catalog_sha256": binding["sha256"],
            "activation_report": activation.to_dict(), "request_constraints": {"risk_tier": 2, "registry_opt_in": True, "safety_disabled": False},
            "request_eligible_candidates": len(pool), "request_eligible_registry_candidates": sum(is_registry_material(row) for row in pool),
            "request_rejections": rejected, "experimental_override_candidates": len(unrestricted),
            "unverified_odor_candidate_leaks": 0, "known_counterexamples": probes,
            "scientific_properties": {
                "existing_records_connected": len(records), "records_after_calculated_structure_overlay": len(connected),
                "rdkit_calculated_records_added": len(connected) - len(records), "overlay_sql_queries": len(queries),
                "vapor_pressure_records": sum(p.vapor_pressure_pa_25c is not None for p in connected.values()),
                "boiling_point_records": sum(p.boiling_point_c is not None for p in connected.values()),
                "odor_threshold_records": sum(p.odor_threshold_ppm is not None for p in connected.values()),
                "new_vapor_pressure_or_boiling_point_or_threshold_measurements": 0,
            },
            "optimistic_single_axis_profile_bounds": {dimension: profile_upper_bound(pool, {dimension: 1.}) for dimension in SCENT_DIMENSIONS},
            "bounds_are_not_achieved_scores_or_human_accuracy": True,
            "source_specific_redistribution_cleared": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("optimistic_single_axis_profile_bounds", "request_rejections", "known_counterexamples")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
