"""Count actual qualified property joins over every audited active ingredient."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PERFUMERY_AI_LOCAL_PROFILE"] = "disabled"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from fragrance_ai.recommender.models import Ingredient
    from fragrance_ai.recommender.science import ScientificPropertyStore
    from fragrance_ai.recommender.physical_evidence_v76 import enrich

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "evidence", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.bank / "manifest.json").read_text(encoding="utf8"))
    if sha(a.bank / "materials.json") != manifest["files"]["materials.json"]:
        raise ValueError("audited material set changed")
    items = [
        Ingredient(**r)
        for r in json.loads((a.bank / "materials.json").read_text(encoding="utf8"))
    ]
    if len(items) != manifest["active_catalog_rows"] or len(
        {i.ingredient_id for i in items}
    ) != len(items):
        raise ValueError("the full unique active ingredient set is required")
    store = ScientificPropertyStore.load_builtin()
    try:
        before = store.with_catalog_structures(
            items, store.get_many([i.ingredient_id for i in items])
        )
    finally:
        store.close()
    after = enrich(items, before, (str(a.evidence), sha(a.evidence)))
    fields = ("vapor_pressure_pa_25c", "boiling_point_c", "odor_threshold_ppm")
    table = []
    for i in items:
        table.append(
            {
                "ingredient_id": i.ingredient_id,
                "before": {
                    f: getattr(before.get(i.ingredient_id), f, None) for f in fields
                },
                "after": {
                    f: getattr(after.get(i.ingredient_id), f, None) for f in fields
                },
            }
        )
    summary = {
        f: {
            "before_qualified": sum(r["before"][f] is not None for r in table),
            "after_qualified": sum(r["after"][f] is not None for r in table),
            "still_prior_only": sum(r["after"][f] is None for r in table),
        }
        for f in fields
    }
    result = {
        "schema": "active_scientific_coverage/v76",
        "active_materials": len(items),
        "counts_scope": "entire_audited_active_generation_pool_not_all_registry_rows_or_request_specific_eligibility",
        "source_materials_sha256": sha(a.bank / "materials.json"),
        "source_bank_manifest_sha256": sha(a.bank / "manifest.json"),
        "evidence_index_sha256": sha(a.evidence),
        "summary": summary,
        "measurement_values_fabricated": False,
        "unknowns_retained_as_priors": True,
        "rows": table,
    }
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "coverage.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
