"""Validate a supplier CSV and emit a registry JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.catalog import IngredientCatalog  # noqa: E402
from fragrance_ai.recommender.supplier import SupplierRegistry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registry = SupplierRegistry.from_csv(args.csv)
    catalog = IngredientCatalog.load_builtin()
    valid_ids = {item.ingredient_id for item in catalog.ingredients}
    unknown = sorted({record.ingredient_id for record in registry.records} - valid_ids)
    if unknown:
        raise SystemExit(f"Unknown ingredient ids: {', '.join(unknown)}")
    registry.metadata.update(
        {
            "registry_status": "operator_provided",
            "validation": "schema_and_catalog_id_checked",
        }
    )
    registry.to_json(args.output)
    print(f"validated_records={len(registry.records)} output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
