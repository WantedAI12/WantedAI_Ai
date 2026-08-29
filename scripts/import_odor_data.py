"""Validate and import panel-derived ingredient odor observations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.catalog import IngredientCatalog  # noqa: E402
from fragrance_ai.recommender.odor_profiles import OdorProfileStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    catalog = IngredientCatalog.load_builtin()
    valid_ids = {item.ingredient_id for item in catalog.ingredients}
    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unknown = sorted({row.get("ingredient_id", "") for row in rows} - valid_ids)
    if unknown:
        raise SystemExit(f"Unknown ingredient ids: {', '.join(unknown)}")
    store = OdorProfileStore(args.db)
    count = store.import_csv(args.csv)
    print(
        f"imported_observations={count} observed_ingredients="
        f"{store.stats(catalog)['odor_observed_ingredients']} db={Path(args.db).resolve()}"
    )


if __name__ == "__main__":
    main()
