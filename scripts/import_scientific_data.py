"""Import verified molecular, volatility and odor-threshold properties."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.catalog import IngredientCatalog  # noqa: E402
from fragrance_ai.recommender.science import ScientificPropertyStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    catalog = IngredientCatalog.load_builtin()
    valid_ids = {item.ingredient_id for item in catalog.ingredients}
    store = ScientificPropertyStore(args.db)
    imported = store.import_csv(args.csv, allowed_ingredient_ids=valid_ids)
    print(f"imported_scientific_records={imported} db={Path(args.db).resolve()}")


if __name__ == "__main__":
    main()
