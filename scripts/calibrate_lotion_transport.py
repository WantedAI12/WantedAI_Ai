"""Fit one lotion coefficient offline from explicit context-bound observations."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fragrance_ai.research.lotion_calibration import LotionCalibrationRequest, fit_lotion_transport
from fragrance_ai.recommender.catalog import IngredientCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path; fitted evidence is not overwritten")
    request = LotionCalibrationRequest.model_validate_json(args.input.read_text(encoding="utf-8"))
    result = fit_lotion_transport(request, IngredientCatalog.load_builtin())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "parameter": result["parameter"],
        "fitted_value": result["fitted_value"], "validation_after": result["validation_after"],
        "automatic_promotion_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
