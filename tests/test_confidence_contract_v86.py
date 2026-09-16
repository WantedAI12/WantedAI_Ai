import json
from dataclasses import fields

import pytest

from fragrance_ai.recommender.confidence_contract import confidence_fields


@pytest.mark.parametrize("label", ["heuristic_only", "heuristic", "synthetic_structural_proxy", "", "0.87"])
def test_labels_never_become_invented_probability(label):
    payload = confidence_fields(label)
    assert payload["confidence"] is None
    assert payload["confidence_kind"] == (label or "unavailable")
    assert json.loads(json.dumps(payload, allow_nan=False))["confidence"] is None


@pytest.mark.parametrize("value", [None, 0., .87, 1., 1])
def test_nullable_double_contract(value):
    payload = confidence_fields(value)
    assert payload["confidence"] is None or type(payload["confidence"]) is float
    assert payload["confidence"] == value


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -.1, 1.1, {}, []])
def test_invalid_numeric_values_do_not_escape_as_json(value):
    with pytest.raises(ValueError):
        confidence_fields(value)


def test_recipe_serialization_keeps_internal_evidence_level_but_never_sends_a_string(monkeypatch):
    from types import SimpleNamespace
    from fragrance_ai.recommender import models
    # Exercise the exact RecipeResult serializer without an expensive search.
    cls = next(value for value in vars(models).values() if isinstance(value, type)
               and hasattr(value, "__dataclass_fields__") and "confidence" in value.__dataclass_fields__)
    instance = object.__new__(cls)
    for field in fields(cls):
        object.__setattr__(instance, field.name, None)
    object.__setattr__(instance, "confidence", "heuristic_only")
    object.__setattr__(instance, "brief", SimpleNamespace(constraints=SimpleNamespace(explicit_bans=set())))
    monkeypatch.setattr(models, "asdict", lambda value: {"confidence": value.confidence, "brief": {"constraints": {}}})
    monkeypatch.setattr("fragrance_ai.recommender.regulatory_status.regulatory_summary", lambda _: {})
    result = instance.to_dict()
    assert result["confidence"] is None and result["confidence_kind"] == "heuristic_only"
    assert instance.confidence == "heuristic_only"
