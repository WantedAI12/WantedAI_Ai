from __future__ import annotations

from scripts.acquire_universal_intensity_physchem_v1 import _annotation_strings


def test_annotation_strings_preserve_text_and_numeric_units():
    payload = {
        "Record": {
            "Section": [
                {"Information": [{"Value": {"StringWithMarkup": [{"String": "1.2 mmHg at 25 C"}]}}]},
                {"Information": [{"Value": {"Number": [3.4], "Unit": "Pa"}}]},
            ]
        }
    }
    values = _annotation_strings(payload)
    assert "1.2 mmHg at 25 C" in values
    assert "[3.4] Pa" in values


def test_annotation_strings_deduplicate_without_reordering():
    payload = {
        "StringWithMarkup": [
            {"String": "same"},
            {"String": "same"},
            {"String": "later"},
        ]
    }
    assert _annotation_strings(payload) == ["same", "later"]
