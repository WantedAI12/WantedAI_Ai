"""Operator-retired compositions, not banned scents, ingredients or test cases.

Only the ten recorded blend results selected by the operator are retired.
Identity uses product, finished-product concentration and actual material doses
at four-decimal-percent output precision. Scores and targets are never edited.
English/Korean records sharing a blend deliberately share one fingerprint.
"""
from collections.abc import Mapping
import hashlib
import math

POLICY_VERSION = "operator-retired-blends/v89-1"
SOURCE_AUDIT_SHA256 = "0b7ad7b24a8ea083f8b09ef8ea2377ffb36af827fc35052371858f1724e2772a"
# Provenance only; request text and case IDs are NOT used by the selector.
SOURCE_RECORDS = [
    {
        "case_id": "en-4",
        "product": "body_lotion",
        "brief": "aquatic scent",
        "concentration_percent": 0.5,
        "composition_sha256": "aa48bc88d4fed1fb8e34eddd7097c769091154d91f264bb3374cd5e7fd941e5b",
        "formula_id": "95b1e54173ace754dcf4081ecb517be37defd8b9af4f427a861c4103c54ac278",
        "source_response_sha256": "275a9adc07ab2c58966816d59cc1cb6b218c939383f9d9dfe9f84ae550cb524e"
    },
    {
        "case_id": "extra-13",
        "product": "body_lotion",
        "brief": "opening citrus, heart floral, drydown woody",
        "concentration_percent": 0.5,
        "composition_sha256": "752010279e42c9103c302193002262a4845c049a6e9b122bf7abbb9d451ad374",
        "formula_id": "a867022d73b0c48aa87141ebb97bfb34785392b913ce28fd03567c83f305bbdb",
        "source_response_sha256": "2864f3a618f919318cec797f08c5f83ae66455ec36a32ac14ca8a88150d215ca"
    },
    {
        "case_id": "extra-14",
        "product": "body_lotion",
        "brief": "첫향은 시트러스, 잔향은 우디 머스크",
        "concentration_percent": 0.5,
        "composition_sha256": "8b5b515bd5bb570e6ca8288f9d5595a798084c9b5e7d2230ff7ee397bb2bb5ab",
        "formula_id": "230a81359a88120421b1410d69d2af3342b79491a6a7af5fd8a82f16e7e3c2f8",
        "source_response_sha256": "4dba742fa25f1ed1766ab6a81ea56edd6bc79ab94d79faf7477ef0108d19caae"
    },
    {
        "case_id": "extra-15",
        "product": "body_lotion",
        "brief": "첫향은 우디, 잔향은 시트러스",
        "concentration_percent": 0.5,
        "composition_sha256": "c27b8fcd4c5ecd102eac4dc8ff2a7977ef7ef18950a9e6d9266f7c49f29debf9",
        "formula_id": "c3ddf4e277602f7d7cbbef5b343c290614056286f6443df3767ab53d20f4c1f4",
        "source_response_sha256": "d54a2e45aa8eace4923c5ff9aff65e07cfdbfbc6e62898963b714b8fb03835c0"
    },
    {
        "case_id": "extra-6",
        "product": "body_lotion",
        "brief": "opening citrus, drydown woody musk",
        "concentration_percent": 0.5,
        "composition_sha256": "e88841cfdb0c6278253dc1ed1c9b3b538ba734a04350ffb41fc43eb75b122295",
        "formula_id": "ac56c33377444cf06e0057d1668bd9e897f9634b5ac9be7ddd5d6cf5233a58bd",
        "source_response_sha256": "546ef2d39395cb0489b521d88065c63d9fea5aea1aa39c61d5afd889e9630f22"
    },
    {
        "case_id": "extra-8",
        "product": "body_lotion",
        "brief": "opening woody, drydown citrus",
        "concentration_percent": 0.5,
        "composition_sha256": "669cafaabfba00e0d413389f0adf7d1f3084568e0b2a85cebd86b0fbcbbe8e8e",
        "formula_id": "f170d044f2f4389bf0f87dc96c084d8a8b38ed14e15fd9244077d8280d0da6b9",
        "source_response_sha256": "422b13b7d85016521dc78b4b59483aec5440af2d96864c947032a50c6cf06ccc"
    },
    {
        "case_id": "ko-4",
        "product": "body_lotion",
        "brief": "아쿠아틱 향",
        "concentration_percent": 0.5,
        "composition_sha256": "aa48bc88d4fed1fb8e34eddd7097c769091154d91f264bb3374cd5e7fd941e5b",
        "formula_id": "29f4e4941cb0c542208d4361a3efcf0970b0c8f66c20eb89e4e88a5936706f57",
        "source_response_sha256": "4bb84bb922731b8ff6208248539b9cbf0d7f70b382602ce6ee9aa97578b6bee7"
    },
    {
        "case_id": "en-4",
        "product": "perfume",
        "brief": "aquatic scent",
        "concentration_percent": 15,
        "composition_sha256": "b9baf57f83d7c144b5b5022d9502c943c085617ba34d2e9ea8d1a9a573c7602d",
        "formula_id": "sha256:7396655abec70f762ca5cb19d4a8800a62ac7e0f64b0babe1fadc387158e2234",
        "source_response_sha256": "b7ff4df8d955d9eb3e0621de22cb3bfa9a5c316aca14478858cf30eda3f6b440"
    },
    {
        "case_id": "extra-13",
        "product": "perfume",
        "brief": "opening citrus, heart floral, drydown woody",
        "concentration_percent": 15,
        "composition_sha256": "ece2fa543d1e81b010966328333e055a3aa4cba92f21d40e051e82530795f350",
        "formula_id": "sha256:4dfbd595a1f846fc97e21284732f8084e5faac7520d3862fa0c8750b572a9f99",
        "source_response_sha256": "4e042356ceafb8a85c6066dbaa336b0cc9abfd47812c5a99ee19aefaea2f9581"
    },
    {
        "case_id": "ko-4",
        "product": "perfume",
        "brief": "아쿠아틱 향",
        "concentration_percent": 15,
        "composition_sha256": "b9baf57f83d7c144b5b5022d9502c943c085617ba34d2e9ea8d1a9a573c7602d",
        "formula_id": "sha256:7396655abec70f762ca5cb19d4a8800a62ac7e0f64b0babe1fadc387158e2234",
        "source_response_sha256": "596308668f2fafa5da847a6ef5945dcab50df8ba1eb65433e5620123e9d4b1fe"
    }
]
_RETIRED = frozenset(row["composition_sha256"] for row in SOURCE_RECORDS)
_PRODUCTS = frozenset(row["product"] for row in SOURCE_RECORDS)


def composition_signature(product, concentration_percent, ingredient_amounts):
    """Hash supplied material percentages without changing or renormalizing them."""
    dose = float(concentration_percent)
    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("finite positive finished-product concentration required")
    amounts = []
    seen = set()
    for identifier, percent in ingredient_amounts:
        value = float(percent)
        if not isinstance(identifier, str) or not identifier or not math.isfinite(value):
            raise ValueError("finite material percentages and ingredient IDs required")
        if identifier in seen:
            raise ValueError("duplicate material in composition fingerprint")
        seen.add(identifier)
        units = round(value * 10000)
        if units > 0:
            amounts.append((identifier, units))
    body = "\n".join(f"{identifier}\t{units}" for identifier, units in sorted(amounts))
    canonical = f"{product}\n{round(dose * 1000000)}\n{body}"
    return hashlib.sha256(canonical.encode("utf8")).hexdigest()


def _field(row, name):
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


class RetiredBlendFilter:
    """Request-local selection filter; the candidate ingredient pool is untouched."""

    def __init__(self, product, concentration_percent):
        self.product = product
        self.concentration_percent = concentration_percent
        self.rejected = 0

    def _reject(self, amounts):
        if self.product not in _PRODUCTS:
            return False
        signature = composition_signature(self.product, self.concentration_percent, amounts)
        if signature not in _RETIRED:
            return False
        self.rejected += 1
        return True

    def reject_lines(self, lines):
        return self._reject((_field(row, "ingredient_id"), _field(row, "concentrate_percent"))
                            for row in lines)

    def reject_weights(self, ingredients, fractions):
        if len(ingredients) != len(fractions):
            raise ValueError("material and weight counts differ")
        return self._reject((_field(item, "ingredient_id"), float(weight) * 100.)
                            for item, weight in zip(ingredients, fractions))

    def report(self):
        return {"policy": POLICY_VERSION, "excluded_records": len(SOURCE_RECORDS),
                "excluded_compositions": len(_RETIRED), "rejected_candidates": self.rejected,
                "ingredient_pool_changed": False, "scent_requests_blocked": False,
                "score_or_threshold_changed": False}
