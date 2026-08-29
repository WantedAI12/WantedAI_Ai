# fragrance_ai/rules/ifra_rules.py
"""
IFRA (International Fragrance Association) Rules and Compliance
성분별 상한, 제품 카테고리별 허용치 관리
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


# ============================================================================
# Product Categories
# ============================================================================

class ProductCategory(str, Enum):
    """IFRA product categories"""
    # Category 4: Hydroalcoholic products
    EAU_DE_PARFUM = "eau_de_parfum"  # 15-20% fragrance
    EAU_DE_TOILETTE = "eau_de_toilette"  # 5-15% fragrance
    EAU_DE_COLOGNE = "eau_de_cologne"  # 2-5% fragrance

    # Category 5: Facial products
    FACE_CREAM = "face_cream"
    FACE_TONER = "face_toner"

    # Category 6: Mouthwash
    MOUTHWASH = "mouthwash"

    # Category 9: Rinse-off products
    SHAMPOO = "shampoo"
    BODY_WASH = "body_wash"

    # Category 11: Non-skin contact
    CANDLE = "candle"
    ROOM_SPRAY = "room_spray"
    DIFFUSER = "diffuser"


# ============================================================================
# IFRA Limit Data
# ============================================================================

@dataclass
class IFRALimit:
    """IFRA limit for specific ingredient in product category"""
    ingredient_name: str
    cas_number: Optional[str]
    category: ProductCategory
    max_percentage: float  # Maximum allowed percentage
    restriction_type: str  # "prohibited", "restricted", "specification"
    amendment: int  # IFRA amendment number (e.g., 49, 50)
    notes: Optional[str] = None


class IFRADatabase:
    """Small, explicit demonstration subset of IFRA screening limits.

    This is deliberately *not* a machine-readable replacement for the IFRA
    Standards Library. An ingredient absent from this table is ``unknown`` to
    this local screen; it is not evidence that the ingredient is unrestricted.
    """

    EMBEDDED_AMENDMENT = 50
    RULESET_LABEL = "IFRA Amendment 50 embedded demonstration subset"
    COVERAGE_STATUS = "partial_embedded_subset_not_a_complete_ifra_rule_pack"

    def __init__(self):
        self.limits: Dict[str, Dict[ProductCategory, IFRALimit]] = {}
        self._load_ifra_limits()

    def _load_ifra_limits(self):
        """Load the deliberately limited Amendment 50 demonstration subset.

        Do not update the label to a newer IFRA amendment unless every limit is
        replaced from an appropriately licensed and versioned official rule
        pack. Supplier certificates are evaluated separately by the safety
        gate and do not upgrade this local subset.
        """

        # Common restricted materials with limits
        ifra_data = [
            # Citrus oils (phototoxic)
            {
                "ingredient": "Bergamot Oil",
                "cas": "8007-75-8",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 2.0,
                    ProductCategory.EAU_DE_TOILETTE: 2.0,
                    ProductCategory.FACE_CREAM: 0.4,
                    ProductCategory.BODY_WASH: 2.0,
                    ProductCategory.CANDLE: 100.0,  # No restriction for non-skin
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Lemon Oil Cold Pressed",
                "cas": "8008-56-8",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 3.0,
                    ProductCategory.EAU_DE_TOILETTE: 3.0,
                    ProductCategory.FACE_CREAM: 0.6,
                    ProductCategory.BODY_WASH: 3.0,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Allergens
            {
                "ingredient": "Oakmoss Absolute",
                "cas": "90028-68-5",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.1,
                    ProductCategory.EAU_DE_TOILETTE: 0.1,
                    ProductCategory.FACE_CREAM: 0.1,
                    ProductCategory.BODY_WASH: 0.1,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Treemoss Absolute",
                "cas": "90028-67-4",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.2,
                    ProductCategory.EAU_DE_TOILETTE: 0.2,
                    ProductCategory.FACE_CREAM: 0.2,
                    ProductCategory.BODY_WASH: 0.2,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Rose/Jasmine (sensitizers)
            {
                "ingredient": "Rose Absolute",
                "cas": "8007-01-0",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.6,
                    ProductCategory.EAU_DE_TOILETTE: 0.6,
                    ProductCategory.FACE_CREAM: 0.02,
                    ProductCategory.BODY_WASH: 0.6,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Jasmine Absolute",
                "cas": "8022-96-6",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.7,
                    ProductCategory.EAU_DE_TOILETTE: 0.7,
                    ProductCategory.FACE_CREAM: 0.02,
                    ProductCategory.BODY_WASH: 0.7,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Musks
            {
                "ingredient": "Musk Xylene",
                "cas": "81-15-2",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.0,  # Prohibited
                    ProductCategory.EAU_DE_TOILETTE: 0.0,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 0.0,
                    ProductCategory.CANDLE: 0.0,
                },
                "type": "prohibited",
                "amendment": 50
            },
            {
                "ingredient": "Musk Ketone",
                "cas": "81-14-1",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 1.4,
                    ProductCategory.EAU_DE_TOILETTE: 1.4,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 1.4,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Other common materials
            {
                "ingredient": "Coumarin",
                "cas": "91-64-5",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 1.6,
                    ProductCategory.EAU_DE_TOILETTE: 1.6,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 1.6,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Eugenol",
                "cas": "97-53-0",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.5,
                    ProductCategory.EAU_DE_TOILETTE: 0.5,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 0.5,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            }
        ]

        # Load into database
        for item in ifra_data:
            ingredient = item["ingredient"]
            self.limits[ingredient] = {}

            for category, limit in item["limits"].items():
                self.limits[ingredient][category] = IFRALimit(
                    ingredient_name=ingredient,
                    cas_number=item["cas"],
                    category=category,
                    max_percentage=limit,
                    restriction_type=item["type"],
                    amendment=item["amendment"],
                    notes=item.get("notes")
                )

    def get_limit(self, ingredient: str, category: ProductCategory) -> Optional[float]:
        """Get IFRA limit for ingredient in product category"""
        if ingredient in self.limits:
            if category in self.limits[ingredient]:
                return self.limits[ingredient][category].max_percentage
        return None  # No embedded rule; this is unknown, not unrestricted.

    def coverage(self, recipe: Dict[str, Any], category: ProductCategory) -> Dict[str, Any]:
        """Report the bounded coverage of this embedded screen.

        ``uncovered_ingredients`` means that this package has no rule for the
        material/category pair. It must never be interpreted as an IFRA pass.
        """
        covered: list[str] = []
        uncovered: list[str] = []
        unsupported_categories: list[str] = []
        for item in recipe.get("ingredients", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            category_limits = self.limits.get(name)
            if category_limits is None:
                uncovered.append(name)
            elif category not in category_limits:
                uncovered.append(name)
                unsupported_categories.append(name)
            else:
                covered.append(name)
        return {
            "rule_set": self.RULESET_LABEL,
            "embedded_amendment": self.EMBEDDED_AMENDMENT,
            "coverage_status": self.COVERAGE_STATUS,
            "is_complete_ifra_rule_pack": False,
            "covered_ingredients": sorted(set(covered)),
            "uncovered_ingredients": sorted(set(uncovered)),
            "category_unsupported_ingredients": sorted(set(unsupported_categories)),
        }

    def is_prohibited(self, ingredient: str, category: ProductCategory) -> bool:
        """Check if ingredient is prohibited in category"""
        limit = self.get_limit(ingredient, category)
        return limit == 0.0 if limit is not None else False


# ============================================================================
# IFRA Compliance Checker
# ============================================================================

class IFRAComplianceChecker:
    """Check formulation compliance with IFRA standards"""

    def __init__(self):
        self.database = IFRADatabase()

    def check_ifra_violations(
        self,
        recipe: Dict[str, Any],
        product_category: ProductCategory = ProductCategory.EAU_DE_PARFUM
    ) -> Dict[str, Any]:
        """
        Check recipe for IFRA violations

        Args:
            recipe: Recipe with ingredients list
            product_category: Target product category

        Returns:
            Dictionary with violation count, penalty, and details
        """
        violations = []
        total_penalty = 0.0

        # Get ingredients from recipe
        ingredients = recipe.get("ingredients", [])

        for ingredient in ingredients:
            name = ingredient.get("name", "")
            concentration = float(ingredient.get("concentration", 0.0))

            if not math.isfinite(concentration) or concentration < 0:
                violations.append({
                    "ingredient": name,
                    "concentration": concentration,
                    "limit": None,
                    "excess": None,
                    "severity": "critical",
                    "reason": "invalid_concentration",
                })
                total_penalty += 100.0
                continue

            # Check IFRA limit
            limit = self.database.get_limit(name, product_category)

            if limit is not None:
                if concentration > limit:
                    violation = {
                        "ingredient": name,
                        "concentration": concentration,
                        "limit": limit,
                        "excess": concentration - limit,
                        "severity": "critical" if limit == 0 else "warning"
                    }
                    violations.append(violation)

                    # Calculate penalty (exponential for severe violations)
                    if limit == 0:  # Prohibited
                        penalty = 100.0 * concentration
                    else:
                        excess_ratio = (concentration - limit) / limit
                        penalty = 10.0 * (1 + excess_ratio) ** 2

                    total_penalty += penalty

        return {
            "count": len(violations),
            "penalty": total_penalty,
            "details": violations,
            "compliant": len(violations) == 0,
            "product_category": product_category.value,
            "coverage": self.database.coverage(recipe, product_category),
        }

    def apply_ifra_limits(
        self,
        recipe: Dict[str, Any],
        product_category: ProductCategory = ProductCategory.EAU_DE_PARFUM,
        mode: str = "clip"
    ) -> Dict[str, Any]:
        """Apply limits without reintroducing violations during rebalancing.

        ``clip`` caps restricted materials and ``remove`` removes all violating
        materials. Any resulting deficit is reallocated only to an unbounded
        material or to a bounded material with remaining headroom. If that is
        impossible, the formula is marked incomplete rather than silently
        normalized back above an IFRA cap.
        """
        if mode not in {"clip", "remove"}:
            raise ValueError("mode must be either 'clip' or 'remove'")

        ingredients = [dict(item) for item in recipe.get("ingredients", [])]
        modified_ingredients: list[Dict[str, Any]] = []
        removed_ingredients: list[Dict[str, Any]] = []
        original_total = 0.0

        for ingredient in ingredients:
            name = str(ingredient.get("name", ""))
            concentration = float(ingredient.get("concentration", 0.0))
            if not math.isfinite(concentration) or concentration < 0:
                raise ValueError(f"invalid concentration for {name or 'unnamed ingredient'}")
            ingredient["concentration"] = concentration
            original_total += concentration
            limit = self.database.get_limit(name, product_category)
            # Keep even a zero-concentration prohibited material out of the
            # returned formula. Leaving it in an export creates an avoidable
            # reweighing/substitution hazard downstream.
            if limit == 0.0:
                removed_ingredients.append(ingredient)
                continue
            if limit is not None and concentration > limit:
                if mode == "remove":
                    removed_ingredients.append(ingredient)
                    continue
                ingredient["concentration"] = limit
                ingredient["ifra_clipped"] = True
            modified_ingredients.append(ingredient)

        current_total = sum(float(ingredient["concentration"]) for ingredient in modified_ingredients)
        deficit = max(0.0, original_total - current_total)

        def distribute(
            indices: list[int], amount: float, capacities: list[float] | None
        ) -> float:
            """Bounded water-filling; return the amount actually allocated."""
            if amount <= 1e-12 or not indices:
                return 0.0
            if capacities is None:
                weights = [max(float(modified_ingredients[index]["concentration"]), 1.0) for index in indices]
                weight_total = sum(weights)
                for index, weight in zip(indices, weights):
                    modified_ingredients[index]["concentration"] += amount * weight / weight_total
                return amount
            remaining = amount
            active = list(zip(indices, capacities))
            while remaining > 1e-10 and active:
                capacity_total = sum(max(capacity, 0.0) for _, capacity in active)
                if capacity_total <= 1e-12:
                    break
                next_active: list[tuple[int, float]] = []
                assigned = 0.0
                for index, capacity in active:
                    addition = min(remaining * max(capacity, 0.0) / capacity_total, capacity)
                    modified_ingredients[index]["concentration"] += addition
                    assigned += addition
                    residual = capacity - addition
                    if residual > 1e-10:
                        next_active.append((index, residual))
                remaining -= assigned
                if assigned <= 1e-12:
                    break
                active = next_active
            return amount - remaining

        unbounded = [
            index
            for index, ingredient in enumerate(modified_ingredients)
            if self.database.get_limit(str(ingredient.get("name", "")), product_category) is None
        ]
        reallocated = distribute(unbounded, deficit, capacities=None)
        remaining = max(0.0, deficit - reallocated)
        if remaining > 1e-12:
            bounded: list[int] = []
            headroom: list[float] = []
            for index, ingredient in enumerate(modified_ingredients):
                limit = self.database.get_limit(str(ingredient.get("name", "")), product_category)
                if limit is None:
                    continue
                capacity = max(0.0, limit - float(ingredient["concentration"]))
                if capacity > 1e-12:
                    bounded.append(index)
                    headroom.append(capacity)
            reallocated += distribute(bounded, remaining, capacities=headroom)

        unallocated = max(0.0, deficit - reallocated)
        modified_recipe = recipe.copy()
        modified_recipe["ingredients"] = modified_ingredients
        final_check = self.check_ifra_violations(modified_recipe, product_category)
        formula_complete = unallocated <= 1e-8
        embedded_limits_compliant = bool(final_check["compliant"])
        modified_recipe["removed_ingredients"] = removed_ingredients
        modified_recipe["original_total_concentration"] = original_total
        modified_recipe["result_total_concentration"] = sum(
            float(ingredient["concentration"]) for ingredient in modified_ingredients
        )
        modified_recipe["reallocated_concentration"] = reallocated
        modified_recipe["unallocated_concentration"] = unallocated
        modified_recipe["formula_complete"] = formula_complete
        modified_recipe["embedded_limits_compliant"] = embedded_limits_compliant
        # The embedded table is intentionally incomplete, so a local pass must
        # not become an IFRA certification claim. Consumers that need the
        # limited engineering result can use ``embedded_limits_compliant``.
        modified_recipe["ifra_compliant"] = False
        modified_recipe["ifra_coverage"] = final_check["coverage"]
        modified_recipe["ifra_application_status"] = (
            "embedded_subset_pass_not_ifra_certified"
            if embedded_limits_compliant and formula_complete
            else "incomplete_or_embedded_limit_failure"
        )
        return modified_recipe


# ============================================================================
# Allergen Declaration
# ============================================================================

class AllergenChecker:
    """EU allergen declaration requirements"""

    # 26 EU allergens that must be declared
    EU_ALLERGENS = {
        "Alpha-Isomethyl Ionone": 10.0,  # ppm threshold in final product
        "Amyl Cinnamal": 10.0,
        "Amylcinnamyl Alcohol": 10.0,
        "Anise Alcohol": 10.0,
        "Benzyl Alcohol": 10.0,
        "Benzyl Benzoate": 10.0,
        "Benzyl Cinnamate": 10.0,
        "Benzyl Salicylate": 10.0,
        "Butylphenyl Methylpropional": 10.0,  # Lilial - now banned
        "Cinnamal": 10.0,
        "Cinnamyl Alcohol": 10.0,
        "Citral": 10.0,
        "Citronellol": 10.0,
        "Coumarin": 10.0,
        "Eugenol": 10.0,
        "Evernia Furfuracea": 10.0,  # Treemoss
        "Evernia Prunastri": 10.0,  # Oakmoss
        "Farnesol": 10.0,
        "Geraniol": 10.0,
        "Hexyl Cinnamal": 10.0,
        "Hydroxycitronellal": 10.0,
        "Isoeugenol": 10.0,
        "Limonene": 10.0,
        "Linalool": 10.0,
        "Methyl 2-Octynoate": 10.0,
        "Hydroxyisohexyl 3-Cyclohexene Carboxaldehyde": 10.0  # Lyral - now banned
    }

    @classmethod
    def check_allergens(cls, recipe: Dict[str, Any], product_concentration: float = 15.0) -> Dict[str, Any]:
        """
        Check which allergens need declaration

        Args:
            recipe: Fragrance recipe
            product_concentration: % of fragrance in final product (e.g., 15% for EDP)

        Returns:
            Dictionary with allergen information
        """
        allergens_to_declare = []

        for ingredient in recipe.get("ingredients", []):
            name = ingredient.get("name", "")
            concentration_in_fragrance = ingredient.get("concentration", 0.0)

            # Calculate concentration in final product (ppm)
            concentration_in_product = (concentration_in_fragrance / 100) * (product_concentration / 100) * 1000000

            if name in cls.EU_ALLERGENS:
                threshold = cls.EU_ALLERGENS[name]
                if concentration_in_product > threshold:
                    allergens_to_declare.append({
                        "name": name,
                        "concentration_ppm": concentration_in_product,
                        "threshold_ppm": threshold,
                        "must_declare": True
                    })

        return {
            "allergens": allergens_to_declare,
            "count": len(allergens_to_declare),
            "compliant": True  # Allergens can be present if declared
        }


# ============================================================================
# Main Module Interface
# ============================================================================

# Global instances
_ifra_checker = None
_allergen_checker = None


def get_ifra_checker() -> IFRAComplianceChecker:
    """Get global IFRA checker instance"""
    global _ifra_checker
    if _ifra_checker is None:
        _ifra_checker = IFRAComplianceChecker()
    return _ifra_checker


def get_allergen_checker() -> AllergenChecker:
    """Get global allergen checker instance"""
    global _allergen_checker
    if _allergen_checker is None:
        _allergen_checker = AllergenChecker()
    return _allergen_checker


def check_compliance(
    recipe: Dict[str, Any],
    product_category: ProductCategory = ProductCategory.EAU_DE_PARFUM,
    product_concentration: float = 15.0
) -> Dict[str, Any]:
    """
    Complete compliance check for recipe

    Returns:
        Dictionary with IFRA and allergen compliance results
    """
    ifra_checker = get_ifra_checker()
    allergen_checker = get_allergen_checker()

    ifra_result = ifra_checker.check_ifra_violations(recipe, product_category)
    allergen_result = allergen_checker.check_allergens(recipe, product_concentration)

    coverage = ifra_result.get("coverage", {})
    return {
        "ifra": ifra_result,
        "allergens": allergen_result,
        "overall_compliant": False,
        "overall_status": "prototype_partial_screen_only",
        "embedded_limits_compliant": ifra_result["compliant"],
        "ifra_rule_pack_complete": bool(coverage.get("is_complete_ifra_rule_pack", False)),
        "commercial_release_eligible": False,
    }


# Export main classes and functions
__all__ = [
    'ProductCategory',
    'IFRALimit',
    'IFRADatabase',
    'IFRAComplianceChecker',
    'AllergenChecker',
    'get_ifra_checker',
    'get_allergen_checker',
    'check_compliance'
]
