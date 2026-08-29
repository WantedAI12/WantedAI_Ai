"""IFRA and allergen rule checks."""

from .ifra_rules import ProductCategory, check_compliance

__all__ = ["ProductCategory", "check_compliance"]
