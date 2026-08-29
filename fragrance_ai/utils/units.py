# fragrance_ai/utils/units.py

from __future__ import annotations

from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Unit Enums
# ============================================================================

class MassUnit(str, Enum):
    """Mass units"""
    GRAM = "g"
    KILOGRAM = "kg"
    MILLIGRAM = "mg"
    POUND = "lb"
    OUNCE = "oz"


class VolumeUnit(str, Enum):
    """Volume units"""
    MILLILITER = "ml"
    LITER = "l"
    MICROLITER = "ul"
    GALLON = "gal"
    FLUID_OUNCE = "fl_oz"


class ConcentrationUnit(str, Enum):
    """Concentration units"""
    PERCENT = "%"
    PPM = "ppm"  # Parts per million
    PPB = "ppb"  # Parts per billion
    GRAM_PER_LITER = "g/L"
    MILLIGRAM_PER_LITER = "mg/L"


# ============================================================================
# Density Database
# ============================================================================

@dataclass
class MaterialDensity:
    """Density information for a material"""
    name: str
    density: float  # g/ml at 20°C
    temperature: float = 20.0  # °C
    notes: Optional[str] = None


class DensityDatabase:
    """Database of material densities for fragrance industry"""

    def __init__(self):
        self.densities: Dict[str, MaterialDensity] = {}
        self._load_default_densities()

    def _load_default_densities(self):
        """Load default density values for common materials"""

        # Solvents
        solvents = [
            MaterialDensity("ethanol", 0.789, notes="95% ethanol"),
            MaterialDensity("ethanol_absolute", 0.785, notes="99.9% ethanol"),
            MaterialDensity("dipropylene_glycol", 1.023, notes="DPG"),
            MaterialDensity("isopropyl_myristate", 0.853, notes="IPM"),
            MaterialDensity("benzyl_benzoate", 1.118, notes="BB"),
            MaterialDensity("diethyl_phthalate", 1.118, notes="DEP"),
            MaterialDensity("water", 0.998),
        ]

        # Essential oils
        essential_oils = [
            MaterialDensity("bergamot_oil", 0.875),
            MaterialDensity("lemon_oil", 0.851),
            MaterialDensity("orange_oil", 0.845),
            MaterialDensity("lavender_oil", 0.885),
            MaterialDensity("rose_oil", 0.855),
            MaterialDensity("jasmine_absolute", 0.945),
            MaterialDensity("sandalwood_oil", 0.973),
            MaterialDensity("patchouli_oil", 0.955),
            MaterialDensity("ylang_ylang_oil", 0.925),
            MaterialDensity("vetiver_oil", 0.988),
        ]

        # Aromatic compounds
        aromatics = [
            MaterialDensity("linalool", 0.858),
            MaterialDensity("limonene", 0.841),
            MaterialDensity("citral", 0.888),
            MaterialDensity("geraniol", 0.879),
            MaterialDensity("eugenol", 1.064),
            MaterialDensity("vanillin", 1.056),
            MaterialDensity("coumarin", 0.935),
            MaterialDensity("benzyl_acetate", 1.055),
            MaterialDensity("phenylethyl_alcohol", 1.017),
            MaterialDensity("iso_e_super", 0.946),
            MaterialDensity("hedione", 0.998),
            MaterialDensity("galaxolide", 1.015),
        ]

        # Add all to database
        for material in solvents + essential_oils + aromatics:
            self.densities[material.name.lower()] = material

    def get_density(self, material_name: str) -> Optional[float]:
        """Get density for a material"""
        return self.densities.get(material_name.lower())

    def add_custom_density(self, material: MaterialDensity):
        """Add custom density to database"""
        self.densities[material.name.lower()] = material


# ============================================================================
# Unit Converter
# ============================================================================

class UnitConverter:
    """Converter for fragrance industry units"""

    def __init__(self):
        self.density_db = DensityDatabase()

    # Mass conversions
    def convert_mass(self, value: float, from_unit: MassUnit, to_unit: MassUnit) -> float:
        """Convert between mass units"""

        # Convert to grams first
        if from_unit == MassUnit.GRAM:
            grams = value
        elif from_unit == MassUnit.KILOGRAM:
            grams = value * 1000
        elif from_unit == MassUnit.MILLIGRAM:
            grams = value / 1000
        elif from_unit == MassUnit.POUND:
            grams = value * 453.592
        elif from_unit == MassUnit.OUNCE:
            grams = value * 28.3495
        else:
            raise ValueError(f"Unknown mass unit: {from_unit}")

        # Convert from grams to target unit
        if to_unit == MassUnit.GRAM:
            return grams
        elif to_unit == MassUnit.KILOGRAM:
            return grams / 1000
        elif to_unit == MassUnit.MILLIGRAM:
            return grams * 1000
        elif to_unit == MassUnit.POUND:
            return grams / 453.592
        elif to_unit == MassUnit.OUNCE:
            return grams / 28.3495
        else:
            raise ValueError(f"Unknown mass unit: {to_unit}")

    # Volume conversions
    def convert_volume(self, value: float, from_unit: VolumeUnit, to_unit: VolumeUnit) -> float:
        """Convert between volume units"""

        # Convert to milliliters first
        if from_unit == VolumeUnit.MILLILITER:
            ml = value
        elif from_unit == VolumeUnit.LITER:
            ml = value * 1000
        elif from_unit == VolumeUnit.MICROLITER:
            ml = value / 1000
        elif from_unit == VolumeUnit.GALLON:
            ml = value * 3785.41
        elif from_unit == VolumeUnit.FLUID_OUNCE:
            ml = value * 29.5735
        else:
            raise ValueError(f"Unknown volume unit: {from_unit}")

        # Convert from milliliters to target unit
        if to_unit == VolumeUnit.MILLILITER:
            return ml
        elif to_unit == VolumeUnit.LITER:
            return ml / 1000
        elif to_unit == VolumeUnit.MICROLITER:
            return ml * 1000
        elif to_unit == VolumeUnit.GALLON:
            return ml / 3785.41
        elif to_unit == VolumeUnit.FLUID_OUNCE:
            return ml / 29.5735
        else:
            raise ValueError(f"Unknown volume unit: {to_unit}")

    # Mass to volume conversions (requires density)
    def mass_to_volume(
        self,
        mass: float,
        mass_unit: MassUnit,
        material_name: str,
        volume_unit: VolumeUnit = VolumeUnit.MILLILITER
    ) -> Optional[float]:
        """Convert mass to volume using density"""

        density_info = self.density_db.get_density(material_name)
        if density_info is None:
            return None

        # Convert mass to grams
        mass_grams = self.convert_mass(mass, mass_unit, MassUnit.GRAM)

        # Calculate volume in ml (density is in g/ml)
        volume_ml = mass_grams / density_info.density

        # Convert to desired volume unit
        return self.convert_volume(volume_ml, VolumeUnit.MILLILITER, volume_unit)

    # Volume to mass conversions (requires density)
    def volume_to_mass(
        self,
        volume: float,
        volume_unit: VolumeUnit,
        material_name: str,
        mass_unit: MassUnit = MassUnit.GRAM
    ) -> Optional[float]:
        """Convert volume to mass using density"""

        density_info = self.density_db.get_density(material_name)
        if density_info is None:
            return None

        # Convert volume to ml
        volume_ml = self.convert_volume(volume, volume_unit, VolumeUnit.MILLILITER)

        # Calculate mass in grams (density is in g/ml)
        mass_grams = volume_ml * density_info.density

        # Convert to desired mass unit
        return self.convert_mass(mass_grams, MassUnit.GRAM, mass_unit)

    # Concentration conversions
    def convert_concentration(
        self,
        value: float,
        from_unit: ConcentrationUnit,
        to_unit: ConcentrationUnit,
        density: float = 1.0
    ) -> float:
        """
        Convert between concentration units

        Args:
            value: Concentration value
            from_unit: Source unit
            to_unit: Target unit
            density: Density for g/L conversions (default 1.0 for water)
        """

        # Convert to percent first
        if from_unit == ConcentrationUnit.PERCENT:
            percent = value
        elif from_unit == ConcentrationUnit.PPM:
            percent = value / 10000
        elif from_unit == ConcentrationUnit.PPB:
            percent = value / 10000000
        elif from_unit == ConcentrationUnit.GRAM_PER_LITER:
            # Assuming density in g/ml, 1L = 1000ml
            percent = (value / (density * 1000)) * 100
        elif from_unit == ConcentrationUnit.MILLIGRAM_PER_LITER:
            percent = (value / (density * 1000000)) * 100
        else:
            raise ValueError(f"Unknown concentration unit: {from_unit}")

        # Convert from percent to target unit
        if to_unit == ConcentrationUnit.PERCENT:
            return percent
        elif to_unit == ConcentrationUnit.PPM:
            return percent * 10000
        elif to_unit == ConcentrationUnit.PPB:
            return percent * 10000000
        elif to_unit == ConcentrationUnit.GRAM_PER_LITER:
            return (percent / 100) * density * 1000
        elif to_unit == ConcentrationUnit.MILLIGRAM_PER_LITER:
            return (percent / 100) * density * 1000000
        else:
            raise ValueError(f"Unknown concentration unit: {to_unit}")

    # Batch size calculations
    def calculate_batch_quantities(
        self,
        formula: Dict[str, float],  # ingredient -> percentage
        batch_size: float,
        batch_unit: Union[MassUnit, VolumeUnit],
        output_unit: Union[MassUnit, VolumeUnit]
    ) -> Dict[str, float]:
        """
        Calculate ingredient quantities for a batch

        Args:
            formula: Dictionary of ingredient names to percentages
            batch_size: Total batch size
            batch_unit: Unit of batch size
            output_unit: Desired output unit for ingredients

        Returns:
            Dictionary of ingredient names to quantities
        """

        # Normalize percentages to sum to 100
        total_percent = sum(formula.values())
        normalized_formula = {
            ing: (pct / total_percent) * 100
            for ing, pct in formula.items()
        }

        quantities = {}

        for ingredient, percent in normalized_formula.items():
            # Calculate quantity based on batch size
            quantity = (percent / 100) * batch_size

            # Convert units if necessary
            if isinstance(batch_unit, MassUnit) and isinstance(output_unit, MassUnit):
                quantities[ingredient] = self.convert_mass(quantity, batch_unit, output_unit)
            elif isinstance(batch_unit, VolumeUnit) and isinstance(output_unit, VolumeUnit):
                quantities[ingredient] = self.convert_volume(quantity, batch_unit, output_unit)
            elif isinstance(batch_unit, MassUnit) and isinstance(output_unit, VolumeUnit):
                # Need density for conversion
                vol = self.mass_to_volume(quantity, batch_unit, ingredient, output_unit)
                if vol is not None:
                    quantities[ingredient] = vol
                else:
                    # Fallback: assume density of 1.0 g/ml
                    mass_g = self.convert_mass(quantity, batch_unit, MassUnit.GRAM)
                    vol_ml = mass_g  # Assuming density = 1.0
                    quantities[ingredient] = self.convert_volume(
                        vol_ml, VolumeUnit.MILLILITER, output_unit
                    )
            elif isinstance(batch_unit, VolumeUnit) and isinstance(output_unit, MassUnit):
                # Need density for conversion
                mass = self.volume_to_mass(quantity, batch_unit, ingredient, output_unit)
                if mass is not None:
                    quantities[ingredient] = mass
                else:
                    # Fallback: assume density of 1.0 g/ml
                    vol_ml = self.convert_volume(quantity, batch_unit, VolumeUnit.MILLILITER)
                    mass_g = vol_ml  # Assuming density = 1.0
                    quantities[ingredient] = self.convert_mass(mass_g, MassUnit.GRAM, output_unit)
            else:
                quantities[ingredient] = quantity

        return quantities


# ============================================================================
# Helper Functions
# ============================================================================

def format_quantity(value: float, unit: Union[MassUnit, VolumeUnit], precision: int = 2) -> str:
    """Format quantity with unit for display"""
    return f"{value:.{precision}f} {unit.value}"


def parse_quantity(quantity_str: str) -> tuple[float, str]:
    """
    Parse a quantity string like "10.5 g" or "250 ml"

    Returns:
        Tuple of (value, unit)
    """
    parts = quantity_str.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid quantity format: {quantity_str}")

    try:
        value = float(parts[0])
    except ValueError:
        raise ValueError(f"Invalid number in quantity: {parts[0]}")

    unit = parts[1].lower()
    return value, unit


def calculate_dilution(
    concentrate_percent: float,
    target_percent: float,
    concentrate_volume: float = 100.0
) -> float:
    """
    Calculate dilution required

    Args:
        concentrate_percent: Original concentration (%)
        target_percent: Desired concentration (%)
        concentrate_volume: Volume of concentrate (ml)

    Returns:
        Volume of diluent to add (ml)
    """
    if target_percent >= concentrate_percent:
        return 0.0

    dilution_factor = concentrate_percent / target_percent
    total_volume = concentrate_volume * dilution_factor
    diluent_volume = total_volume - concentrate_volume

    return diluent_volume


# ============================================================================
# Mass Conservation Checker
# ============================================================================

class MassConservationChecker:
    """Verify mass conservation in formulation operations"""

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize checker

        Args:
            tolerance: Acceptable deviation (1% default)
        """
        self.tolerance = tolerance

    def check_percentage_sum(
        self,
        components: Dict[str, float],
        expected_sum: float = 100.0
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Check if percentages sum to expected value

        Args:
            components: Dictionary of ingredient to percentage
            expected_sum: Expected sum (usually 100%)

        Returns:
            Dictionary with check results
        """
        actual_sum = sum(components.values())
        deviation = abs(actual_sum - expected_sum)
        is_valid = deviation <= self.tolerance

        return {
            "valid": is_valid,
            "actual_sum": actual_sum,
            "expected_sum": expected_sum,
            "deviation": deviation,
            "tolerance": self.tolerance,
            "message": "OK" if is_valid else f"Sum deviation {deviation:.2f}% exceeds tolerance"
        }

    def check_mass_balance(
        self,
        input_masses: Dict[str, float],
        output_masses: Dict[str, float]
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Check mass balance (input = output)

        Args:
            input_masses: Dictionary of input ingredient to mass
            output_masses: Dictionary of output ingredient to mass

        Returns:
            Dictionary with balance results
        """
        total_input = sum(input_masses.values())
        total_output = sum(output_masses.values())

        if total_input == 0:
            return {
                "valid": False,
                "message": "Total input mass is zero"
            }

        deviation_g = abs(total_output - total_input)
        deviation_pct = (deviation_g / total_input) * 100
        is_valid = deviation_pct <= self.tolerance

        return {
            "valid": is_valid,
            "total_input_g": total_input,
            "total_output_g": total_output,
            "deviation_g": deviation_g,
            "deviation_percent": deviation_pct,
            "tolerance_percent": self.tolerance,
            "message": "Mass conserved" if is_valid else f"Mass imbalance: {deviation_pct:.2f}%"
        }

    def check_dilution(
        self,
        concentrate_g: float,
        solvent_g: float,
        final_g: float
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Check dilution mass conservation

        Args:
            concentrate_g: Concentrate mass in grams
            solvent_g: Solvent mass in grams
            final_g: Final product mass in grams

        Returns:
            Validation results
        """
        expected_final = concentrate_g + solvent_g
        deviation = abs(final_g - expected_final)
        deviation_pct = (deviation / expected_final * 100) if expected_final > 0 else 100
        is_valid = deviation_pct <= self.tolerance

        return {
            "valid": is_valid,
            "concentrate_g": concentrate_g,
            "solvent_g": solvent_g,
            "expected_final_g": expected_final,
            "actual_final_g": final_g,
            "deviation_percent": deviation_pct,
            "message": "Dilution valid" if is_valid else f"Dilution error: {deviation_pct:.2f}%"
        }

    def validate_formulation(
        self,
        formula: Dict[str, float],
        batch_size_g: float,
        actual_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[bool, float, str, list]]:
        """
        Complete formulation validation

        Args:
            formula: Dictionary of ingredient to percentage
            batch_size_g: Target batch size in grams
            actual_weights: Optional actual measured weights

        Returns:
            Comprehensive validation results
        """
        results = {
            "percentage_check": self.check_percentage_sum(formula),
            "issues": []
        }

        # Check percentage sum
        if not results["percentage_check"]["valid"]:
            results["issues"].append("Percentages do not sum to 100%")

        # Calculate theoretical weights
        converter = UnitConverter()
        theoretical_weights = {}
        for ingredient, percent in formula.items():
            theoretical_weights[ingredient] = (percent / 100) * batch_size_g

        results["theoretical_weights"] = theoretical_weights
        results["theoretical_total"] = sum(theoretical_weights.values())

        # Check actual vs theoretical if provided
        if actual_weights:
            results["mass_balance"] = self.check_mass_balance(
                theoretical_weights, actual_weights
            )
            if not results["mass_balance"]["valid"]:
                results["issues"].append("Mass balance deviation exceeds tolerance")

        results["valid"] = len(results["issues"]) == 0
        return results


# ============================================================================
# Batch Scaler
# ============================================================================

class BatchScaler:
    """Scale formulas between different batch sizes"""

    @staticmethod
    def scale_formula(
        formula: Dict[str, float],
        from_batch_g: float,
        to_batch_g: float,
        maintain_percentages: bool = True
    ) -> Dict[str, float]:
        """
        Scale formula to different batch size

        Args:
            formula: Dictionary of ingredient to amount
            from_batch_g: Original batch size in grams
            to_batch_g: Target batch size in grams
            maintain_percentages: If True, formula contains percentages; if False, contains weights

        Returns:
            Scaled formula
        """
        if maintain_percentages:
            # Formula contains percentages - they stay the same
            return formula.copy()
        else:
            # Formula contains weights - scale proportionally
            scale_factor = to_batch_g / from_batch_g
            return {
                ingredient: amount * scale_factor
                for ingredient, amount in formula.items()
            }

    @staticmethod
    def calculate_yield(
        theoretical_g: float,
        actual_g: float
    ) -> Dict[str, float]:
        """
        Calculate production yield

        Args:
            theoretical_g: Theoretical yield in grams
            actual_g: Actual yield in grams

        Returns:
            Yield metrics
        """
        if theoretical_g <= 0:
            return {
                "yield_percent": 0.0,
                "loss_g": 0.0,
                "loss_percent": 0.0
            }

        yield_pct = (actual_g / theoretical_g) * 100
        loss_g = theoretical_g - actual_g
        loss_pct = (loss_g / theoretical_g) * 100

        return {
            "yield_percent": yield_pct,
            "loss_g": loss_g,
            "loss_percent": loss_pct,
            "efficiency": "good" if yield_pct >= 98 else "acceptable" if yield_pct >= 95 else "poor"
        }


# ============================================================================
# Enhanced Unit Converter with % conversions
# ============================================================================

# Add methods to existing UnitConverter class
def percentage_to_grams(self, percentage: float, batch_size_g: float) -> float:
    """
    Convert percentage to grams

    Args:
        percentage: Percentage in formula (0-100)
        batch_size_g: Total batch size in grams

    Returns:
        Weight in grams
    """
    return (percentage / 100.0) * batch_size_g

def grams_to_percentage(self, grams: float, batch_size_g: float) -> float:
    """
    Convert grams to percentage

    Args:
        grams: Weight in grams
        batch_size_g: Total batch size in grams

    Returns:
        Percentage in formula
    """
    if batch_size_g <= 0:
        return 0.0
    return (grams / batch_size_g) * 100.0

def percentage_to_ml(
    self,
    percentage: float,
    batch_size_g: float,
    material_name: str
) -> Optional[float]:
    """
    Convert percentage to milliliters

    Args:
        percentage: Percentage in formula
        batch_size_g: Total batch size in grams
        material_name: Material name for density lookup

    Returns:
        Volume in milliliters
    """
    grams = self.percentage_to_grams(percentage, batch_size_g)
    return self.mass_to_volume(grams, MassUnit.GRAM, material_name, VolumeUnit.MILLILITER)

# Add these methods to UnitConverter class
UnitConverter.percentage_to_grams = percentage_to_grams
UnitConverter.grams_to_percentage = grams_to_percentage
UnitConverter.percentage_to_ml = percentage_to_ml


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'MassUnit',
    'VolumeUnit',
    'ConcentrationUnit',
    'MaterialDensity',
    'DensityDatabase',
    'UnitConverter',
    'MassConservationChecker',
    'BatchScaler',
    'format_quantity',
    'parse_quantity',
    'calculate_dilution'
]