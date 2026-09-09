"""Project reference composition; never supplies invented transport coefficients.

Only factual ratios and project choices are retained, not supplier prose/PDFs.
The adapted formula is unqualified for manufacture and is not a fitted model.
"""
from .application_context import ApplicationContext, BaseComponent, assess_application_context


REFERENCE_ID = "LC-CCT-OW-01"


def lotion_reference(reference_id=REFERENCE_ID):
    if reference_id != REFERENCE_ID:
        raise KeyError(reference_id)
    context = ApplicationContext(
        emulsion_type="oil_in_water",
        formula_reference="LC-CCT-OW-01 v1; project adaptation of Lotioncrafter Basic Lotion with Simulgel EG",
        base_components=[BaseComponent(name=name, mass_percent=percent, role=role) for name, percent, role in (
            ("Distilled water", 84., "water"), ("Glycerin", 3., "humectant"),
            ("Caprylic/Capric Triglyceride", 10., "oil"), ("Simulgel EG", 2., "emulsifier"),
            ("Lotioncrafter PE 9010", 1., "preservative"))])
    return {
        "schema_version": "lotion-reference-1", "reference_id": REFERENCE_ID, "version": 1,
        "status": "reference_composition_only", "application_context": context.model_dump(mode="json"),
        "assessment": assess_application_context(context),
        "sources": [
            {"url": "https://lotioncrafter.com/blogs/skin-care/basic-lotion-with-simulgel-eg",
             "kind": "supplier_formula", "accessed_on": "2026-09-07"},
            {"url": "https://lotioncrafter.com/products/lotioncrafter-pe-9010",
             "kind": "supplier_blend_information", "accessed_on": "2026-09-07"},
            {"url": "https://www.seppic.com/en-US/product/simulgel-eg",
             "kind": "manufacturer_ingredient_information", "accessed_on": "2026-09-07"}],
        "project_choices": ["Glycerin selected from published humectant alternatives.",
            "CCT selected for the unspecified oil; grade and C8/C10 ratio not established.",
            "Optional fragrance 0.5 percent replaced with water; no added fragrance does not mean no intrinsic odor.",
            "Percent column used: source preservative 1 percent and 5 grams per 400 grams are inconsistent."],
        "unresolved_model_inputs": ["application conditions and product density",
            "documented phase portions and densities for supplied blends",
            "context-specific material partition and transfer coefficients",
            "water loss and retained water fraction", "headspace geometry and ventilation"],
        "measured_release_data_available": False, "ready_to_simulate": False,
        "ready_to_optimize": False, "manufacturing_approved": False,
        "human_similarity_percent": None,
    }
